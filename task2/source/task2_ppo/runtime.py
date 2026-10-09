"""Pinned offline model loading for the approved Task 2 continuation.

Loads weights only when called from a prepared CUDA job. All public and course
files are verified before deserialization. No fallback downloads or new defaults.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, BitsAndBytesConfig)
from transformers.modeling_rope_utils import _compute_default_rope_parameters

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.models import make_value_lora_config, trainable_parameters, value_parameter_groups
from task2_ppo.critic_precision import promote_critic_head
from task2_ppo.preflight import (make_schedule, verify_assets, verify_config,
                                  verify_environment, verify_source)
from task2_ppo.reward_compat import compatible_reward_config
from task2_ppo.storage import sha256


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_models(models_dir, approval):
    root = Path(models_dir).resolve()
    manifest_path = root / "model_manifest.json"
    preparation = json.loads(repo_path("docs/task2_cpu_preparation.json").read_text())
    require(sha256(manifest_path) == preparation["model_manifest_sha256"], "Prepared public-model manifest differs")
    manifest = json.loads(manifest_path.read_text())
    folders = {}
    for kind in ("policy", "reward"):
        entry = manifest[kind]
        require((entry["model_id"], entry["revision"]) ==
                (approval[kind + "_model"], approval[kind + "_revision"]), "Pinned model identity differs")
        folder = (root / entry["directory"]).resolve()
        require(root in folder.parents, "Prepared model folder is outside its root")
        require("config.json" in entry["sha256"] and
                any(n.endswith(".safetensors") for n in entry["sha256"]), "Prepared model lacks weights/config")
        for name, digest in entry["sha256"].items():
            path = (folder / name).resolve()
            require(folder in path.parents and path.is_file() and sha256(path) == digest,
                    f"Prepared model file differs: {kind}/{name}")
        for index in folder.glob("*.safetensors.index.json"):
            shards = set(json.loads(index.read_text())["weight_map"].values())
            require(shards <= entry["sha256"].keys(), "Weight index references missing shards")
        folders[kind] = folder
    return folders, {"model_manifest_sha256": sha256(manifest_path), "identities": {
        k: {"model_id": manifest[k]["model_id"], "revision": manifest[k]["revision"]}
        for k in folders}}


def validate_job(models_dir):
    approval = json.loads(repo_path("docs/task2_approval.json").read_text())
    precision = json.loads(repo_path("docs/task2_precision_approval.json").read_text())
    require(approval["approved"] and approval["approved_decisions"] == list("ABCDEFGHIJK"), "Missing A-K approval")
    require(precision["approved"] and precision["decision_id"] == "L", "Missing precision approval L")
    source = verify_source(approval)
    installed = json.loads(repo_path("docs/task2_chunk2_files.json").read_text())
    for name, digest in installed.items():
        require(sha256(repo_path(name)) == digest, f"Installed Chunk 2 source differs: {name}")
    cfg = load_yaml("configs/ppo.yaml")
    verify_config(cfg)
    environment = verify_environment()
    assets, train, evaluation = verify_assets(cfg, approval)
    schedule = json.loads(repo_path("docs/task2_schedule.json").read_text())
    require(schedule == make_schedule(train, evaluation, cfg["seed"]), "Frozen prompt schedule differs")
    models, model_audit = verify_models(models_dir, approval)
    return cfg, train, schedule, models, {
        "source": source, "assets": assets, "environment": environment,
        "public_models": model_audit, "approvals": [approval, precision],
        "chunk2_files": installed, "schedule_sha256": sha256(repo_path("docs/task2_schedule.json")),
    }


def _loading_ok(info, label):
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"{label} checkpoint was not loaded exactly: {info}")


def load_tokenizer_local(folder):
    tok = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, padding_side="left", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    require(tok.eos_token_id is not None and tok.pad_token_id is not None, "Tokenizer lacks EOS/PAD")
    tok.padding_side = "left"
    return tok


def load_actor_critic(cfg, folders):
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer_local(folders["policy"])
    base, policy_info = AutoModelForCausalLM.from_pretrained(
        str(folders["policy"]), dtype=torch.float16, low_cpu_mem_usage=True,
        local_files_only=True, output_loading_info=True)
    _loading_ok(policy_info, "Policy base")
    base.config.pad_token_id = tokenizer.pad_token_id
    policy = PeftModel.from_pretrained(base, str(repo_path(cfg["paths"]["ppo_midpoint_policy"])),
                                       is_trainable=True, local_files_only=True)
    policy.cuda()
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.enable_input_require_grads()
    # The course loader enables checkpointing, but approved eval forward mode
    # disables its training-mode checkpoint branch. Do not silently change mode.
    policy.eval()
    critic_base, critic_info = AutoModelForSequenceClassification.from_pretrained(
        str(repo_path(cfg["paths"]["ppo_midpoint_value"])), num_labels=1,
        dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True, output_loading_info=True)
    _loading_ok(critic_info, "Merged midpoint critic")
    critic_base.config.pad_token_id = tokenizer.pad_token_id
    critic_base.config.use_cache = False
    critic = get_peft_model(critic_base, make_value_lora_config(cfg)).cuda().eval()
    precision_audit = promote_critic_head(critic)
    for label, model in (("policy", policy), ("critic", critic)):
        require(all(p.dtype == torch.float32 for p in trainable_parameters(model)),
                f"Unexpected trainable {label} precision; stop rather than silently cast")
    actor_optimizer = torch.optim.AdamW(trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]), betas=(.9, .999), eps=1e-8, weight_decay=.01)
    critic_optimizer = torch.optim.AdamW(value_parameter_groups(critic,
        lora_lr=cfg["value_lora_learning_rate"], head_lr=cfg["value_head_learning_rate"]),
        betas=(.9, .999), eps=1e-8, weight_decay=0.)
    audit = {"policy_loading": policy_info, "critic_loading": critic_info,
             "critic_precision": precision_audit, "forward_mode": "eval with gradients enabled",
             "policy_attention": policy.config._attn_implementation,
             "critic_attention": critic.config._attn_implementation,
             "policy_trainable_parameters": sum(p.numel() for p in trainable_parameters(policy)),
             "critic_trainable_parameters": sum(p.numel() for p in trainable_parameters(critic))}
    return policy, critic, tokenizer, actor_optimizer, critic_optimizer, audit


def load_reward_local(cfg, folders, tokenizer):
    effective, audit = compatible_reward_config(folders["reward"] / "config.json")
    require(effective.max_position_embeddings >= cfg["reward_max_length"], "Reward context is too small")
    model, info = AutoModelForSequenceClassification.from_pretrained(
        str(folders["reward"]), config=effective, low_cpu_mem_usage=True,
        local_files_only=True, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto", output_loading_info=True)
    _loading_ok(info, "Reward model")
    require(getattr(model, "is_loaded_in_8bit", False), "Reward must be loaded in released 8-bit mode")
    require(all(p.device.type == "cuda" for p in model.parameters()), "Reward offloading detected; stop for review")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval().requires_grad_(False)
    expected, _ = _compute_default_rope_parameters(effective, torch.device("cpu"))
    actual = model.model.rotary_emb.inv_freq.detach().float().cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    audit.update({"model_weights_loaded": True, "loading_info": info, "frozen": True,
                  "quantization": "8bit", "attention": model.config._attn_implementation,
                  "loaded_position_frequencies_match": True})
    return model, audit
