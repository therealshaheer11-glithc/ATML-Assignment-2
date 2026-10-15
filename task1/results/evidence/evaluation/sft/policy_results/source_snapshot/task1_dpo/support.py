"""Task 1 run budgets, pinned loading and verified artifact copying.

Reuses course data, seed, LoRA and scoring infrastructure. No GPU training
starts on import. Download preparation is a separate CPU step.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil

import torch
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from common.data import repo_path, read_jsonl
from common.models import make_lora_config

DATA = {
    "dpo_standard_train": (1500, "8a0e52eb0927b261b35dab887f8f49f5d81bc666adb4621a91481ded277a3c42"),
    "dpo_length_train": (1500, "15e1126ef1befe35d1011a713d223ed28044f3cb0ab8cbe4acecc033fdbd9bbf"),
    "dpo_standard_eval": (300, "c1f9e7c7e140b4671efd67173a4434946780cf04232aabc03ef317ae621e5079"),
    "dpo_length_eval": (246, "89e2c56ee2fdaec75fcf1f56ae4802d710ed4ad08f127e097990923017ca7405"),
    "word_limit_prompts": (10, "4252fb01e078eaa835f9d4b3d6abaaba8033fe68e30dfa2d17c4d6886dd2ea49"),
}
RUNS = {
    "standard": ("dpo_standard_train", 1500, .10),
    "beta_003": ("dpo_standard_train", 600, .03),
    "beta_010": ("dpo_standard_train", 600, .10),
    "beta_030": ("dpo_standard_train", 600, .30),
    "length_balanced": ("dpo_length_train", 1500, .10),
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def verified_copy(source, destination):
    """Copy to a new folder and verify every source file; never overwrite."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Preserve existing backup: {destination}")
    source_resolved, destination_resolved = source.resolve(), destination.resolve()
    if destination_resolved == source_resolved or source_resolved in destination_resolved.parents:
        raise ValueError("Backup cannot be inside its source.")
    before = {str(p.relative_to(source)): file_sha(p)
              for p in sorted(source.rglob("*")) if p.is_file()}
    if not before:
        raise ValueError("Refuse an empty artifact backup.")
    shutil.copytree(source, destination)
    after = {str(p.relative_to(destination)): file_sha(p)
             for p in sorted(destination.rglob("*")) if p.is_file()}
    unchanged = {str(p.relative_to(source)): file_sha(p)
                 for p in sorted(source.rglob("*")) if p.is_file()}
    if before != after or before != unchanged:
        raise IOError("Artifact copy verification failed; do not release runtime.")
    return {"status": "BACKUP_VERIFIED", "files": len(before), "sha256": before,
            "source": str(source), "destination": str(destination)}


def validate_config(cfg):
    fixed = {"seed": 6304, "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
             "reward_model": "yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback",
             "reward_tokenizer": "Qwen/Qwen2.5-1.5B-Instruct", "quantize_frozen_models": True,
             "dtype": "float16", "batch_size": 2, "grad_accum_steps": 8,
             "epochs": 1, "max_sequence_length": 4096, "max_prompt_length": 4096,
             "reward_max_length": 4096, "max_generation_tokens": 256,
             "generation_batch_size": 1, "learning_rate": 2e-5,
             "weight_decay": 0., "max_grad_norm": 1., "short_ablation_examples": 600,
             "base_model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
             "reward_tokenizer_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
             "reward_model_revision": "f98759a4a1ccdc47a3136748b6bfea9a32ba8fcc"}
    for key, expected in fixed.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Unapproved configuration: {key}={cfg.get(key)!r}")
    lc, gen = cfg["lora"], cfg["generation"]
    if (lc["r"], lc["alpha"], lc["dropout"], set(lc["target_modules"])) != (8, 16, .05, {"q_proj", "v_proj"}):
        raise ValueError("LoRA configuration differs from approval.")
    if (gen["do_sample"], gen["temperature"], gen["top_p"]) != (True, .7, .9):
        raise ValueError("Decoding configuration differs from approval.")
    if list(cfg["betas"]) != [.03, .10, .30] or cfg["beta"] != .10:
        raise ValueError("DPO beta protocol differs from approval.")


def validated_rows(cfg, path_key):
    count, expected = DATA[path_key]
    path = repo_path(cfg["paths"][path_key])
    if file_sha(path) != expected:
        raise ValueError(f"Released dataset hash mismatch: {path}")
    rows = read_jsonl(path)
    if len(rows) != count:
        raise ValueError(f"Released dataset count mismatch: {path}")
    return rows


def run_plan(cfg, name):
    validate_config(cfg)
    if name not in RUNS:
        raise ValueError(f"Select one of {list(RUNS)}")
    path_key, count, beta = RUNS[name]
    rows = validated_rows(cfg, path_key)[:count]
    output = (cfg["standard_output"] if name == "standard" else
              cfg["length_output"] if name == "length_balanced" else
              str(Path(cfg["standard_output"]).parent / name))
    return rows, {"name": name, "path_key": path_key, "pairs": count,
                  "beta": beta, "epochs": 1, "output": output,
                  "dataset_sha256": DATA[path_key][1],
                  "selection": "original first 600 before shuffle" if count == 600 else "all released rows"}


def model_source(cfg, models_dir=None):
    """Use a pinned HF cache, or a CPU-prepared flat snapshot directory."""
    if models_dir is None:
        return cfg["base_model"], {"revision": cfg["base_model_revision"], "local_files_only": True}
    root = Path(models_dir)
    manifest = json.loads((root / "model_manifest.json").read_text())
    entry = manifest["policy"]
    if (entry["model_id"], entry["revision"]) != (cfg["base_model"], cfg["base_model_revision"]):
        raise ValueError("Prepared policy snapshot has a different identity.")
    folder = (root / entry["directory"]).resolve()
    if root.resolve() not in folder.parents:
        raise ValueError("Invalid snapshot directory.")
    for name, digest in entry["sha256"].items():
        path = (folder / name).resolve()
        if folder not in path.parents or file_sha(path) != digest:
            raise ValueError(f"Prepared policy snapshot changed: {name}")
    return str(folder), {"local_files_only": True}


def load_pinned_tokenizer(cfg, source, kwargs):
    tok = AutoTokenizer.from_pretrained(source, use_fast=True, padding_side="left", **kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.eos_token_id is None:
        raise ValueError("Task 1 requires the verified EOS token.")
    provenance = json.loads(repo_path("docs/task1_tokenizer_manifest.json").read_text())
    if (provenance["model_id"], provenance["resolved_revision"]) != (cfg["base_model"], cfg["base_model_revision"]):
        raise ValueError("Tokenizer provenance differs from the approved model.")
    if hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest() != provenance["tokenizer_backend_sha256"]:
        raise ValueError("Tokenizer backend differs from the completed data audit.")
    if hashlib.sha256(json.dumps(tok.chat_template, sort_keys=True).encode()).hexdigest() != provenance["chat_template_sha256"]:
        raise ValueError("Chat template differs from the completed data audit.")
    if (tok.pad_token_id, tok.eos_token_id, tok.padding_side, tok.truncation_side) != (
        provenance["pad_token_id"], provenance["eos_token_id"], provenance["padding_side"], provenance["truncation_side"]):
        raise ValueError("Tokenizer special tokens or padding/truncation sides differ.")
    return tok


def load_pinned_policy(cfg, source, kwargs, tokenizer, *, trainable=False, adapter=None):
    if not torch.cuda.is_available():
        raise RuntimeError("Official model operations require the approved GPU runtime.")
    if str(torch.__version__).split("+")[0] != "2.11.0":
        raise RuntimeError("GPU PyTorch differs from the validated 2.11.0 environment; stop and review.")
    for package, expected in {"transformers": "4.57.1", "tokenizers": "0.22.1", "peft": "0.17.1", "bitsandbytes": "0.50.2"}.items():
        if metadata.version(package) != expected:
            raise RuntimeError(f"Unexpected GPU package version: {package}; stop and review.")
    base = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.float16, low_cpu_mem_usage=True, **kwargs,
    )
    if "revision" in kwargs and base.config._commit_hash != cfg["base_model_revision"]:
        raise ValueError("Resolved model revision differs from approval.")
    base.config.pad_token_id = tokenizer.pad_token_id
    if int(base.config.max_position_embeddings) < int(cfg["max_prompt_length"]) + int(cfg["max_generation_tokens"]):
        raise ValueError("Policy context capacity is too small for approved generation limits.")
    if trainable:
        if adapter is not None:
            raise ValueError("Each training run must start with fresh adapters.")
        model = get_peft_model(base, make_lora_config(cfg))
    elif adapter is not None:
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    else:
        model = base
        model.requires_grad_(False)
    if isinstance(model, PeftModel):
        model.peft_config["default"].base_model_name_or_path = cfg["base_model"]
        model.peft_config["default"].revision = cfg["base_model_revision"]
    model = model.cuda()
    if trainable:
        model.train()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not named or any("lora_" not in n or p.dtype != torch.float32 for n, p in named):
            raise ValueError("Only FP32 LoRA parameters may train.")
    else:
        model.eval()
        model.requires_grad_(False)
    if any(p.dtype != torch.float16 for n, p in model.named_parameters() if "lora_" not in n):
        raise ValueError("Frozen base must remain FP16.")
    return model


def parameter_digest(model, *, adapters=False):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ("lora_" in name) != adapters:
            continue
        digest.update(name.encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(str(tuple(parameter.shape)).encode())
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()
