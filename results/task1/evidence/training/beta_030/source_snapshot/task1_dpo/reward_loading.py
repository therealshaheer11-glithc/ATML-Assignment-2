"""Pinned course reward loading with the user-approved RoPE format correction.

The immutable checkpoint declares default RoPE theta in a newer nested field.
Translate only that field in memory for Transformers 4.57.1. No model loads
on import; the real 8-bit reward model requires the approved GPU environment.
"""
import copy
import importlib.metadata as metadata
import json
import math
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, BitsAndBytesConfig
from transformers.modeling_rope_utils import _compute_default_rope_parameters
from common.data import repo_path
from task1_dpo.support import file_sha, load_pinned_tokenizer, model_source, validate_config


def compatible_reward_config(path):
    """CPU-only configuration translation; never modify the source file."""
    if metadata.version("transformers") != "4.57.1":
        raise RuntimeError("Configuration correction is validated for Transformers 4.57.1 only.")
    path = Path(path)
    digest = file_sha(path)
    raw = json.loads(path.read_text())
    rope = raw.get("rope_parameters")
    if (raw.get("model_type") != "qwen2" or
            raw.get("architectures") != ["Qwen2ForSequenceClassification"] or
            not isinstance(rope, dict) or set(rope) != {"rope_type", "rope_theta"} or
            rope["rope_type"] != "default"):
        raise ValueError("Reward configuration differs from the audited default-RoPE format.")
    theta = float(rope["rope_theta"])
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("Invalid checkpoint-declared RoPE theta.")
    if raw.get("rope_theta") is not None and float(raw["rope_theta"]) != theta:
        raise ValueError("Conflicting nested and top-level RoPE settings.")
    original = AutoConfig.from_pretrained(str(path.parent), local_files_only=True)
    if original.num_labels != 1 or original.rope_scaling is not None:
        raise ValueError("Reward head or RoPE scaling differs from the audited checkpoint.")
    effective = copy.deepcopy(original)
    effective.rope_theta = theta
    before, after = original.to_dict(), effective.to_dict()
    changed = {k: {"before": before.get(k), "after": after.get(k)}
               for k in set(before) | set(after) if before.get(k) != after.get(k)}
    if set(changed) - {"rope_theta"}:
        raise RuntimeError("Configuration translation changed an unapproved field.")
    frequencies, _ = _compute_default_rope_parameters(effective, torch.device("cpu"))
    head_dim = getattr(effective, "head_dim", None) or effective.hidden_size // effective.num_attention_heads
    dim = int(head_dim * getattr(effective, "partial_rotary_factor", 1.0))
    expected = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    torch.testing.assert_close(frequencies, expected, rtol=0, atol=0)
    if file_sha(path) != digest:
        raise RuntimeError("Original reward configuration changed during validation.")
    return effective, {"raw_config_sha256": digest, "raw_config": raw,
        "original_effective_config": before, "corrected_effective_config": after,
        "changed_fields": changed, "declared_theta": theta,
        "position_frequencies_match_checkpoint": True, "original_file_unchanged": True,
        "correction": "Copy declared nested default-RoPE theta into the supported top-level field in memory.",
        "user_approval": "okay i approve, make sure to document it, and proceed"}


def reward_source(cfg, models_dir):
    """Verify the prepared pinned snapshot; never download or accept new weights."""
    validate_config(cfg)
    root = Path(models_dir).resolve()
    preparation = json.loads(repo_path("docs/task1_model_preparation.json").read_text())
    manifest_path = root / "model_manifest.json"
    if file_sha(manifest_path) != preparation["model_manifest_sha256"]:
        raise ValueError("Prepared model manifest differs from verified CPU preparation.")
    entry = json.loads(manifest_path.read_text())["reward"]
    if (entry["model_id"], entry["revision"]) != (cfg["reward_model"], cfg["reward_model_revision"]):
        raise ValueError("Prepared reward identity differs from approval.")
    folder = (root / entry["directory"]).resolve()
    if root not in folder.parents or not {"config.json", "model.safetensors"} <= entry["sha256"].keys():
        raise ValueError("Reward snapshot is incomplete or outside the prepared folder.")
    for name, digest in entry["sha256"].items():
        path = (folder / name).resolve()
        if folder not in path.parents or not path.is_file() or file_sha(path) != digest:
            raise ValueError(f"Prepared reward file changed: {name}")
    return folder


def load_pinned_reward(cfg, models_dir):
    """Same course 8-bit frozen RM/tokenizer, with the approved config translation."""
    if not torch.cuda.is_available():
        raise RuntimeError("Reward model weights require the approved GPU runtime.")
    validate_config(cfg)
    if str(torch.__version__).split("+")[0] != "2.11.0":
        raise RuntimeError("GPU PyTorch differs from the validated 2.11.0 environment.")
    for name, version in {"transformers": "4.57.1", "tokenizers": "0.22.1",
                          "peft": "0.17.1", "bitsandbytes": "0.50.2"}.items():
        if metadata.version(name) != version:
            raise RuntimeError(f"Unexpected GPU package: {name}; stop and review.")
    folder = reward_source(cfg, models_dir)
    effective, record = compatible_reward_config(folder / "config.json")
    if effective.max_position_embeddings < cfg["reward_max_length"]:
        raise ValueError("Reward context capacity is below the approved input limit.")
    policy_source, kwargs = model_source(cfg, models_dir)
    tokenizer = load_pinned_tokenizer(cfg, policy_source, kwargs)
    quantization = BitsAndBytesConfig(load_in_8bit=True)
    # num_labels=1 is already validated in the explicit config; do not pass it
    # again as a constructor keyword. Preserve the course quantization defaults.
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        str(folder), config=effective, local_files_only=True, low_cpu_mem_usage=True,
        quantization_config=quantization, device_map="auto", output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Reward weights were not loaded completely: {loading}")
    if not getattr(model, "is_loaded_in_8bit", False):
        raise RuntimeError("Reward model was not loaded with course 8-bit quantization.")
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise RuntimeError("Reward parameters were offloaded; stop and review available GPU memory.")
    if model.config.rope_theta != record["declared_theta"] or model.config.num_labels != 1:
        raise RuntimeError("Loaded reward configuration differs from the validated correction.")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.requires_grad_(False)
    expected, _ = _compute_default_rope_parameters(effective, torch.device("cpu"))
    actual = model.model.rotary_emb.inv_freq.detach().float().cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    record.update({"reward_model": cfg["reward_model"], "reward_revision": cfg["reward_model_revision"],
        "tokenizer_model": cfg["reward_tokenizer"], "tokenizer_revision": cfg["reward_tokenizer_revision"],
        "quantization": quantization.to_dict(), "loading_info": loading,
        "loaded_effective_config": model.config.to_dict(),
        "attention_implementation": model.config._attn_implementation,
        "loaded_position_frequencies_match": True, "frozen": True})
    return model, tokenizer, record
