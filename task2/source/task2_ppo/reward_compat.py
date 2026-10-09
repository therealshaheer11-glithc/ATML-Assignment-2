"""Approved, in-memory default-RoPE format translation; no weights loaded here.

Adapted from this assignment's Task 1 reward-loading compatibility check.
"""
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.modeling_rope_utils import _compute_default_rope_parameters


def compatible_reward_config(path):
    if importlib.metadata.version("transformers") != "4.57.1":
        raise RuntimeError("Reward translation requires pinned Transformers 4.57.1.")
    path = Path(path)
    before_bytes = path.read_bytes()
    raw = json.loads(before_bytes)
    rope = raw.get("rope_parameters")
    if (raw.get("model_type") != "qwen2" or
            raw.get("architectures") != ["Qwen2ForSequenceClassification"] or
            not isinstance(rope, dict) or set(rope) != {"rope_type", "rope_theta"} or
            rope["rope_type"] != "default"):
        raise ValueError("Reward config differs from the approved default-RoPE format.")
    theta = float(rope["rope_theta"])
    if not math.isfinite(theta) or theta != 1_000_000.:
        raise ValueError("Reward theta differs from the approved checkpoint.")
    if raw.get("rope_theta") is not None and float(raw["rope_theta"]) != theta:
        raise ValueError("Conflicting positional configurations.")
    original = AutoConfig.from_pretrained(str(path.parent), local_files_only=True)
    if original.num_labels != 1 or original.rope_scaling is not None:
        raise ValueError("Reward head or RoPE scaling differs from the approved checkpoint.")
    effective = copy.deepcopy(original)
    effective.rope_theta = theta
    before, after = original.to_dict(), effective.to_dict()
    changed = {key: {"before": before.get(key), "after": after.get(key)}
               for key in before.keys() | after.keys()
               if before.get(key) != after.get(key)}
    if set(changed) - {"rope_theta"}:
        raise RuntimeError("Unexpected configuration changes.")
    actual, _ = _compute_default_rope_parameters(effective, torch.device("cpu"))
    head_dim = getattr(effective, "head_dim", None) or effective.hidden_size // effective.num_attention_heads
    dim = int(head_dim * getattr(effective, "partial_rotary_factor", 1.))
    expected = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if path.read_bytes() != before_bytes:
        raise RuntimeError("Original reward configuration was modified.")
    return effective, {
        "raw_config_sha256": hashlib.sha256(before_bytes).hexdigest(),
        "changed_fields": changed, "declared_theta": theta,
        "effective_config": after, "position_frequencies_match_checkpoint": True,
        "original_file_unchanged": True, "model_weights_loaded": False,
        "approval": "User approved Task 2 decisions A-K on 2026-10-08.",
    }
