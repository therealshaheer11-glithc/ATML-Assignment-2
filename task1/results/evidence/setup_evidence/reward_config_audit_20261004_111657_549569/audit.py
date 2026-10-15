
"""Read-only CPU audit of pinned checkpoint configuration interpretation."""
from pathlib import Path
import json, sys
import torch, transformers
from transformers import AutoConfig
from transformers.modeling_rope_utils import _compute_default_rope_parameters

assert transformers.__version__ == "4.57.1", transformers.__version__
assert not torch.cuda.is_available(), "Keep GPU off for this configuration audit."
stage = Path(sys.argv[1])
report = {"status": "COMPATIBLE", "transformers": transformers.__version__,
          "torch": str(torch.__version__), "gpu_available": False,
          "model_weights_loaded": False, "settings_changed": False,
          "official_training_started": False, "models": {}}
for kind in ("policy", "reward"):
    folder = stage / "configs" / kind
    raw = json.loads((folder / "config.json").read_text())
    parsed = AutoConfig.from_pretrained(str(folder), local_files_only=True)
    assert parsed.model_type == "qwen2", "Unexpected checkpoint architecture."
    nested = raw.get("rope_parameters") or {}
    declared = raw.get("rope_theta", nested.get("rope_theta"))
    assert declared is not None and float(declared) > 0, "Checkpoint RoPE setting missing."
    assert nested.get("rope_type", "default") == "default" and parsed.rope_scaling is None
    actual, factor = _compute_default_rope_parameters(parsed, torch.device("cpu"))
    head_dim = getattr(parsed, "head_dim", None) or parsed.hidden_size // parsed.num_attention_heads
    dim = int(head_dim * getattr(parsed, "partial_rotary_factor", 1.0))
    expected = 1.0 / (float(declared) ** (torch.arange(0, dim, 2).float() / dim))
    matches = bool(torch.equal(actual, expected))
    item = {"checkpoint_transformers_version": raw.get("transformers_version"),
            "checkpoint_rope_theta": float(declared),
            "parsed_rope_theta": float(parsed.rope_theta),
            "checkpoint_rope_parameters": raw.get("rope_parameters"),
            "parsed_rope_scaling": parsed.rope_scaling,
            "rotary_frequency_vectors_match": matches,
            "maximum_frequency_difference": float((actual - expected).abs().max()),
            "context_capacity": int(parsed.max_position_embeddings),
            "architectures": parsed.architectures,
            "num_labels": int(parsed.num_labels)}
    report["models"][kind] = item
    if not matches:
        report["status"] = "REVIEW_REQUIRED"
    print(f"{kind}: checkpoint position setting={declared}; "
          f"library uses={parsed.rope_theta}; matches={matches}", flush=True)
report["interpretation"] = (
    "Pinned Transformers interprets a checkpoint position setting differently. "
    "No correction applied; review and approval required before reward scoring."
    if report["status"] == "REVIEW_REQUIRED" else
    "Both checkpoint position settings match the installed library interpretation.")
(stage / "config_audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print("REWARD_CONFIG_AUDIT_DONE: no weights loaded and no active settings changed.")
