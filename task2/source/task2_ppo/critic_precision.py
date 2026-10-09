"""Task 2 decision L, approved 2026-10-08: float32 trainable value head.

Integrate after the released critic loader and BEFORE optimizer construction.
Use token_values_fp32_head instead of common.models.token_values for this critic.
The course source is not modified by this module.
"""
from __future__ import annotations

import torch
from peft import PeftModel


def _parts(value_model):
    base = value_model.get_base_model() if isinstance(value_model, PeftModel) else value_model
    if not hasattr(base, "score"):
        raise RuntimeError("Decision L expects the released Qwen scalar score head")
    head = base.score
    if hasattr(head, "modules_to_save"):
        if getattr(head, "disable_adapters", False):
            raise RuntimeError("The critic scalar-head adapter must be enabled")
        active = head.active_adapter
        if not isinstance(active, str) or active not in head.modules_to_save:
            raise RuntimeError("Expected one active saved critic scalar head")
        active_head = head.modules_to_save[active]
    else:
        active_head = head
    if getattr(active_head, "out_features", None) != 1:
        raise RuntimeError("Expected a scalar linear value head")
    backbone = getattr(base, base.base_model_prefix)
    return backbone, head, active_head


def promote_critic_head(value_model):
    """Preserve Parameter identities and values; call before creating AdamW."""
    _, _, active_head = _parts(value_model)
    params = list(active_head.parameters())
    if not params or any(not p.requires_grad for p in params):
        raise RuntimeError("Expected a fully trainable active scalar head")
    if any(p.grad is not None for p in params):
        raise RuntimeError("Promote the critic head before computing gradients")
    if any(p.dtype not in (torch.float16, torch.float32) for p in params):
        raise RuntimeError("Unexpected scalar-head precision")
    with torch.no_grad():
        for p in params:
            p.data = p.data.to(torch.float32)
    return critic_dtype_audit(value_model)


def critic_dtype_audit(value_model):
    """Return a serializable per-parameter audit for the run record."""
    return {
        "approval_id": "L",
        "parameters": [
            {"name": name, "dtype": str(p.dtype), "trainable": p.requires_grad,
             "numel": p.numel()}
            for name, p in value_model.named_parameters()
        ],
    }


def token_values_fp32_head(value_model, input_ids, attention_mask):
    """Values at each input state; caller performs the PPO causal alignment."""
    backbone, head, active_head = _parts(value_model)
    if any(p.dtype != torch.float32 for p in active_head.parameters()):
        raise RuntimeError("Apply approved decision L before scoring the critic")
    hidden = backbone(
        input_ids=input_ids, attention_mask=attention_mask,
        return_dict=True, use_cache=False,
    ).last_hidden_state
    return head(hidden.to(torch.float32)).squeeze(-1)
