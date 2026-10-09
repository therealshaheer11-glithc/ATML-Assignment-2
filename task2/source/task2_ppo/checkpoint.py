"""Save only trainable weights plus optimizer/RNG state, never frozen weights."""
from __future__ import annotations

import random

import numpy as np
import torch


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore_trainables(model, state):
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        raise RuntimeError("Saved trainable parameter names differ")
    with torch.no_grad():
        for name, p in parameters.items():
            saved = state[name]
            if p.shape != saved.shape or p.dtype != saved.dtype:
                raise RuntimeError(f"Checkpoint shape/dtype differs: {name}; promote head BEFORE restore")
            if not torch.isfinite(saved).all():
                raise FloatingPointError(f"Nonfinite checkpoint parameter: {name}")
            p.copy_(saved.to(p.device))


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def optimizer_dtype_audit(optimizer):
    return [{"group": group.get("name", "actor"), "learning_rate": group["lr"],
             "weight_decay": group["weight_decay"], "betas": list(group["betas"]), "eps": group["eps"],
             "parameter_dtypes": sorted({str(p.dtype) for p in group["params"]}),
             "state_dtypes": {key: sorted({str(optimizer.state[p][key].dtype)
                                          for p in group["params"] if key in optimizer.state[p]})
                              for key in ("step", "exp_avg", "exp_avg_sq")}}
            for group in optimizer.param_groups]


def snapshot(policy, critic, actor_optimizer, critic_optimizer, completed_updates, contract):
    return {"format": "ATML_TASK2_TRAINABLES_V1", "completed_updates": completed_updates,
            "contract": contract, "policy": trainable_state(policy), "critic": trainable_state(critic),
            "actor_optimizer": cpu_tree(actor_optimizer.state_dict()),
            "critic_optimizer": cpu_tree(critic_optimizer.state_dict()),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}


def restore(state, policy, critic, actor_optimizer, critic_optimizer, contract):
    if state["format"] != "ATML_TASK2_TRAINABLES_V1" or state["contract"] != contract:
        raise RuntimeError("Saved run contract differs; checkpoint not restored")
    restore_trainables(policy, state["policy"])
    restore_trainables(critic, state["critic"])
    actor_optimizer.load_state_dict(state["actor_optimizer"])
    critic_optimizer.load_state_dict(state["critic_optimizer"])
    for opt in (actor_optimizer, critic_optimizer):
        for p, values in opt.state.items():
            if any(values[k].dtype != p.dtype for k in ("exp_avg", "exp_avg_sq")):
                raise RuntimeError("Restored AdamW moments differ from parameter precision")
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"]:
        if len(state["rng"]["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Saved CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return int(state["completed_updates"])
