"""Task 3 trainables/AdamW/RNG persistence; excludes frozen model weights."""
from __future__ import annotations
import random
import numpy as np
import torch


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


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def snapshot(policy, optimizer, completed, contract):
    return {"format": "ATML_TASK3_V1", "completed_updates": completed, "contract": contract,
            "policy": trainable_state(policy), "optimizer": cpu_tree(optimizer.state_dict()),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}


def restore(state, policy, optimizer, contract):
    if state["format"] != "ATML_TASK3_V1" or state["contract"] != contract:
        raise RuntimeError("Checkpoint run contract differs")
    parameters = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    if parameters.keys() != state["policy"].keys():
        raise RuntimeError("Checkpoint trainable names differ")
    with torch.no_grad():
        for name, p in parameters.items():
            saved = state["policy"][name]
            if p.shape != saved.shape or p.dtype != saved.dtype or not torch.isfinite(saved).all():
                raise RuntimeError("Checkpoint trainable shape/dtype/finiteness differs: " + name)
            p.copy_(saved.to(p.device))
    optimizer.load_state_dict(state["optimizer"])
    for p, values in optimizer.state.items():
        for k in ("exp_avg", "exp_avg_sq"):
            if values[k].dtype != p.dtype or not torch.isfinite(values[k]).all():
                raise RuntimeError("Restored optimizer precision/finiteness differs")
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"]:
        if len(state["rng"]["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Saved CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return int(state["completed_updates"])


def optimizer_audit(optimizer):
    return [{"lr": g["lr"], "weight_decay": g["weight_decay"], "betas": list(g["betas"]),
             "eps": g["eps"], "parameter_dtypes": sorted({str(p.dtype) for p in g["params"]}),
             "moment_dtypes": sorted({str(v.dtype) for p in g["params"]
                                      for k, v in optimizer.state[p].items() if k in ("exp_avg", "exp_avg_sq")})}
            for g in optimizer.param_groups]
