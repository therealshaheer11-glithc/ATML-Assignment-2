"""DPO objective corrected against the PA2 equation.

Use summed response-token log-probabilities.
Reference values are constants. Zero adjusted margins are ties.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta: float):
    if beta <= 0:
        raise ValueError("DPO beta must be positive.")

    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp.detach() - ref_rejected_logp.detach()
    margin = policy_margin - ref_margin
    logits = float(beta) * margin
    loss = -F.logsigmoid(logits).mean()

    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "reference_margin_mean": ref_margin.detach().mean(),
        "adjusted_margin_mean": margin.detach().mean(),
        "preference_accuracy": (margin > 0).float().mean().detach(),
    }
