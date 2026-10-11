from __future__ import annotations

import torch
import math

from common.metrics import masked_mean, sampled_kl, sample_entropy


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6):
    """Return one scalar advantage per sampled completion.

    `group_ids[i]` identifies which prompt produced reward `rewards[i]`.
    Validate this implementation against the group-relative definition in the assignment manual.
    """
    if rewards.ndim != 1 or group_ids.shape != rewards.shape or not rewards.numel():
        raise ValueError("Expected nonempty one-dimensional rewards and matching group IDs")
    if not torch.isfinite(rewards).all() or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Finite rewards and positive epsilon are required")
    if not torch.isfinite(group_ids).all():
        raise ValueError("Finite prompt-group IDs are required")
    rewards = rewards.detach().float()
    if not torch.isfinite(rewards).all():
        raise ValueError("Rewards cannot be represented in the approved float32 statistics")
    group_ids = group_ids.to(rewards.device)
    advantages = torch.empty_like(rewards)
    for group in torch.unique(group_ids):
        selected = group_ids == group
        values = rewards[selected]
        # PA p. 9: population statistics WITHIN one prompt, sigma + epsilon.
        advantages[selected] = (values - values.mean()) / (values.std(unbiased=False) + eps)
    return advantages


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    `token_mask` may be all-zero for a completion that was deliberately masked because it hit the
    maximum generation length.
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask).sum(-1)
    if loss_type == "grpo":
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        # Constant normalization rather than dividing by each response's realized length.
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "loss_kl_penalty": kl.detach(),
        "reference_kl": sampled_kl(new_logp.detach(), ref_logp, token_mask).detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
