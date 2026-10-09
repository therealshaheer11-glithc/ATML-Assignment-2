"""Approved PPO math and two-epoch optimization; usable with tiny CPU models.

Reuses the course's corrected objective, GAE, masks, and raw token log-probs.
No scheduler, entropy bonus, value clipping, extra KL loss, or gradient scaler.
"""
from __future__ import annotations

import math

import torch

from common.generation import response_token_logprobs
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import reference_mode, trainable_parameters
from task2_ppo.critic_precision import token_values_fp32_head
from task2_ppo.ppo import (compute_gae, normalize_advantages, ppo_policy_loss,
                           shaped_rewards, value_mse_loss)


def finite(name, tensor):
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"Nonfinite {name}; no update was silently skipped")


def ordinary_rollout(generation):
    # generate() creates inference tensors. Autograd needs ordinary saved inputs.
    return {k: v.detach().clone() if isinstance(v, torch.Tensor) else v
            for k, v in generation.items()}


def response_values(value_model, batch):
    width, steps = int(batch["prompt_width"]), batch["response_ids"].shape[1]
    if width < 1:
        raise RuntimeError("A response must have at least one prompt state")
    all_values = token_values_fp32_head(value_model, batch["sequences"], batch["attention_mask"])
    values = all_values[:, width - 1:width - 1 + steps]
    if values.shape != batch["response_mask"].shape:
        raise RuntimeError("Critic response-state alignment differs")
    return values.float()


def chosen_logprobs(policy, batch):
    return response_token_logprobs(policy, batch["sequences"], batch["attention_mask"],
                                  batch["prompt_width"], batch["response_ids"])


@torch.no_grad()
def categorical_entropy(logits, mask):
    # Report exact distribution entropy separately from the release sampled estimate.
    logp = torch.log_softmax(logits.detach().float(), dim=-1)
    return masked_mean(-(logp.exp() * logp).sum(-1), mask)


@torch.no_grad()
def freeze_rollout(policy, value_model, batch, raw_rewards, cfg):
    if policy.training or value_model.training:
        raise RuntimeError("Approved PPO forward mode is eval() with gradients enabled later")
    mask = batch["response_mask"].float()
    if mask.ndim != 2 or (mask.sum(-1) < 1).any() or not ((mask == 0) | (mask == 1)).all():
        raise RuntimeError("Empty or non-binary response mask")
    if (mask[:, 1:] > mask[:, :-1]).any():
        raise RuntimeError("Response mask must be a contiguous valid prefix")
    old_logp, logits = chosen_logprobs(policy, batch)
    rollout_entropy = categorical_entropy(logits, mask)
    del logits
    with reference_mode(policy):
        ref_logp, ref_logits = chosen_logprobs(policy, batch)
        del ref_logits
    values = response_values(value_model, batch)
    raw_rewards = raw_rewards.detach().float()
    if raw_rewards.shape != (mask.shape[0],):
        raise RuntimeError("Reward count differs from rollout count")
    penalty = torch.tensor([not flag for flag in batch["terminated_with_eos"]],
                           device=raw_rewards.device, dtype=torch.float32)
    effective = raw_rewards - float(cfg["missing_eos_penalty"]) * penalty
    rewards = shaped_rewards(effective, old_logp.float(), ref_logp.float(), mask, cfg["kl_beta"])
    raw_adv, returns = compute_gae(rewards, values, mask, cfg["gamma"], cfg["gae_lambda"])
    advantages = normalize_advantages(raw_adv, mask)
    fixed = {"old_logprobs": old_logp.float(), "ref_logprobs": ref_logp.float(),
             "old_values": values, "raw_rewards": raw_rewards, "effective_rewards": effective,
             "shaped_rewards": rewards, "raw_advantages": raw_adv,
             "advantages": advantages, "returns": returns}
    for name, tensor in fixed.items():
        finite(name, tensor)
    fixed = {k: v.detach().clone() for k, v in fixed.items()}
    diagnostics = {
        "raw_reward": float(raw_rewards.mean()), "effective_reward": float(effective.mean()),
        "rollout_reference_kl": float(sampled_kl(old_logp, ref_logp, mask)),
        "sampled_entropy": float(sample_entropy(old_logp, mask)),
        "categorical_entropy": float(rollout_entropy),
        "old_value_mean": float(masked_mean(values, mask)),
        "return_mean": float(masked_mean(returns, mask)),
        "critic_rollout_mse": float(value_mse_loss(values, returns, mask)),
        "response_length": float(mask.sum(-1).mean()),
        "valid_generated_tokens": int(mask.sum()),
    }
    return fixed, diagnostics


def _check_gradients(params, label):
    for index, p in enumerate(params):
        if p.grad is None:
            raise RuntimeError(f"Missing {label} gradient for trainable parameter {index}")
        finite(f"{label} gradient {index}", p.grad)


def _check_optimizer(opt, label):
    for group in opt.param_groups:
        for p in group["params"]:
            finite(label + " parameter after AdamW", p)
            for key, value in opt.state[p].items():
                if isinstance(value, torch.Tensor):
                    finite(label + " AdamW state " + key, value)


def optimize_rollout(policy, value_model, policy_optimizer, value_optimizer, batch, fixed, cfg):
    """Keep old/ref log-probs, GAE advantages and returns fixed for both epochs."""
    if policy.training or value_model.training:
        raise RuntimeError("PPO optimization requires the approved eval forward mode")
    if int(cfg["ppo_epochs"]) != 2:
        raise RuntimeError("Release requires exactly two epochs per rollout")
    for value in fixed.values():
        if value.requires_grad:
            raise RuntimeError("Rollout statistics must be detached")
    actor_params, critic_params = trainable_parameters(policy), trainable_parameters(value_model)
    mask, records = batch["response_mask"].float(), []
    for epoch in range(2):
        policy_optimizer.zero_grad(set_to_none=True)
        value_optimizer.zero_grad(set_to_none=True)
        new_logp, logits = chosen_logprobs(policy, batch)
        actor_loss, ratio, clip_fraction = ppo_policy_loss(
            new_logp.float(), fixed["old_logprobs"], fixed["advantages"], mask, cfg["clip_epsilon"])
        finite("policy loss", actor_loss)
        finite("importance ratio", ratio)
        entropy = categorical_entropy(logits, mask)
        del logits
        actor_loss.backward()
        predicted = response_values(value_model, batch)
        mse = value_mse_loss(predicted, fixed["returns"], mask)
        critic_loss = float(cfg["value_coef"]) * mse
        finite("value loss", critic_loss)
        critic_loss.backward()
        # Check BOTH models before either optimizer changes parameters.
        _check_gradients(actor_params, "actor")
        _check_gradients(critic_params, "critic")
        actor_norm = torch.nn.utils.clip_grad_norm_(actor_params, cfg["max_grad_norm"], error_if_nonfinite=True)
        critic_norm = torch.nn.utils.clip_grad_norm_(critic_params, cfg["max_grad_norm"], error_if_nonfinite=True)
        finite("actor preclip norm", actor_norm)
        finite("critic preclip norm", critic_norm)
        record = {
            "ppo_epoch": epoch, "policy_loss": float(actor_loss.detach()),
            "value_mse": float(mse.detach()), "weighted_value_loss": float(critic_loss.detach()),
            "reference_kl_on_rollout": float(sampled_kl(new_logp.detach(), fixed["ref_logprobs"], mask)),
            "sampled_entropy": float(sample_entropy(new_logp.detach(), mask)),
            "categorical_entropy": float(entropy), "clip_fraction": float(clip_fraction),
            "ratio_mean": float(masked_mean(ratio, mask)),
            "ratio_max_abs_change": float((ratio[mask.bool()] - 1).abs().max()),
            "actor_grad_norm_preclip": float(actor_norm),
            "critic_grad_norm_preclip": float(critic_norm),
        }
        if not all(math.isfinite(v) for v in record.values()):
            raise FloatingPointError("Nonfinite PPO diagnostic")
        policy_optimizer.step()
        value_optimizer.step()
        _check_optimizer(policy_optimizer, "actor")
        _check_optimizer(value_optimizer, "critic")
        records.append(record)
    policy_optimizer.zero_grad(set_to_none=True)
    value_optimizer.zero_grad(set_to_none=True)
    return records
