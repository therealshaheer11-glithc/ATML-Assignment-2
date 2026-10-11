"""Approved critic-free math and one-epoch update; tiny-CPU-model testable."""
from __future__ import annotations
import torch
from common.generation import response_token_logprobs, batch_generate
from common.logging_utils import set_seed
from common.metrics import masked_mean, sampled_kl, sample_entropy
from common.models import reference_mode, trainable_parameters
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def finite(name, tensor):
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError("Nonfinite " + name + "; no silent skip or stabilization")


def ordinary(batch):
    return {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def chosen_logprobs(policy, batch):
    return response_token_logprobs(policy, batch["sequences"], batch["attention_mask"],
                                  batch["prompt_width"], batch["response_ids"])


@torch.no_grad()
def categorical_entropy(logits, mask):
    p = torch.log_softmax(logits.detach().float(), dim=-1)
    return masked_mean(-(p.exp() * p).sum(-1), mask)


def merge_completions(batches, pad_id):
    if not batches or len({b["prompt_width"] for b in batches}) != 1:
        raise ValueError("Expected completions with identical prompt widths")
    width = batches[0]["prompt_width"]
    n = max(b["response_ids"].shape[1] for b in batches)
    result = {"prompt_width": width}
    for name in ("sequences", "attention_mask", "response_ids", "response_mask"):
        tensors = []
        for b in batches:
            t = b[name]
            if t.shape[0] != 1:
                raise ValueError("Per-completion seed semantics require one generated row")
            extra = n - b["response_ids"].shape[1]
            value = pad_id if name in ("sequences", "response_ids") else 0
            tensors.append(torch.nn.functional.pad(t, (0, extra), value=value))
        result[name] = torch.cat(tensors)
    for name in ("responses", "terminated_with_eos", "truncated", "response_lengths"):
        result[name] = [v for b in batches for v in b[name]]
    result["attention_mask"][:, width:] = result["response_mask"].long()
    return result


def generate_group(policy, tokenizer, messages, entry, cfg):
    batches = []
    if len(entry["completion_seeds"]) != cfg["num_generations"]:
        raise RuntimeError("Group generation seeds differ")
    for seed in entry["completion_seeds"]:
        set_seed(seed)
        batches.append(ordinary(batch_generate(policy, tokenizer, [messages],
            max_prompt_length=cfg["max_prompt_length"], max_new_tokens=cfg["max_completion_length"],
            **cfg["generation"])))
    return merge_completions(batches, tokenizer.pad_token_id)


@torch.no_grad()
def freeze_rollout(policy, batch, rewards, cfg, group_ids=None):
    if policy.training:
        raise RuntimeError("Approved forward mode requires eval()")
    mask = batch["response_mask"].float()
    if mask.ndim != 2 or (mask.sum(-1) < 1).any() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Empty or nonbinary generation mask")
    if (mask[:, 1:] > mask[:, :-1]).any():
        raise ValueError("Generation mask must be a contiguous prefix")
    old, logits = chosen_logprobs(policy, batch)
    entropy = categorical_entropy(logits, mask)
    del logits
    with reference_mode(policy):
        ref, logits = chosen_logprobs(policy, batch)
        del logits
    if rewards.shape != (mask.shape[0],):
        raise ValueError("One reward per completion is required")
    ids = torch.zeros_like(rewards, dtype=torch.long) if group_ids is None else group_ids
    advantages = group_relative_advantages(rewards, ids)
    loss_mask = mask_truncated_sequences(mask, batch["truncated"])
    fixed = {"old_logprobs": old.float(), "ref_logprobs": ref.float(), "rewards": rewards.float(),
             "advantages": advantages, "group_ids": ids, "loss_mask": loss_mask}
    for name, value in fixed.items():
        finite(name, value)
    lengths = mask.sum(-1)
    stds = torch.stack([rewards[ids == g].float().std(unbiased=False) for g in torch.unique(ids)])
    return {k: v.detach().clone() for k, v in fixed.items()}, {
        "raw_reward": float(rewards.float().mean()),
        "reference_kl": float(sampled_kl(old, ref, mask)),
        "sampled_entropy": float(sample_entropy(old, mask)), "categorical_entropy": float(entropy),
        "group_reward_std": float(stds.mean()), "uninformative_group_fraction": float((stds <= 1e-6).float().mean()),
        "informative_group_fraction": float((stds > 1e-6).float().mean()),
        "response_length_mean": float(lengths.mean()), "response_length_std_population": float(lengths.std(unbiased=False)),
        "valid_generated_tokens": int(mask.sum()), "loss_bearing_tokens": int(loss_mask.sum()),
        "truncated_completions": sum(batch["truncated"]), "completions": len(lengths)}


def length_gradient_records(logp, fixed, cfg, loss_type):
    loss, diagnostic = grpo_policy_loss(logp, fixed["old_logprobs"], fixed["advantages"],
        fixed["loss_mask"], fixed["ref_logprobs"], cfg["clip_epsilon"], cfg["kl_beta"],
        loss_type=loss_type, max_completion_length=cfg["max_completion_length"])
    # policy_term is detached in the public diagnostic; recompute with beta=0.
    policy_term, _ = grpo_policy_loss(logp, fixed["old_logprobs"], fixed["advantages"],
        fixed["loss_mask"], fixed["ref_logprobs"], cfg["clip_epsilon"], 0.,
        loss_type=loss_type, max_completion_length=cfg["max_completion_length"])
    gradient, = torch.autograd.grad(policy_term, logp, retain_graph=True)
    records = []
    for i, (g, mask, advantage) in enumerate(zip(gradient, fixed["loss_mask"], fixed["advantages"])):
        n = int(mask.sum())
        records.append({"completion_index": i, "loss_type": loss_type, "valid_loss_tokens": n,
            "length_bin": "masked" if n == 0 else "1-128" if n <= 128 else "129-256" if n <= 256 else "257-512",
            "advantage": float(advantage), "advantage_sign": "positive" if advantage > 0 else "negative" if advantage < 0 else "zero",
            "policy_logprob_gradient_l2": float(torch.linalg.vector_norm(g.detach() * mask)),
            "definition": "L2 of d(policy surrogate)/d(response-token logp), excluding KL"})
    return loss, diagnostic, records


def optimize_rollout(policy, optimizer, batch, fixed, cfg, loss_type):
    if policy.training or cfg["policy_epochs"] != 1:
        raise RuntimeError("Approved eval forward mode and exactly one policy epoch required")
    if any(v.requires_grad for v in fixed.values()):
        raise RuntimeError("Old/reference probabilities and advantages must be detached")
    optimizer.zero_grad(set_to_none=True)
    logp, logits = chosen_logprobs(policy, batch)
    finite("new log-probabilities", logp)
    entropy = categorical_entropy(logits, batch["response_mask"])
    del logits
    loss, diagnostics, gradients = length_gradient_records(logp, fixed, cfg, loss_type)
    finite("loss", loss)
    loss.backward()
    parameters = trainable_parameters(policy)
    for p in parameters:
        if p.grad is None:
            raise RuntimeError("Missing trainable policy gradient")
        finite("policy gradient", p.grad)
    norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["max_grad_norm"], error_if_nonfinite=True)
    finite("pre-clipping gradient norm", norm)
    result = {k: float(v) for k, v in diagnostics.items()}
    result.update({"loss": float(loss.detach()), "gradient_norm_preclip": float(norm),
        "categorical_entropy": float(entropy), "length_gradient_statistics": gradients,
        "ratio_max_abs_change_before_update": float((torch.exp(logp.detach() - fixed["old_logprobs"]) - 1).abs()[batch["response_mask"].bool()].max()),
        "all_completions_masked": bool(fixed["loss_mask"].sum() == 0)})
    optimizer.step()
    for p in parameters:
        finite("policy parameter after AdamW", p)
        for k, value in optimizer.state[p].items():
            if isinstance(value, torch.Tensor):
                finite("AdamW state " + k, value)
    optimizer.zero_grad(set_to_none=True)
    return result
