"""Task 1 numerical routines; no model downloads or data selection."""
from __future__ import annotations

import math
import torch
from common.generation import response_sequence_logprobs
from common.models import reference_mode
from task1_dpo.dpo import dpo_loss


def update_pair_counts(total_pairs, batch_size, accumulation_steps):
    """Actual denominators, including the final partial window."""
    if min(total_pairs, batch_size, accumulation_steps) <= 0:
        raise ValueError("Counts must be positive.")
    width = batch_size * accumulation_steps
    return [
        min(width, total_pairs - offset)
        for offset in range(0, total_pairs, width)
    ]


def device_batch(batch, device):
    return {key: tensor.to(device) for key, tensor in batch.items()}


def pair_scores(model, batches):
    return tuple(response_sequence_logprobs(model, batch)[0] for batch in batches)


def train_epoch(
    model, loader, optimizer, scaler, *,
    beta, total_pairs, batch_size, accumulation_steps, max_grad_norm, emit,
):
    """One epoch; equal pair weights within each optimizer window.

    No scheduler, autocast, retries, filtering, or early stopping.
    Invalid numerical results stop the run immediately.
    """
    targets = update_pair_counts(total_pairs, batch_size, accumulation_steps)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("The policy has no trainable parameters.")
    device = parameters[0].device
    seen = in_window = update_index = microbatches = 0
    loss_sum = correct = 0.0
    ordered_ids = []

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for chosen, rejected, ids in loader:
        count = len(ids)
        if count != min(batch_size, total_pairs - seen):
            raise RuntimeError("Unexpected batch size or dataset budget.")
        if (
            update_index >= len(targets)
            or in_window + count > targets[update_index]
        ):
            raise RuntimeError("Accumulation window does not match the budget.")

        batches = tuple(
            device_batch(b, device) for b in (chosen, rejected)
        )
        with torch.no_grad(), reference_mode(model):
            reference = pair_scores(model, batches)
        if not model.training:
            raise RuntimeError("Reference scoring did not restore training mode.")

        policy = pair_scores(model, batches)
        loss, diagnostics = dpo_loss(*policy, *reference, beta)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite DPO loss; stop and review.")

        margin = (
            (policy[0] - policy[1])
            - (reference[0] - reference[1])
        )
        emit({
            "event": "microbatch",
            "microbatch": microbatches + 1,
            "update": update_index + 1,
            "pairs": count,
            "prompt_ids": ids,
            "window_pairs": targets[update_index],
            "mean_loss": loss.item(),
            "policy_chosen_logp": policy[0].detach().cpu().tolist(),
            "policy_rejected_logp": policy[1].detach().cpu().tolist(),
            "reference_chosen_logp": reference[0].cpu().tolist(),
            "reference_rejected_logp": reference[1].cpu().tolist(),
            "reference_adjusted_margin": margin.detach().cpu().tolist(),
            "diagnostics": {k: v.item() for k, v in diagnostics.items()},
        })

        loss_sum += loss.item() * count
        correct += (margin.detach() > 0).sum().item()

        # Use the actual number of pairs, including the partial final window.
        scaler.scale(loss * count / targets[update_index]).backward()
        seen += count
        in_window += count
        microbatches += 1
        ordered_ids.extend(ids)
        del loss, diagnostics, margin, policy, reference, batches

        if in_window == targets[update_index]:
            scaler.unscale_(optimizer)
            bad = [
                i for i, p in enumerate(parameters)
                if p.grad is not None
                and not torch.isfinite(p.grad).all().item()
            ]
            if bad:
                emit({
                    "event": "nonfinite_gradients",
                    "update": update_index + 1,
                    "parameter_indices": bad,
                })
                raise FloatingPointError(
                    "Nonfinite unscaled gradients; stop and review."
                )

            norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_grad_norm, error_if_nonfinite=True,
            )
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()

            if new_scale < old_scale:
                raise FloatingPointError(
                    "Optimizer update skipped; stop and review."
                )
            if any(not torch.isfinite(p).all().item() for p in parameters):
                raise FloatingPointError(
                    "Nonfinite adapter weights; stop and review."
                )

            update_index += 1
            emit({
                "event": "optimizer_update",
                "update": update_index,
                "pairs": in_window,
                "seen_pairs": seen,
                "gradient_norm_before_clipping": norm.item(),
                "scale_before": old_scale,
                "scale_after": new_scale,
            })
            optimizer.zero_grad(set_to_none=True)
            in_window = 0

    if seen != total_pairs or in_window or update_index != len(targets):
        raise RuntimeError("Incomplete epoch cannot be a successful final policy.")

    return {
        "pairs": seen,
        "optimizer_updates": update_index,
        "microbatches": microbatches,
        "training_mean_loss": loss_sum / seen,
        "training_preference_accuracy": correct / seen,
        "ordered_prompt_ids": ordered_ids,
        "update_pair_counts": targets,
    }


class PairMetrics:
    """Strict reference-adjusted accuracy, with ties in the denominator."""
    def __init__(self, beta):
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        self.beta = beta
        self.count = self.correct = self.ties = 0
        self.loss_sum = 0.0

    def add(
        self, policy_chosen, policy_rejected,
        reference_chosen, reference_rejected,
    ):
        tensors = (
            policy_chosen, policy_rejected,
            reference_chosen, reference_rejected,
        )
        if any(not torch.isfinite(t).all().item() for t in tensors):
            raise FloatingPointError("Nonfinite pair scores")
        if (
            policy_chosen.numel() == 0
            or any(t.shape != policy_chosen.shape for t in tensors)
        ):
            raise ValueError("Pair score shapes differ or batch is empty")

        margin = (
            (policy_chosen - policy_rejected)
            - (reference_chosen - reference_rejected)
        )
        loss, _ = dpo_loss(*tensors, self.beta)
        count = margin.numel()
        self.count += count
        self.correct += (margin > 0).sum().item()
        self.ties += (margin == 0).sum().item()
        self.loss_sum += loss.item() * count

    def result(self):
        if self.count == 0:
            raise ValueError("Cannot evaluate an empty pool")
        return {
            "pairs": self.count,
            "beta": self.beta,
            "mean_dpo_loss": self.loss_sum / self.count,
            "preference_accuracy": self.correct / self.count,
            "correct_pairs": self.correct,
            "ties": self.ties,
        }


class SampledKL:
    """Global response-token-weighted sampled KL; finite negatives retained."""
    def __init__(self):
        self.logp_difference_sum = 0.0
        self.token_count = 0

    def add(self, policy_logp, reference_logp, response_mask):
        if (
            policy_logp.shape != reference_logp.shape
            or policy_logp.shape != response_mask.shape
        ):
            raise ValueError("Generated token score shapes differ")
        active = response_mask.bool()
        if not torch.all(
            (response_mask == 0) | (response_mask == 1)
        ).item():
            raise ValueError("Response mask must contain only zero and one")

        differences = (policy_logp - reference_logp)[active]
        if not torch.isfinite(differences).all().item():
            raise FloatingPointError("Nonfinite generated token log probabilities")

        self.logp_difference_sum += differences.double().sum().item()
        self.token_count += active.sum().item()

    def result(self):
        if self.token_count == 0:
            raise ValueError("No valid response tokens in evaluation pool")
        return {
            "sampled_kl": self.logp_difference_sum / self.token_count,
            "logp_difference_sum": self.logp_difference_sum,
            "valid_response_tokens": self.token_count,
        }
