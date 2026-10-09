"""Strict fixed-cache reconstruction and epsilon geometry; never trains a model."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import torch

from common.data import prompt_messages
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards
from task2_ppo.storage import atomic_json, commit_directory, publish_directory, restore_published, verify_directory
from task2_ppo.training_core import chosen_logprobs, finite


def reconstruct_cached(tokenizer, cached, row, cfg):
    if (row["prompt_id"], row["source_index"]) != (cached["prompt_id"], cached["source_index"]):
        raise RuntimeError("Cached prompt provenance differs")
    ids = tokenizer(cached["response"], add_special_tokens=False)["input_ids"]
    if cached["terminated_with_eos"]:
        ids += [tokenizer.eos_token_id]
    if len(ids) != int(cached["response_tokens"]):
        raise RuntimeError("Cached response token count differs; no trimming/padding is allowed")
    if tokenizer.decode(ids, skip_special_tokens=True) != cached["response"]:
        raise RuntimeError("Cached response text round-trip differs")
    if not ids or (cached["terminated_with_eos"] and ids[-1] != tokenizer.eos_token_id):
        raise RuntimeError("Cached EOS reconstruction differs")
    if any(token == tokenizer.eos_token_id for token in ids[:-1]):
        raise RuntimeError("Unexpected interior EOS in reconstructed cache")
    rendered = tokenizer.apply_chat_template(prompt_messages(row), tokenize=False, add_generation_prompt=True)
    enc = tokenizer([rendered], return_tensors="pt", padding=True, truncation=True,
                    max_length=cfg["max_prompt_length"])
    response = torch.tensor([ids], dtype=torch.long)
    width = enc["input_ids"].shape[1]
    sequences = torch.cat([enc["input_ids"], response], dim=1)
    attention = torch.cat([enc["attention_mask"], torch.ones_like(response)], dim=1)
    batch = {"sequences": sequences, "attention_mask": attention, "prompt_width": width,
             "response_ids": response, "response_mask": torch.ones_like(response, dtype=torch.float32)}
    return batch


def fixed_cached_advantages(cached, cfg):
    n = int(cached["response_tokens"])
    old = cached["old_logprobs"].float().reshape(1, n)
    ref = cached["ref_logprobs"].float().reshape(1, n)
    values = cached["values"].float().reshape(1, n)
    mask = torch.ones_like(old)
    effective = torch.tensor([cached["effective_terminal_reward"]], dtype=torch.float32)
    rewards = shaped_rewards(effective, old, ref, mask, .1)
    raw_adv, returns = compute_gae(rewards, values, mask, cfg["gamma"], cfg["gae_lambda"])
    # One response is one rollout, matching the release's one-prompt update unit.
    advantages = normalize_advantages(raw_adv, mask)
    for name, value in (("cache old", old), ("cache ref", ref), ("cache values", values),
                        ("cache advantages", advantages), ("cache returns", returns)):
        finite(name, value)
    return {"old_logprobs": old, "ref_logprobs": ref, "values": values,
            "advantages": advantages, "raw_advantages": raw_adv, "returns": returns,
            "shaped_rewards": rewards, "mask": mask}


def geometry_summary(records, epsilons=(.05, .2, .5)):
    total = sum(len(r["new_logprobs"]) for r in records)
    if total < 1:
        raise RuntimeError("No cached tokens")
    results = []
    for epsilon in epsilons:
        objective_sum, affected, active, unclipped = 0., 0, 0, 0.
        for row in records:
            new, old, advantage = (torch.tensor(row[k], dtype=torch.float32).reshape(1, -1)
                                   for k in ("new_logprobs", "old_logprobs", "advantages"))
            mask = torch.ones_like(old)
            loss, ratio, fraction = ppo_policy_loss(new, old, advantage, mask, epsilon)
            finite("cache geometry ratio", ratio)
            n = ratio.numel()
            objective_sum += -float(loss) * n
            unclipped += float((ratio * advantage).sum())
            affected += int(((ratio < 1. - epsilon) | (ratio > 1. + epsilon)).sum())
            active += int((ratio.clamp(1. - epsilon, 1. + epsilon) * advantage < ratio * advantage).sum())
        results.append({"epsilon": epsilon, "valid_tokens": total,
            "clipped_surrogate": objective_sum / total, "policy_loss": -objective_sum / total,
            "unclipped_surrogate": unclipped / total, "affected_token_count": affected,
            "affected_token_fraction": affected / total,
            "active_clipped_branch_count": active, "active_clipped_branch_fraction": active / total})
    return {"rows": len(records), "tokens": total, "beta_kl": .1,
            "advantage_normalization": "Once per cached response/rollout; same fixed advantages for every epsilon",
            "aggregation": "Masked token mean across all cached responses",
            "candidate": "standard final after exactly 20 updates", "results": results,
            "training_on_cache": False,
            "reconstruction_limit": "Recorded token counts and decoded text verified. Original token IDs are absent, so text/token-count reconstruction alone cannot prove uniqueness of the original tokenization."}


@torch.no_grad()
def score_cache(model, tokenizer, cached_rows, eval_rows, cfg, contract, output_root, persistent_root, budget, session_id=None):
    out, saved = Path(output_root), Path(persistent_root)
    out.mkdir(parents=True, exist_ok=True)
    saved.mkdir(parents=True, exist_ok=True)
    by_id = {r["prompt_id"]: r for r in eval_rows}
    records = []
    for index, cached in enumerate(cached_rows):
        budget.check()
        folder = out / f"row_{index:04d}"
        if not folder.exists() and (saved / (folder.name + ".zip")).exists():
            restore_published(saved / (folder.name + ".zip"), folder)
        if folder.exists():
            verify_directory(folder)
            record = json.loads((folder / "record.json").read_text())
            if record["contract"] != contract or record["prompt_id"] != cached["prompt_id"]:
                raise RuntimeError("Saved cache geometry provenance differs")
        else:
            budget.check()
            batch = reconstruct_cached(tokenizer, cached, by_id[cached["prompt_id"]], cfg)
            fixed = fixed_cached_advantages(cached, cfg)
            device = next(model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            new_logp, logits = chosen_logprobs(model, batch)
            del logits
            finite("cached candidate logprobs", new_logp)
            record = {"row_index": index, "prompt_id": cached["prompt_id"], "source_index": cached["source_index"],
                "contract": contract, "environment_session_id": session_id,
                "response_token_ids": batch["response_ids"][0].cpu().tolist(),
                "new_logprobs": new_logp[0].cpu().tolist(),
                **{k: v[0].cpu().tolist() for k, v in fixed.items() if k != "mask"}}
            with tempfile.TemporaryDirectory(prefix=".cache_", dir=out) as temporary:
                atomic_json(Path(temporary) / "record.json", record)
                commit_directory(temporary, folder)
        publish_directory(folder, saved)
        records.append(record)
    summary = {"contract": contract, **geometry_summary(records, cfg["clip_values"])}
    atomic_json(out / "summary.json", summary)
    atomic_json(saved / "summary.json", summary)
    print(f"Cache geometry complete: {len(records)} fixed rows, all three epsilon values", flush=True)
    return summary
