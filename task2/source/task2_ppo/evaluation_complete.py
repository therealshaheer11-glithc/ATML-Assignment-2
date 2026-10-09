"""Frozen 200-prompt evaluation with matched per-prompt sampling seeds.

One response at a time preserves approved seed semantics. No checkpoint choice
or training follows from these measurements. Async persistence overlaps CPU
archive copying with the next GPU generation; models stay on the main thread.
"""
from __future__ import annotations

import gc
import json
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from common.data import prompt_messages, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import reference_mode
from task2_ppo.runtime import _loading_ok
from task2_ppo.storage import atomic_json, commit_directory, publish_directory, restore_published, verify_directory
from task2_ppo.training_core import categorical_entropy, chosen_logprobs, finite, ordinary_rollout


def load_evaluation_policy(cfg, folders, tokenizer):
    base, loading = AutoModelForCausalLM.from_pretrained(
        str(folders["policy"]), dtype=torch.float16, local_files_only=True,
        low_cpu_mem_usage=True, output_loading_info=True)
    _loading_ok(loading, "Evaluation policy base")
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, str(repo_path(cfg["paths"]["ppo_midpoint_policy"])),
        adapter_name="midpoint", is_trainable=False, local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    return model, loading


def activate_adapter(model, name, output_root):
    if name != "midpoint" and name not in model.peft_config:
        final = Path(output_root) / name / "final"
        verify_directory(final)
        model.load_adapter(str(final / "policy"), adapter_name=name, is_trainable=False,
                           local_files_only=True)
    model.set_adapter(name)
    model.eval().requires_grad_(False)


@torch.no_grad()
def evaluate_prompt(model, tokenizer, reward, row, entry, cfg):
    if model.training:
        raise RuntimeError("Frozen evaluation requires eval mode")
    if (row["prompt_id"], row["source_index"]) != (entry["prompt_id"], entry["source_index"]):
        raise RuntimeError("Held-out schedule provenance differs")
    messages = prompt_messages(row)
    set_seed(entry["generation_seed"])
    started = time.perf_counter()
    batch = ordinary_rollout(batch_generate(model, tokenizer, [messages],
        max_prompt_length=cfg["max_prompt_length"], max_new_tokens=cfg["eval_max_response_length"],
        **cfg["generation"]))
    logp, logits = chosen_logprobs(model, batch)
    mask = batch["response_mask"].float()
    entropy = categorical_entropy(logits, mask)
    del logits
    with reference_mode(model):
        ref, logits = chosen_logprobs(model, batch)
        del logits
    raw = score_reward_pairs(reward, tokenizer, [messages], batch["responses"],
                              max_length=cfg["reward_max_length"])
    for name, value in (("evaluation logp", logp), ("evaluation reference", ref), ("raw reward", raw)):
        finite(name, value)
    n = int(mask.sum())
    terminated = bool(batch["terminated_with_eos"][0])
    return {"schedule": entry, "prompt_messages": messages, "response": batch["responses"][0],
        "prompt_token_ids": batch["sequences"][0, :batch["prompt_width"]].cpu().tolist(),
        "response_token_ids": batch["response_ids"][0, :n].cpu().tolist(),
        "response_length": n, "terminated_with_eos": terminated, "truncated": bool(batch["truncated"][0]),
        "raw_reward": float(raw[0]),
        "effective_reward": float(raw[0]) - (0. if terminated else float(cfg["missing_eos_penalty"])),
        "reference_kl": float(sampled_kl(logp, ref, mask)),
        "sampled_entropy": float(sample_entropy(logp, mask)),
        "categorical_entropy": float(entropy),
        "policy_token_logprobs": logp[0, :n].cpu().tolist(),
        "reference_token_logprobs": ref[0, :n].cpu().tolist(),
        "elapsed_generation_scoring_seconds": time.perf_counter() - started}


def summarize_evaluation(records):
    if not records:
        raise RuntimeError("Cannot summarize an empty evaluation")
    result = {"prompts": len(records), "valid_tokens": sum(r["response_length"] for r in records),
        "aggregation": "Primary means weight each prompt equally; std uses population denominator. Token-weighted KL/entropy also reported distinctly.",
        "eos_rate": statistics.mean(float(r["terminated_with_eos"]) for r in records),
        "truncation_rate": statistics.mean(float(r["truncated"]) for r in records)}
    for key in ("raw_reward", "effective_reward", "reference_kl", "sampled_entropy", "categorical_entropy", "response_length"):
        values = [r[key] for r in records]
        result[key + "_mean"] = statistics.mean(values)
        result[key + "_std_population"] = statistics.pstdev(values)
    for key in ("reference_kl", "sampled_entropy", "categorical_entropy"):
        result[key + "_token_weighted"] = sum(r[key] * r["response_length"] for r in records) / result["valid_tokens"]
    return result


def evaluate_condition(model, tokenizer, reward, rows, schedule, cfg, name, contract,
                       output_root, persistent_root, budget, session_id=None):
    out, saved = Path(output_root) / name, Path(persistent_root) / name
    out.mkdir(parents=True, exist_ok=True)
    saved.mkdir(parents=True, exist_ok=True)
    for path in (out / "contract.json", saved / "contract.json"):
        if path.exists() and json.loads(path.read_text()) != contract:
            raise RuntimeError("Evaluation contract changed; existing evidence was retained")
        atomic_json(path, contract)
    records = []
    # Each prompt is independently seeded, so no unrecorded RNG replay is needed.
    with ThreadPoolExecutor(max_workers=1) as uploader:
        pending = None
        try:
            for entry in schedule["evaluation"]:
                budget.check()
                index = entry["row_index"]
                folder = out / f"prompt_{index:04d}"
                archive = saved / (folder.name + ".zip")
                if not folder.exists() and archive.exists():
                    restore_published(archive, folder)
                if folder.exists():
                    verify_directory(folder)
                    record = json.loads((folder / "record.json").read_text())
                    if record["schedule"] != entry or record["condition"] != name or record["contract"] != contract:
                        raise RuntimeError("Saved evaluation prompt provenance differs")
                else:
                    budget.check()
                    record = evaluate_prompt(model, tokenizer, reward, rows[index], entry, cfg)
                    record.update({"condition": name, "contract": contract, "environment_session_id": session_id})
                    with tempfile.TemporaryDirectory(prefix=".prompt_", dir=out) as temporary:
                        atomic_json(Path(temporary) / "record.json", record)
                        commit_directory(temporary, folder)
                records.append(record)
                if pending is not None:
                    pending.result()  # Surface copy failures; never claim unsaved progress.
                pending = uploader.submit(publish_directory, folder, saved)
                if len(records) % 10 == 0 or len(records) == len(rows):
                    print(f"Evaluation {name}: {len(records)}/{len(rows)} prompts; "
                          f"mean reward {statistics.mean(r['raw_reward'] for r in records):.4f}", flush=True)
        finally:
            if pending is not None:
                pending.result()
    if len(records) != len(rows):
        raise RuntimeError("Evaluation did not cover the entire fixed held-out pool")
    summary = {"condition": name, "contract": contract, "metrics": summarize_evaluation(records),
               "status": "COMPLETE", "selection_or_tuning": False}
    atomic_json(out / "summary.json", summary)
    atomic_json(saved / "summary.json", summary)
    return summary
