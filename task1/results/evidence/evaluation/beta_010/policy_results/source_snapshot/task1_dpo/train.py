"""Complete one approved Task 1 run, save it, reload it, and verify its backup."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import gc
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from common.data import encode_prompt_response, load_yaml, pad_batch, preference_responses, prompt_messages_from_preference, repo_path
from common.logging_utils import set_seed
from common.models import reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.runtime import pair_scores, train_epoch, update_pair_counts
from task1_dpo.support import file_sha, load_pinned_policy, load_pinned_tokenizer, model_source, parameter_digest, run_plan, save_json, verified_copy


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            for target, answer in zip((chosen, rejected), preference_responses(row)):
                target.append(encode_prompt_response(tokenizer, prompt, answer, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected), [r["prompt_id"] for r in rows]
    return collate


def validate_overrides(cfg, plan, dataset_path, beta, max_examples):
    if dataset_path is not None and repo_path(dataset_path).resolve() != repo_path(cfg["paths"][plan["path_key"]]).resolve():
        raise ValueError("Dataset override differs from the approved condition.")
    if beta is not None and beta != plan["beta"]:
        raise ValueError("Beta override differs from the approved condition.")
    if max_examples is not None and max_examples != plan["pairs"]:
        raise ValueError("Pair-budget override differs from the approved condition.")


def _train_once(cfg, rows, plan, output, models_dir, record):
    set_seed(int(cfg["seed"]))
    source, kwargs = model_source(cfg, models_dir)
    tok = load_pinned_tokenizer(cfg, source, kwargs)
    model = load_pinned_policy(cfg, source, kwargs, tok, trainable=True)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    optimizer = AdamW(parameters, lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.,
                                growth_factor=2., backoff_factor=.5, growth_interval=2000)
    collate = make_collate(tok, int(cfg["max_sequence_length"]))
    loader = DataLoader(rows, batch_size=int(cfg["batch_size"]), shuffle=True,
                        drop_last=False, collate_fn=collate)
    record.update({"optimizer_defaults": optimizer.defaults,
                   "scaler_settings": {"enabled": True, "init_scale": 1024., "growth_factor": 2.,
                                       "backoff_factor": .5, "growth_interval": 2000},
                   "autocast": False, "trainable_names": [n for n, _ in named],
                   "trainable_parameters": sum(p.numel() for p in parameters),
                   "trainable_dtypes": dict(Counter(str(p.dtype) for p in parameters)),
                   "attention_implementation": model.config._attn_implementation,
                   "initial_adapter_sha256": parameter_digest(model, adapters=True),
                   "frozen_base_before_sha256": parameter_digest(model),
                   "context_capacity": model.config.max_position_embeddings})
    cpu_probe = collate(rows[:2])[:2]
    probe = tuple({k: v.cuda() for k, v in b.items()} for b in cpu_probe)
    model.eval()
    with torch.no_grad():
        initial_policy = pair_scores(model, probe)
    with torch.no_grad(), reference_mode(model):
        initial_reference = pair_scores(model, probe)
    for a, b in zip(initial_policy, initial_reference):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-4)
    initial_loss, _ = dpo_loss(*initial_policy, *initial_reference, plan["beta"])
    if abs(initial_loss.item() - math.log(2)) >= 1e-5:
        raise RuntimeError("Fresh policy does not match the frozen reference.")
    record["initial_probe_loss"] = initial_loss.item()
    torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}, output / "rng_before_epoch.pt")
    save_json(output / "run_record.json", record)
    torch.cuda.reset_peak_memory_stats()
    training_start = time.perf_counter()
    last_update_time = training_start
    with (output / "training_trace.jsonl").open("w", encoding="utf-8") as trace:
        def emit(event):
            nonlocal last_update_time
            if event["event"] == "optimizer_update":
                torch.cuda.synchronize()
                now = time.perf_counter()
                event.update({"training_wall_seconds": now - training_start,
                              "update_wall_seconds": now - last_update_time,
                              "peak_allocated_VRAM_GiB": torch.cuda.max_memory_allocated() / 2**30,
                              "peak_reserved_VRAM_GiB": torch.cuda.max_memory_reserved() / 2**30})
                last_update_time = now
                record["completed_updates"] = event["update"]
                record["completed_pairs"] = event["seen_pairs"]
                if event["update"] == 1 or event["update"] % 10 == 0 or event["seen_pairs"] == len(rows):
                    print(f"{plan['name']}: update {event['update']}; pairs {event['seen_pairs']}/{len(rows)}; "
                          f"training {event['training_wall_seconds'] / 60:.1f} min", flush=True)
            trace.write(json.dumps(event, allow_nan=False) + "\n")
            trace.flush()
        result = train_epoch(model, loader, optimizer, scaler, beta=plan["beta"], total_pairs=len(rows),
                             batch_size=int(cfg["batch_size"]), accumulation_steps=int(cfg["grad_accum_steps"]),
                             max_grad_norm=float(cfg["max_grad_norm"]), emit=emit)
    torch.cuda.synchronize()
    record["training_wall_seconds"] = time.perf_counter() - training_start
    if Counter(result["ordered_prompt_ids"]) != Counter(r["prompt_id"] for r in rows):
        raise RuntimeError("Epoch membership changed.")
    record["training"] = result
    record["frozen_base_after_sha256"] = parameter_digest(model)
    if record["frozen_base_after_sha256"] != record["frozen_base_before_sha256"]:
        raise RuntimeError("Frozen base parameters changed.")
    record["final_adapter_sha256"] = parameter_digest(model, adapters=True)
    if record["final_adapter_sha256"] == record["initial_adapter_sha256"]:
        raise RuntimeError("No adapter update occurred.")
    model.eval()
    with torch.no_grad():
        final_policy = pair_scores(model, probe)
    with torch.no_grad(), reference_mode(model):
        final_reference = pair_scores(model, probe)
    for a, b in zip(initial_reference, final_reference):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-4)
    torch.save({"batches": cpu_probe, "prompt_ids": [r["prompt_id"] for r in rows[:2]],
                "policy_logp": [t.cpu() for t in final_policy],
                "reference_logp": [t.cpu() for t in final_reference]}, output / "reload_probe.pt")
    # Embeddings are unchanged and not trainable; save only the LoRA adapter.
    model.save_pretrained(output, safe_serialization=True, save_embedding_layers=False)
    tok.save_pretrained(output / "tokenizer")
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "completed_pairs": len(rows), "completed_updates": result["optimizer_updates"]},
               output / "final_training_state.pt")
    record.update({"status": "TRAINED_PENDING_RELOAD", "peak_allocated_VRAM_GiB": torch.cuda.max_memory_allocated() / 2**30,
                   "peak_reserved_VRAM_GiB": torch.cuda.max_memory_reserved() / 2**30})
    save_json(output / "run_record.json", record)


def run_training(config_path, run_name, dataset_path=None, output_path=None, beta=None,
                 max_examples=None, *, backup_root=None, models_dir=None):
    cfg = load_yaml(config_path)
    rows, plan = run_plan(cfg, run_name)
    validate_overrides(cfg, plan, dataset_path, beta, max_examples)
    if not torch.cuda.is_available():
        raise RuntimeError("Use --plan on CPU; official training requires a GPU.")
    if backup_root is None:
        raise ValueError("Provide --backup-root for durable verified artifacts.")
    backup_root = Path(backup_root)
    if not backup_root.is_dir():
        raise ValueError("Create/mount the persistent backup root before launching.")
    output = repo_path(output_path or plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    stamp = datetime.now(ZoneInfo("Asia/Karachi")).strftime("%Y%m%d_%H%M%S_%f")
    record = {"status": "STARTED", "official_training_started": True,
              "started_at": stamp, "plan": plan, "effective_config": cfg,
              "models_dir": str(Path(models_dir).resolve()) if models_dir else None,
              "selected_prompt_ids": [r["prompt_id"] for r in rows],
              "expected_update_pair_counts": update_pair_counts(len(rows), 2, 8),
              "seed": 6304, "python": platform.python_version(), "torch": str(torch.__version__),
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "total_VRAM_GiB": torch.cuda.get_device_properties(0).total_memory / 2**30,
              "packages": {p: metadata.version(p) for p in ("transformers", "tokenizers", "peft", "bitsandbytes")},
              "matmul_precision": torch.get_float32_matmul_precision(),
              "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "completed_updates": 0, "completed_pairs": 0}
    watched = list(repo_path("common").glob("*.py")) + list(repo_path("task1_dpo").glob("*.py"))
    watched += [repo_path("configs/base.yaml"), repo_path(config_path)]
    record["source_sha256"] = {}
    for path in watched:
        relative = path.relative_to(repo_path(".")) if path.is_relative_to(repo_path(".")) else Path("external_config") / path.name
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        record["source_sha256"][str(relative)] = file_sha(path)
    save_json(output / "run_record.json", record)
    (output / "pip_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    start = time.perf_counter()
    try:
        _train_once(cfg, rows, plan, output, models_dir, record)
        gc.collect()
        torch.cuda.empty_cache()
        command = [sys.executable, "-m", "task1_dpo.verify_checkpoint", "--checkpoint", str(output)]
        verified = subprocess.run(command, cwd=repo_path("."), text=True, capture_output=True)
        (output / "reload_verification.log").write_text(verified.stdout + "\n" + verified.stderr)
        print(verified.stdout, end="", flush=True)
        if verified.returncode:
            print(verified.stderr, flush=True)
            raise RuntimeError("Fresh-process checkpoint verification failed.")
        for path in watched:
            relative = path.relative_to(repo_path(".")) if path.is_relative_to(repo_path(".")) else Path("external_config") / path.name
            if file_sha(path) != record["source_sha256"][str(relative)]:
                raise RuntimeError(f"Source changed during training: {relative}")
        record["status"] = "COMPLETE_RELOAD_VERIFIED"
    except BaseException as error:
        record["status"] = "FAILED"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["total_wall_seconds"] = time.perf_counter() - start
        save_json(output / "run_record.json", record)
        destination = backup_root / f"{run_name}_{stamp}_{record['status']}"
        backup_start = time.perf_counter()
        receipt = verified_copy(output, destination)
        receipt["backup_wall_seconds"] = time.perf_counter() - backup_start
        receipt["job_wall_seconds_including_backup"] = time.perf_counter() - start
        save_json(backup_root / (destination.name + "_backup_receipt.json"), receipt)
        print("BACKUP_VERIFIED:", destination, flush=True)
    print("DPO_RUN_COMPLETE:", run_name, flush=True)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--backup-root")
    ap.add_argument("--models-dir")
    ap.add_argument("--plan", action="store_true", help="CPU-only budget validation; no model loading")
    args = ap.parse_args()
    if args.plan:
        cfg = load_yaml(args.config)
        _, plan = run_plan(cfg, args.run_name)
        validate_overrides(cfg, plan, args.dataset, args.beta, args.max_examples)
        plan["update_pair_counts"] = update_pair_counts(plan["pairs"], 2, 8)
        plan["optimizer_updates"] = len(plan["update_pair_counts"])
        print(json.dumps(plan, indent=2))
        return
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta,
                 args.max_examples, backup_root=args.backup_root, models_dir=args.models_dir)


if __name__ == "__main__":
    main()
