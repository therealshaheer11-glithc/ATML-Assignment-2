"""Run one fixed Task 1 evaluation phase and verify its persistent backup.

Policy scoring/generation and reward replay are separate invocations. Importing
this module or requesting --plan never loads model weights or creates results.
"""
import argparse
from contextlib import contextmanager
import copy
from datetime import datetime
from zoneinfo import ZoneInfo
import gc
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import time

import torch
from common.data import load_yaml, read_jsonl, repo_path
from task1_dpo.evaluation_core import (evaluate_pair_pool, generate_pool,
                                     pool_messages, score_reward_pool)
from task1_dpo.evaluation_support import evaluation_plan, reward_input
from task1_dpo.reward_loading import load_pinned_reward
from task1_dpo.runtime import update_pair_counts
from task1_dpo.support import (DATA, RUNS, file_sha, load_pinned_policy,
    load_pinned_tokenizer, model_source, parameter_digest, run_plan,
    save_json, validated_rows, verified_copy)


def fixed_pools(cfg, name):
    plan = evaluation_plan(cfg, name)
    keys = list(dict.fromkeys(plan["pair_pools"] + plan["generation_pools"]))
    pools = {key: validated_rows(cfg, key) for key in keys}
    plan["datasets"] = {key: {"rows": len(rows), "sha256": DATA[key][1],
        "ordered_prompt_ids": [row["prompt_id"] for row in rows]}
        for key, rows in pools.items()}
    return plan, pools


def checkpoint_for_condition(cfg, name, adapter=None):
    if name == "sft":
        if adapter is not None:
            raise ValueError("The untouched SFT baseline cannot receive an adapter.")
        return None, None, {}
    rows, expected = run_plan(cfg, name)
    folder = repo_path(adapter or expected["output"])
    training = json.loads((folder / "run_record.json").read_text())
    reload_check = json.loads((folder / "reload_verification.json").read_text())
    counts = update_pair_counts(len(rows), cfg["batch_size"], cfg["grad_accum_steps"])
    if (training["status"] != "COMPLETE_RELOAD_VERIFIED" or
            training["effective_config"] != cfg or training["plan"] != expected or
            training["selected_prompt_ids"] != [row["prompt_id"] for row in rows] or
            training["completed_pairs"] != len(rows) or
            training["completed_updates"] != len(counts) or
            training["expected_update_pair_counts"] != counts or
            reload_check["status"] != "PASS" or reload_check["held_out_examples_used"] is not False or
            reload_check["probe_prompt_ids"] != [row["prompt_id"] for row in rows[:2]]):
        raise ValueError("Checkpoint is incomplete, mismatched or not reload verified.")
    names = ("run_record.json", "reload_verification.json", "adapter_config.json",
             "adapter_model.safetensors")
    hashes = {name: file_sha(folder / name) for name in names}
    return folder, training, hashes


def checked_policy_results(folder, cfg, plan):
    folder = Path(folder)
    record = json.loads((folder / "evaluation_record.json").read_text())
    if (record["status"] != "POLICY_COMPLETE" or record["phase"] != "policy" or
            record["plan"] != plan or record["effective_config"] != cfg):
        raise ValueError("Reward replay requires matching, complete policy results.")
    hashes = record["artifact_sha256"]
    required = {"metrics.json"} | {f"generation_{key}.jsonl" for key in plan["generation_pools"]}
    required |= {f"pairs_{key}.jsonl" for key in plan["pair_pools"]}
    if not required <= hashes.keys():
        raise ValueError("Policy result manifest is incomplete.")
    for name, digest in hashes.items():
        path = (folder / name).resolve()
        if folder.resolve() not in path.parents or file_sha(path) != digest:
            raise ValueError(f"Policy result artifact changed: {name}")
    metrics = json.loads((folder / "metrics.json").read_text())
    if (metrics["name"] != plan["name"] or metrics["beta"] != plan["beta"] or
            set(metrics["pairs"]) != set(plan["pair_pools"]) or
            set(metrics["generation"]) != set(plan["generation_pools"])):
        raise ValueError("Policy metric scopes differ from the approved plan.")
    for key in plan["pair_pools"]:
        if metrics["pairs"][key]["pairs"] != plan["datasets"][key]["rows"]:
            raise ValueError("Policy pair metric denominator is incomplete.")
    generated = {}
    for key in plan["generation_pools"]:
        generated[key] = read_jsonl(folder / f"generation_{key}.jsonl")
        if (len(generated[key]) != plan["datasets"][key]["rows"] or
                metrics["generation"][key]["responses"] != len(generated[key])):
            raise ValueError("Policy generation metric denominator is incomplete.")
    return record, metrics, generated


def reward_preflight(tokenizer, generated, pools, cfg):
    """Validate every complete saved input before any reward weights are loaded."""
    audit = {}
    for key, records in generated.items():
        rows = pools[key]
        if len(records) != len(rows):
            raise ValueError("Reward replay membership is incomplete.")
        lengths = []
        for index, (record, row) in enumerate(zip(records, rows)):
            if (record["row_index"] != index or record["prompt_id"] != row["prompt_id"] or
                    record["messages"] != pool_messages(row, key) or
                    record["response_length"] != len(record["response_token_ids"]) or
                    not 1 <= record["response_length"] <= cfg["max_generation_tokens"] or
                    tokenizer.decode(record["response_token_ids"], skip_special_tokens=True) != record["response"]):
                raise ValueError("Reward replay row/order/prompt/tokens changed.")
            lengths.append(len(reward_input(tokenizer, record["messages"],
                                           record["response"], cfg["reward_max_length"])[1]))
        audit[key] = {"responses": len(records), "maximum_complete_input_tokens": max(lengths),
                      "input_token_counts": lengths, "truncated": 0}
    return audit


@contextmanager
def row_writer(path, count, label):
    """Flush each row; callers must finish every row before a successful phase."""
    emitted = 0
    with Path(path).open("x", encoding="utf-8") as handle:
        def emit(row):
            nonlocal emitted
            if row["row_index"] != emitted:
                raise ValueError("Evaluation emitter row order changed.")
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            emitted += 1
            if emitted == 1 or emitted % 25 == 0 or emitted == count:
                print(f"{label}: {emitted}/{count}", flush=True)
        yield emit
    if emitted != count:
        raise RuntimeError(f"Incomplete evaluation output: {label}")


def policy_phase(cfg, plan, pools, output, models_dir, adapter, training, record):
    source, kwargs = model_source(cfg, models_dir)
    tokenizer = load_pinned_tokenizer(cfg, source, kwargs)
    model = load_pinned_policy(cfg, source, kwargs, tokenizer, adapter=adapter)
    base_before = parameter_digest(model)
    adapter_before = parameter_digest(model, adapters=True)
    if training is not None and (base_before != training["frozen_base_before_sha256"] or
                                adapter_before != training["final_adapter_sha256"]):
        raise RuntimeError("Loaded checkpoint parameter hashes differ from training.")
    record.update({"base_parameter_sha256": base_before, "adapter_parameter_sha256": adapter_before,
                   "raw_policy_config": json.loads((Path(source) / "config.json").read_text()),
                   "loaded_policy_config": model.config.to_dict(),
                   "attention_implementation": model.config._attn_implementation})
    save_json(output / "evaluation_record.json", record)
    metrics = {"name": plan["name"], "beta": plan["beta"], "pairs": {}, "generation": {}}
    for key in plan["pair_pools"]:
        with row_writer(output / f"pairs_{key}.jsonl", len(pools[key]), f"Pairs {key}") as emit:
            metrics["pairs"][key] = evaluate_pair_pool(model, tokenizer, pools[key], cfg,
                plan["beta"], emit, stratified=key == "dpo_length_eval")
        save_json(output / "metrics.json", metrics)
    for key in plan["generation_pools"]:
        with row_writer(output / f"generation_{key}.jsonl", len(pools[key]), f"Generate {key}") as emit:
            metrics["generation"][key] = generate_pool(model, tokenizer, pools[key], cfg, key, emit)
        save_json(output / "metrics.json", metrics)
    if parameter_digest(model) != base_before or parameter_digest(model, adapters=True) != adapter_before:
        raise RuntimeError("Evaluation changed frozen policy parameters.")
    record["policy_parameters_unchanged"] = True


def reward_phase(cfg, plan, pools, output, models_dir, policy_results, record):
    # Preserve a verified independent copy, including text/token IDs and policy
    # metrics, so final reward artifacts remain usable after runtime deletion.
    verified_copy(policy_results, output / "policy_results")
    _, metrics, generated = checked_policy_results(output / "policy_results", cfg, plan)
    source, kwargs = model_source(cfg, models_dir)
    tokenizer = load_pinned_tokenizer(cfg, source, kwargs)
    audit = reward_preflight(tokenizer, generated, pools, cfg)
    save_json(output / "reward_input_audit.json", audit)
    record["policy_results_record_sha256"] = file_sha(output / "policy_results/evaluation_record.json")
    model, tokenizer, loading = load_pinned_reward(cfg, models_dir)
    save_json(output / "reward_loading.json", loading)
    metrics = copy.deepcopy(metrics)
    for key in plan["generation_pools"]:
        with row_writer(output / f"rewards_{key}.jsonl", len(pools[key]), f"Reward {key}") as emit:
            result = score_reward_pool(model, tokenizer, generated[key], pools[key], cfg, key, emit)
        metrics["generation"][key].update(result)
        save_json(output / "metrics.json", metrics)
    checked_policy_results(output / "policy_results", cfg, plan)
    record["complete_reward_inputs_verified"] = True


def source_files(config_path):
    watched = list(repo_path("common").glob("*.py")) + list(repo_path("task1_dpo").glob("*.py"))
    watched += [repo_path("configs/base.yaml"), repo_path(config_path),
                repo_path("docs/task1_tokenizer_manifest.json"),
                repo_path("docs/task1_model_preparation.json"),
                repo_path("docs/task1_reward_configuration_fix.md")]
    return list(dict.fromkeys(watched))


def run_evaluation(config_path, name, phase, *, adapter=None, models_dir=None,
                   output_path=None, backup_root=None, policy_results=None):
    cfg = load_yaml(config_path)
    plan, pools = fixed_pools(cfg, name)
    if phase not in {"policy", "reward"}:
        raise ValueError("Select policy or reward phase.")
    if name == "sft" and adapter is not None:
        raise ValueError("The untouched SFT baseline cannot receive an adapter.")
    if not torch.cuda.is_available():
        raise RuntimeError("Use --plan on CPU; actual evaluation requires the approved GPU.")
    if models_dir is None or backup_root is None or not Path(backup_root).is_dir():
        raise ValueError("Provide prepared local models and an existing persistent backup root.")
    models_dir = Path(models_dir).resolve()
    prepared = json.loads(repo_path("docs/task1_model_preparation.json").read_text())
    if file_sha(models_dir / "model_manifest.json") != prepared["model_manifest_sha256"]:
        raise ValueError("Model manifest differs from verified CPU preparation.")
    if phase == "policy":
        if policy_results is not None:
            raise ValueError("--policy-results belongs to the reward phase.")
        adapter, training, checkpoint_hashes = checkpoint_for_condition(cfg, name, adapter)
    else:
        if adapter is not None or policy_results is None:
            raise ValueError("Reward replay requires --policy-results and loads no policy adapter.")
        policy_results = repo_path(policy_results).resolve()
        checked_policy_results(policy_results, cfg, plan)
        adapter, training, checkpoint_hashes = None, None, {}
    output = repo_path(output_path or str(Path(cfg["results_dir"]) / name / phase)).resolve()
    if output == Path(backup_root).resolve() or output in Path(backup_root).resolve().parents:
        raise ValueError("Persistent backups cannot be inside evaluation output.")
    if phase == "reward" and (output == policy_results or policy_results in output.parents or output in policy_results.parents):
        raise ValueError("Policy results and reward output must be separate folders.")
    output.mkdir(parents=True, exist_ok=False)
    stamp = datetime.now(ZoneInfo("Asia/Karachi")).strftime("%Y%m%d_%H%M%S_%f")
    record = {"status": "STARTED", "phase": phase, "started_at": stamp,
        "plan": plan, "effective_config": cfg, "models_dir": str(models_dir),
        "model_manifest_sha256": prepared["model_manifest_sha256"],
        "adapter": str(adapter) if adapter else None, "checkpoint_file_sha256": checkpoint_hashes,
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "packages": {p: metadata.version(p) for p in ("transformers", "tokenizers", "peft", "bitsandbytes")},
        "source_sha256": {}}
    start = time.perf_counter()
    try:
        watched = source_files(config_path)
        for path in watched:
            relative = path.relative_to(repo_path(".")) if path.is_relative_to(repo_path(".")) else Path("external_config") / path.name
            target = output / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            record["source_sha256"][str(path)] = file_sha(path)
        if training is not None:
            save_json(output / "training_run_record.json", training)
            shutil.copyfile(adapter / "reload_verification.json", output / "checkpoint_reload_verification.json")
        save_json(output / "evaluation_record.json", record)
        torch.cuda.reset_peak_memory_stats()
        if phase == "policy":
            policy_phase(cfg, plan, pools, output, models_dir, adapter, training, record)
        else:
            reward_phase(cfg, plan, pools, output, models_dir, policy_results, record)
        torch.cuda.synchronize()
        for path, digest in record["source_sha256"].items():
            if file_sha(path) != digest:
                raise RuntimeError("Evaluation source/configuration changed during execution.")
        for filename, digest in checkpoint_hashes.items():
            if file_sha(adapter / filename) != digest:
                raise RuntimeError("Checkpoint files changed during evaluation.")
        for key, info in plan["datasets"].items():
            if file_sha(repo_path(cfg["paths"][key])) != info["sha256"]:
                raise RuntimeError("Evaluation dataset changed during execution.")
        record["artifact_sha256"] = {str(path.relative_to(output)): file_sha(path)
            for path in sorted(output.rglob("*")) if path.is_file() and path != output / "evaluation_record.json"}
        record["status"] = "POLICY_COMPLETE" if phase == "policy" else "COMPLETE"
    except BaseException as error:
        record["status"] = "FAILED"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record.update({"wall_seconds_before_backup": time.perf_counter() - start,
            "peak_allocated_VRAM_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_VRAM_GiB": torch.cuda.max_memory_reserved() / 2**30})
        save_json(output / "evaluation_record.json", record)
        gc.collect()
        torch.cuda.empty_cache()
        destination = Path(backup_root) / f"eval_{name}_{phase}_{stamp}_{record['status']}"
        receipt = verified_copy(output, destination)
        receipt["job_wall_seconds_including_backup"] = time.perf_counter() - start
        save_json(Path(backup_root) / (destination.name + "_backup_receipt.json"), receipt)
        print("BACKUP_VERIFIED:", destination, flush=True)
    print("EVALUATION_PHASE_COMPLETE:", name, phase, flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--name", choices=[*RUNS, "sft"], required=True)
    parser.add_argument("--phase", choices=["policy", "reward"], default="policy")
    parser.add_argument("--adapter")
    parser.add_argument("--models-dir")
    parser.add_argument("--output")
    parser.add_argument("--backup-root")
    parser.add_argument("--policy-results")
    parser.add_argument("--plan", action="store_true", help="CPU-only full-pool validation; no model loading")
    args = parser.parse_args()
    if args.plan:
        if args.name == "sft" and args.adapter is not None:
            raise ValueError("The untouched SFT baseline cannot receive an adapter.")
        plan, _ = fixed_pools(load_yaml(args.config), args.name)
        print(json.dumps(plan, indent=2))
        return
    run_evaluation(args.config, args.name, args.phase, adapter=args.adapter,
        models_dir=args.models_dir, output_path=args.output,
        backup_root=args.backup_root, policy_results=args.policy_results)


if __name__ == "__main__":
    main()
