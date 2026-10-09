"""Prepared training -> frozen evaluation -> cache scoring; resumable GPU work.

CPU plotting and qualitative review are deliberately separate from this job.
The session limit schedules work across sessions, never reduces required budgets.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from uuid import uuid4

import torch

from common.data import read_jsonl, repo_path
from task2_ppo import continuation
from task2_ppo.cache_geometry import score_cache
from task2_ppo.evaluation_complete import (activate_adapter, evaluate_condition,
                                          load_evaluation_policy)
from task2_ppo.generation_audit import generation_protocol
from task2_ppo.runtime import load_reward_local, load_tokenizer_local, validate_job
from task2_ppo.session_budget import SessionBudget, SessionBudgetReached
from task2_ppo.storage import atomic_json, publish_directory, restore_published, sha256, verify_directory


def run_pipeline(models_dir, persistent_root, minutes, start_epoch=None):
    budget = SessionBudget(minutes, start_epoch)
    saved = Path(persistent_root)
    saved.mkdir(parents=True, exist_ok=True)
    progress_path = saved / "progress.json"
    progress = {"status": "STARTING", "session_minutes": minutes,
                "started_at_epoch": time.time(), "phase": "validation"}

    def record(status, phase):
        progress.update({"status": status, "phase": phase, "recorded_at_epoch": time.time()})
        atomic_json(progress_path, progress)
        atomic_json(repo_path("results/task2_ppo/gpu_pipeline.json"), progress)

    policy = reward = None
    try:
        budget.check()
        cfg, train, schedule, folders, audit = validate_job(models_dir)
        session_id = str(int(time.time())) + "_" + uuid4().hex[:8]
        atomic_json(saved / "environment_sessions" / (session_id + ".json"), {
            "session_id": session_id, "environment": audit["environment"],
            "session_minutes": minutes, "recorded_at_epoch": time.time()})
        extra = json.loads(repo_path("docs/task2_chunk3_files.json").read_text())
        for name, digest in extra.items():
            if sha256(repo_path(name)) != digest:
                raise RuntimeError(f"Installed final-chunk source differs: {name}")
        resource = json.loads(repo_path("docs/task2_resource_approval.json").read_text())
        if not resource["approved"] or minutes != resource["session_minutes"]:
            raise RuntimeError("Session limit differs from the user's explicit approval")
        approval = json.loads(repo_path("docs/task2_approval.json").read_text())
        specifications = approval["required_runs"]
        rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
        job_contract = {"protocol": "TASK2_APPROVED_COMPLETE_V1", "approvals": audit["approvals"],
            "source": audit["source"], "chunk2_files": audit["chunk2_files"], "chunk3_files": extra,
            "schedule_sha256": audit["schedule_sha256"], "models": audit["public_models"],
            "assets": audit["assets"]["verified_files"], "release_config": cfg,
            "resource_approval": resource}
        contract_path = saved / "job_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != job_contract:
            raise RuntimeError("Existing experiment source/settings differ; evidence not overwritten")
        atomic_json(contract_path, job_contract)
        record("RUNNING", "loading_reward")
        budget.check()
        tokenizer = load_tokenizer_local(folders["policy"])
        reward, reward_audit = load_reward_local(cfg, folders, tokenizer)
        # Narrow resource hook: check immediately before each new rollout, then
        # invoke the original course generation helper unchanged. No notebook state.
        original_generate = continuation.batch_generate
        generation_records = {}
        def bounded_generate(*args, **kwargs):
            budget.check()
            name = progress["phase"].split(":", 1)[-1]
            protocol = generation_protocol(args[0], args[1], kwargs["max_new_tokens"], cfg)
            if name not in generation_records:
                atomic_json(saved / "generation_protocol" / (name + ".json"), protocol)
                generation_records[name] = protocol
            elif generation_records[name] != protocol:
                raise RuntimeError("Generation protocol changed within a training run")
            return original_generate(*args, **kwargs)
        continuation.batch_generate = bounded_generate
        try:
            for specification in specifications:
                budget.check()
                record("RUNNING", "training:" + specification["name"])
                local_final = repo_path("outputs/task2_ppo") / specification["name"] / "final"
                remote_run = saved / "training" / specification["name"]
                if not local_final.exists() and (remote_run / "final.zip").exists():
                    restore_published(remote_run / "final.zip", local_final)
                if local_final.exists():
                    verify_directory(local_final)
                    saved_summary = json.loads((local_final / "summary.json").read_text())
                    run_cfg = {**cfg, "updates": specification["updates"],
                               "clip_epsilon": specification["clip_epsilon"], "kl_beta": specification["kl_beta"]}
                    expected = {"run": specification, "release_config": run_cfg,
                        "approvals": audit["approvals"], "source": audit["source"],
                        "chunk2_files": audit["chunk2_files"], "assets": audit["assets"]["verified_files"],
                        "models": audit["public_models"], "schedule_sha256": audit["schedule_sha256"]}
                    if saved_summary["contract"] != expected or saved_summary["completed_updates"] != specification["updates"]:
                        raise RuntimeError("Completed training endpoint contract differs")
                    publish_directory(local_final, remote_run)
                    print("Retained verified completed run:", specification["name"], flush=True)
                    continue  # Avoid restoring large optimizer archives for finished runs.
                continuation.run_one(cfg, specification, train, schedule, folders, audit, reward, reward_audit,
                                     repo_path("outputs/task2_ppo"), saved / "training", resume=True)
        finally:
            continuation.batch_generate = original_generate
        # Freeze all six endpoints before any held-out evaluation is inspected.
        candidates = {}
        for specification in specifications:
            final = repo_path("outputs/task2_ppo") / specification["name"] / "final"
            files = verify_directory(final)
            summary = json.loads((final / "summary.json").read_text())
            if summary["completed_updates"] != specification["updates"]:
                raise RuntimeError("Final training endpoint has the wrong budget")
            candidates[specification["name"]] = {k: v for k, v in files.items() if k.startswith("policy/")}
        atomic_json(saved / "frozen_candidates.json", {"candidates": candidates,
            "standard_is_task4_policy": True, "held_out_selection": False})
        budget.check()
        record("RUNNING", "loading_evaluation_policy")
        policy, policy_loading = load_evaluation_policy(cfg, folders, tokenizer)
        for name in ["midpoint"] + [s["name"] for s in specifications]:
            budget.check()
            activate_adapter(policy, name, repo_path("outputs/task2_ppo"))
            record("RUNNING", "evaluation:" + name)
            contract = {"job": job_contract, "condition": name,
                "adapter": candidates[name] if name != "midpoint" else {
                    k: v for k, v in audit["assets"]["verified_files"].items()
                    if k.startswith("checkpoints/ppo_midpoint_policy/")},
                "generation": {**cfg["generation"], "max_prompt_length": cfg["max_prompt_length"],
                               "max_response_length": cfg["eval_max_response_length"]},
                "full_generation_protocol": generation_protocol(policy, tokenizer, cfg["eval_max_response_length"], cfg),
                "reward_loading": reward_audit, "policy_loading": policy_loading,
                "forward_mode": "eval", "raw_logprobs": True,
                "aggregation": "Equal prompt means, plus separately named token-weighted metrics"}
            evaluate_condition(policy, tokenizer, reward, rows, schedule, cfg, name, contract,
                repo_path("results/task2_ppo/evaluation"), saved / "evaluation", budget, session_id)
        budget.check()
        activate_adapter(policy, "standard", repo_path("outputs/task2_ppo"))
        record("RUNNING", "cache_geometry")
        cache = torch.load(repo_path(cfg["cached_rollouts"]), map_location="cpu", weights_only=False)
        cache_contract = {"job": job_contract, "candidate": candidates["standard"],
                          "cache_sha256": audit["assets"]["verified_files"][cfg["cached_rollouts"]],
                          "reconstruction": "Pinned tokenizer text+recorded EOS, exact token count and text round-trip"}
        score_cache(policy, tokenizer, cache, rows, cfg, cache_contract,
                    repo_path("results/task2_ppo/cache_geometry"), saved / "cache_geometry", budget, session_id)
        record("GPU_WORK_COMPLETE", "CPU aggregation and qualitative review remain")
        print("GPU WORK COMPLETE: all six training runs, seven 200-prompt evaluations and fixed-cache scoring saved.", flush=True)
        return 0
    except SessionBudgetReached:
        record("SAVED_AT_SESSION_LIMIT", progress["phase"])
        print("SESSION LIMIT: saved completed work. Resume the same launcher to continue the remaining required work.", flush=True)
        return 75
    except BaseException as error:
        progress["error"] = {"type": type(error).__name__, "message": str(error)}
        record("FAILED", progress["phase"])
        raise
    finally:
        del policy, reward
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--persistent-root", required=True)
    parser.add_argument("--session-minutes", required=True, type=float)
    parser.add_argument("--session-start-epoch", type=float)
    args = parser.parse_args()
    if not str(Path(args.persistent_root).resolve()).startswith("/content/drive/") or not Path("/content/drive/MyDrive").is_dir():
        parser.error("Mount Drive and use a persistent experiment directory")
    raise SystemExit(run_pipeline(args.models_dir, args.persistent_root, args.session_minutes, args.session_start_epoch))


if __name__ == "__main__":
    main()
