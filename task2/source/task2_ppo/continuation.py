"""Task 2 standard continuation and five approved independent short forks.

No held-out data or cached diagnostic rows enter this training loop.
Requires CPU preparation first; GPU launch instructions follow in a later chunk.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from common.data import prompt_messages, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from task2_ppo.checkpoint import cpu_tree, optimizer_dtype_audit, restore, snapshot
from task2_ppo.runtime import load_actor_critic, load_reward_local, validate_job
from task2_ppo.storage import (atomic_json, commit_directory, publish_directory,
                               restore_published, sha256, verify_directory)
from task2_ppo.training_core import freeze_rollout, optimize_rollout, ordinary_rollout


def stamp():
    return datetime.now(timezone.utc).isoformat()


def stability(updates):
    steps = [s for update in updates for s in update["optimization"]]
    result = {"optimizer_steps": len(steps), "nonfinite_steps": 0,
              "definition": "population std and max across pre-clipping optimizer-step gradient norms"}
    for label in ("actor", "critic"):
        norms = [s[label + "_grad_norm_preclip"] for s in steps]
        result[label + "_std_preclip"] = statistics.pstdev(norms) if norms else None
        result[label + "_max_preclip"] = max(norms) if norms else None
    return result


def _restore_missing_updates(out, persistent):
    if not persistent.exists():
        return
    for archive in sorted(persistent.glob("step_*.zip")):
        destination = out / archive.stem
        if not destination.exists():
            restore_published(archive, destination)
        else:
            verify_directory(destination)


def run_one(cfg, specification, train, schedule, folders, audit, reward, reward_audit,
            output_root, persistent_root, resume=False):
    cfg = dict(cfg)
    cfg.update({"updates": specification["updates"], "clip_epsilon": specification["clip_epsilon"],
                "kl_beta": specification["kl_beta"]})
    name = specification["name"]
    out, persistent = Path(output_root) / name, Path(persistent_root) / name
    contract = {"run": specification, "release_config": cfg,
                "approvals": audit["approvals"], "source": audit["source"],
                "chunk2_files": audit["chunk2_files"],
                "assets": audit["assets"]["verified_files"],
                "models": audit["public_models"], "schedule_sha256": audit["schedule_sha256"]}
    if out.exists() and not resume:
        raise FileExistsError(f"Run already exists; use explicit --resume: {out}")
    out.mkdir(parents=True, exist_ok=resume)
    persistent.mkdir(parents=True, exist_ok=True)
    contract_path = out / "contract.json"
    remote_contract = persistent / "contract.json"
    for path in (contract_path, remote_contract):
        if path.exists() and json.loads(path.read_text()) != contract:
            raise RuntimeError(f"Existing run contract differs: {path}")
    if resume:
        _restore_missing_updates(out, persistent)
    elif remote_contract.exists():
        raise FileExistsError("Persistent run already exists; choose --resume")
    atomic_json(contract_path, contract)
    atomic_json(remote_contract, contract)
    if (out / "final").exists():
        verify_directory(out / "final")
        if json.loads((out / "final/summary.json").read_text())["contract"] != contract:
            raise RuntimeError("Final export contract differs")
        publish_directory(out / "final", persistent)
        print(f"{name}: verified completed run retained", flush=True)
        return
    policy, critic, tokenizer, actor_opt, critic_opt, loading = load_actor_critic(cfg, folders)
    loading["reward"] = reward_audit
    completed, history = 0, []
    updates = sorted(out.glob("step_[0-9][0-9][0-9][0-9]"))
    for index, folder in enumerate(updates, 1):
        verify_directory(folder)
        if folder.name != f"step_{index:04d}":
            raise RuntimeError("Checkpoint update sequence has a gap")
        record = json.loads((folder / "update.json").read_text())
        if record["completed_updates"] != index:
            raise RuntimeError("Checkpoint update number differs")
        history.append(record)
        # Repair a local commit whose Drive copy was interrupted; never retrain it.
        publish_directory(folder, persistent)
    if updates:
        state = torch.load(updates[-1] / "checkpoint.pt", map_location="cpu", weights_only=False)
        completed = restore(state, policy, critic, actor_opt, critic_opt, contract)
        if completed != len(history) or completed > cfg["updates"]:
            raise RuntimeError("Checkpoint budget/history differs")
        del state
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    clock_path = out / ("session_" + session_id + ".json")
    remote_clock = persistent / clock_path.name
    session = {"id": session_id, "started_at": stamp(), "status": "RUNNING",
               "start_completed_updates": completed, "environment": audit["environment"],
               "model_loading": loading, "elapsed_seconds": 0., "peak_allocated_gib": 0.,
               "peak_reserved_gib": 0., "wall_clock_complete": False}
    # Synchronize at timing boundaries; no model-load/download time is called training time.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    def record_clock(status, complete=False):
        torch.cuda.synchronize()
        session.update({"status": status, "elapsed_seconds": time.perf_counter() - started,
            "completed_updates": completed, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "wall_clock_complete": complete, "recorded_at": stamp()})
        atomic_json(clock_path, session)
        atomic_json(remote_clock, session)

    try:
        record_clock("RUNNING")
        for update_index in range(completed, int(cfg["updates"])):
            entry = schedule["training"][update_index]
            row = train[entry["row_index"]]
            if (row["prompt_id"], row["source_index"]) != (entry["prompt_id"], entry["source_index"]):
                raise RuntimeError("Training prompt schedule mismatch")
            prompts = [prompt_messages(row)]
            set_seed(entry["rollout_seed"])
            torch.cuda.synchronize()
            start_generation = time.perf_counter()
            batch = ordinary_rollout(batch_generate(policy, tokenizer, prompts,
                max_prompt_length=cfg["max_prompt_length"], max_new_tokens=cfg["max_response_length"],
                **cfg["generation"]))
            raw_rewards = score_reward_pairs(reward, tokenizer, prompts, batch["responses"],
                                             max_length=cfg["reward_max_length"])
            fixed, diagnostics = freeze_rollout(policy, critic, batch, raw_rewards, cfg)
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - start_generation
            start_optimization = time.perf_counter()
            steps = optimize_rollout(policy, critic, actor_opt, critic_opt, batch, fixed, cfg)
            torch.cuda.synchronize()
            optimization_seconds = time.perf_counter() - start_optimization
            record = {"completed_updates": update_index + 1, "schedule": entry,
                "prompt_messages": prompts[0], "responses": batch["responses"],
                "terminated_with_eos": batch["terminated_with_eos"], "truncated": batch["truncated"],
                "rollout": diagnostics, "optimization": steps,
                "generation_scoring_seconds": generation_seconds, "optimization_seconds": optimization_seconds,
                "actor_optimizer": optimizer_dtype_audit(actor_opt),
                "critic_optimizer": optimizer_dtype_audit(critic_opt)}
            with tempfile.TemporaryDirectory(prefix=".uncommitted_", dir=out) as temporary:
                temporary = Path(temporary)
                atomic_json(temporary / "update.json", record)
                torch.save(cpu_tree({"batch": batch, "fixed": fixed}), temporary / "rollout.pt")
                torch.save(snapshot(policy, critic, actor_opt, critic_opt, update_index + 1, contract),
                           temporary / "checkpoint.pt")
                destination = out / f"step_{update_index + 1:04d}"
                commit_directory(temporary, destination)
            completed = update_index + 1
            history.append(record)
            publish_directory(destination, persistent)
            record_clock("RUNNING")
            print(f"{name} {completed}/{cfg['updates']} | reward {diagnostics['raw_reward']:.4f} | "
                  f"KL {diagnostics['rollout_reference_kl']:.4f} | tokens {diagnostics['valid_generated_tokens']} | "
                  "checkpoint verified", flush=True)
            del batch, fixed, raw_rewards
        record_clock("TRAINING_COMPLETE", complete=True)
        # Previous sessions are copied to Drive individually; include them in the final summary.
        sessions = {p.name: json.loads(p.read_text()) for p in persistent.glob("session_*.json")}
        sessions[clock_path.name] = session
        summary = {"status": "COMPLETE", "contract": contract, "completed_updates": completed,
            "optimization_steps": 2 * completed, "allocated_response_tokens": 512 * completed,
            "actual_response_tokens": sum(r["rollout"]["valid_generated_tokens"] for r in history),
            "stability": stability(history), "sessions": list(sessions.values()),
            "nonfinite_failure_events": sum(bool(s.get("error", {}).get("nonfinite_failure")) for s in sessions.values()),
            "nonfinite_count_definition": "Completed optimizer-step records must all be finite. Failure events count recorded sessions stopped for nonfinite arithmetic, separately from completed-step stability statistics.",
            "wall_clock_seconds": sum(s["elapsed_seconds"] for s in sessions.values()),
            "wall_clock_complete": all(s["wall_clock_complete"] for s in sessions.values()),
            "wall_clock_definition": "Active continuation sessions after model loading; includes generation, scoring, optimization and per-update checkpoint persistence; excludes final adapter export, setup, model reload and offline analysis. A hard disconnect can leave an incomplete timing interval, explicitly flagged.",
            "peak_allocated_gib": max(s["peak_allocated_gib"] for s in sessions.values()),
            "peak_reserved_gib": max(s["peak_reserved_gib"] for s in sessions.values()),
            "final_policy_for_task4": name == "standard", "held_out_used_for_training": False,
            "history": history}
        with tempfile.TemporaryDirectory(prefix=".export_", dir=out) as temporary:
            temporary = Path(temporary)
            policy.save_pretrained(temporary / "policy", safe_serialization=True)
            critic.save_pretrained(temporary / "critic", safe_serialization=True)
            tokenizer.save_pretrained(temporary / "tokenizer")
            atomic_json(temporary / "summary.json", summary)
            atomic_json(temporary / "critic_loading_note.json", {
                "decision": "L", "head_dtype": "float32",
                "resume": "Use the hashed step checkpoint with trainable-state restore after head promotion. Do not load the saved float32 head into a float16 destination.",
                "base_critic": cfg["paths"]["ppo_midpoint_value"]})
            commit_directory(temporary, out / "final")
        publish_directory(out / "final", persistent)
        record_clock("COMPLETE", complete=True)
    except BaseException as error:
        try:
            message = str(error).lower()
            session["error"] = {"type": type(error).__name__, "message": str(error),
                                "nonfinite_failure": isinstance(error, FloatingPointError)
                                    or "nonfinite" in message or "non-finite" in message}
            # Mark a closed failure interval complete as a timer interval, but retain failure status.
            record_clock("FAILED", complete=True)
        except BaseException:
            pass  # Preserve the original exception; never report success after backup failure.
        raise
    finally:
        del policy, critic, actor_opt, critic_opt, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--persistent-root", required=True, help="Mounted Drive directory for verified run archives")
    parser.add_argument("--output-root", default="outputs/task2_ppo")
    parser.add_argument("--runs", nargs="+", default=["standard"],
                        choices=["standard", "clip_005", "central_8", "clip_050", "kl_000", "kl_020"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(args.runs) != len(set(args.runs)):
        parser.error("Duplicate run names")
    persistent = Path(args.persistent_root).resolve()
    if not str(persistent).startswith("/content/drive/") or not Path("/content/drive/MyDrive").is_dir():
        parser.error("Persistence must use mounted Drive, not the ephemeral Colab disk")
    cfg, train, schedule, folders, audit = validate_job(args.models_dir)
    approval = json.loads(repo_path("docs/task2_approval.json").read_text())
    specifications = {r["name"]: r for r in approval["required_runs"]}
    # Keep the frozen RM resident across runs; actor/critic are independently reloaded each time.
    from task2_ppo.runtime import load_tokenizer_local
    tokenizer = load_tokenizer_local(folders["policy"])
    reward, reward_audit = load_reward_local(cfg, folders, tokenizer)
    for name in args.runs:
        run_one(cfg, specifications[name], train, schedule, folders, audit, reward, reward_audit,
                repo_path(args.output_root), persistent, resume=args.resume)


if __name__ == "__main__":
    main()
