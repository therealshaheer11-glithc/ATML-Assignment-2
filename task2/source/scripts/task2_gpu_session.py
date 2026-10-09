"""Run one approved GPU session; always save logs before returning to notebook."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from scripts.task2_stage import restore_files, stage_models
from scripts.task2_finish import finish_session
from task2_ppo.storage import atomic_json


def run_visible(command, log, cwd):
    process = subprocess.Popen(list(map(str, command)), cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, start_new_session=True)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise


def main():
    def stop_requested(signum, frame):
        raise KeyboardInterrupt("Session interrupted; stop model process and preserve completed work")
    signal.signal(signal.SIGTERM, stop_requested)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready", required=True, type=Path)
    parser.add_argument("--start-epoch", required=True, type=float)
    parser.add_argument("--finish-receipt", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    ready = json.loads(args.ready.read_text())
    approval = ready["resource_approval"]
    if ready["status"] != "GPU_JOBS_READY" or approval["session_minutes"] != 120 or not approval["automatic_disconnect"] or not approval["approved"]:
        raise RuntimeError("Expected the approved GPU-ready receipt")
    experiment = Path(ready["experiment_root"])
    if not str(experiment.resolve()).startswith("/content/drive/"):
        raise RuntimeError("Experiment evidence must be saved on mounted Drive")
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    drafts = repo / "results/task2_ppo/gpu_sessions"
    drafts.mkdir(parents=True, exist_ok=True)
    draft = drafts / (".draft_" + session_id)
    draft.mkdir()
    metadata = {"session_id": session_id, "start_epoch": args.start_epoch,
        "resource_approval": approval, "ready_receipt": str(args.ready), "process_stopped": False}
    exit_code = 1
    with (draft / "console.log").open("w", encoding="utf-8") as log:
        try:
            # No heavyweight package is imported by this orchestrator. The fresh
            # pipeline process sees exactly the installed course package versions.
            exit_code = run_visible([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                                      "-r", repo / "requirements.txt"], log, repo)
            if exit_code:
                raise RuntimeError("GPU environment package installation failed")
            exit_code = 1  # Any subsequent staging/launch exception is a failure.
            if time.time() - args.start_epoch >= 120 * 60:
                exit_code = 75
            else:
                restore_files(ready, repo)
                local_models = Path("/content/ATML_PA2_Task2_models")
                stage_models(ready, local_models)
                exit_code = run_visible([sys.executable, "-u", "-m", "task2_ppo.pipeline",
                    "--models-dir", local_models, "--persistent-root", experiment,
                    "--session-minutes", "120", "--session-start-epoch", str(args.start_epoch)], log, repo)
        except BaseException as error:
            metadata["launcher_error"] = {"type": type(error).__name__, "message": str(error)}
            details = traceback.format_exc()
            print(details, flush=True)
            log.write(details)
        finally:
            metadata.update({"process_stopped": True, "return_code": exit_code,
                "end_epoch": time.time(), "elapsed_session_seconds": time.time() - args.start_epoch,
                "status": "COMPLETE" if exit_code == 0 else "SESSION_LIMIT" if exit_code == 75 else "FAILED"})
    # A persistence error propagates: no verified finish receipt, hence no unassign.
    result = finish_session(repo, experiment, draft, session_id, metadata)
    result.update({"pipeline_return_code": exit_code, "status": metadata["status"],
                   "resource_approval": approval, "experiment_root": str(experiment)})
    atomic_json(args.finish_receipt, result)
    print("SESSION BACKUP VERIFIED:", result["session_archive"], flush=True)
    print("Session status:", result["status"], flush=True)


if __name__ == "__main__":
    main()
