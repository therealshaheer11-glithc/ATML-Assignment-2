"""Rescue committed local evidence, verify session logs, then permit disconnect."""
from pathlib import Path
import json
import shutil

from task2_ppo.storage import atomic_json, commit_directory, publish_directory, verify_directory


def rescue_commits(repo, experiment):
    repo, experiment = Path(repo), Path(experiment)
    locations = [
        (repo / "outputs/task2_ppo", experiment / "training", "*/step_[0-9][0-9][0-9][0-9]"),
        (repo / "outputs/task2_ppo", experiment / "training", "*/final"),
        (repo / "results/task2_ppo/evaluation", experiment / "evaluation", "*/prompt_[0-9][0-9][0-9][0-9]"),
        (repo / "results/task2_ppo/cache_geometry", experiment / "cache_geometry", "row_[0-9][0-9][0-9][0-9]"),
    ]
    retained = recovered = 0
    for local, saved, pattern in locations:
        for folder in sorted(local.glob(pattern)):
            verify_directory(folder)
            relative_parent = folder.parent.relative_to(local)
            target = saved / relative_parent
            archive = target / (folder.name + ".zip")
            receipt = target / (folder.name + ".sha256.json")
            if archive.exists() and receipt.exists():
                prior = json.loads(receipt.read_text())
                if prior["archive"] != archive.name or prior["bytes"] != archive.stat().st_size:
                    raise RuntimeError("Previously verified persistence receipt differs")
                # This upload was already read back and verified by the running
                # job. Do not reread every old optimizer archive from Drive.
                retained += 1
            else:
                publish_directory(folder, target)
                recovered += 1
    return {"previously_verified_units_retained": retained, "interrupted_uploads_recovered": recovered}


def finish_session(repo, experiment, draft, session_id, metadata):
    metadata["persistence"] = rescue_commits(repo, experiment)
    draft = Path(draft)
    # Preserve the final progress state and small timing/contracts alongside the
    # console log, including closed failure/limit intervals.
    repo, experiment = Path(repo), Path(experiment)
    for source_root, pattern, label in [
        (repo / "outputs/task2_ppo", "*/session_*.json", "training_clocks"),
        (experiment, "progress.json", "pipeline"),
        (experiment, "job_contract.json", "pipeline"),
        (experiment, "frozen_candidates.json", "pipeline"),
        (experiment, "environment_sessions/*.json", "pipeline"),
    ]:
        for source in source_root.glob(pattern):
            target = draft / label / source.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    atomic_json(draft / "session.json", metadata)
    final = draft.parent / session_id
    commit_directory(draft, final)
    receipt = publish_directory(final, Path(experiment) / "gpu_sessions")
    return {"verified": True, "session_archive": receipt}


def disconnect_after_verified(approval, backup_verified, flush_drive, disconnect):
    if not backup_verified:
        raise RuntimeError("Backup is not verified; automatic disconnection is forbidden")
    flush_drive()  # A flush error stops here and leaves the runtime available.
    if approval.get("approved") and approval.get("automatic_disconnect"):
        disconnect()
