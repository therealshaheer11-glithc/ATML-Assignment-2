"""Download/verify Task 2 files and persist a ready project; never load weights.

Public-file checksum preparation adapts the earlier approved Task 1 CPU workflow.
Existing Task 1 snapshots are reused only after identity and checksum validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from huggingface_hub import HfApi, snapshot_download

from common.data import load_yaml, repo_path
from task2_ppo.preflight import make_schedule, verify_assets, verify_config, verify_source
from task2_ppo.reward_compat import compatible_reward_config
from task2_ppo.storage import atomic_json, sha256


ROOT = repo_path(".").resolve()
TASK1_MANIFEST_SHA = "7ad306ec2e4ba0f2c12f5d9e9ce16f1eec34de418388f2a4ef776196f72e4b2e"


def verify_public(root, approval, expected_manifest=None):
    root = Path(root)
    manifest_path = root / "model_manifest.json"
    if expected_manifest and sha256(manifest_path) != expected_manifest:
        raise RuntimeError("Existing public model manifest differs from the saved Task 1 evidence")
    manifest = json.loads(manifest_path.read_text())
    for kind in ("policy", "reward"):
        entry = manifest[kind]
        if (entry["model_id"], entry["revision"]) != (approval[kind + "_model"], approval[kind + "_revision"]):
            raise RuntimeError(f"Model identity differs: {kind}")
        folder = (root / entry["directory"]).resolve()
        if root.resolve() not in folder.parents:
            raise RuntimeError("Model folder escapes its root")
        for name, digest in entry["sha256"].items():
            path = (folder / name).resolve()
            if folder not in path.parents or not path.is_file() or sha256(path) != digest:
                raise RuntimeError(f"Saved model file differs: {kind}/{name}")
        if not {"config.json"} <= entry["sha256"].keys() or not any(n.endswith(".safetensors") for n in entry["sha256"]):
            raise RuntimeError("Incomplete prepared public model")
        for index in folder.glob("*.safetensors.index.json"):
            if not set(json.loads(index.read_text())["weight_map"].values()) <= entry["sha256"].keys():
                raise RuntimeError("Missing model shard")
    return manifest


def download_public(destination, approval):
    destination = Path(destination)
    identity = {k: {"model_id": approval[k + "_model"], "revision": approval[k + "_revision"]}
                for k in ("policy", "reward")}
    identity_path = destination / "identity.json"
    if destination.exists():
        if not identity_path.exists() or json.loads(identity_path.read_text()) != identity:
            raise RuntimeError("Existing public-model destination has a different identity")
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(identity_path, identity)
    api, manifest = HfApi(token=False), {}
    with tempfile.TemporaryDirectory(prefix="task2_public_") as temporary:
        for kind, spec in identity.items():
            print(f"CPU: preparing pinned {kind} model files", flush=True)
            info = api.model_info(spec["model_id"], revision=spec["revision"], files_metadata=True)
            if info.sha != spec["revision"]:
                raise RuntimeError("Public model revision differs")
            selected = {item.rfilename: item for item in info.siblings
                        if len(Path(item.rfilename).parts) == 1 and
                        Path(item.rfilename).suffix in {".json", ".safetensors", ".txt", ".model", ".tiktoken", ".jinja"}}
            if "config.json" not in selected or not any(n.endswith(".safetensors") for n in selected):
                raise RuntimeError("Public snapshot lacks expected files")
            sizes = [item.size for item in selected.values()]
            if any(size is None for size in sizes) or shutil.disk_usage(temporary).free < sum(sizes) + 2**30:
                raise RuntimeError("Missing size metadata or insufficient temporary disk")
            folder = Path(temporary) / kind
            snapshot_download(spec["model_id"], revision=spec["revision"], token=False,
                              local_dir=folder, allow_patterns=list(selected), max_workers=4)
            hashes = {}
            for name, item in selected.items():
                path = folder / name
                if not path.is_file() or path.is_symlink() or path.stat().st_size != item.size:
                    raise RuntimeError(f"Incomplete/nonportable public file: {name}")
                digest = sha256(path)
                if item.lfs is not None:
                    if digest != item.lfs.sha256:
                        raise RuntimeError(f"Public LFS checksum differs: {name}")
                elif item.blob_id:
                    blob = hashlib.sha1(f"blob {item.size}\0".encode())
                    with path.open("rb") as f:
                        for block in iter(lambda: f.read(2**20), b""):
                            blob.update(block)
                    if blob.hexdigest() != item.blob_id:
                        raise RuntimeError(f"Public Git blob checksum differs: {name}")
                else:
                    raise RuntimeError(f"No public checksum for {name}")
                hashes[name] = digest
                saved = destination / kind / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                if saved.exists() and sha256(saved) != digest:
                    raise RuntimeError(f"Existing persistent model file differs: {saved}")
                if not saved.exists():
                    shutil.copyfile(path, saved)
                if sha256(saved) != digest:
                    raise RuntimeError(f"Persistent model copy differs: {saved}")
            config = json.loads((folder / "config.json").read_text())
            manifest[kind] = {**spec, "directory": kind, "sha256": hashes,
                              "bytes": sum(sizes), "max_position_embeddings": config["max_position_embeddings"]}
            # Free local files after each model to reduce ephemeral disk requirements.
            shutil.rmtree(folder)
    atomic_json(destination / "model_manifest.json", manifest)
    verify_public(destination, approval)
    return destination


def save_archive(paths, root, destination, extra=None):
    """One local archive then one verified Drive copy; minimal Drive small-file IO."""
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary) / destination.name
        with zipfile.ZipFile(local, "w", zipfile.ZIP_STORED) as z:
            for path in sorted(paths):
                if path.is_symlink():
                    raise RuntimeError(f"Do not persist symlinks: {path}")
                z.write(path, path.relative_to(root).as_posix())
            if extra:
                for name, value in extra.items():
                    z.writestr(name, json.dumps(value, indent=2))
        shutil.copyfile(local, destination)
        digest = sha256(local)
        if sha256(destination) != digest:
            raise RuntimeError(f"Persistent archive copy failed: {destination}")
        with zipfile.ZipFile(destination) as z:
            if z.testzip() is not None:
                raise RuntimeError("Persistent archive CRC check failed")
        return {"path": str(destination), "sha256": digest, "bytes": destination.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent-root", type=Path, default=Path("/content/drive/MyDrive/ATML-Assignment-2/task2"))
    parser.add_argument("--reuse-models", type=Path, default=Path(
        "/content/drive/MyDrive/ATML-Assignment-2/model_snapshots/20261004_002851_903477"))
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("This is CPU file preparation; disconnect the GPU first")
    if not str(args.persistent_root.resolve()).startswith("/content/drive/") or not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Mount Google Drive before preparing persistent files")
    approval = json.loads(repo_path("docs/task2_approval.json").read_text())
    verify_source(approval)
    installed = json.loads(repo_path("docs/task2_chunk2_files.json").read_text())
    for name, digest in installed.items():
        if sha256(repo_path(name)) != digest:
            raise RuntimeError(f"Installed source changed: {name}")
    checks = sorted(repo_path("results/task2_ppo/setup").glob("chunk2_*/validation.json"))
    validation = json.loads(checks[-1].read_text()) if checks else {}
    if (not validation.get("passed") or validation.get("tests_run") != 37
            or validation.get("installed_sha256") != installed):
        raise RuntimeError("Complete Chunk 2 CPU tests before preparation")
    cfg = load_yaml("configs/ppo.yaml")
    verify_config(cfg)
    asset_spec = dict(repo_id="AbDu11aHHH/ATML-PA2-assets", repo_type="dataset",
                      revision=approval["asset_revision"], local_dir=ROOT)
    print("CPU: restoring only the Task 2 course files", flush=True)
    snapshot_download(**asset_spec, allow_patterns=["manifests/**"])
    manifest_path = repo_path("manifests/sha256.json")
    if sha256(manifest_path) != approval["asset_manifest_sha256"]:
        raise RuntimeError("Pinned course asset manifest differs")
    manifest = json.loads(manifest_path.read_text())
    selected = [name for name in manifest if name.startswith(("checkpoints/ppo_midpoint_policy/", "checkpoints/ppo_midpoint_value/"))
                or name in {cfg["paths"]["rl_prompt_train"], cfg["paths"]["rl_prompt_eval"], cfg["cached_rollouts"],
                            "checkpoints/ppo_midpoint_acceptance.json"}]
    snapshot_download(**asset_spec, allow_patterns=selected)
    assets, train, evaluation = verify_assets(cfg, approval)
    schedule_path = repo_path("docs/task2_schedule.json")
    if json.loads(schedule_path.read_text()) != make_schedule(train, evaluation, cfg["seed"]):
        raise RuntimeError("Saved prompt schedule differs")
    if args.reuse_models.exists():
        print("CPU: checking the existing Task 1 public model files before reuse", flush=True)
        verify_public(args.reuse_models, approval, expected_manifest=TASK1_MANIFEST_SHA)
        public = args.reuse_models
    else:
        public = args.persistent_root / "public_models"
        if (public / "model_manifest.json").exists():
            verify_public(public, approval)
        else:
            public = download_public(public, approval)
    _, reward_config_audit = compatible_reward_config(public / "reward/config.json")
    if reward_config_audit["raw_config_sha256"] != approval["reward_config_sha256"]:
        raise RuntimeError("Prepared reward config differs")
    # Tokenizer loading is CPU-safe and ensures files suffice for offline loading.
    from task2_ppo.runtime import load_tokenizer_local
    tokenizer = load_tokenizer_local(public / "policy")
    del tokenizer
    stage = args.persistent_root / "cpu_prepared" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    stage.mkdir(parents=True, exist_ok=False)
    asset_paths = [repo_path(name) for name in selected]
    asset_paths += [p for p in repo_path("manifests").rglob("*") if p.is_file()]
    asset_paths = sorted(set(asset_paths))
    asset_archive = save_archive(asset_paths, ROOT, stage / "task2_assets.zip")
    preparation = {"status": "CPU_PREPARATION_VERIFIED", "gpu_used": False, "weights_loaded": False,
        "training_started": False, "gpu_launch_ready": False,
        "models_drive_dir": str(public), "model_manifest_sha256": sha256(public / "model_manifest.json"),
        "course_assets": asset_archive, "asset_files_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in asset_paths},
        "reward_config_translation": reward_config_audit, "verified_course_assets": assets,
        "note": "Training code is prepared; await the remaining evaluation/analysis chunk before allocating GPU time."}
    atomic_json(repo_path("docs/task2_cpu_preparation.json"), preparation)
    excluded = {"data", "cached", "checkpoints", "outputs", ".cache", ".venv", "__pycache__"}
    project_paths = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix != ".pyc"
                     and not excluded.intersection(p.relative_to(ROOT).parts)
                     and p.name != "TASK2_BACKUP_MANIFEST.json"]
    backup_manifest = {"files_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in project_paths},
                       "training_executed": False, "stage": "Chunk 2 CPU preparation"}
    project_archive = save_archive(project_paths, ROOT, stage / "project.zip",
        {"TASK2_BACKUP_MANIFEST.json": backup_manifest})
    preparation["project_archive"] = project_archive
    atomic_json(stage / "preparation.json", preparation)
    print("CPU PREPARATION VERIFIED:", stage / "preparation.json", flush=True)
    print("Public weights reused/prepared; course assets and updated code are saved and verified.", flush=True)
    print("No model weights loaded, no training run. Keep GPU off until the remaining jobs are ready.", flush=True)


if __name__ == "__main__":
    main()
