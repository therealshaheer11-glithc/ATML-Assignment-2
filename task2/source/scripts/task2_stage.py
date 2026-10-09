"""Portable CPU/GPU file staging; streamed checksums, no model loading."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from task2_ppo.storage import safe_extract, sha256


def checked_copy(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if sha256(destination) != expected:
            raise RuntimeError(f"Existing staged file differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    h = hashlib.sha256()
    with source.open("rb") as src, temporary.open("wb") as dst:
        for block in iter(lambda: src.read(2**20), b""):
            dst.write(block)
            h.update(block)
    if h.hexdigest() != expected:
        raise RuntimeError(f"Source/copy checksum differs: {source}")
    os.replace(temporary, destination)


def restore_archive(spec, destination, expected_files=None, project=False):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task2_restore_", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        local = temporary / "snapshot.zip"
        checked_copy(spec["path"], local, spec["sha256"])
        with zipfile.ZipFile(local) as archive:
            if project:
                manifest = json.loads(archive.read("TASK2_BACKUP_MANIFEST.json"))
                expected_files = manifest["files_sha256"]
                names = set(expected_files) | {"TASK2_BACKUP_MANIFEST.json"}
            else:
                names = set(expected_files)
            if set(archive.namelist()) != names or len(archive.namelist()) != len(names):
                raise RuntimeError("Staged archive inventory differs")
            for name, digest in expected_files.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise RuntimeError(f"Staged archive member differs: {name}")
            draft = temporary / "extracted"
            draft.mkdir()
            safe_extract(archive, draft)
        # Never replace modified source or evidence. Missing matching files can
        # be restored into an existing interrupted stage without resetting it.
        for name in expected_files:
            target = destination / name
            if target.exists() and sha256(target) != expected_files[name]:
                raise RuntimeError(f"Existing restored file differs: {name}")
        destination.mkdir(parents=True, exist_ok=True)
        for name in expected_files:
            target = destination / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(draft / name, target)


def restore_files(preparation, repo):
    repo = Path(repo)
    restore_archive(preparation["project_archive"], repo, project=True)
    files = preparation["asset_files_sha256"]
    missing = False
    for name, expected in files.items():
        path = repo / name
        if not path.exists():
            missing = True
        elif sha256(path) != expected:
            raise RuntimeError(f"Existing course asset differs: {name}")
    if missing:
        restore_archive(preparation["course_assets"], repo, files)


def stage_models(preparation, destination):
    source, destination = Path(preparation["models_drive_dir"]), Path(destination)
    manifest_path = source / "model_manifest.json"
    if sha256(manifest_path) != preparation["model_manifest_sha256"]:
        raise RuntimeError("Prepared model manifest differs")
    manifest = json.loads(manifest_path.read_text())
    checked_copy(manifest_path, destination / "model_manifest.json", preparation["model_manifest_sha256"])
    for kind in ("policy", "reward"):
        print(f"Staging verified {kind} files from Drive to local disk", flush=True)
        entry = manifest[kind]
        folder = Path(entry["directory"])
        if folder.is_absolute() or ".." in folder.parts:
            raise RuntimeError("Unsafe model directory")
        for name, digest in entry["sha256"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Unsafe model filename")
            checked_copy(source / folder / name, destination / folder / name, digest)
