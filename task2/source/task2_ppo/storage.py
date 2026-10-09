"""Atomic, hash-verified Task 2 records; no model or network operations."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        f.write(json_bytes(value))
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_directory(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "FILES_SHA256.json").read_text())
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*")
              if p.is_file() and p.name != "FILES_SHA256.json"}
    if actual != set(manifest):
        raise RuntimeError(f"Checkpoint file set differs: {folder}")
    for name, digest in manifest.items():
        path = folder / name
        if path.is_symlink() or sha256(path) != digest:
            raise RuntimeError(f"Checkpoint checksum differs: {name}")
    return manifest


def commit_directory(temporary, destination):
    temporary, destination = Path(temporary), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Committed evidence already exists: {destination}")
    files = {p.relative_to(temporary).as_posix(): sha256(p)
             for p in temporary.rglob("*") if p.is_file()}
    atomic_json(temporary / "FILES_SHA256.json", files)
    verify_directory(temporary)
    os.rename(temporary, destination)


def safe_extract(archive, destination):
    """Reject traversal, links, and duplicate names before extracting a ZIP."""
    destination = Path(destination).resolve()
    names = set()
    for item in archive.infolist():
        name = item.filename
        target = (destination / name).resolve()
        if target != destination and destination not in target.parents:
            raise RuntimeError(f"Unsafe archive member: {name}")
        if name in names or (item.external_attr >> 16) & 0o170000 == 0o120000:
            raise RuntimeError(f"Duplicate or symlink archive member: {name}")
        names.add(name)
    archive.extractall(destination)


def publish_directory(folder, destination):
    """Upload one archive per committed update; verify the saved bytes."""
    folder, destination = Path(folder), Path(destination)
    verify_directory(folder)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (folder.name + ".zip")
    receipt_path = destination / (folder.name + ".sha256.json")
    if target.exists():
        # Restoring a ZIP changes filesystem timestamps. Compare file contents,
        # rather than requiring a recompressed ZIP to have identical metadata.
        files = verify_directory(folder)
        with zipfile.ZipFile(target) as z:
            expected_names = set(files) | {"FILES_SHA256.json"}
            if len(z.namelist()) != len(expected_names) or set(z.namelist()) != expected_names:
                raise RuntimeError(f"Existing persistent archive file set differs: {target}")
            if json.loads(z.read("FILES_SHA256.json")) != files:
                raise RuntimeError(f"Existing persistent checkpoint differs: {target}")
            for name, digest in files.items():
                if hashlib.sha256(z.read(name)).hexdigest() != digest:
                    raise RuntimeError(f"Existing persistent checkpoint file differs: {name}")
        digest = sha256(target)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["sha256"] != digest or receipt["archive"] != target.name:
                raise RuntimeError(f"Persistent archive receipt differs: {target}")
        else:
            receipt = {"archive": target.name, "sha256": digest, "bytes": target.stat().st_size}
            atomic_json(receipt_path, receipt)
        return receipt
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary) / target.name
        with zipfile.ZipFile(local, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for p in sorted(folder.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(folder).as_posix())
        digest = sha256(local)
        staging = destination / (target.name + ".partial")
        shutil.copyfile(local, staging)
        if sha256(staging) != digest:
            raise RuntimeError(f"Persistent copy failed verification: {target}")
        os.replace(staging, target)
        receipt = {"archive": target.name, "sha256": digest, "bytes": target.stat().st_size}
        atomic_json(receipt_path, receipt)
        return receipt


def restore_published(archive_path, destination):
    archive_path, destination = Path(archive_path), Path(destination)
    receipt = json.loads(archive_path.with_suffix(".sha256.json").read_text())
    if receipt["archive"] != archive_path.name or sha256(archive_path) != receipt["sha256"]:
        raise RuntimeError("Persistent checkpoint archive checksum differs")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        extracted = Path(temporary) / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive_path) as z:
            safe_extract(z, extracted)
        verify_directory(extracted)
        os.rename(extracted, destination)
