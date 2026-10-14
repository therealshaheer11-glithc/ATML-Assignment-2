# Task 5 — Block 1: CPU inventory
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import hashlib, json, os, platform, shutil, subprocess, sys, uuid

print("1/4: checking that this is a CPU runtime.", flush=True)
if not Path("/content").is_dir():
    raise RuntimeError("Run this block in Google Colab.")

smi = shutil.which("nvidia-smi")
gpu = subprocess.run(
    [smi, "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True, text=True, timeout=15,
) if smi else None

if gpu and gpu.returncode == 0 and gpu.stdout.strip():
    raise RuntimeError(
        "Select Runtime > Change runtime type > Hardware accelerator: None, then rerun."
    )
if os.environ.get("COLAB_TPU_ADDR") or os.environ.get("TPU_NAME"):
    raise RuntimeError("Select a CPU runtime for this block, then rerun.")

from google.colab import drive
if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive")
if not Path("/content/drive/MyDrive").is_dir():
    raise RuntimeError("Google Drive was not mounted successfully.")

project = Path("/content/drive/MyDrive/ATML-Assignment-2")
root = project / "task5/experiment_v1"
attempt = root / "inventory_attempts" / (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    + "_" + uuid.uuid4().hex[:8]
)
attempt.mkdir(parents=True, exist_ok=False)

report = {
    "format": "ATML_PA2_TASK5_INITIAL_INVENTORY_V1",
    "status": "RUNNING",
    "attempt_utc": datetime.now(timezone.utc).isoformat(),
    "approved_scope": "Stage 1 CPU inventory and Task 5 progress-folder creation only",
    "python": sys.version,
    "platform": platform.platform(),
    "progress_folder": str(root),
    "installs": 0,
    "downloads": 0,
    "model_loads": 0,
    "policy_responses": 0,
    "ai_judgments": 0,
    "existing_files_modified": False,
    "asset_discovery_exhaustive": False,
    "full_asset_integrity_validated": False,
    "expected_course_source_commit": "1d64ac65acd5e45d1e4e1f415edc80455e21274e",
    "expected_course_asset_revision": "0b350481fb03f5525a35bcdec4131bd4fe487f98",
    "packages": {},
    "repositories": [],
    "data_files": [],
    "adapter_files": [],
    "inspection_errors": [],
}

try:
    print("2/4: recording installed versions; no packages are being changed.", flush=True)
    pins = {
        "transformers": "4.57.1",
        "tokenizers": "0.22.1",
        "trl": "0.27.2",
        "peft": "0.17.1",
    }
    names = list(pins) + [
        "torch", "datasets", "accelerate", "bitsandbytes", "huggingface_hub",
        "safetensors", "pyyaml", "pandas", "numpy", "scikit-learn", "matplotlib",
    ]
    for name in names:
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = None
        report["packages"][name] = {"installed": version}
        if name in pins:
            report["packages"][name].update(
                required=pins[name], matches=(version == pins[name])
            )

    check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True, text=True, timeout=30,
    )
    report["pip_check"] = {
        "returncode": check.returncode,
        "stdout": check.stdout,
        "stderr": check.stderr,
    }
    report["local_disk_free_gib"] = round(
        shutil.disk_usage("/content").free / 2**30, 2
    )

    print("3/4: inspecting existing project storage and nearby asset locations.", flush=True)
    candidates = {project}
    excluded = {"drive", "sample_data", ".config", "task5"}
    for parent in (project, Path("/content")):
        children = sorted(parent.iterdir(), key=lambda p: p.name)
        label = "project_children" if parent == project else "runtime_children"
        report[label] = [p.name for p in children]
        folders = [
            p for p in children
            if p.is_dir() and not p.is_symlink() and p.name not in excluded
        ]
        if len(folders) > 30:
            report["inspection_errors"].append(
                "Initial folder scan limited to 30 children of " + str(parent)
            )
        for folder in folders[:30]:
            candidates.add(folder)
            for suffix in (
                "repo", "repository", "source",
                "course-starter", "course_assets", "assets",
            ):
                nested = folder / suffix
                if nested.is_dir() and not nested.is_symlink():
                    candidates.add(nested)

    report["candidate_roots_inspected"] = [
        str(p) for p in sorted(candidates)
    ]
    wanted = (
        "gsm8k_eval.jsonl",
        "math_transfer_eval.jsonl",
        "task5_controlled_reward_diagnostics.jsonl",
    )
    for base in sorted(candidates):
        if (base / ".git").exists() and shutil.which("git"):
            result = subprocess.run(
                ["git", "-C", str(base), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=15,
            )
            report["repositories"].append({
                "path": str(base),
                "head": result.stdout.strip(),
                "returncode": result.returncode,
                "error": result.stderr.strip(),
            })

        for filename in wanted:
            path = base / "data" / filename
            if not path.is_file() or path.is_symlink():
                continue
            try:
                if path.stat().st_size > 16 * 2**20:
                    raise ValueError(
                        "Unexpectedly large evaluation file; inspect separately."
                    )
                payload = path.read_bytes()
                rows = [
                    json.loads(line)
                    for line in payload.decode("utf-8").splitlines()
                    if line.strip()
                ]
                report["data_files"].append({
                    "path": str(path),
                    "rows": len(rows),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "first_row_keys": (
                        sorted(rows[0])
                        if rows and isinstance(rows[0], dict) else []
                    ),
                })
            except Exception as exc:
                report["inspection_errors"].append(
                    str(path) + ": " + str(exc)
                )

        for policy in ("rlvr", "rlaif"):
            folder = base / "checkpoints" / (policy + "_policy")
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                path = folder / filename
                if path.is_file() and not path.is_symlink():
                    report["adapter_files"].append({
                        "policy": policy,
                        "path": str(path),
                        "bytes": path.stat().st_size,
                    })

    report["status"] = "CPU_INITIAL_INVENTORY_SAVED_REVIEW_PENDING"

except Exception as exc:
    report["status"] = "CPU_INVENTORY_STOPPED_REVIEW_PENDING"
    report["error"] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }

finally:
    print("4/4: saving this inventory attempt to Google Drive.", flush=True)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    temporary = attempt / "REPORT.pending"
    destination = attempt / "REPORT.json"
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    if destination.read_bytes() != payload:
        raise RuntimeError("Saved report failed its readback check.")

    print("\nStatus:", report["status"])
    print("Python:", platform.python_version())
    print("Project folders:", report.get("project_children", []))
    print("Runtime folders:", report.get("runtime_children", []))
    print("Packages:", json.dumps(report["packages"], indent=2))
    print("Dependency check:", json.dumps(report.get("pip_check", {}), indent=2))
    print("Repositories:", json.dumps(report["repositories"], indent=2))
    print("Evaluation files:", json.dumps(report["data_files"], indent=2))
    print(
        "Adapter files (presence/size only):",
        json.dumps(report["adapter_files"], indent=2),
    )
    print("Inspection notes:", report["inspection_errors"])
    if "error" in report:
        print("Error:", report["error"])
    print("Saved report:", destination)
    print(
        "Scope: 0 installs; 0 downloads; 0 model loads; "
        "0 policy responses; 0 AI judgments."
    )
    print("Initial inventory only. Send the printed output before continuing.")