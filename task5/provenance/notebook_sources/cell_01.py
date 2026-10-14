# Task 5 — Block 2: targeted CPU asset inventory
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, json, os, shutil, subprocess, uuid

project = Path("/content/drive/MyDrive/ATML-Assignment-2")
root = project / "task5/experiment_v1"
previous = root / (
    "inventory_attempts/"
    "20261012T162435.079975Z_249d0dc3/REPORT.json"
)
if not previous.is_file():
    raise RuntimeError("The saved Block 1 report is unavailable. Keep Drive mounted.")

smi = shutil.which("nvidia-smi")
probe = subprocess.run(
    [smi, "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True, text=True, timeout=15,
) if smi else None

if (
    (probe and probe.returncode == 0 and probe.stdout.strip())
    or os.environ.get("COLAB_TPU_ADDR")
    or os.environ.get("TPU_NAME")
):
    raise RuntimeError("Keep hardware accelerator set to None for this inventory.")

baseline = json.loads(previous.read_text())
if baseline["status"] != "CPU_INITIAL_INVENTORY_SAVED_REVIEW_PENDING":
    raise RuntimeError("Block 1 status differs from the returned result.")

assets = project / "task3/experiment_v1/course_assets"

# Recorded fingerprints from the preserved original course asset manifest.
expected = {
    "data/gsm8k_eval.jsonl":
        "24f712750928c18c1a0a6849eebe97b3e01a6c0c0ebf2410d101a3df1872bf4a",
    "data/task5_controlled_reward_diagnostics.jsonl":
        "9008b96595bb1419f1326bf02beefd689f09028c2aa6598d7c427394602f036c",
    "checkpoints/rlvr_policy/adapter_config.json":
        "b7c4427b92d1514135b9f3530e3fc9134a1f9f460f980c8798a22c7d5fd0d4ae",
    "checkpoints/rlvr_policy/adapter_model.safetensors":
        "b309564b45c0268e4e5ffb0db5cb100e557a2c407a68ea3fa3de91fe964d92c8",
    "checkpoints/rlaif_policy/adapter_config.json":
        "b7c4427b92d1514135b9f3530e3fc9134a1f9f460f980c8798a22c7d5fd0d4ae",
    "checkpoints/rlaif_policy/adapter_model.safetensors":
        "e5a0eaacb3ce32a73fe86fd6a823be21c60a460eee8682493e5cc0f2e1019713",
}

stamp = (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    + "_" + uuid.uuid4().hex[:8]
)
attempt = root / "asset_inventory_attempts" / stamp
attempt.mkdir(parents=True, exist_ok=False)

report = {
    "status": "RUNNING",
    "asset_root": str(assets),
    "files": {},
    "datasets": {},
    "saved_records": {},
    "model_locations": {},
    "errors": [],
    "installs": 0,
    "downloads": 0,
    "model_loads": 0,
    "policy_responses": 0,
    "ai_judgments": 0,
    "previous_files_modified": False,
}

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()

try:
    report["previous_report_sha256"] = digest(previous)

    for index, (relative, wanted) in enumerate(expected.items(), 1):
        print(
            f"Checking pinned course file {index}/{len(expected)}: {relative}",
            flush=True,
        )
        path = assets / relative
        if not path.is_file() or path.is_symlink():
            report["files"][relative] = {
                "status": "NOT_FOUND_AT_RECORDED_LOCATION"
            }
            continue

        actual = digest(path)
        report["files"][relative] = {
            "path": str(path),
            "sha256": actual,
            "matches_recorded_course_hash": actual == wanted,
            "bytes": path.stat().st_size,
        }
        if actual != wanted:
            report["errors"].append("Course checksum differs: " + relative)

    for filename in (
        "gsm8k_eval.jsonl",
        "task5_controlled_reward_diagnostics.jsonl",
        "math_transfer_eval.jsonl",
    ):
        path = assets / "data" / filename
        if not path.is_file() or path.is_symlink():
            report["datasets"][filename] = {
                "status": "NOT_FOUND_AT_RECORDED_LOCATION"
            }
            continue

        if path.stat().st_size > 16 * 2**20:
            raise RuntimeError("Unexpected evaluation-file size: " + str(path))

        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        info = {
            "path": str(path),
            "rows": len(rows),
            "sha256": digest(path),
            "row_keys": sorted(set().union(*(row.keys() for row in rows))),
        }

        if "diagnostics" in filename:
            pairs = [
                (str(row["problem_id"]), row["variant_type"])
                for row in rows
            ]
            info.update(
                problems=len({pid for pid, variant in pairs}),
                variant_counts=dict(Counter(variant for pid, variant in pairs)),
                duplicate_problem_variant_pairs=len(pairs) - len(set(pairs)),
                all_marked_manually_validated=(
                    bool(rows)
                    and all(row.get("manual_validation") is True for row in rows)
                ),
            )

        if filename == "math_transfer_eval.jsonl":
            info["source_indices_in_file_order"] = [
                row.get("source_index") for row in rows
            ]
            info["official_source_order_independently_verified"] = False

        report["datasets"][filename] = info

    records = [
        "task3/experiment_v1/PREPARATION.json",
        "task3/experiment_v1/models/model_manifest.json",
        "task4/experiment_v1/CPU_MODEL_INPUT_CONTRACT.json",
        "task4/experiment_v1/JUDGE_SNAPSHOT_CONTRACT.json",
        "task4/experiment_v1/environment_wheelhouse/"
        "VERIFIED_WHEELHOUSE_CONTRACT.json",
    ]
    for relative in records:
        path = project / relative
        if path.is_file() and not path.is_symlink():
            value = json.loads(path.read_text())
            report["saved_records"][relative] = {
                "sha256": digest(path),
                "top_level_keys": sorted(value),
                "historical_status": value.get("status"),
                "historical_format": value.get("format"),
            }
        else:
            report["saved_records"][relative] = {
                "status": "NOT_FOUND_AT_RECORDED_LOCATION"
            }

    for label, relative in [
        ("base_policy", "task3/experiment_v1/models/policy"),
        ("judge", "task4/experiment_v1/models/judge"),
    ]:
        folder = project / relative
        report["model_locations"][label] = {
            "path": str(folder),
            "exists": folder.is_dir(),
            "files": [
                {"name": p.name, "bytes": p.stat().st_size}
                for p in sorted(folder.iterdir())
                if p.is_file() and not p.is_symlink()
            ] if folder.is_dir() else [],
            "weights_freshly_integrity_verified": False,
        }

    report["status"] = "TARGETED_CPU_INVENTORY_SAVED_REVIEW_PENDING"

except Exception as exc:
    report["status"] = "TARGETED_CPU_INVENTORY_STOPPED_REVIEW_PENDING"
    report["errors"].append(type(exc).__name__ + ": " + str(exc))

finally:
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    temporary = attempt / "REPORT.pending"
    destination = attempt / "REPORT.json"
    temporary.write_bytes(payload)
    os.replace(temporary, destination)

    if destination.read_bytes() != payload:
        raise RuntimeError("Saved inventory failed its readback check.")

    print(json.dumps(report, indent=2))
    print("Saved report:", destination)
    print("Send this output before continuing.")
    print("No installations, downloads, or model inference performed.")