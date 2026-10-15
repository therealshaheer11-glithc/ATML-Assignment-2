"""CPU-only preparation of the exact public Task 1 data and model snapshots.

Publication utility added after the experiments. Checks recorded hashes,
never imports Torch, and never overwrites an existing mismatched asset.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from task1_dpo.analyze_results import read, require, sha

ROOT = Path(__file__).resolve().parents[1]


def requests(models_dir, scope):
    assets = read(ROOT / "docs/task1_assets_manifest.json")
    models = read(ROOT / "docs/task1_model_manifest.json")
    result = []
    if scope in ("all", "data"):
        for filename, digest in assets["sha256"].items():
            result.append({"repo_id": assets["repo_id"], "repo_type": "dataset", "revision": assets["revision"],
                           "filename": filename, "destination": str(ROOT / filename), "sha256": digest,
                           "tracked_only": filename == "data/word_limit_prompts.jsonl"})
    if scope in ("all", "models"):
        for entry in models.values():
            for filename, digest in entry["sha256"].items():
                result.append({"repo_id": entry["model_id"], "repo_type": "model", "revision": entry["revision"],
                               "filename": filename, "destination": str(models_dir / entry["directory"] / filename), "sha256": digest})
    return result


def prepare(models_dir, scope, verify_only=False):
    jobs = requests(models_dir, scope)
    manifest = models_dir / "model_manifest.json"
    source_manifest = ROOT / "docs/task1_model_manifest.json"
    if scope in ("all", "models") and manifest.exists():
        require(sha(manifest) == sha(source_manifest), "Existing model manifest differs; use a new directory.")
    for job in jobs:
        target = Path(job["destination"])
        if target.exists():
            require(sha(target) == job["sha256"], f"Existing file differs; preserve and review: {target}")
        elif verify_only or job.get("tracked_only"):
            raise FileNotFoundError(target)
    for index, job in enumerate(jobs, 1):
        target = Path(job["destination"])
        if not target.exists():
            from huggingface_hub import hf_hub_download
            cached = Path(hf_hub_download(**{k: job[k] for k in ("repo_id", "repo_type", "revision", "filename")}))
            require(sha(cached) == job["sha256"], f"Pinned download hash mismatch: {job['filename']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".prepare_", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                shutil.copyfile(cached, temporary)
                require(sha(temporary) == job["sha256"], "Local copy hash mismatch")
                require(not target.exists(), f"Destination appeared during copy: {target}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Verified {index}/{len(jobs)}: {target}", flush=True)
    if scope in ("all", "models"):
        if verify_only:
            require(manifest.is_file() and sha(manifest) == sha(source_manifest), "Prepared model manifest missing/different")
        elif not manifest.exists():
            manifest.write_bytes(source_manifest.read_bytes())
        require(sha(manifest) == sha(source_manifest), "Model manifest copy failed")
    print("TASK1_ASSETS_VERIFIED; no model loaded or training started.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models/task1")
    parser.add_argument("--scope", choices=("all", "data", "models"), default="all")
    parser.add_argument("--plan", action="store_true", help="Print requests; no downloads/writes")
    parser.add_argument("--verify-only", action="store_true", help="Check existing files without downloading")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(requests(args.models_dir, args.scope), indent=2))
    else:
        prepare(args.models_dir, args.scope, args.verify_only)


if __name__ == "__main__":
    main()
