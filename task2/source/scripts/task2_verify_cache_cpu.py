"""Check all real cached rows with the pinned tokenizer; load no model weights."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from common.data import load_yaml, read_jsonl, repo_path
from task2_ppo.cache_geometry import fixed_cached_advantages, reconstruct_cached
from task2_ppo.preflight import verify_assets, verify_config, verify_source
from task2_ppo.runtime import load_tokenizer_local
from task2_ppo.storage import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-folder", required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("Use CPU for cache reconstruction")
    approval = json.loads(repo_path("docs/task2_approval.json").read_text())
    verify_source(approval)
    cfg = load_yaml("configs/ppo.yaml")
    verify_config(cfg)
    assets, train, evaluation = verify_assets(cfg, approval)
    folder = Path(args.policy_folder)
    preparation = json.loads(repo_path("docs/task2_cpu_preparation.json").read_text())
    manifest_path = folder.parent / "model_manifest.json"
    if sha256(manifest_path) != preparation["model_manifest_sha256"]:
        raise RuntimeError("Prepared tokenizer manifest differs")
    manifest = json.loads(manifest_path.read_text())["policy"]
    if manifest["revision"] != approval["policy_revision"]:
        raise RuntimeError("Tokenizer revision differs")
    for name, digest in manifest["sha256"].items():
        if name in {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "generation_config.json", "config.json"}:
            if sha256(folder / name) != digest:
                raise RuntimeError("Prepared tokenizer/config file differs: " + name)
    tokenizer = load_tokenizer_local(args.policy_folder)
    by_id = {r["prompt_id"]: r for r in evaluation}
    cached = torch.load(repo_path(cfg["cached_rollouts"]), map_location="cpu", weights_only=False)
    result = []
    for index, row in enumerate(cached):
        batch = reconstruct_cached(tokenizer, row, by_id[row["prompt_id"]], cfg)
        fixed_cached_advantages(row, cfg)
        result.append({"row": index, "prompt_id": row["prompt_id"], "tokens": row["response_tokens"],
            "exact_length_and_text": True, "response_ids_sha256": hashlib.sha256(
                json.dumps(batch["response_ids"][0].tolist()).encode()).hexdigest()})
    atomic_json(repo_path("results/task2_ppo/setup/cache_reconstruction_cpu.json"), {
        "rows": result, "passed": len(result) == 32, "course_weights_loaded": False,
        "cache_sha256": sha256(repo_path(cfg["cached_rollouts"])),
        "limitation": "Original cache token IDs are absent; length/text agreement does not prove tokenization uniqueness."})
    print(f"CACHE CPU CHECK: {len(result)}/32 exact token-count and text round-trips; advantages finite.", flush=True)


if __name__ == "__main__":
    main()
