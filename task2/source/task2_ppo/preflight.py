"""Task 2 setup validation and frozen schedules; loads no model weights."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from common.data import load_yaml, read_jsonl, repo_path
from task2_ppo.reward_compat import compatible_reward_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_source(approval):
    head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    require(head == approval["course_commit"], "Course commit differs from the approved source.")
    hashes = {}
    for relative, expected in approval["source_sha256"].items():
        path = REPO_ROOT / relative
        require(path.is_file(), f"Missing source file: {relative}")
        observed = sha256(path)
        require(observed == expected, f"Unexpected source modification: {relative}")
        hashes[relative] = observed
    return {"head": head, "verified_files": hashes}


def verify_config(cfg):
    expected = {
        "updates": 20, "fork_updates": 8, "prompts_per_update": 1, "ppo_epochs": 2,
        "policy_learning_rate": 3e-6, "value_lora_learning_rate": 1e-4,
        "value_head_learning_rate": 3e-4, "value_train_mode": "lora_head",
        "clip_epsilon": .2, "clip_values": [.05, .2, .5],
        "kl_beta": .1, "kl_values": [0., .1, .2], "gamma": 1., "gae_lambda": .95,
        "value_coef": .5, "missing_eos_penalty": 1., "max_prompt_length": 256,
        "max_response_length": 512, "eval_max_response_length": 768,
        "reward_max_length": 1280, "max_grad_norm": 1., "seed": 6304,
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "reward_model": "yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback",
        "reward_tokenizer": "Qwen/Qwen2.5-1.5B-Instruct",
        "dtype": "float16", "quantize_frozen_models": True,
        "generation": {"temperature": .7, "top_p": .9, "do_sample": True},
    }
    for key, value in expected.items():
        require(cfg.get(key) == value, f"Release setting differs: {key}")
    lora = {"r": 8, "alpha": 16, "dropout": .05, "target_modules": ["q_proj", "v_proj"]}
    require(cfg["lora"] == lora and cfg["value_lora"] == lora, "LoRA settings differ.")
    for key, path in {
        "ppo_midpoint_policy": "checkpoints/ppo_midpoint_policy",
        "ppo_midpoint_value": "checkpoints/ppo_midpoint_value",
        "rl_prompt_train": "data/rl_prompt_pool_train.jsonl",
        "rl_prompt_eval": "data/rl_prompt_pool_eval.jsonl",
    }.items():
        require(cfg["paths"].get(key) == path, f"Release asset path differs: {key}")
    return expected


def verify_environment():
    pins = {"transformers": "4.57.1", "tokenizers": "0.22.1", "peft": "0.17.1", "trl": "0.27.2"}
    for package, expected in pins.items():
        actual = importlib.metadata.version(package)
        require(actual == expected, f"{package}: expected {expected}, found {actual}")
    require(torch.cuda.is_available(), "No CUDA GPU is available. Select a GPU runtime before setup.")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    require(tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 4), "PyTorch is below the course minimum.")
    packages = json.loads(subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"], text=True
    ))
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "gpu": props.name,
            "gpu_total_vram_gib": props.total_memory / 2**30,
            "logical_gpu_index": device, "packages": packages}


def verify_assets(cfg, approval):
    manifest_path = REPO_ROOT / "manifests/sha256.json"
    require(sha256(manifest_path) == approval["asset_manifest_sha256"], "Pinned asset manifest differs.")
    manifest = json.loads(manifest_path.read_text())
    relative_files = sorted(p for p in manifest if
        p.startswith(("checkpoints/ppo_midpoint_policy/", "checkpoints/ppo_midpoint_value/")) or
        p in {cfg["paths"]["rl_prompt_train"], cfg["paths"]["rl_prompt_eval"], cfg["cached_rollouts"]})
    require("checkpoints/ppo_midpoint_value/model.safetensors" in relative_files, "Manifest lacks the released critic weights.")
    checked = {}
    for relative in relative_files:
        path = REPO_ROOT / relative
        require(path.is_file(), f"Missing PPO asset: {relative}")
        observed = sha256(path)
        require(observed == manifest[relative], f"PPO asset hash mismatch: {relative}")
        checked[relative] = observed
    train = read_jsonl(cfg["paths"]["rl_prompt_train"])
    evaluation = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    require(len(train) == 1200 and len(evaluation) == 200, "Unexpected RL prompt pool sizes.")
    train_ids = {row["prompt_id"] for row in train}
    eval_ids = {row["prompt_id"] for row in evaluation}
    require(len(train_ids) == len(train) and len(eval_ids) == len(evaluation), "Duplicate prompt IDs.")
    require(train_ids.isdisjoint(eval_ids), "Training and evaluation IDs overlap.")
    # Hash verification precedes deserialization of the trusted, pinned course cache.
    rows = torch.load(repo_path(cfg["cached_rollouts"]), map_location="cpu", weights_only=False)
    require(isinstance(rows, list) and len(rows) == 32, "Unexpected fixed PPO cache size.")
    for row in rows:
        n = int(row["response_tokens"])
        require(row["prompt_id"] in eval_ids and row["prompt_id"] not in train_ids, "Cache prompt provenance differs.")
        for key in ("old_logprobs", "ref_logprobs", "values"):
            require(isinstance(row[key], torch.Tensor) and tuple(row[key].shape) == (n,), f"Cache shape mismatch: {key}")
            require(torch.isfinite(row[key]).all().item(), f"Nonfinite cached {key}.")
        penalty = 0. if row["terminated_with_eos"] else cfg["missing_eos_penalty"]
        require(abs(row["raw_terminal_reward"] - penalty - row["effective_terminal_reward"]) < 1e-6,
                "Cache missing-EOS penalty semantics differ.")
    return {"verified_files": checked, "cache_rows": len(rows),
            "cache_is_held_out": True, "train_rows": len(train), "eval_rows": len(evaluation)}, train, evaluation


def make_schedule(train, evaluation, seed):
    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)
    training = [{"update_index": step, "row_index": index,
                 "prompt_id": train[index]["prompt_id"], "source_index": train[index]["source_index"],
                 "rollout_seed": seed + step}
                for step, index in enumerate(indices[:20])]
    held_out = [{"row_index": index, "prompt_id": row["prompt_id"],
                 "source_index": row["source_index"], "generation_seed": seed + 100_000 + index}
                for index, row in enumerate(evaluation)]
    return {"base_seed": seed, "algorithm": "random.Random(seed).shuffle(row_indices)",
            "training": training, "short_forks_use_first_n_updates": 8, "evaluation": held_out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-config", required=True, help="Pinned reward config.json; no weights needed.")
    parser.add_argument("--objective-report", required=True, help="Fresh Chunk 1 unittest result JSON.")
    args = parser.parse_args()
    approval = json.loads((REPO_ROOT / "docs/task2_approval.json").read_text())
    source = verify_source(approval)
    objective = json.loads(Path(args.objective_report).read_text())
    require(objective.get("passed") is True and objective.get("return_code") == 0 and objective.get("tests_run") == 21,
            "Objective/setup checks did not pass.")
    require(objective["test_file_sha256"] == sha256(REPO_ROOT / "tests/test_task2_objective.py"),
            "Test file changed after validation.")
    cfg = load_yaml("configs/ppo.yaml")
    verify_config(cfg)
    environment = verify_environment()
    assets, train, evaluation = verify_assets(cfg, approval)
    _, reward = compatible_reward_config(args.reward_config)
    require(reward["raw_config_sha256"] == approval["reward_config_sha256"], "Pinned reward config bytes differ.")
    schedule = make_schedule(train, evaluation, int(cfg["seed"]))
    schedule_path = REPO_ROOT / "docs/task2_schedule.json"
    encoded_schedule = (json.dumps(schedule, indent=2) + "\n").encode()
    if schedule_path.exists():
        require(schedule_path.read_bytes() == encoded_schedule, "Existing frozen prompt schedule differs; do not overwrite.")
    else:
        schedule_path.write_bytes(encoded_schedule)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    out = REPO_ROOT / "results/task2_ppo/setup" / run_id
    out.mkdir(parents=True, exist_ok=False)
    record = {"chunk1_complete": True, "training_executed": False, "model_weights_loaded": False,
              "source": source, "environment": environment, "assets": assets,
              "reward_config_translation": reward, "release_config": cfg,
              "schedule_sha256": hashlib.sha256(encoded_schedule).hexdigest(),
              "objective_tests": objective,
              "approval": approval}
    (out / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    print("CHUNK 1 COMPLETE: objective tests and setup checks passed.")
    print("GPU:", environment["gpu"], "| VRAM GiB:", round(environment["gpu_total_vram_gib"], 2))
    print("Results:", out.relative_to(REPO_ROOT))
    print("No model weights were loaded and no PPO training was run.")


if __name__ == "__main__":
    main()
