"""Fresh-process, training-only probe check of a final Task 1 adapter."""
import argparse
import json
from pathlib import Path
import torch
from common.models import reference_mode
from task1_dpo.runtime import pair_scores
from task1_dpo.support import load_pinned_policy, load_pinned_tokenizer, model_source, parameter_digest, save_json, validate_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    folder = Path(args.checkpoint)
    result = {"status": "RUNNING", "held_out_examples_used": False}
    try:
        record = json.loads((folder / "run_record.json").read_text())
        if record["status"] not in {"TRAINED_PENDING_RELOAD", "COMPLETE_RELOAD_VERIFIED"}:
            raise ValueError("Checkpoint is not a completed training budget.")
        cfg = record["effective_config"]
        validate_config(cfg)
        source, kwargs = model_source(cfg, record["models_dir"])
        tokenizer = load_pinned_tokenizer(cfg, source, kwargs)
        model = load_pinned_policy(cfg, source, kwargs, tokenizer, adapter=folder)
        if parameter_digest(model) != record["frozen_base_before_sha256"]:
            raise RuntimeError("Reloaded frozen-base hash mismatch.")
        if parameter_digest(model, adapters=True) != record["final_adapter_sha256"]:
            raise RuntimeError("Reloaded adapter hash mismatch.")
        saved = torch.load(folder / "reload_probe.pt", map_location="cpu", weights_only=True)
        batches = tuple({k: t.cuda() for k, t in b.items()} for b in saved["batches"])
        with torch.no_grad():
            policy = pair_scores(model, batches)
        with torch.no_grad(), reference_mode(model):
            reference = pair_scores(model, batches)
        for actual, expected in zip(policy, saved["policy_logp"]):
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=1e-4)
        for actual, expected in zip(reference, saved["reference_logp"]):
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=1e-4)
        result.update({"status": "PASS", "probe_prompt_ids": saved["prompt_ids"],
                       "policy_max_abs_changes": [(a.cpu() - b).abs().max().item() for a, b in zip(policy, saved["policy_logp"])],
                       "reference_max_abs_changes": [(a.cpu() - b).abs().max().item() for a, b in zip(reference, saved["reference_logp"])]})
        print("FRESH_CHECKPOINT_RELOAD_OK", flush=True)
    except Exception as error:
        result["status"] = "FAIL"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save_json(folder / "reload_verification.json", result)


if __name__ == "__main__":
    main()
