"""One disposable DPO update on 16 long training pairs; no official training.

Uses the approved prompt-preserving 4096-token rule only inside this process.
Course source/config files and original datasets remain unchanged. Fresh base
and tokenizer loads are pinned to the revision recorded in the data audit.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from peft import get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.data import (
    load_yaml, pad_batch, preference_responses,
    prompt_messages_from_preference, read_jsonl, repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import set_seed
from common.models import make_lora_config, reference_mode
from task1_dpo.dpo import dpo_loss


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(str(parameter.dtype).encode())
            digest.update(str(tuple(parameter.shape)).encode())
            array = parameter.detach().cpu().contiguous().numpy()
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def trial_collate(tok, cap):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = tok.apply_chat_template(
                prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True,
            )
            assert 0 < len(prompt) < cap - 1
            for target, answer in zip((chosen, rejected), preference_responses(row)):
                raw = tok(answer, add_special_tokens=False)["input_ids"]
                kept = raw[:cap - len(prompt) - 1]
                assert len(kept) == len(raw), "The completed audit reported no truncation."
                response = kept + [tok.eos_token_id]
                target.append((prompt + response, [0] * len(prompt) + [1] * len(response)))
        return pad_batch(tok, chosen), pad_batch(tok, rejected), [r["prompt_id"] for r in rows]
    return collate


def scores(model, batches):
    return [response_sequence_logprobs(model, batch)[0] for batch in batches]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    report = {"status": "RUNNING", "official_training_started": False,
              "diagnostic_optimizer_updates": 0, "stage": "setup"}
    start = time.perf_counter()
    watched, before = [], {}
    try:
        assert torch.cuda.is_available(), "Reconnect the approved A100 GPU runtime."
        audits = sorted(repo_path("results/setup").glob("cap4096_audit_*/audit_summary.json"))
        assert audits, "The completed 4096-token audit is missing."
        audit_path = audits[-1]
        audit = json.loads(audit_path.read_text())
        assert audit["status"] == "AUDIT_COMPLETE" and audit["trial_combined_cap"] == 4096
        assert audit["removed_pairs"] == 0
        for item in audit["summaries"].values():
            assert item["prompt_truncated"] == 0
            assert all(side["truncated"] == 0 and side["only_EOS_retained"] == 0
                       and side["empty_original_content"] == 0 for side in item["responses"].values())
        for relative, digest in audit["dataset_sha256"].items():
            assert sha256(repo_path(relative)) == digest, f"Dataset changed: {relative}"
        for relative, digest in audit["unchanged_source_sha256"].items():
            assert sha256(repo_path(relative)) == digest, f"Source/config changed: {relative}"
        for name in ("standard_eval", "word_limit_prompts"):
            assert audit["generation"][name]["prompts_exceeding_trial_cap"] == 0

        cfg = load_yaml("configs/dpo.yaml")
        assert cfg["dtype"] == "float16"
        assert (int(cfg["batch_size"]), int(cfg["grad_accum_steps"])) == (2, 8)
        assert float(cfg["beta"]) == .10 and float(cfg["max_grad_norm"]) == 1.0
        set_seed(int(cfg["seed"]))
        provenance = audit["tokenizer"]
        assert cfg["base_model"] == provenance["model_id"]
        tok = AutoTokenizer.from_pretrained(
            cfg["base_model"], revision=provenance["resolved_revision"],
            use_fast=True, padding_side="left", local_files_only=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        assert hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest() == provenance["tokenizer_backend_sha256"]
        assert hashlib.sha256(json.dumps(tok.chat_template, sort_keys=True).encode()).hexdigest() == provenance["chat_template_sha256"]
        assert (tok.pad_token_id, tok.eos_token_id) == (provenance["pad_token_id"], provenance["eos_token_id"])

        # Stress selection uses training lengths only, never held-out scores.
        with (audit_path.parent / "per_pair_token_counts.csv").open(newline="") as handle:
            candidates = [r for r in csv.DictReader(handle)
                          if r["dataset"] in ("standard_train", "balanced_train")]
        candidates.sort(key=lambda r: (
            -max(int(r["chosen_sequence_tokens"]), int(r["rejected_sequence_tokens"])),
            r["dataset"], int(r["row_index"]),
        ))
        selected = candidates[:16]
        training_sets = {
            "standard_train": read_jsonl(cfg["paths"]["dpo_standard_train"]),
            "balanced_train": read_jsonl(cfg["paths"]["dpo_length_train"]),
        }
        rows = [training_sets[r["dataset"]][int(r["row_index"])] for r in selected]
        assert len(rows) == 16
        assert all(row["prompt_id"] == saved["prompt_id"] for row, saved in zip(rows, selected))

        watched = ["configs/base.yaml", "configs/dpo.yaml", "common/models.py",
                   "common/generation.py", "common/data.py", "common/logging_utils.py",
                   "task1_dpo/dpo.py"]
        before = {name: sha256(repo_path(name)) for name in watched}
        report.update({
            "active_course_config": cfg, "trial_combined_cap": 4096,
            "audit_source": str(audit_path.relative_to(repo_path("."))),
            "audit_summary_sha256": sha256(audit_path),
            "pair_csv_sha256": sha256(audit_path.parent / "per_pair_token_counts.csv"),
            "stress_selection": "16 longest retained pairs from both training sets; ties by dataset name, then original row index; shuffled using course seed.",
            "selected_training_pairs": selected, "source_sha256": before,
            "trial_script_sha256": sha256(__file__), "tokenizer": provenance,
            "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
            "total_VRAM_GiB": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "autocast": False, "reference_tolerance": {"rtol": 0, "atol": 1e-4},
            "grad_scaler": {"init_scale": 65536.0, "growth_factor": 2.0,
                            "backoff_factor": .5, "growth_interval": 2000},
            "microbatches": [],
        })

        report["stage"] = "model_load"
        print("Loading pinned Qwen weights and fresh course LoRA adapters...", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"], revision=provenance["resolved_revision"],
            dtype=torch.float16, low_cpu_mem_usage=True,
        )
        base.config.pad_token_id = tok.pad_token_id
        model = get_peft_model(base, make_lora_config(cfg)).cuda()
        model.train()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        assert model.config._commit_hash == provenance["resolved_revision"]
        assert int(model.config.max_position_embeddings) >= 4096

        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        assert named and all("lora_" in name and p.dtype == torch.float32 for name, p in named)
        assert all(p.dtype == torch.float16 for p in model.parameters() if not p.requires_grad)
        lora = model.peft_config["default"]
        assert (lora.r, lora.lora_alpha, lora.lora_dropout, lora.bias) == (8, 16, .05, "none")
        assert set(lora.target_modules) == {"q_proj", "v_proj"}

        parameters = [p for _, p in named]
        initial_adapters = {name: p.detach().clone() for name, p in named}
        optimizer = AdamW(parameters, lr=float(cfg["learning_rate"]),
                          weight_decay=float(cfg.get("weight_decay", 0.0)))
        collate = trial_collate(tok, 4096)
        loader = DataLoader(rows, batch_size=2, shuffle=True, collate_fn=collate)
        report.update({
            "model_revision": model.config._commit_hash,
            "attention_implementation": model.config._attn_implementation,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "trainable_dtypes": dict(Counter(str(p.dtype) for p in parameters)),
            "trainable_names": [name for name, _ in named],
            "optimizer_defaults": optimizer.defaults,
        })
        torch.cuda.reset_peak_memory_stats()

        report["stage"] = "initial_reference_probe"
        print("Checking initial policy/reference agreement on two long pairs...", flush=True)
        probe = tuple({k: v.cuda() for k, v in b.items()} for b in collate(rows[:2])[:2])
        model.eval()
        with torch.no_grad():
            policy_before = scores(model, probe)
        model.train()
        with torch.no_grad(), reference_mode(model):
            reference_before = scores(model, probe)
        assert model.training
        for policy, reference in zip(policy_before, reference_before):
            torch.testing.assert_close(policy, reference, rtol=0, atol=1e-4)
        loss0, _ = dpo_loss(*policy_before, *reference_before, float(cfg["beta"]))
        assert abs(loss0.item() - math.log(2)) < 1e-5
        report["initial_probe_loss"] = loss0.item()
        report["reference_before"] = [t.cpu().tolist() for t in reference_before]
        report["frozen_before_sha256"] = frozen_digest(model)

        scaler = torch.amp.GradScaler(
            "cuda", enabled=True, init_scale=65536.0,
            growth_factor=2.0, backoff_factor=.5, growth_interval=2000,
        )
        optimizer.zero_grad(set_to_none=True)
        seen = 0
        print("Testing eight batches of two long pairs, then one optimizer update...", flush=True)
        for number, (chosen, rejected, ids) in enumerate(loader, start=1):
            report["stage"] = f"microbatch_{number}_reference"
            batches = tuple({k: v.cuda() for k, v in b.items()} for b in (chosen, rejected))
            with torch.no_grad(), reference_mode(model):
                reference = scores(model, batches)
            assert model.training

            report["stage"] = f"microbatch_{number}_policy"
            policy = scores(model, batches)
            loss, _ = dpo_loss(*policy, *reference, float(cfg["beta"]))
            assert torch.isfinite(loss).item(), f"Non-finite loss in batch {number}."
            report["microbatches"].append({
                "number": number, "prompt_ids": ids, "pairs": len(ids), "mean_loss": loss.item(),
                "chosen_sequence_tokens": chosen["attention_mask"].sum(-1).tolist(),
                "rejected_sequence_tokens": rejected["attention_mask"].sum(-1).tolist(),
                "chosen_scored_response_tokens": chosen["response_mask"][:, 1:].sum(-1).tolist(),
                "rejected_scored_response_tokens": rejected["response_mask"][:, 1:].sum(-1).tolist(),
            })
            report["stage"] = f"microbatch_{number}_backward"
            (output / "gpu_trial_report.json").write_text(json.dumps(report, indent=2))
            scaler.scale(loss * (len(ids) / 16)).backward()
            seen += len(ids)
            print(f"Batch {number}/8: loss={loss.item():.6f}; "
                  f"peak VRAM={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
            del policy, reference, loss, batches
        assert seen == 16

        report["stage"] = "unscale_clip_update"
        scaler.unscale_(optimizer)
        bad = [name for name, p in named
               if p.grad is not None and not torch.isfinite(p.grad).all().item()]
        report["nonfinite_gradient_names"] = bad
        assert not bad, "Non-finite gradients: stop and review numerical settings."
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(cfg["max_grad_norm"]), error_if_nonfinite=True,
        )
        assert norm.item() > 0
        report["gradient_norm_before_clipping"] = norm.item()
        report["scale_before"] = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        report["scale_after"] = scaler.get_scale()
        assert report["scale_after"] >= report["scale_before"], "Optimizer update was skipped."
        report["diagnostic_optimizer_updates"] = 1
        optimizer.zero_grad(set_to_none=True)

        changed = [name for name, p in named
                   if not torch.equal(p.detach(), initial_adapters[name])]
        assert changed, "No adapter weights changed."
        report["changed_adapter_parameters"] = changed
        report["stage"] = "frozen_reference_recheck"
        report["frozen_after_sha256"] = frozen_digest(model)
        assert report["frozen_before_sha256"] == report["frozen_after_sha256"]
        with torch.no_grad(), reference_mode(model):
            reference_after = scores(model, probe)
        for a, b in zip(reference_before, reference_after):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-4)
        report["reference_max_abs_changes"] = [
            (a - b).abs().max().item() for a, b in zip(reference_before, reference_after)
        ]
        assert before == {name: sha256(repo_path(name)) for name in watched}

        report["status"], report["stage"] = "PASS", "complete"
        print("Initial DPO loss:", report["initial_probe_loss"], flush=True)
        print("Gradient norm before clipping:", report["gradient_norm_before_clipping"], flush=True)
        print("Gradient scale:", report["scale_before"], "->", report["scale_after"], flush=True)
        print("Adapter tensors changed:", len(changed), flush=True)
        print("Frozen base: exact hash match; reference scores unchanged.", flush=True)
        print("CAP_4096_GPU_TRIAL_OK", flush=True)
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - start
        if torch.cuda.is_available():
            report["peak_allocated_VRAM_GiB"] = torch.cuda.max_memory_allocated() / 2**30
            report["peak_reserved_VRAM_GiB"] = torch.cuda.max_memory_reserved() / 2**30
        if before:
            report["active_source_unchanged"] = before == {
                name: sha256(repo_path(name)) for name in watched
            }
        (output / "gpu_trial_report.json").write_text(json.dumps(report, indent=2))
        if "peak_allocated_VRAM_GiB" in report:
            print("Peak allocated/reserved VRAM (GiB):",
                  round(report["peak_allocated_VRAM_GiB"], 2),
                  round(report["peak_reserved_VRAM_GiB"], 2), flush=True)


if __name__ == "__main__":
    main()