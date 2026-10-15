"""Read-only token audit of the approved 4096-token feasibility trial.

This implements the proposed TA encoding rule inside an auditor only. It does
not replace the course encoder or configuration, load model weights, filter
examples, or train a model. Held-out data is used only for length diagnostics.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from common.data import (
    load_yaml, preference_responses, prompt_messages,
    prompt_messages_from_preference, read_jsonl, repo_path,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sign(value):
    return (value > 0) - (value < 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out, cap = Path(args.output), 4096
    cfg = load_yaml("configs/dpo.yaml")
    provenance = json.loads(repo_path("results/setup/task1_tokenizer_provenance.json").read_text())
    assert cfg["base_model"] == provenance["model_id"]
    assert cfg["reward_tokenizer"] == cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(
        cfg["base_model"], revision=provenance["resolved_revision"],
        use_fast=True, padding_side=provenance["padding_side"], local_files_only=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    assert tok.eos_token_id is not None
    assert hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest() == provenance["tokenizer_backend_sha256"]
    assert hashlib.sha256(json.dumps(tok.chat_template, sort_keys=True).encode()).hexdigest() == provenance["chat_template_sha256"]
    assert (tok.pad_token_id, tok.eos_token_id, tok.truncation_side) == (
        provenance["pad_token_id"], provenance["eos_token_id"], provenance["truncation_side"],
    )
    receipts = sorted(repo_path("results/setup").glob("task1_asset_validation_*.json"))
    assert receipts, "The earlier asset-validation receipt is missing."
    assets = json.loads(receipts[-1].read_text())
    assert assets["status"] == "PASS"
    watched = ["configs/base.yaml", "configs/dpo.yaml", "common/data.py", "common/generation.py"]
    before = {name: sha256(repo_path(name)) for name in watched}
    sets = {
        "standard_train": ("dpo_standard_train", 1500),
        "balanced_train": ("dpo_length_train", 1500),
        "standard_eval": ("dpo_standard_eval", 300),
        "stratified_eval": ("dpo_length_eval", 246),
    }
    responses, pairs, summaries, data_hashes, generation_rows = [], [], {}, {}, []

    def summarize(name, selected):
        rr = [r for r in responses if r["dataset"] == ("standard_train" if name == "short_train_first_600" else name)
              and (name != "short_train_first_600" or r["row_index"] < 600)]
        item = {
            "pairs": len(selected), "removed_pairs": 0,
            "longest_prompt": max(r["prompt_tokens"] for r in selected),
            "longest_original_combined_sequence": max(r["original_total_tokens"] for r in rr),
            "longest_retained_sequence": max(r["retained_total_tokens"] for r in rr),
            "prompt_truncated": 0,
            "answer_length_order_changed": sum(r["answer_length_order_changed"] for r in selected),
            "supplied_length_strata": dict(Counter(r["length_stratum"] for r in selected)),
            "ordered_prompt_ids": [r["prompt_id"] for r in selected],
            "responses": {},
        }
        for side in ("chosen", "rejected"):
            subset = [r for r in rr if r["side"] == side]
            item["responses"][side] = {
                "truncated": sum(r["response_truncated"] for r in subset),
                "content_tokens_removed": sum(r["content_tokens_removed"] for r in subset),
                "empty_original_content": sum(r["original_content_tokens"] == 0 for r in subset),
                "only_EOS_retained": sum(r["retained_content_tokens"] == 0 for r in subset),
                "truncated_by_supplied_stratum": dict(Counter(
                    r["length_stratum"] for r in subset if r["response_truncated"]
                )),
            }
        summaries[name] = item
        print(f"{name}: {len(selected)} pairs; longest question={item['longest_prompt']}; "
              f"answers shortened chosen/rejected="
              f"{item['responses']['chosen']['truncated']}/{item['responses']['rejected']['truncated']}; "
              f"length ordering changed={item['answer_length_order_changed']}", flush=True)

    def audit_generation(name, rows, is_preference):
        for index, row in enumerate(rows):
            messages = prompt_messages_from_preference(row) if is_preference else prompt_messages(row)
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tok(text, truncation=False)["input_ids"]
            direct = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            assert ids == direct, "Generation and DPO prompt serialization differ."
            empty_rm = tok.apply_chat_template(
                list(messages) + [{"role": "assistant", "content": ""}],
                tokenize=False, add_generation_prompt=False,
            )
            generation_rows.append({
                "dataset": name, "row_index": index, "prompt_id": row["prompt_id"],
                "generation_prompt_tokens": len(ids), "fits_trial_prompt_cap": len(ids) <= cap,
                "prompt_at_or_above_existing_reward_cap": len(ids) >= 1024,
                "reward_serialized_tokens_with_empty_answer": len(tok(empty_rm, truncation=False)["input_ids"]),
            })

    print("Auditing the 4096-token proposal; original files remain unchanged.", flush=True)
    for name, (key, expected_count) in sets.items():
        relative = cfg["paths"][key]
        rows = read_jsonl(relative)
        data_hashes[relative] = sha256(repo_path(relative))
        assert data_hashes[relative] == assets["task1_files"][relative]["sha256"]
        assert len(rows) == expected_count == assets["task1_files"][relative]["rows"]
        current = []
        for index, row in enumerate(rows):
            messages = prompt_messages_from_preference(row)
            prompt_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            p = len(prompt_ids)
            assert 0 < p < cap - 1, f"Prompt {row['prompt_id']} leaves no room for content and EOS."
            lengths = {}
            for side, answer in zip(("chosen", "rejected"), preference_responses(row)):
                raw = tok(answer, add_special_tokens=False)["input_ids"]
                kept = raw[:cap - p - 1]
                ids = prompt_ids + kept + [tok.eos_token_id]
                mask = [0] * p + [1] * (len(kept) + 1)
                assert ids[:p] == prompt_ids and len(ids) <= cap
                assert ids[-1] == tok.eos_token_id and sum(mask[1:]) == len(kept) + 1
                responses.append({
                    "dataset": name, "row_index": index, "prompt_id": row["prompt_id"],
                    "length_stratum": row.get("length_stratum", ""), "side": side,
                    "prompt_tokens": p, "original_content_tokens": len(raw),
                    "retained_content_tokens": len(kept), "original_total_tokens": p + len(raw) + 1,
                    "retained_total_tokens": len(ids), "scored_response_tokens_including_EOS": len(kept) + 1,
                    "response_truncated": len(kept) < len(raw), "content_tokens_removed": len(raw) - len(kept),
                })
                lengths[side] = len(raw), len(kept), len(ids)
            difference_before = lengths["chosen"][0] - lengths["rejected"][0]
            difference_after = lengths["chosen"][1] - lengths["rejected"][1]
            current.append({
                "dataset": name, "row_index": index, "prompt_id": row["prompt_id"],
                "length_stratum": row.get("length_stratum", ""), "prompt_tokens": p,
                "chosen_sequence_tokens": lengths["chosen"][2], "rejected_sequence_tokens": lengths["rejected"][2],
                "chosen_minus_rejected_before": difference_before, "chosen_minus_rejected_after": difference_after,
                "answer_length_order_changed": sign(difference_before) != sign(difference_after),
            })
        pairs.extend(current)
        summarize(name, current)
        if name == "standard_train":
            summarize("short_train_first_600", current[:600])
        if name == "standard_eval":
            audit_generation(name, rows, True)

    relative = cfg["paths"]["word_limit_prompts"]
    data_hashes[relative] = sha256(repo_path(relative))
    assert data_hashes[relative] == assets["task1_files"][relative]["sha256"]
    words = read_jsonl(relative)
    assert len(words) == 10
    audit_generation("word_limit_prompts", words, False)
    generation_summary = {}
    for name in ("standard_eval", "word_limit_prompts"):
        rr = [r for r in generation_rows if r["dataset"] == name]
        generation_summary[name] = {
            "prompts": len(rr), "longest_prompt": max(r["generation_prompt_tokens"] for r in rr),
            "prompts_exceeding_trial_cap": sum(not r["fits_trial_prompt_cap"] for r in rr),
            "prompts_at_or_above_existing_reward_cap": sum(r["prompt_at_or_above_existing_reward_cap"] for r in rr),
            "ordered_prompt_ids": [r["prompt_id"] for r in rr],
        }
        print(f"Generation {name}: {len(rr)} prompts; longest={generation_summary[name]['longest_prompt']}; "
              f"question >= current RM limit (1024)="
              f"{generation_summary[name]['prompts_at_or_above_existing_reward_cap']}", flush=True)
    stress = sorted(
        [r for r in pairs if r["dataset"] == "standard_train"],
        key=lambda r: (-max(r["chosen_sequence_tokens"], r["rejected_sequence_tokens"]), r["row_index"]),
    )[:16]
    after = {name: sha256(repo_path(name)) for name in watched}
    assert before == after, "The audit unexpectedly changed an active source/config file."
    assert all(sha256(repo_path(name)) == digest for name, digest in data_hashes.items())
    report = {
        "status": "AUDIT_COMPLETE", "trial_combined_cap": cap, "tokenizer": provenance,
        "active_config_combined_cap": int(cfg["max_sequence_length"]),
        "summaries": summaries, "generation": generation_summary,
        "existing_reward_cap_audited": 1024,
        "reward_audit_limit": "Prompt length only; actual generated answers and RM memory are not yet checked.",
        "training_only_stress_pairs": stress, "stress_selection": "Longest retained standard-training sequences, ties by original row index; diagnostic use only.",
        "dataset_sha256": data_hashes, "unchanged_source_sha256": before,
        "audit_script_sha256": sha256(__file__), "removed_pairs": 0,
        "official_training_started": False, "gpu_feasibility": "NOT_YET_TESTED",
        "final_configuration": "PENDING_USER_REVIEW_AFTER_TRIAL",
    }
    save_csv(out / "per_response_token_counts.csv", responses)
    save_csv(out / "per_pair_token_counts.csv", pairs)
    save_csv(out / "generation_prompt_lengths.csv", generation_rows)
    (out / "audit_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("CAP_4096_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()