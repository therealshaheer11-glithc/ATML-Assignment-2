"""Validate installed encoding against the approved 4096-token audit."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from common.data import (
    encode_prompt_response, load_yaml, preference_responses, prompt_messages,
    prompt_messages_from_preference, read_jsonl, repo_path,
)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    output = Path(parser.parse_args().output)
    report = {
        "status": "RUNNING", "official_training_started": False,
        "model_weights_loaded": False, "summaries": {},
    }
    try:
        assert not torch.cuda.is_available()
        cfg = load_yaml("configs/dpo.yaml")
        assert (
            cfg["max_sequence_length"], cfg["max_prompt_length"],
            cfg["reward_max_length"], cfg["max_generation_tokens"]
        ) == (4096, 4096, 4096, 256)

        settings_path = sorted(repo_path("results/setup").glob(
            "final_settings_*/settings_receipt.json"
        ))[-1]
        settings = json.loads(settings_path.read_text())
        assert settings["status"] == "PASS"
        assert settings["approval"]["configuration_applied"]
        for name, digest in settings["after_sha256"].items():
            assert sha(repo_path(name)) == digest, f"Active file changed: {name}"
        assert sha(repo_path("common/generation.py")) == (
            settings["before_sha256"]["common/generation.py"]
        )

        audit_path = sorted(repo_path("results/setup").glob(
            "cap4096_audit_*/audit_summary.json"
        ))[-1]
        audit = json.loads(audit_path.read_text())
        assert audit["status"] == "AUDIT_COMPLETE"
        assert audit["trial_combined_cap"] == 4096

        for name, digest in audit["dataset_sha256"].items():
            assert sha(repo_path(name)) == digest, f"Dataset changed: {name}"

        with (audit_path.parent / "per_response_token_counts.csv").open(
            newline=""
        ) as handle:
            prior_rows = list(csv.DictReader(handle))
        prior = {
            (r["dataset"], int(r["row_index"]), r["side"]): r
            for r in prior_rows
        }
        assert len(prior) == len(prior_rows) == 7092

        provenance = audit["tokenizer"]
        assert cfg["base_model_revision"] == provenance["resolved_revision"]
        assert cfg["base_model"] == provenance["model_id"]

        print("Loading pinned tokenizer files only; no model weights...", flush=True)
        tok = AutoTokenizer.from_pretrained(
            cfg["base_model"], revision=cfg["base_model_revision"],
            use_fast=True, padding_side=provenance["padding_side"],
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        assert hashlib.sha256(
            tok.backend_tokenizer.to_str().encode()
        ).hexdigest() == provenance["tokenizer_backend_sha256"]
        assert hashlib.sha256(
            json.dumps(tok.chat_template, sort_keys=True).encode()
        ).hexdigest() == provenance["chat_template_sha256"]
        assert (tok.pad_token_id, tok.eos_token_id, tok.truncation_side) == (
            provenance["pad_token_id"], provenance["eos_token_id"],
            provenance["truncation_side"],
        )

        report.update({
            "settings_receipt_sha256": sha(settings_path),
            "earlier_audit_sha256": sha(audit_path),
            "tokenizer": provenance,
            "dataset_sha256": audit["dataset_sha256"],
            "active_encoder_sha256": sha(repo_path("common/data.py")),
            "validation_script_sha256": sha(__file__),
        })

        counts, empty_ids, visited = [], [], set()
        pools = [
            ("standard_train", "dpo_standard_train", 1500),
            ("balanced_train", "dpo_length_train", 1500),
            ("standard_eval", "dpo_standard_eval", 300),
            ("stratified_eval", "dpo_length_eval", 246),
        ]

        for name, path_key, expected_count in pools:
            rows = read_jsonl(cfg["paths"][path_key])
            assert len(rows) == expected_count
            assert [r["prompt_id"] for r in rows] == (
                audit["summaries"][name]["ordered_prompt_ids"]
            )
            start = len(counts)
            print(f"Checking {name}: {len(rows)} original pairs...", flush=True)

            for index, row in enumerate(rows):
                messages = prompt_messages_from_preference(row)
                prompt = tok.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                assert prompt

                for side, answer in zip(
                    ("chosen", "rejected"), preference_responses(row)
                ):
                    raw = tok(answer, add_special_tokens=False)["input_ids"]
                    ids, mask = encode_prompt_response(
                        tok, messages, answer, 4096
                    )
                    assert ids == prompt + raw + [tok.eos_token_id], (
                        f"Tokens lost or altered: {name}/{index}/{side}"
                    )
                    assert mask == [0] * len(prompt) + [1] * (len(raw) + 1)
                    assert len(ids) <= 4096
                    assert sum(mask[1:]) == len(raw) + 1

                    key = (name, index, side)
                    previous = prior[key]
                    assert previous["prompt_id"] == row["prompt_id"]
                    assert (
                        int(previous["prompt_tokens"]),
                        int(previous["original_content_tokens"]),
                        int(previous["retained_content_tokens"]),
                        int(previous["retained_total_tokens"]),
                    ) == (len(prompt), len(raw), len(raw), len(ids))

                    visited.add(key)
                    counts.append({
                        "dataset": name, "row_index": index,
                        "prompt_id": row["prompt_id"], "side": side,
                        "prompt_tokens": len(prompt),
                        "answer_content_tokens": len(raw),
                        "sequence_tokens": len(ids),
                        "scored_response_tokens": len(raw) + 1,
                    })
                    if not raw:
                        assert answer == ""
                        empty_ids.append({
                            "dataset": name, "row_index": index,
                            "side": side, "prompt_id": row["prompt_id"],
                        })

            current = counts[start:]
            report["summaries"][name] = {
                "pairs": len(rows), "responses": len(current),
                "prompt_tokens_removed": 0, "answer_tokens_removed": 0,
                "EOS_only_original_answers": sum(
                    r["answer_content_tokens"] == 0 for r in current
                ),
                "length_strata": dict(Counter(
                    r.get("length_stratum", "") for r in rows
                )),
            }
            print(
                f"PASS: {name}; prompts/answers unchanged; "
                f"{len(rows)} pairs retained.", flush=True
            )

            if name == "standard_train":
                assert [r["prompt_id"] for r in rows[:600]] == (
                    audit["summaries"]["short_train_first_600"]["ordered_prompt_ids"]
                )
                report["summaries"]["short_train_first_600"] = {
                    "pairs": 600, "selection": "original first 600",
                    "prompt_tokens_removed": 0, "answer_tokens_removed": 0,
                    "EOS_only_original_answers": sum(
                        r["row_index"] < 600 for r in empty_ids
                    ),
                }

        assert visited == set(prior)
        assert len(empty_ids) == 4
        assert {r["row_index"] for r in empty_ids} == {265, 543, 589, 1381}
        assert all(
            r["dataset"] == "standard_train" and r["side"] == "rejected"
            for r in empty_ids
        )
        diagnostics = sorted(repo_path("results/setup").glob(
            "cap4096_preflight_diagnostic_*/diagnostic_report.json"
        ))
        confirmed = json.loads(diagnostics[-1].read_text())["affected_responses"]
        assert {r["prompt_id"] for r in empty_ids} == {
            r["prompt_id"] for r in confirmed
        }
        report["empty_original_answers"] = empty_ids

        report["generation_prompts"] = {}
        for name, path_key, preference, expected_count in (
            ("standard_eval", "dpo_standard_eval", True, 300),
            ("word_limit_prompts", "word_limit_prompts", False, 10),
        ):
            rows = read_jsonl(cfg["paths"][path_key])
            assert len(rows) == expected_count
            assert [r["prompt_id"] for r in rows] == (
                audit["generation"][name]["ordered_prompt_ids"]
            )
            lengths = []
            for row in rows:
                messages = (
                    prompt_messages_from_preference(row)
                    if preference else prompt_messages(row)
                )
                rendered = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                ids = tok(rendered, truncation=False)["input_ids"]
                assert ids == tok.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                assert len(ids) <= cfg["max_prompt_length"]
                lengths.append(len(ids))
            report["generation_prompts"][name] = {
                "prompts": len(rows), "longest": max(lengths), "truncated": 0,
                "ordered_prompt_ids": [r["prompt_id"] for r in rows],
                "ordered_prompt_token_lengths": lengths,
            }

        with (output / "active_response_counts.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(counts[0]))
            writer.writeheader()
            writer.writerows(counts)

        report["active_response_counts_sha256"] = sha(
            output / "active_response_counts.csv"
        )
        report["reward_scoring"] = (
            "Full generated inputs and GPU memory remain pending; "
            "no reward scores computed."
        )
        report["status"] = "PASS"
        print(
            "ACTIVE_ENCODING_OK: 7092 responses match the earlier audit; "
            "zero truncation.", flush=True
        )
        print(
            "First 600 rows unchanged; four original empty rejected answers "
            "score EOS only.", flush=True
        )
        print("All 310 generation prompts fit without truncation.", flush=True)

    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "active_encoding_report.json").write_text(
            json.dumps(report, indent=2)
        )

if __name__ == "__main__":
    main()
