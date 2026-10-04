"""Record released DPO encoding behavior on fixed Task 1 data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from huggingface_hub import hf_hub_download
from common.data import (
    encode_prompt_response, load_yaml, preference_responses,
    prompt_messages_from_preference, read_jsonl, repo_path,
)
from common.models import load_tokenizer


def summarize(rows):
    output = {"pairs": len(rows) // 2}
    for side in ("chosen", "rejected"):
        selected = [r for r in rows if r["side"] == side]
        output[side] = {
            "responses": len(selected),
            "prompt_shortened": sum(r["prompt_shortened"] for r in selected),
            "response_shortened": sum(r["response_shortened"] for r in selected),
            "prompt_fully_removed": sum(r["prompt_fully_removed"] for r in selected),
            "retained_token_without_preceding_position": sum(
                r["retained_response_tokens"] - r["scored_response_tokens"]
                for r in selected
            ),
            "original_response_tokens_mean": mean(
                r["original_response_tokens"] for r in selected
            ),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    cfg = load_yaml("configs/dpo.yaml")
    cap = int(cfg["max_sequence_length"])
    assert cap == 768

    print("Loading the course Qwen tokenizer...", flush=True)
    tok = load_tokenizer(cfg["base_model"])
    cached_config = Path(hf_hub_download(
        cfg["base_model"], "tokenizer_config.json", local_files_only=True,
    ))
    revision = cached_config.parent.name
    assert len(revision) == 40 and all(
        c in "0123456789abcdef" for c in revision
    )
    assert tok.eos_token_id is not None

    provenance = {
        "model_id": cfg["base_model"],
        "resolved_revision": revision,
        "tokenizer_config_sha256": hashlib.sha256(
            cached_config.read_bytes()
        ).hexdigest(),
        "tokenizer_backend_sha256": hashlib.sha256(
            tok.backend_tokenizer.to_str().encode()
        ).hexdigest(),
        "chat_template_sha256": hashlib.sha256(
            json.dumps(tok.chat_template, sort_keys=True).encode()
        ).hexdigest(),
        "padding_side": tok.padding_side,
        "truncation_side": tok.truncation_side,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    previous = repo_path("results/setup/task1_tokenizer_provenance.json")
    if previous.exists():
        assert json.loads(previous.read_text()) == provenance, (
            "Tokenizer provenance changed. Stop and review before continuing."
        )
    saved = json.dumps(provenance, indent=2)
    (output / "tokenizer_provenance.json").write_text(saved)
    previous.write_text(saved)

    asset_records = sorted(
        repo_path("results/setup").glob("task1_asset_validation_*.json")
    )
    assert asset_records, "The completed asset-validation receipt is missing."
    assets = json.loads(asset_records[-1].read_text())
    assert assets["status"] == "PASS"

    sets = {
        "standard_train": "dpo_standard_train",
        "balanced_train": "dpo_length_train",
        "standard_eval": "dpo_standard_eval",
        "stratified_eval": "dpo_length_eval",
    }
    records, summaries, file_hashes = [], {}, {}

    for name, key in sets.items():
        relative = cfg["paths"][key]
        file_hashes[relative] = hashlib.sha256(
            repo_path(relative).read_bytes()
        ).hexdigest()
        assert file_hashes[relative] == assets["task1_files"][relative]["sha256"]
        rows = read_jsonl(relative)
        assert len(rows) == assets["task1_files"][relative]["rows"]
        current = []
        print(f"Auditing {name}: {len(rows)} pairs...", flush=True)

        for index, row in enumerate(rows):
            messages = prompt_messages_from_preference(row)
            prompt_ids = tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            chosen, rejected = preference_responses(row)

            for side, text in (("chosen", chosen), ("rejected", rejected)):
                raw_response = tok(
                    text + (tok.eos_token or ""),
                    add_special_tokens=False,
                )["input_ids"]
                ids, mask = encode_prompt_response(tok, messages, text, cap)
                retained_response = sum(mask)
                retained_prompt = len(ids) - retained_response
                scored_response = sum(mask[1:])
                assert len(ids) <= cap and scored_response > 0
                assert ids[-1] == tok.eos_token_id

                current.append({
                    "dataset": name,
                    "row_index": index,
                    "prompt_id": row["prompt_id"],
                    "side": side,
                    "length_stratum": row.get("length_stratum", ""),
                    "original_prompt_tokens": len(prompt_ids),
                    "original_response_tokens": len(raw_response),
                    "retained_prompt_tokens": retained_prompt,
                    "retained_response_tokens": retained_response,
                    "scored_response_tokens": scored_response,
                    "prompt_shortened": retained_prompt < len(prompt_ids),
                    "response_shortened": retained_response < len(raw_response),
                    "prompt_fully_removed": retained_prompt == 0,
                })

        records.extend(current)
        summaries[name] = summarize(current)
        if name == "standard_train":
            summaries["short_train_first_600"] = summarize(
                [r for r in current if r["row_index"] < 600]
            )

    with (output / "per_response_token_counts.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    result = {
        "status": "PASS",
        "max_sequence_length": cap,
        "tokenizer": provenance,
        "dataset_sha256": file_hashes,
        "summaries": summaries,
        "official_training_started": False,
    }
    (output / "audit_summary.json").write_text(json.dumps(result, indent=2))

    print("Tokenizer revision:", revision)
    print("Sequence limit:", cap)
    for name, summary in summaries.items():
        print(f"\n{name}: {summary['pairs']} pairs")
        for side in ("chosen", "rejected"):
            item = summary[side]
            print(
                f"  {side}: prompt shortened={item['prompt_shortened']}; "
                f"response shortened={item['response_shortened']}; "
                f"prompt fully removed={item['prompt_fully_removed']}"
            )
    print("TOKENIZATION_AUDIT_OK")


if __name__ == "__main__":
    main()
