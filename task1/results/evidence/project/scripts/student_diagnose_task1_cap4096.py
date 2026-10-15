"""Inspect the failed trial's data checks; no model loading or data changes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from common.data import load_yaml, preference_responses, read_jsonl, repo_path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    output = Path(parser.parse_args().output)
    audits = sorted(repo_path("results/setup").glob("cap4096_audit_*/audit_summary.json"))
    assert audits, "The completed 4096-token audit is missing."
    audit_path = audits[-1]
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "AUDIT_COMPLETE"
    cfg = load_yaml("configs/dpo.yaml")
    for relative, expected in audit["dataset_sha256"].items():
        assert sha256(repo_path(relative)) == expected, f"Dataset changed: {relative}"
    assert sha256(repo_path("common/data.py")) == audit["unchanged_source_sha256"]["common/data.py"]

    checks, failures = [], []
    print("Inspecting the three checks that stopped the GPU trial:", flush=True)
    for name, group in audit["summaries"].items():
        for side, values in group["responses"].items():
            record = {
                "dataset": name, "side": side,
                **{key: int(values[key]) for key in (
                    "truncated", "only_EOS_retained", "empty_original_content",
                )},
            }
            checks.append(record)
            if any(record[key] != 0 for key in (
                "truncated", "only_EOS_retained", "empty_original_content",
            )):
                failures.append(record)
            print(f"{name}/{side}: shortened={record['truncated']}; "
                  f"empty original answer={record['empty_original_content']}; "
                  f"EOS-only answer={record['only_EOS_retained']}", flush=True)

    paths = {
        "standard_train": "dpo_standard_train", "balanced_train": "dpo_length_train",
        "standard_eval": "dpo_standard_eval", "stratified_eval": "dpo_length_eval",
    }
    source_rows = {name: read_jsonl(cfg["paths"][key]) for name, key in paths.items()}
    csv_path = audit_path.parent / "per_response_token_counts.csv"
    affected = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            original = int(record["original_content_tokens"])
            retained = int(record["retained_content_tokens"])
            if original > 0 and retained > 0:
                continue
            row = source_rows[record["dataset"]][int(record["row_index"])]
            assert row["prompt_id"] == record["prompt_id"]
            chosen, rejected = preference_responses(row)
            answer = chosen if record["side"] == "chosen" else rejected
            affected.append({
                "dataset": record["dataset"], "row_index": int(record["row_index"]),
                "prompt_id": record["prompt_id"], "side": record["side"],
                "length_stratum": record["length_stratum"],
                "original_content_tokens": original, "retained_content_tokens": retained,
                "raw_answer_characters": len(answer),
                "raw_answer_is_empty_string": answer == "",
                "raw_answer_is_whitespace_only": bool(answer) and not answer.strip(),
                "raw_answer_preview": repr(answer[:80]),
                "released_response_format": type(row[record["side"]]).__name__,
            })
    trials = sorted(repo_path("results/setup").glob("cap4096_gpu_trial_*/gpu_trial_report.json"))
    last_trial = json.loads(trials[-1].read_text()) if trials else None
    print("\nAffected responses (first 20; all IDs saved):", flush=True)
    for record in affected[:20]:
        print(f"{record['dataset']} row={record['row_index']} id={record['prompt_id']} "
              f"side={record['side']} original/retained tokens="
              f"{record['original_content_tokens']}/{record['retained_content_tokens']} "
              f"raw answer={record['raw_answer_preview']}", flush=True)
    print("Total affected responses:", len(affected), flush=True)
    report = {
        "status": "CHECK_COMPLETE", "audit_source": str(audit_path),
        "audit_summary_sha256": sha256(audit_path), "audit_csv_sha256": sha256(csv_path),
        "summary_checks": checks, "failed_summary_checks": failures,
        "affected_responses": affected,
        "latest_trial_report": str(trials[-1]) if trials else None,
        "latest_trial_stage": last_trial.get("stage") if last_trial else None,
        "latest_trial_error": last_trial.get("error") if last_trial else None,
        "latest_trial_updates": last_trial.get("diagnostic_optimizer_updates") if last_trial else None,
        "script_sha256": sha256(__file__), "dataset_sha256": audit["dataset_sha256"],
        "training_performed_by_this_diagnostic": False,
        "preprocessing_or_filtering_changed": False,
    }
    (output / "diagnostic_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("CAP_4096_PREFLIGHT_DIAGNOSTIC_DONE", flush=True)


if __name__ == "__main__":
    main()