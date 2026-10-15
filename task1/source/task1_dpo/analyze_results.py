"""Independent CPU audit of saved Task 1 evidence; standard library only.

Added after the experiments. Never loads models, trains, changes run artifacts,
or imports the implementation being checked. FP32 subtractions reproduce the
recorded arithmetic; stable scalar log-sigmoid is an independent loss check.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct

NAMES = ("standard", "beta_003", "beta_010", "beta_030", "length_balanced")
BETAS = dict(zip(NAMES, (.1, .03, .1, .3, .1)))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a, b, label, tol=1e-9):
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=tol, abs_tol=tol),
            f"{label}: {a} != {b}")


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def margin_loss(pc, pr, rc, rr, beta):
    m = f32(f32(pc - pr) - f32(rc - rr))
    z = f32(beta * m)
    return m, max(-z, 0) + math.log1p(math.exp(-abs(z)))


def stats(values):
    return {"count": len(values), "mean": statistics.mean(values),
            "population_std": statistics.pstdev(values)}


def verify_stats(values, saved, label):
    result = stats(values)
    for k, v in result.items():
        close(v, saved[k], label + "/" + k)
    return result


def verify_sources(folder, record, project, *, failed=False):
    for key, digest in record["source_sha256"].items():
        relative = key.removeprefix("/content/ATML-Assignment-2/")
        require(sha(folder / "source_snapshot" / relative) == digest, f"Source snapshot drift: {relative}")
        if not failed:
            require(sha(project / relative) == digest, f"Executed source differs: {relative}")
    for relative, digest in record.get("artifact_sha256", {}).items():
        require(sha(folder / relative) == digest, f"Artifact drift: {relative}")


def pair_summary(records, beta):
    margins, losses = [], []
    for r in records:
        m, loss = margin_loss(*(r[k] for k in ("policy_chosen_logp", "policy_rejected_logp",
                             "reference_chosen_logp", "reference_rejected_logp")), beta)
        close(m, r["reference_adjusted_margin"], "pair margin", 0)
        close(loss, r["dpo_loss"], "pair scalar loss", 2e-6)
        require((m > 0) == r["correct"], "Pair accuracy uses wrong margin")
        require(min(r["chosen_response_tokens"], r["rejected_response_tokens"]) >= 1, "Missing EOS response score")
        margins.append(m)
        losses.append(r["dpo_loss"])
    return {"pairs": len(records), "beta": beta, "mean_dpo_loss": statistics.mean(losses),
            "preference_accuracy": sum(m > 0 for m in margins) / len(records),
            "correct_pairs": sum(m > 0 for m in margins), "ties": sum(m == 0 for m in margins)}


def audit(evidence):
    manifest = read(evidence / "SHA256SUMS.json")
    for relative, digest in manifest.items():
        # Keep the original bytes/hash but deactivate this historical ignore
        # file so it cannot hide evidence files during the student's Git push.
        stored = "project/.gitignore.archived.txt" if relative == "project/.gitignore" else relative
        require(sha(evidence / stored) == digest, f"Export file drift: {relative}")
    project = evidence / "project"
    trains, train_table, all_traces = {}, [], {}
    canonical = read(evidence / "training/standard/run_record.json")
    for name in (*NAMES, "standard_failed_scale65536"):
        folder = evidence / "training" / name
        rec, trace = read(folder / "run_record.json"), rows(folder / "training_trace.jsonl")
        failed = name == "standard_failed_scale65536"
        verify_sources(folder, rec, project, failed=failed)
        for relative, digest in read(folder / "original_backup_receipt.json")["sha256"].items():
            if (folder / relative).is_file():
                require(sha(folder / relative) == digest, f"Training receipt drift: {name}/{relative}")
        require(rec["effective_config"] == canonical["effective_config"], "Training configs differ")
        require(rec["initial_adapter_sha256"] == canonical["initial_adapter_sha256"], "Fresh initialization differs")
        require(rec["frozen_base_before_sha256"] == canonical["frozen_base_before_sha256"], "Base differs")
        require(rec["optimizer_defaults"] == canonical["optimizer_defaults"], "Optimizer changed")
        require(rec["seed"] == 6304 and rec["autocast"] is False, "Numerical protocol differs")
        require(rec["trainable_dtypes"] == {"torch.float32": 112}, "Adapter dtype differs")
        require(rec["scaler_settings"] == {"enabled": True, "init_scale": 65536. if failed else 1024.,
                "growth_factor": 2., "backoff_factor": .5, "growth_interval": 2000}, "Scaler protocol differs")
        micro = [r for r in trace if r["event"] == "microbatch"]
        updates = [r for r in trace if r["event"] == "optimizer_update"]
        ids, weighted_loss, correct = [], 0., 0
        for i, batch in enumerate(micro, 1):
            require(batch["microbatch"] == i and batch["pairs"] == len(batch["prompt_ids"]), "Microbatch ordering")
            vals = [margin_loss(*v, rec["plan"]["beta"]) for v in zip(
                batch["policy_chosen_logp"], batch["policy_rejected_logp"],
                batch["reference_chosen_logp"], batch["reference_rejected_logp"])]
            require(len(vals) == batch["pairs"], "Training score count")
            for (m, _), saved in zip(vals, batch["reference_adjusted_margin"]):
                close(m, saved, "Training margin", 0)
            close(statistics.mean(v[1] for v in vals), batch["mean_loss"], "Training scalar loss", 2e-6)
            close(sum(v[0] > 0 for v in vals) / len(vals), batch["diagnostics"]["preference_accuracy"], "Train accuracy")
            weighted_loss += batch["pairs"] * batch["mean_loss"]
            correct += sum(v[0] > 0 for v in vals)
            ids += batch["prompt_ids"]
        seen = 0
        for i, update in enumerate(updates, 1):
            window = [m for m in micro if m["update"] == i]
            count = sum(m["pairs"] for m in window)
            seen += count
            require(update["update"] == i and update["pairs"] == count and update["seen_pairs"] == seen, "Update membership")
            require(all(m["window_pairs"] == count for m in window), "Accumulation tail weighting")
            require(math.isfinite(update["gradient_norm_before_clipping"]), "Nonfinite completed update")
            require(update["scale_before"] == update["scale_after"] == rec["scaler_settings"]["init_scale"], "Skipped/rescaled update")
        require(seen == rec["completed_pairs"] and len(updates) == rec["completed_updates"], "Completion totals")
        if failed:
            require(rec["status"] == "FAILED" and seen == 288 and len(updates) == 18 and len(micro) == 152, "Failed attempt budget")
            require(trace[-1] == {"event": "nonfinite_gradients", "update": 19, "parameter_indices": list(range(8))}, "Failed gradient evidence")
        else:
            count = 600 if name.startswith("beta_") else 1500
            require(rec["status"] == "COMPLETE_RELOAD_VERIFIED" and seen == count, "Incomplete run")
            require(rec["plan"]["beta"] == BETAS[name] and rec["plan"]["pairs"] == count and rec["plan"]["epochs"] == 1, "Run budget")
            require(len(updates) == math.ceil(count / 16) and len(micro) == count // 2, "Run window count")
            require(Counter(ids) == Counter(rec["selected_prompt_ids"]), "Training membership changed")
            require(ids == rec["training"]["ordered_prompt_ids"], "Training order differs")
            require([u["pairs"] for u in updates] == rec["expected_update_pair_counts"] == rec["training"]["update_pair_counts"], "Tail omitted")
            require(rec["frozen_base_after_sha256"] == rec["frozen_base_before_sha256"], "Frozen base changed")
            require(rec["final_adapter_sha256"] != rec["initial_adapter_sha256"], "No adapter update")
            reload = read(folder / "reload_verification.json")
            require(reload["status"] == "PASS" and reload["held_out_examples_used"] is False, "Reload probe invalid")
            require(all(v == 0 for k in ("policy_max_abs_changes", "reference_max_abs_changes") for v in reload[k]), "Reload score drift")
            close(weighted_loss/count, rec["training"]["training_mean_loss"], "Mean training loss")
            close(correct/count, rec["training"]["training_preference_accuracy"], "Mean training accuracy")
            train_table.append({"condition": name, "pairs": count, "updates": len(updates), "tail_pairs": updates[-1]["pairs"],
                                "minutes": rec["training_wall_seconds"]/60, "loss": weighted_loss/count, "accuracy": correct/count})
        trains[name], all_traces[name] = rec, ids
    for name in NAMES[1:4]:
        require(trains[name]["selected_prompt_ids"] == canonical["selected_prompt_ids"][:600], "Wrong short subset")
        require(all_traces[name] == all_traces["beta_003"], "Beta training order differs")
    require(all_traces["standard_failed_scale65536"] == all_traces["standard"][:304], "Retry order differs")

    tables = {"training": train_table, "summary": [], "length_strata": [], "word_limits": [], "generation_ceiling": []}
    totals = {"held_out_pairs": 0, "generated_answers": 0, "reward_scores": 0}
    reference_pools, decoding, source_manifest = {}, None, None
    for name in ("sft", *NAMES):
        folder = evidence / "evaluation" / name
        policy = folder / "policy_results"
        rec, pr, metrics = read(folder / "evaluation_record.json"), read(policy / "evaluation_record.json"), read(folder / "metrics.json")
        require(rec["status"] == "COMPLETE" and pr["status"] == "POLICY_COMPLETE", "Incomplete evaluation")
        for path, record in ((folder, rec), (policy, pr)):
            verify_sources(path, record, project)
            require(record["effective_config"] == canonical["effective_config"], "Evaluation config changed")
            require(record["model_manifest_sha256"] == sha(evidence / "model_metadata/model_manifest.json"), "Model manifest drift")
        if source_manifest is None:
            source_manifest = pr["source_sha256"]
        require(source_manifest == pr["source_sha256"] == rec["source_sha256"], "Evaluation source differs")
        require(rec["policy_results_record_sha256"] == sha(policy / "evaluation_record.json"), "Reward replay mismatch")
        for relative, digest in read(folder / "original_backup_receipt.json")["sha256"].items():
            require(sha(folder / relative) == digest, f"Evaluation receipt drift: {name}/{relative}")
        if name != "sft":
            require(pr["base_parameter_sha256"] == trains[name]["frozen_base_after_sha256"] and
                    pr["adapter_parameter_sha256"] == trains[name]["final_adapter_sha256"], "Evaluated wrong model")
            for file in ("run_record.json", "reload_verification.json", "adapter_config.json"):
                require(pr["checkpoint_file_sha256"][file] == sha(evidence / "training" / name / file), "Checkpoint mismatch")
        else:
            require(pr["adapter"] is None and not metrics["pairs"], "Baseline has adapter or invented pairs")
        rm = read(folder / "reward_loading.json")
        require(rm["loaded_position_frequencies_match"] and rm["position_frequencies_match_checkpoint"] and
                rm["original_file_unchanged"] and rm["frozen"], "Reward fix/freeze not verified")
        require(rm["changed_fields"] == {"rope_theta": {"before": 10000., "after": 1000000.}}, "Unexpected RM translation")
        require(all(v == [] for v in rm["loading_info"].values()), "Reward checkpoint load mismatch")
        require(rm["raw_config_sha256"] == sha(evidence / "model_metadata/reward/config.json"), "Reward source config differs")
        input_audit = read(folder / "reward_input_audit.json")
        expected_pairs = [] if name == "sft" else ["dpo_standard_eval"] + (["dpo_length_eval"] if name in ("standard", "length_balanced") else [])
        expected_gen = ["dpo_standard_eval"] + (["word_limit_prompts"] if name in ("standard", "length_balanced", "sft") else [])
        require(list(metrics["pairs"]) == expected_pairs and list(metrics["generation"]) == expected_gen, "Wrong evaluation scope")
        for pool, saved in metrics["pairs"].items():
            records = rows(policy / f"pairs_{pool}.jsonl")
            require([r["row_index"] for r in records] == list(range(len(records))), "Pair rows reordered/dropped")
            require([r["prompt_id"] for r in records] == pr["plan"]["datasets"][pool]["ordered_prompt_ids"], "Pair IDs differ")
            require(len(records) == (300 if pool == "dpo_standard_eval" else 246), "Pair budget differs")
            result = pair_summary(records, BETAS[name])
            for k, v in result.items():
                close(v, saved[k], "Pair aggregate/"+k, 2e-6 if k == "mean_dpo_loss" else 1e-12)
            if pool == "dpo_length_eval":
                require(Counter(r["length_stratum"] for r in records) == {s: 82 for s in saved["strata"]}, "Stratum sizes")
                for stratum, target in saved["strata"].items():
                    group = pair_summary([r for r in records if r["length_stratum"] == stratum], BETAS[name])
                    for k, v in group.items():
                        close(v, target[k], "Stratum/"+k, 2e-6 if k == "mean_dpo_loss" else 1e-12)
                    tables["length_strata"].append({"condition": name, "stratum": stratum, **group})
            totals["held_out_pairs"] += len(records)
        for pool, saved in metrics["generation"].items():
            generated, rewards = rows(policy / f"generation_{pool}.jsonl"), rows(folder / f"rewards_{pool}.jsonl")
            require(len(generated) == len(rewards) == (300 if pool == "dpo_standard_eval" else 10), "Generation budget")
            require([r["prompt_id"] for r in generated] == pr["plan"]["datasets"][pool]["ordered_prompt_ids"], "Generation membership")
            identities = [(r["prompt_id"], r["messages"], r["prompt_tokens"], r["rendered_prompt_sha256"]) for r in generated]
            if pool not in reference_pools:
                reference_pools[pool] = identities
            require(identities == reference_pools[pool], "Prompts differ across models")
            config = saved["effective_generation_config"]
            if decoding is None:
                decoding = config
            require(config == decoding and saved["seed_per_pool"] == 6304, "Decoding differs")
            require((config["do_sample"], config["temperature"], config["top_p"], config["top_k"], config["repetition_penalty"], config["max_new_tokens"]) == (True, .7, .9, 20, 1.1, 256), "Decoding protocol changed")
            token_sum, difference_sum, compliance = 0, 0., 0
            for i, (g, reward) in enumerate(zip(generated, rewards)):
                require(g["row_index"] == reward["row_index"] == i and g["prompt_id"] == reward["prompt_id"], "Reward replay order")
                ids = g["response_token_ids"]
                n = len(ids)
                require(0 < n <= 256 and n == g["response_length"], "Generated length")
                require(all(len(g[k]) == n for k in ("policy_token_logp", "reference_token_logp", "token_logp_difference")), "Token array length")
                eos = config["eos_token_id"]
                require(eos not in ids[:-1] and (ids[-1] == eos) == g["terminated_with_eos"], "EOS boundary")
                require(g["reached_generation_ceiling"] == (n == 256 and ids[-1] != eos), "Ceiling flag")
                differences = [f32(a-b) for a,b in zip(g["policy_token_logp"], g["reference_token_logp"])]
                for x, y in zip(differences, g["token_logp_difference"]):
                    close(x, y, "Token difference", 0)
                close(sum(differences), g["logp_difference_sum"], "Response difference")
                if name == "sft":
                    require(all(d == 0 for d in differences), "SFT self-reference differs")
                wc = len(re.findall(r"\b\w+\b", g["response"]))
                require(wc == g["word_count"] and g["prompt_tokens"] <= 4096, "Word count or prompt cap")
                require(math.isfinite(reward["reward_score"]) and reward["reward_input_tokens"] <= 4096, "Reward value/cap")
                if pool == "word_limit_prompts":
                    ok = wc <= g["word_limit"]
                    require(ok == g["word_limit_compliance"], "Word rule")
                    compliance += ok
                    tables["word_limits"].append({"condition": name, "prompt_id": g["prompt_id"], "limit": g["word_limit"],
                                                 "words": wc, "complies": ok, "reward": reward["reward_score"], "tokens": n})
                token_sum += n
                difference_sum += sum(differences)
            close(difference_sum, saved["logp_difference_sum"], "Pool logp sum")
            close(token_sum, saved["valid_response_tokens"], "Pool token count", 0)
            close(difference_sum/token_sum, saved["sampled_kl"], "Pool sampled KL")
            token_stats = verify_stats([g["response_length"] for g in generated], saved["response_length_tokens"], "Token stats")
            word_stats = verify_stats([g["word_count"] for g in generated], saved["response_length_words"], "Word stats")
            reward_stats = verify_stats([r["reward_score"] for r in rewards], saved["reward_score"], "Reward stats")
            ia = input_audit[pool]
            require(ia["truncated"] == 0 and ia["responses"] == len(generated) and
                    ia["input_token_counts"] == [r["reward_input_tokens"] for r in rewards] and
                    ia["maximum_complete_input_tokens"] == max(ia["input_token_counts"]), "RM input audit")
            if pool == "word_limit_prompts":
                close(compliance/len(generated), saved["word_limit_compliance"], "Word compliance aggregate")
            else:
                pairs = metrics["pairs"].get(pool, {})
                tables["summary"].append({"condition": name, "training_pairs": 0 if name == "sft" else trains[name]["completed_pairs"],
                                          "beta": BETAS.get(name), "dpo_loss": pairs.get("mean_dpo_loss"),
                                          "accuracy": pairs.get("preference_accuracy"), "correct_pairs": pairs.get("correct_pairs"),
                                          "sampled_kl": difference_sum/token_sum, "reward_mean": reward_stats["mean"],
                                          "reward_population_sd": reward_stats["population_std"], "token_mean": token_stats["mean"],
                                          "token_population_sd": token_stats["population_std"]})
            tables["generation_ceiling"].append({"condition": name, "pool": pool, "responses": len(generated),
                                                 "at_ceiling_without_eos": sum(g["reached_generation_ceiling"] for g in generated),
                                                 "word_mean": word_stats["mean"], "maximum_reward_input": ia["maximum_complete_input_tokens"]})
            totals["generated_answers"] += len(generated)
            totals["reward_scores"] += len(rewards)
    require(totals == {"held_out_pairs": 1992, "generated_answers": 1830, "reward_scores": 1830}, "Incomplete Task 1 totals")
    report = {"status": "PASS", "export_files_verified": len(manifest), **totals,
              "completed_training_runs": 5, "failed_attempts_preserved": 1, "matched_evaluation_source_files": len(source_manifest),
              "source_and_settings_consistent": True, "training_budgets_and_trace_metrics_verified": True,
              "frozen_base_and_reload_records_verified": True, "reward_translation_records_verified": True,
              "all_raw_pair_and_generation_metrics_recomputed": True,
              "scope": "Independent saved-record verification; no model execution or fresh GPU replay. Omitted binary artifacts remain on Drive."}
    return report, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("results/task1/evidence"))
    parser.add_argument("--output", type=Path, default=Path("outputs/task1_recomputed"))
    args = parser.parse_args()
    require(not args.output.exists(), "Choose a new output directory; saved results are never overwritten.")
    report, tables = audit(args.evidence)
    args.output.mkdir(parents=True)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, values in tables.items():
        with (args.output / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    print(json.dumps(report, indent=2))
    print("Tables saved:", args.output)


if __name__ == "__main__":
    main()
