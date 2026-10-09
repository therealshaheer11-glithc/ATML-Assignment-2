"""CPU-only quantitative artifacts and an unfilled qualitative-review worksheet.

Never invents quality judgments. Required agreement/disagreement evidence remains
pending until the user supplies reasoned assessments of actual generated text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from task2_ppo.evaluation_complete import summarize_evaluation
from task2_ppo.cache_geometry import geometry_summary
from task2_ppo.storage import atomic_json, sha256


NAMES = ["standard", "clip_005", "central_8", "clip_050", "kl_000", "kl_020"]


def archived_json(archive, member):
    archive = Path(archive)
    receipt = json.loads(archive.with_suffix(".sha256.json").read_text())
    if receipt["sha256"] != sha256(archive):
        raise RuntimeError(f"Archive checksum differs: {archive}")
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("FILES_SHA256.json"))
        payload = z.read(member)
        if hashlib.sha256(payload).hexdigest() != manifest[member]:
            raise RuntimeError(f"Archived JSON checksum differs: {member}")
    return json.loads(payload)


def write_csv(path, rows, columns=None):
    columns = columns or list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def collect(experiment):
    experiment = Path(experiment)
    training, evaluation, records = {}, {}, {}
    for name in NAMES:
        training[name] = archived_json(experiment / "training" / name / "final.zip", "summary.json")
        expected = 20 if name == "standard" else 8
        if training[name]["completed_updates"] != expected or len(training[name]["history"]) != expected:
            raise RuntimeError(f"Training budget differs: {name}")
    for name in ["midpoint"] + NAMES:
        folder = experiment / "evaluation" / name
        candidates = sorted(folder.glob("prompt_*.zip"))
        if len(candidates) != 200:
            raise RuntimeError(f"Evaluation incomplete for {name}: {len(candidates)}/200")
        rows = [archived_json(p, "record.json") for p in candidates]
        indices = [r["schedule"]["row_index"] for r in rows]
        if indices != list(range(200)) or any(r["condition"] != name for r in rows):
            raise RuntimeError("Evaluation prompt provenance differs")
        records[name] = rows
        evaluation[name] = summarize_evaluation(rows)
    baseline_ids = [r["schedule"]["prompt_id"] for r in records["midpoint"]]
    for name in NAMES:
        if [r["schedule"]["prompt_id"] for r in records[name]] != baseline_ids:
            raise RuntimeError("Held-out prompt identities differ across conditions")
        if [r["schedule"]["generation_seed"] for r in records[name]] != [r["schedule"]["generation_seed"] for r in records["midpoint"]]:
            raise RuntimeError("Held-out generation seeds differ")
    cache_records = [archived_json(p, "record.json") for p in sorted((experiment / "cache_geometry").glob("row_*.zip"))]
    if len(cache_records) != 32 or [r["row_index"] for r in cache_records] != list(range(32)):
        raise RuntimeError("Cache row evidence incomplete")
    cache = geometry_summary(cache_records)
    if cache["rows"] != 32 or len(cache["results"]) != 3 or cache["training_on_cache"]:
        raise RuntimeError("Cache geometry incomplete or invalid")
    return training, evaluation, records, cache


def render_plots(destination, training, evaluation, cache):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    history = training["standard"]["history"]
    x = [r["completed_updates"] for r in history]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    def curve(axis, values, label):
        axis.plot(x, values, marker=".", label=label)
        axis.set_xlabel("Rollout update")
        axis.grid(alpha=.2)
    rollout_fields = [("raw_reward", "Raw learned reward"), ("rollout_reference_kl", "Sampled reference KL"),
                      ("response_length", "Response tokens")]
    for ax, (key, title) in zip(axes[0], rollout_fields):
        curve(ax, [r["rollout"][key] for r in history], title)
        ax.set_title(title)
    for ax, key, title in [(axes[1, 0], "policy_loss", "Policy loss"),
                           (axes[1, 1], "value_mse", "Unweighted critic MSE"),
                           (axes[1, 2], "clip_fraction", "Affected-token fraction")]:
        curve(ax, [statistics.mean(s[key] for s in r["optimization"]) for r in history], title)
        ax.set_title(title + " (mean of two epochs)")
    for ax, key, title in [(axes[2, 0], "actor_grad_norm_preclip", "Actor gradient norm"),
                           (axes[2, 1], "critic_grad_norm_preclip", "Critic gradient norm")]:
        for epoch in (0, 1):
            curve(ax, [r["optimization"][epoch][key] for r in history], "PPO epoch " + str(epoch + 1))
        ax.set_title(title + " before clipping")
        ax.legend()
    for key, label in [("sampled_entropy", "Course sampled estimate"), ("categorical_entropy", "Categorical entropy")]:
        curve(axes[2, 2], [r["rollout"][key] for r in history], label)
    axes[2, 2].set_title("Rollout entropy (raw policy)")
    axes[2, 2].legend()
    fig.suptitle("Standard PPO: 20 rollout updates, two optimization epochs each")
    for suffix in ("png", "pdf"):
        fig.savefig(destination / ("standard_trajectories." + suffix), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, field, title in [(axes[0, 0], "raw_reward", "Raw reward"),
                             (axes[0, 1], "rollout_reference_kl", "Sampled reference KL"),
                             (axes[1, 0], "categorical_entropy", "Categorical entropy"),
                             (axes[1, 1], "response_length", "Response tokens")]:
        for name, beta in [("kl_000", 0.), ("central_8", .1), ("kl_020", .2)]:
            h = training[name]["history"]
            ax.plot(range(1, 9), [r["rollout"][field] for r in h], marker=".", label=f"KL beta={beta:.2f}")
        ax.set(title=title, xlabel="Rollout update")
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("KL-pressure forks: matched prompt schedule and eight-update allowance")
    for suffix in ("png", "pdf"):
        fig.savefig(destination / ("kl_fork_trajectories." + suffix), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    epsilon = [r["epsilon"] for r in cache["results"]]
    axes[0].plot(epsilon, [r["affected_token_fraction"] for r in cache["results"]], marker="o", label="Affected")
    axes[0].plot(epsilon, [r["active_clipped_branch_fraction"] for r in cache["results"]], marker="o", label="Active clipped branch")
    axes[0].set(title="Same fixed cached tokens", xlabel="Clip epsilon", ylabel="Token fraction")
    axes[0].legend()
    fork_names = ["clip_005", "central_8", "clip_050"]
    axes[1].plot(epsilon, [training[n]["stability"]["actor_std_preclip"] for n in fork_names], marker="o")
    axes[1].set(title="Actor optimization stability", xlabel="Clip epsilon", ylabel="Population std of pre-clip norm")
    axes[2].plot(epsilon, [evaluation[n]["raw_reward_mean"] for n in fork_names], marker="o")
    axes[2].set(title="Frozen 200-prompt evaluation", xlabel="Clip epsilon", ylabel="Mean raw reward")
    for ax in axes:
        ax.grid(alpha=.2)
    for suffix in ("png", "pdf"):
        fig.savefig(destination / ("clipping_study." + suffix), dpi=180)
    plt.close(fig)


def aggregate(experiment, destination, review=None):
    if torch.cuda.is_available():
        raise RuntimeError("Use CPU for aggregation, plotting and qualitative review")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    training, evaluation, records, cache = collect(experiment)
    rows = []
    for name in ["midpoint"] + NAMES:
        rows.append({"condition": name, **evaluation[name]})
    write_csv(destination / "held_out_metrics.csv", rows)
    clipping = []
    for name, cache_row in zip(["clip_005", "central_8", "clip_050"], cache["results"]):
        t = training[name]
        clipping.append({"condition": name, **cache_row, **evaluation[name], **t["stability"],
            "allocated_tokens": t["allocated_response_tokens"], "actual_tokens": t["actual_response_tokens"]})
    write_csv(destination / "clipping_study.csv", clipping)
    pressure = []
    for name in ["kl_000", "central_8", "kl_020"]:
        t = training[name]
        pressure.append({"condition": name, "beta_kl": t["contract"]["run"]["kl_beta"],
            **evaluation[name], **t["stability"], "allocated_tokens": t["allocated_response_tokens"],
            "actual_tokens": t["actual_response_tokens"]})
    write_csv(destination / "kl_pressure_study.csv", pressure)
    write_csv(destination / "standard_optimizer_steps.csv", [
        {"rollout_update": r["completed_updates"], **s} for r in training["standard"]["history"] for s in r["optimization"]])
    atomic_json(destination / "standard_timing_memory.json", {k: training["standard"][k] for k in
        ("wall_clock_seconds", "wall_clock_complete", "wall_clock_definition", "peak_allocated_gib", "peak_reserved_gib", "sessions")})
    qualitative = []
    for name in NAMES:
        for baseline, candidate in zip(records["midpoint"], records[name]):
            qualitative.append({"condition": name, "prompt_id": candidate["schedule"]["prompt_id"],
                "source_index": candidate["schedule"]["source_index"],
                "prompt_messages": json.dumps(candidate["prompt_messages"], ensure_ascii=False),
                "midpoint_response": baseline["response"], "candidate_response": candidate["response"],
                "midpoint_reward": baseline["raw_reward"], "candidate_reward": candidate["raw_reward"],
                "reward_delta": candidate["raw_reward"] - baseline["raw_reward"],
                "quality_direction": "", "criterion": "", "reasoning": "", "verification_sources": ""})
    write_csv(destination / "qualitative_review.csv", qualitative)
    agreement = disagreement = 0
    if review:
        known = {(r["condition"], r["prompt_id"]): r for r in qualitative}
        reviewed = set()
        with Path(review).open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = r["condition"], r["prompt_id"]
                if key not in known:
                    raise RuntimeError("Qualitative review refers to an unknown result")
                if key in reviewed:
                    raise RuntimeError("Duplicate qualitative review row")
                reviewed.add(key)
                if not r.get("reasoning") or not r.get("criterion"):
                    continue
                delta, quality = known[key]["reward_delta"], r.get("quality_direction")
                agreement += int((delta > 0 and quality == "better") or (delta < 0 and quality == "worse"))
                disagreement += int((delta > 0 and quality == "worse") or (delta < 0 and quality == "better"))
    render_plots(destination, training, evaluation, cache)
    atomic_json(destination / "evidence_inventory.json", {
        "quantitative_complete": True, "qualitative_complete": agreement > 0 and disagreement > 0,
        "reviewed_agreement_examples": agreement, "reviewed_disagreement_examples": disagreement,
        "remaining": [] if agreement and disagreement else ["Human-supported reward/quality agreement and disagreement examples"],
        "central_8_reused_between_sweeps": True, "held_out_selection": False,
        "safety_evidence": "Task 4 remains separate; do not infer safety from learned reward alone"})
    (destination / "README.md").write_text(
        "# Task 2 measured evidence\n\nAll quantitative values were computed from verified saved runs. "
        "The independent central_8 fork appears in both sweeps. Raw learned reward is the primary held-out reward metric; "
        "EOS penalties are reported separately. KL is the signed sampled log-probability difference, not an exact nonnegative KL. "
        "Actual generated tokens can differ despite equal allowances. Standard and short-fork budgets differ.\n\n"
        "Fill qualitative_review.csv using better/same/worse/unclear in quality_direction, a criterion, and supporting reasoning/sources. "
        "The blank worksheet is not quality evidence. No model judge or correctness claim is invented.\n")
    print("CPU evidence saved:", destination, flush=True)
    print("Qualitative evidence complete:", agreement > 0 and disagreement > 0, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review")
    args = parser.parse_args()
    aggregate(args.experiment, args.output, args.review)


if __name__ == "__main__":
    main()
