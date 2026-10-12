# Task 4 approved protocol and current progress

Approval recorded on 11 October 2026. Actual Colab execution remains pending. Full source/provenance/test receipts are in ATML_PA2_Task4_CPU_Preflight.zip.


---

```json
{
  "format": "ATML_TASK4_APPROVAL_V1",
  "approved": true,
  "date_local": "2026-10-11",
  "timezone": "Asia/Karachi",
  "user_approval_messages": [
    "Sure I approve",
    "Yes I approve but document everything"
  ],
  "approved_recommendations": {
    "1": "Task 4 reproducible scripts with preserved course helpers; Colab cells are launchers",
    "2": "Generation batches of 4; policies sequential; preserve released precision and judge quantization",
    "3": "Save every generation batch and every judgment under the approved Drive root; checksums and identity-checked resume",
    "4": "Mean and population SD for response length; agreement over all 60 responses per policy including ambiguous labels; report counts and percentages"
  },
  "drive_relative_root": "ATML-Assignment-2/task4/experiment_v1",
  "interaction": "Exactly one executable Colab block; wait for its actual output before the next block",
  "documentation": "Record requirements, choices, provenance, implementation, checks, failures, actual progress, and resume state",
  "execution_environment": "Google Colab; CPU when possible; A100 for model inference",
  "new_discretionary_changes": "Seek explicit user approval before acting",
  "manual_labels": "User-provided blind human labels; no AI labels presented as manual annotations",
  "training_required": false,
  "session_time_limit": null,
  "automatic_disconnect": false,
  "note": "Task 3 time limits, disconnect behavior, optimizer choices and approvals are not inherited by Task 4."
}
```


---

# Assignment requirements and evidence map

Authority: ATML PA2 manual supplied by the user, SHA-256 in `provenance/course_reference.json`. Read as coursework specifications; the user's explicit approval and execution constraints govern assistance. Task 4 pages 12-14 were text-extracted and visually checked during planning; the complete manual was read.

| Source | Fixed requirement | Implementation/evidence destination |
|---|---|---|
| pp. 1-4, 18 | Individual work; understand all code; attribute material reuse; reproducible scripts rather than a monolithic notebook | README, source attribution, standalone entry points |
| pp. 1-4, 18 | Preserve fixed data/IDs; record hyperparameters and machine-readable results; use released environment/helpers | Input contract, environment/model receipts, frozen runtime contract |
| pp. 12-13 | Complete earlier alignment experiments before safety comparison; no Task 4 checkpoint selection or retuning | Standard-run completion records; endpoint identities |
| pp. 12-13 | SFT is untouched Qwen2.5-1.5B-Instruct; DPO is standard one-epoch final; PPO/GRPO are standard 20-update finals | SFT adapter-null check; three standard adapter checks |
| p. 13 | Every fixed XSTest prompt, same order/IDs, one deterministic response per policy, common cap | Frozen prompt IDs; unique policy-ID coverage; saved generations |
| p. 13 | Use supplied categorical judge, prompt and parser; confidence for audit only | Preserved course judge; saved labels/confidence/tags |
| pp. 13-14 | Safe-answer, over-refusal, unsafe-compliance, justified-refusal, ambiguous rates; length; category distribution | Aggregate JSON/CSV and report tables |
| pp. 13-14 | Fixed 60-prompt manual audit, label blindly with same five labels; agreement, confusion breakdown and ambiguity | Blind human sheet; later merged comparison |
| p. 14 | Examples of harmful compliance, justified refusal and over-refusal; identify policy difference/judge error/both | Reviewed qualitative evidence with IDs |
| pp. 13-14 | Judge is not ground truth; policy differences are observational; connect earlier scores with calibration cautiously | Report interpretation and limitations |
| pp. 1, 18-20 | Public repository code/results; no raw datasets/public weights/large checkpoints; combined 8-page NeurIPS report, references excluded | Small results package and reproduction commands; later combined report |

## Exact released settings

Seed 6304. Base policy `Qwen/Qwen2.5-1.5B-Instruct`. AI judge `Qwen/Qwen2.5-3B-Instruct`. Safety response cap 256. Prompt cap 256 in the supplied generation helper. Generation batch-size default 4. Deterministic policy generation `do_sample=False`; the helper ignores temperature/top-p arguments in that mode. Judge cap 64, deterministic decoding. Policy base dtype float16. GPU judge loader uses NF4 4-bit quantization with float16 compute under the released configuration. Preserve inherited model generation defaults and record them; do not silently replace them.

Labels: SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS. Parser failures are labeled AMBIGUOUS by the course parser. Confidence is not a reward, threshold, filter, or weight.

The supplied audit function uses NumPy default_rng(6304), selects 30 SAFE then 30 UNSAFE IDs without replacement in preserved source order, and sorts the combined IDs. Use those same 60 prompt IDs for each policy, giving 240 human response labels. It does not enforce equal counts for every `type`; preserve this released procedure and disclose that limitation.

Rates use all prompts of the appropriate benchmark class in their denominators. Ambiguous labels remain in the denominators. Required lengths include mean; approved dispersion is population SD. Match the released response-token accounting, including generated EOS and excluding padding. Report counts and percentages, all five labels by category, ambiguity overall/by class, and all-60 exact agreement per policy with a five-by-five confusion table.

## Deliberate errors and unfinished work

The manual and course README explicitly identify one deliberate core-objective defect in each of Tasks 1-3. They do not specify an additional Task 4 deliberate objective defect. Task 4's generation/judging main functions and evaluation aggregation are intentionally incomplete scaffolds. Preserve fixed scoring semantics while completing orchestration. The default base sampling configuration must not leak into deterministic Task 4 generation. Avoid ablation adapter substitution, dropping ambiguous examples, or confusing the learned reward model with the categorical safety judge.

## Pending before GPU

Verify actual Drive identities; install/check the released environment and record resolved versions; verify/pin base and judge artifacts; select/verify the exact course audit IDs; validate orchestration, masks/lengths, parser handling, denominators, resumability and output coverage on CPU. Do not use held-out safety behavior to choose settings.


---

# Decisions and approval boundaries

Approved by the user on 11 October 2026: “Sure I approve” followed by “Yes I approve but document everything”. The approved recommendations are reproduced in `approval.json`; these are Task 4 approvals, separate from historical Task 1-3 approvals.

| Decision | Reason and impact | Status |
|---|---|---|
| Separate Task 4 script pipeline, preserved course helpers, Colab launch cells | Avoid hidden notebook state and accidental changes from earlier task-specific helpers | Approved recommendation 1 |
| Batch 4, sequential policies; preserve precision/quantization | Retain released behavior and keep memory requirements controlled | Approved recommendation 2 |
| Approved Drive root; save each generation batch/judgment; checksummed source/data/model contract | Survive Colab interruptions and prevent mixing conditions or stale records | Approved recommendation 3 |
| Mean plus population SD, exact all-60 agreement including ambiguity, counts plus percentages | Consistent denominators and transparent audit reliability | Approved recommendation 4 |
| Exactly one executable block followed by returned output | User controls staged Colab execution and progress is based on verified outputs | Direct user requirement |
| Human labels made without AI-label exposure | Manual audit remains independent of the AI evaluation instrument | Assignment requirement |
| Document all choices, checks, changes and actual progress | Reproducibility and reviewability | Direct user requirement |

No Task 3 session budget or automatic disconnect is adopted. No training optimizer, scheduler, clipping, reward normalization, AMP, safety retuning, alternate judge, parser change, retry policy, or checkpoint selection is introduced. If an unexpected condition requires a consequential change, explain it and seek explicit approval first.

The first block's standard-library-only verifier is an implementation of the approved input audit. It uses recorded paths, preserves prior-task files, saves immutable source/report files with readback, and stops for missing/mismatched inputs. It does not silently search for or substitute another adapter, repair a receipt, or fetch replacements. File layout, UTC attempt timestamps and checksums are bookkeeping supporting the approved persistence approach, not additional experimental settings.

Approval of this plan is not a fabricated verification result. Colab, Drive and model checks remain pending until their actual output is returned. Future stages will extend documentation with exact executed commands, versions, artifact hashes, tests, counts and failure/resume records.


---

# Task 4 progress

## Verified locally

- Complete assignment reviewed; Task 4 pages 12-14 visually inspected during planning.
- Four Task 4 course scripts, common generation/model/data helpers, configs and requirements available from the clean course checkout at recorded commit.
- Prior preserved Task 4 files/configs/generation helper compared across two local copies and matched.
- DPO standard completion metadata and backup identity available; PPO and GRPO standard handoff identities available.
- User explicitly approved recommendations 1-4 and requested documentation.
- First CPU verifier and documents prepared. Actual local validation is recorded separately in `validation/local_tests.json` and its transcript; consult those records for results rather than treating this document as a test receipt.

## Pending actual Colab output

First CPU block: verify XSTest and all three standard adapter inputs on Drive, and save input identities/work counts. Four input groups are planned. No actual Drive checks are claimed in this local release.

The number of course prompts N remains unverified here. Planned generation is 4N responses and judging is 4N calls. The fixed audit covers 60 IDs across four policies, or 240 manual response labels. No Task 4 generation, judging, training or manual annotation has been performed by this work.

## Subsequent stages

1. Finish CPU environment/model preparation and exact audit-ID freezing after the first block's output.
2. Implement and validate CPU orchestration/analysis in reviewable stages.
3. A100 deterministic generation with per-batch persistence.
4. A100 fixed categorical judging with per-response persistence.
5. Blind user manual audit followed by agreement analysis on CPU.
6. CPU evidence/reproduction export and requirement coverage review.

For every returned Colab output, append observed completed/remaining units, successful saved receipts, unresolved issues and the next authorized block. Never infer progress from a launched cell alone. A failed block preserves its successful checks and error report; it does not mark the stage complete.

Latest local outcome: 20/20 synthetic CPU checks passed; zero failures, errors or skips. Source syntax and unchanged reference-file hashes checked. Actual Colab input verification remains 0/4 groups verified by this chat until the first returned output.


---

# Task 4 change and execution record

## 2026-10-11: planning and approval

Read complete PA; visually checked Task 4 pages. Inspected unmodified course safety scripts/configuration and actual prior-task metadata. Distinguished the Tasks 1-3 deliberate defects from Task 4's incomplete orchestration. Presented fixed requirements and recommendations 1-4. User approved and requested documentation.

## 2026-10-11: CPU preflight release 1

Created Task 4's own documentation/provenance tree, preserved course-reference copies, and a standard-library input verifier. Added independent synthetic checks of corrupt/mismatched inputs, archive safety, endpoint completeness and preservation of saved evidence. Local verification results and release hashes are saved in machine-readable records.

This stage reads XSTest and adapter artifacts only. It does not install model packages or deserialize model/optimizer tensors. It saves documentation/source snapshots and per-attempt reports to the approved Task 4 Drive root. Original synced sources and Task 1-3 artifacts remain unchanged. Colab execution is awaiting the user's first output.

Subsequent executed changes, fixes and returned outputs will be documented here or in immutable machine-readable attempt records. Hypotheses will be labeled as hypotheses; a local fixture test is not a Colab/model experiment.

Final local validation: 20 synthetic CPU tests passed, zero failures/errors/skips. Checks cover a full successful first-stage run and matching rerun, failure preservation and recovery, wrong budgets/endpoints, changed input bytes and archive integrity. The RL config check was refined to preserve and report saved metadata while enforcing the exact recorded config hash; it does not assume the RL saved base path equals DPO's public identifier. Final tested source hashes and timestamp are in validation/local_tests.json. No Colab/Drive execution is claimed.


---

Release SHA-256: `3b135537927a0e562a8d5e149a603af021e4231b8a90e5e9e550d82afce694d8`.
