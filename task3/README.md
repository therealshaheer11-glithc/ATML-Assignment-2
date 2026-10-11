# PA2 Task 3 — approved Colab implementation

The deliberate TA defect is corrected: advantages use the reward mean and population standard deviation **within each prompt group**. The PA formula uses `std + 1e-6`. Standard GRPO and the course constant-denominator condition share the corrected advantages and reward function.

User approval: “All approved go ahead”, 10 October 2026, decisions A-M. See [the complete protocol](PROTOCOL_AND_APPROVALS.md) and [authorization record](source/docs/approval.json). This package targets [ATML-Assignment-2](https://github.com/therealshaheer11-glithc/ATML-Assignment-2) and keeps Tasks 1/2 separate.

Completed execution: standard 20/20 updates, both forks 8/8 updates, and 200/200 evaluations for midpoint and each endpoint. The returned small export passed the local audit; the five-pair qualitative score table was approved on 11 October 2026. See [results and limitations](RESULTS.md) and the [approved scores](results/actual_run_20261011/review/SCORES.md). The package is prepared for publication; the final GitHub commit still needs verification. Main-report integration remains separate. See the [PA compliance audit](publication/PA_COMPLIANCE.md).

## Verify this publication without GPU

From the repository root with Python 3.10 or newer:

```bash
python3 scripts/verify_repository.py
python3 task3/verify_package.py
```

These checks require only the standard library. They validate file hashes, the original sealed release, all 800 evaluation records, the 36 training records and the approved score table. They do not load models or rerun training. A fresh independent audit can be saved with:

```bash
python3 task3/analysis/audit_saved_export.py task3/results/actual_run_20261011 --output task3/.runtime/rechecked-audit
```

Rebuild the complete CSV tables and exact reviewed examples with:

```bash
python3 task3/analysis/present_saved_results.py --output task3/.runtime/rebuilt-presentation
```

Add `--plot` with matplotlib installed to recreate the eight-panel diagnostic figure. See [all difficulty-bin statistics](publication/derived/GROUP_SIZE.md), [complete training diagnostics](publication/derived/complete_training_diagnostics.png), [exact reviewed questions and answers](publication/derived/EXACT_REVIEWED_EXAMPLES.md), and the [Task 4 adapter identity](publication/TASK4_HANDOFF.json).

The old root `task3_grpo/` starter is archived under `archive/course-starter/task3_grpo/`. Run the completed code from the isolated Task 3 runtime. The preserved Colab ZIP is the actual original release, including historical preparation-time documentation. Never run its historical package builder to reseal the executed source.

## Run in Colab

1. Download the preserved [original Colab release](provenance/releases/ATML_PA2_Task3_Colab.zip) and both notebooks. Open `notebooks/01_CPU_Prepare.ipynb` in Colab. Choose **no hardware accelerator**, run both cells, and upload the ZIP when prompted. Authorize your Drive mount. This downloads pinned files, validates the course bundle, runs CPU tests and freezes schedules. It does not train or load policy weights.
2. After successful preparation, open `notebooks/02_GPU_Run_and_Resume.ipynb`. Choose a **GPU**, run both cells and upload the same ZIP. It runs the probe, standard continuation, two forks and four endpoint evaluations. All settings are already approved and fixed.
3. If the 120-minute scheduling boundary is reached, run the GPU notebook again. It restores completed units under the same source/config/data contract. A current unit can overrun the boundary. Errors keep the runtime available. After verified saves, a stopped model process and successful Drive flushing, the notebook performs the approved automatic disconnect.
4. Results appear in `MyDrive/ATML-Assignment-2/task3/experiment_v1/task3_small_results.zip`. Return that small ZIP for qualitative review and the Task 3 report. Large weights and checkpoints remain on Drive. The final **standard 20-update** adapter is `training/standard/final.zip`, inside `policy/`, for Task 4.

An existing CPU preparation root requires explicitly setting `RESUME_PREPARATION = True` in the CPU notebook; its ownership and contract must match. A different contract stops for review. Do not rename an old experiment or change scientific settings to bypass a failed check. The recorded production run used an NVIDIA A100-SXM4-40GB; other GPU configurations have not been established by this run.

## Fixed experimental queue

| Job | Initialization | Updates / evaluation |
|---|---|---|
| Shared normalization probe | Supplied GRPO midpoint | Same frozen rollout, both losses, zero optimizer updates |
| Standard GRPO | Supplied GRPO midpoint | 20 updates, K=4 |
| Canonical fork | Independently reload supplied midpoint | 8 updates, K=4 |
| Constant-denominator fork | Independently reload supplied midpoint | 8 updates, K=4, denominator 512 |
| Common evaluation | Midpoint and all three frozen endpoints | All 200 prompts each; identical per-prompt seeds |
| Group-size diagnostic | Supplied fixed cache | K=2/4/8, all 192 cached generations each; no training |

Both short forks have 16,384 allocated response tokens each. Natural EOS means **actual generated tokens can differ**; actual and loss-bearing counts are saved separately. Truncated continuations are masked from training, retained for length/evaluation diagnostics. Raw reward scoring uses the released 1,024-token input cap. Evaluation generation uses the approved 512-token response cap.

## Reviewable code chunks

- `source/task3_grpo/grpo.py`: objective correction and normalization contrast.
- `training_core.py`: detached rollout statistics, truncation masks, one AdamW step and direct token-log-probability gradient statistics.
- `analyze_group_size.py`: conserved cache partitions, fixed difficulty proxy and signal statistics.
- `reward_compat.py`: approved in-memory RoPE translation; original config bytes remain unchanged.
- `checkpoint.py`, `storage.py`, `jobs.py`: optimizer/RNG recovery and verified per-update/per-prompt persistence.
- `prepare.py`, `runtime.py`, `pipeline.py`: CPU preparation receipts, pinned offline model loading and approved GPU queue.
- `report.py`: evidence audit, summaries, trajectory figure and exact qualitative examples exported before the subsequent approved review.

The course `common/`, configs and requirements are copied unchanged. `provenance/COURSE_SOURCE_SHA256.json` records original starter hashes. `provenance/REVIEW_EVIDENCE.json` preserves the earlier pre-approval review; `source/docs/approval.json` records the subsequent authorization. `source/docs/SOURCE_SHA256.json` seals the implemented source. `restore_source.py` reconstructs a separate runtime and refuses to overwrite differing code.

## Validation and limits

`validation/cpu_tests.json` records actual local CPU tests and tested file hashes. They cover multi-prompt independence, population statistics, clipping signs, both length denominators, EOS/padding/truncation, detached statistics, reference isolation, optimizer/RNG recovery, exact resumed adapter parameters, archive integrity and a synthetic 800-record evaluation/export round trip. Synthetic tests are not experimental results.

`validation/group_size.json` is actual analysis of the pinned supplied cache. `validation/pinned_configuration_audit.json` verifies the real pinned public reward configuration and inherited generation defaults. No public model weights were loaded locally. The instructor midpoint metadata is provenance only.

The actual Colab run records package/environment validation, asset checks, Drive persistence, 8-bit CUDA reward loading, all 36 continuation updates and all 800 evaluations. The local small-export audit reproduces saved metrics and verifies source/approval identities, schedules, token counts and internal file hashes; large checkpoint/optimizer tensors remain on Drive and were not independently inspected locally. The original export's pending qualitative flags are preserved; the subsequently approved score table is a separate sidecar. Input truncation, output masking, sparse gradient bins and selected-example limits are documented in [RESULTS.md](RESULTS.md). Nonfinite values, unexpected dtypes, model-loading differences, offload and contract mismatches stop execution; no stabilization or alternate settings are silently introduced.

## Commands behind the notebooks

From the reconstructed runtime:

```bash
python -m task3_grpo.prepare --experiment /content/drive/MyDrive/ATML-Assignment-2/task3/experiment_v1
python -m task3_grpo.pipeline --experiment /content/drive/MyDrive/ATML-Assignment-2/task3/experiment_v1 --resume
```

The GPU notebook additionally passes its startup timestamp so installation/preflight time counts toward the session limit. Separate entrypoints `continue_train`, `compare_normalization` and `evaluate` launch their corresponding fixed stages; they do not accept hyperparameter overrides. Local CPU validation is `python scripts/run_tests.py --output /tmp/task3_tests.json` from `source/`. Dependencies match the course requirements.

## Attribution

Course code: [ATML-PA2-LLM-PostTraining](https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining/tree/1d64ac65acd5e45d1e4e1f415edc80455e21274e). Assets: `AbDu11aHHH/ATML-PA2-assets`, revision `0b350481fb03f5525a35bcdec4131bd4fe487f98`. Policy/tokenizer: Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Reward: yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback revision `f98759a4a1ccdc47a3136748b6bfea9a32ba8fcc`.

The storage/recovery and narrow reward-configuration bridge are adapted from the user's approved Task 2 source snapshot at commit `cddc252d1116de1b26d48823ccfb95dae83b4857`. New Task 3 orchestration and tests were developed with coding assistance under A-M approval. The cached asset manifest has a stale self-checksum; original bytes are preserved and all substantive entries are checked.

Colab persistence/disconnect calls follow Google's [Drive implementation](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py) and [runtime implementation](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py). Successful flush/disconnection is recorded in the completed Colab execution. The Codespaces publication workflow verifies the final index and remote commit separately.
