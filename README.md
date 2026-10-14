# ATML PA2 — LLM Post-Training

Completed experiments compare DPO, PPO, GRPO, safety calibration, and verifier versus AI feedback. Tasks 1–5 were published at **`7e7099d99180ce7e180b09da22fd9ee8fc835566`**. This repository preserves the executed source, fixed IDs, logs, results, and documented limitations. The final course report and LMS submission are separate deliverables.

## Start here

| Task | Executed implementation | Results and interpretation |
|---|---|---|
| 1: DPO | [task1_dpo](task1_dpo/) | [Results](docs/task1/RESULTS.md), [audit](docs/task1/AUDIT.md), [figures](figures/task1/README.md) |
| 2: PPO | [task2/source/task2_ppo](task2/source/task2_ppo/) | [Guide](task2/README.md), [audit](task2/AUDIT.md), [examples](task2/QUALITATIVE_EXAMPLES.md) |
| 3: GRPO | [task3/source/task3_grpo](task3/source/task3_grpo/) | [Results](task3/RESULTS.md), [coverage](task3/publication/PA_COMPLIANCE.md) |
| 4: Safety | [Source and stage index](task4/REPRODUCE.md) | [Results](task4/RESULTS.md), [coverage](task4/PA_COMPLIANCE.md) |
| 5: Feedback | [task5/source/frozen_release](task5/source/frozen_release/) | [Results](task5/RESULTS.md), [coverage](task5/PA_COMPLIANCE.md) |

See [current publication status](docs/PUBLICATION_STATUS.json) and [source attribution](docs/SOURCE_ATTRIBUTION.md). Task 6 and the report are separate future work.

## Verify and reproduce the published results on CPU

Run from the repository root with Python 3.10+; no third-party packages, credentials, model downloads, or GPU are required:

```bash
python3 -B scripts/verify_repository.py
python3 -B scripts/reproduce_saved_results.py --output outputs/reproduced
```

The second command requires a new output directory and saves separate logs plus a checksum receipt. It checks the publication and regenerates the five tasks' saved-result analyses. Task 5's public records contain scores, lengths and judge labels; recalculation from those records does not independently rescore private full answer text. For exact answer rescoring, see its [private-bundle audit](task5/REPRODUCE.md).

Individual commands are:

```bash
python3 -B -m task1_dpo.analyze_results --output outputs/task1_recomputed
python3 -B task2/analysis/audit_export.py --original task2/evidence/original --output outputs/task2_recomputed
python3 -B task3/analysis/present_saved_results.py --output outputs/task3_recomputed
python3 -B task4/reproduce.py --output outputs/task4_recomputed
python3 -B task5/verify_package.py --output outputs/task5_recomputed
```

Task 2's corrected clipping table is `task2/evidence/reviewed/clipping_study_corrected.csv`; the original export is preserved as history. Do not substitute its overwritten metadata fields into the report.

## Model execution: prerequisites and exact entry points

The commands below describe the implemented experiments, not a request to repeat completed runs. Use CPU for preparation and analysis, and the documented A100 environment for model execution. The original experiments used Python 3.13.15 and CUDA PyTorch 2.11.0, with pinned task-specific dependencies. Read each task's environment and persistence instructions before running it. Do not install the CPU verification environment over an active GPU environment.

**Portability boundary:** Tasks 1 and 3 have public asset preparation paths. Tasks 2, 4, and 5 preserve their original staged setup and guards; some stages require owner-held Drive receipts, weights, and earlier checkpoints. The public package supports complete saved-statistic reproduction, but does not yet provide a tested empty-machine bootstrap for every GPU workflow. Exact historical receipts must never be invented, and guards must not be disabled to make a fresh runtime proceed. [Details](docs/REPRODUCTION_SCOPE.md).

### Task 1: standard DPO, beta sweep, and length control

From the repository root, with compatible CUDA PyTorch installed and Drive mounted:

```bash
python -m pip install -r requirements.txt -c requirements-task1-lock.txt
python -m task1_dpo.prepare_assets --models-dir /content/ATML_PA2_models
mkdir -p /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
```

For standard and length-balanced training, plus the untouched SFT evaluation:

```bash
(
set -e
for condition in standard length_balanced; do
  python -m task1_dpo.train --run-name "$condition" --models-dir /content/ATML_PA2_models --output "outputs/reproduce/$condition/checkpoint" --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
  python -m task1_dpo.evaluate --name "$condition" --phase policy --adapter "outputs/reproduce/$condition/checkpoint" --models-dir /content/ATML_PA2_models --output "outputs/reproduce/$condition/policy" --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
  python -m task1_dpo.evaluate --name "$condition" --phase reward --policy-results "outputs/reproduce/$condition/policy" --models-dir /content/ATML_PA2_models --output "outputs/reproduce/$condition/reward" --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
done
)
python -m task1_dpo.evaluate --name sft --phase policy --models-dir /content/ATML_PA2_models --output outputs/reproduce/sft/policy --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
python -m task1_dpo.evaluate --name sft --phase reward --policy-results outputs/reproduce/sft/policy --models-dir /content/ATML_PA2_models --output outputs/reproduce/sft/reward --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
```

The fixed three-condition beta sweep uses:

```bash
python -m task1_dpo.ablate_beta --phase train --run-root outputs/reproduce --models-dir /content/ATML_PA2_models --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
python -m task1_dpo.ablate_beta --phase policy --run-root outputs/reproduce --models-dir /content/ATML_PA2_models --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
python -m task1_dpo.ablate_beta --phase reward --run-root outputs/reproduce --models-dir /content/ATML_PA2_models --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
```

Execute phases separately and stop on failure. The fixed evaluations include the required length strata and word-limit prompts. [Full Task 1 instructions](docs/task1/REPRODUCE.md).

### Task 2: standard PPO, clipping and KL-pressure forks

Restore the exact course checkout and overlay on CPU:

```bash
python3 task2/restore_source.py
```

Follow the [Task 2 setup notebooks](task2/notebooks/) and [preparation constraints](task2/README.md#reproduction-and-checkpoints). After valid preparation, the underlying command from the reconstructed runtime runs the standard 20-update continuation, five independent eight-update forks, seven 200-prompt endpoint evaluations, and the fixed clipping cache:

```bash
cd task2/.runtime/course
python -m scripts.task2_resume_json --pipeline-entry --models-dir /content/ATML_PA2_Task2_models --persistent-root /content/drive/MyDrive/ATML-Assignment-2/task2/experiment_v1 --session-minutes 120
```

The command includes the approved JSON-contract compatibility repair. Use the corrected `Task2_GPU_Resume.ipynb` launcher for staged persistence and automatic release; the lower-level command does not itself disconnect the runtime. The independent central fork is reused in the clipping and KL sweeps. CPU aggregation of a complete original experiment uses:

```bash
python -m task2_ppo.aggregate_cpu --experiment /content/drive/MyDrive/ATML-Assignment-2/task2/experiment_v1 --output /content/drive/MyDrive/ATML-Assignment-2/task2/recomputed_tables
```

That historical aggregator creates a new qualitative worksheet. The approved published examples are separately retained in `task2/QUALITATIVE_EXAMPLES.md`; blank worksheets are not completed qualitative evidence.

### Task 3: standard GRPO, group-size cache and normalization forks

```bash
python3 task3/restore_source.py --destination task3/.runtime/course
```

With the [preserved Colab setup](task3/README.md#run-in-colab), run from the reconstructed runtime:

```bash
cd task3/.runtime/course
python -m task3_grpo.prepare --experiment /content/drive/MyDrive/ATML-Assignment-2/task3/experiment_v1
python -m task3_grpo.pipeline --experiment /content/drive/MyDrive/ATML-Assignment-2/task3/experiment_v1 --resume
```

Preparation computes the fixed K=2/4/8 cache study. The pipeline performs the shared normalization probe, standard 20-update continuation, both eight-update forks, and four 200-prompt evaluations. All group-size conditions retain the same 192 cached generations. The original CPU/GPU notebooks remain thin staged launchers rather than the sole implementation.

### Task 4: fixed-policy safety and audit

Restore the 19 sealed source releases without executing them:

```bash
python3 task4/restore_releases.py --destination outputs/task4_original_releases
```

On the already prepared original Colab runtime, change the notebook working directory to the repository root. Run each of these Python statements in a separate notebook cell, checking completion before continuing. The generation launcher asks for the preserved `ATML_PA2_Task4_A100_Generation.zip` from `task4/provenance/releases/`:

```python
import runpy
runpy.run_path("task4/provenance/launchers/Task4_Block14_A100_Generation.py", run_name="__main__")
runpy.run_path("task4/evidence/execution/judging/JUDGING_SOURCE.py", run_name="__main__")
runpy.run_path("task4/evidence/summaries/AGGREGATION_SOURCE.py", run_name="__main__")
```

These retain historical prerequisite and source guards. The [stage index](task4/REPRODUCE.md#source-stage-index) documents preparation, resume, audit-packet creation, and audit-label import. Standard Tasks 1–3 adapters are required; they are not replaced with ablation winners. Audit labels were AI-generated under a user-reported TA exception and are not independent human validation.

### Task 5: SFT/RLVR/RLAIF and controlled diagnostics

With the original mounted Drive project, run CPU preflight from `task5/source/frozen_release/`:

```bash
python -B -m task5_run.preflight
```

Restore the documented A100 environment using the [preserved setup](task5/REPRODUCE.md#original-model-execution-and-resume). From the frozen release directory, run one phase at a time:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /content/task5_isolated_runtime_v1/bin/python -B -m task5_run.run generate --project /content/drive/MyDrive/ATML-Assignment-2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /content/task5_isolated_runtime_v1/bin/python -B -m task5_run.run judge --project /content/drive/MyDrive/ATML-Assignment-2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /content/task5_isolated_runtime_v1/bin/python -B -m task5_run.run aggregate --project /content/drive/MyDrive/ATML-Assignment-2
```

These cover 1,200 policy answers and 880 comparison events, including all 80 diagnostic pairs. The released four-token judge limit, first-label parsing, cache behavior, and high tie rate remain disclosed. No Task 5 training is required.

## Interpretation and submission

Tasks 2–3 use equal **allocated** short-run token allowances and fixed update counts, preserving natural EOS termination. Actual generated-token totals differ and are reported. This is an explicit experimental choice, not a claim of exactly equal token consumption or instructor endorsement. See [budget rationale](docs/EXPERIMENTAL_DECISIONS.md).

The report, Task 6 synthesis and LMS submission are outside this repository-cleanup update.

## Layout, provenance and attribution

Task 1 uses root `common/` and `configs/`. Later tasks retain isolated helpers to reproduce their executed versions. `archive/` contains historical source and earlier publication documentation; those files intentionally describe earlier states. The root `task5_feedback/` is original course scaffolding, not the completed pipeline. Use `task5/source/frozen_release/`.

Raw downloaded datasets, public model weights, optimizer checkpoints, caches and outer upload ZIPs stay outside Git. Small response evidence, fixed IDs, exact source releases and machine-readable results are intentional publication artifacts. See [layout](docs/REPOSITORY_LAYOUT.md).

Materially reused course code is attributed to [AbDu11aHHH/ATML-PA2-LLM-PostTraining](https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining); pinned revisions and the completed student/Codex contributions are listed in [source attribution](docs/SOURCE_ATTRIBUTION.md), [Task 2](task2/README.md#attribution), [Task 3](task3/README.md#attribution), [Task 4](task4/README.md#attribution), and [Task 5](task5/provenance/README.md). The student remains responsible for understanding the submitted work.
