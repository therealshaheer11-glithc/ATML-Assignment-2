# ATML PA2 — LLM Post-Training

Tasks 1 (DPO), 2 (PPO), 3 (GRPO), 4 (safety calibration), and 5 (feedback sources) have completed experiments, saved evidence, and documented audits. The combined assignment report is separate from this code and evidence repository.

## Start here

| Task | Implementation | Results and audit |
|---|---|---|
| 1: DPO | [task1_dpo](task1_dpo/) | [Results](docs/task1/RESULTS.md), [audit](docs/task1/AUDIT.md), [figures](figures/task1/README.md) |
| 2: PPO | [task2/source/task2_ppo](task2/source/task2_ppo/) | [Task 2 guide](task2/README.md), [audit](task2/AUDIT.md), [approved examples](task2/QUALITATIVE_EXAMPLES.md) |
| 3: GRPO | [task3/source/task3_grpo](task3/source/task3_grpo/) | [Task 3 guide](task3/README.md), [results](task3/RESULTS.md), [PA audit](task3/publication/PA_COMPLIANCE.md) |
| 4: Safety | [Task 4 source and reproduction](task4/README.md) | [Results](task4/RESULTS.md), [coverage and limits](task4/PA_COMPLIANCE.md) |
| 5: Feedback | [Task 5 source and reproduction](task5/README.md) | [Results](task5/RESULTS.md), [coverage and limits](task5/PA_COMPLIANCE.md) |

Task 2 carries the exact helpers and configuration used for its experiments inside `task2/source/`. Follow its [restoration instructions](task2/README.md) before running its source tests or experiments. Task 1 uses the root `common/` and `configs/` directories.

## Verify the published files

From the repository root, using Python 3.10 or newer:

```bash
python3 scripts/verify_repository.py
```

This checks the original 988-file Task 1 package at its documented current locations, the separate Task 1 figures addition, all 133 manifest-listed Task 2 files, the manifest-listed Task 3 files and saved-record audit, all Task 4 files and 1,800 saved response–judgment pairs with 240 audit labels, all Task 5 files with 1,200 policy metric records, 880 comparison events and nine reproduced result files, and the repository maintenance files. It requires no model downloads, third-party Python packages, or GPU. It verifies file integrity; the scientific audit results and their limits are documented in each task's audit.

Task 1 saved statistics can also be recomputed with:

```bash
python3 -m task1_dpo.analyze_results --output outputs/task1_recomputed
```

Choose a fresh output directory. See [Task 1 reproduction](docs/task1/REPRODUCE.md) and [Task 2 verification and reproduction](task2/README.md) for further checks. Task 3 CPU verification and exact Colab reproduction commands are in the [Task 3 guide](task3/README.md).

## Task 3 verification and reproduction

```bash
python3 task3/verify_package.py
python3 task3/restore_source.py --destination task3/.runtime/course
```

Use the preserved [Task 3 Colab ZIP](task3/provenance/releases/ATML_PA2_Task3_Colab.zip) with its [CPU notebook](task3/notebooks/01_CPU_Prepare.ipynb) and [GPU/resume notebook](task3/notebooks/02_GPU_Run_and_Resume.ipynb) for model execution. After notebook setup in Colab, the underlying fixed commands from `/content/task3_runtime/course` are:

```bash
python -m task3_grpo.prepare --experiment /content/drive/MyDrive/ATML-Assignment-2/task3/experiment_v1
python -m task3_grpo.pipeline --experiment /content/drive/MyDrive/ATML-Assignment-2/task3/experiment_v1 --resume
```

These are reproduction instructions, not a request to rerun completed experiments. The notebooks enforce persistence and session limits. Task 3 source must run in its isolated runtime because root helpers belong to Task 1. The final standard 20-update adapter is reserved for Task 4; model files stay on Drive.

## Task 4 verification and reproduction

```bash
python3 task4/verify_package.py
python3 task4/reproduce.py --output outputs/task4_recomputed
```

These CPU commands reproduce all four metric summaries, four audit matrices and six CSV tables. The [Task 4 guide](task4/README.md) links the original execution source, fixed configuration/IDs and logs. The audit applies the fixed five-label rubric to the saved 60-prompt subset for each policy using an automated procedure.

## Task 5 verification and reproduction

```bash
python3 -B task5/verify_package.py
python3 -B task5/verify_package.py --output outputs/task5_recomputed
```

These CPU commands reproduce the saved quantitative results without model loads.
The [Task 5 guide](task5/README.md) links fixed IDs, immutable model-execution
source, recorded settings, small logs and approved qualitative examples. Public
records exclude full raw course data and generated answer text; optional private
bundle verification recomputes exact scores from the saved full evidence. The
judge's four-token limit, frequent ties and mixed-label outputs are documented.
The root `task5_feedback/` remains the original course reference, including its
unfinished scaffolds; the completed implementation is `task5/source/frozen_release/`.

## Organization and provenance

- `docs/task1/`, `results/task1/`, and `figures/task1/` contain Task 1 documentation, evidence, and figures.
- `task2/` contains the complete approved Task 2 publication, including its immutable execution source and original evidence.
- `task3/` contains the completed Task 3 implementation, immutable source/release and original evidence plus approved review sidecars.
- `task4/` contains completed safety evaluation, frozen exports and original source releases; `archive/course-starter/task4_safety/` preserves its five retired scaffolds.
- `task5/` contains completed feedback-source evaluation, CPU reproduction, fixed IDs, metric records and source provenance.
- `archive/repository-maintenance-20261014-task5/` preserves navigation, the combined verifier and maintenance records before Task 5 integration.
- `archive/repository-maintenance-20261011/` preserves navigation and verifiers before Task 4 integration.
- `archive/course-starter/task3_grpo/` preserves the six retired Task 3 starter files.
- `archive/repository-maintenance-20261009/` preserves the previous repository navigation and verifiers.
- `archive/course-starter/task2_ppo/` contains the superseded, incomplete Task 2 course starter. It is historical material, not the implementation used for the completed experiments.
- `archive/task1-publication/` preserves the original Task 1 README and verification script byte for byte. The archived README describes the earlier publication state.
- The original chronological Task 1 notebook export remains in `archive/atml_pa2_task1.py.txt`; it is history and should not be run wholesale.
- Model weights, optimizer checkpoints, runtime backups, downloaded datasets, and outer delivery ZIPs remain outside the published repository.

The intentional evidence and code-release archives inside `task2/` and `task3/provenance/releases/` / `task4/provenance/releases/` are required publication files. They are distinct from outer delivery ZIPs used to upload packages.

See [repository layout and verification changes](docs/REPOSITORY_LAYOUT.md), [Task 1 source attribution](docs/SOURCE_ATTRIBUTION.md), and [Task 2 attribution](task2/README.md#attribution), and [Task 3 attribution](task3/README.md#attribution). Repository maintenance does not change model code, hyperparameters, results, or experimental approvals.
