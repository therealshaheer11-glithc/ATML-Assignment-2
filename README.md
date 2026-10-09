# ATML PA2 — LLM Post-Training

Tasks 1 (DPO) and 2 (PPO) have completed experiments, saved evidence, and documented audits. Tasks 3–5 remain unfinished course starters. The combined assignment report is separate from this code and evidence repository.

## Start here

| Task | Implementation | Results and audit |
|---|---|---|
| 1: DPO | [task1_dpo](task1_dpo/) | [Results](docs/task1/RESULTS.md), [audit](docs/task1/AUDIT.md), [figures](figures/task1/README.md) |
| 2: PPO | [task2/source/task2_ppo](task2/source/task2_ppo/) | [Task 2 guide](task2/README.md), [audit](task2/AUDIT.md), [approved examples](task2/QUALITATIVE_EXAMPLES.md) |
| 3–5 | task3_grpo, task4_safety, task5_feedback | Unfinished course starters |

Task 2 carries the exact helpers and configuration used for its experiments inside `task2/source/`. Follow its [restoration instructions](task2/README.md) before running its source tests or experiments. Task 1 uses the root `common/` and `configs/` directories.

## Verify the published files

From the repository root, using Python 3.10 or newer:

```bash
python3 scripts/verify_repository.py
```

This checks the original 988-file Task 1 package at its documented current locations, the separate Task 1 figures addition, all 133 manifest-listed Task 2 files, and the repository maintenance files. It requires no model downloads, third-party Python packages, or GPU. It verifies file integrity; the scientific audit results and their limits are documented in each task's audit.

Task 1 saved statistics can also be recomputed with:

```bash
python3 -m task1_dpo.analyze_results --output outputs/task1_recomputed
```

Choose a fresh output directory. See [Task 1 reproduction](docs/task1/REPRODUCE.md) and [Task 2 verification and reproduction](task2/README.md) for further checks.

## Organization and provenance

- `docs/task1/`, `results/task1/`, and `figures/task1/` contain Task 1 documentation, evidence, and figures.
- `task2/` contains the complete approved Task 2 publication, including its immutable execution source and original evidence.
- `archive/course-starter/task2_ppo/` contains the superseded, incomplete Task 2 course starter. It is historical material, not the implementation used for the completed experiments.
- `archive/task1-publication/` preserves the original Task 1 README and verification script byte for byte. The archived README describes the earlier publication state.
- The original chronological Task 1 notebook export remains in `archive/atml_pa2_task1.py.txt`; it is history and should not be run wholesale.
- Model weights, optimizer checkpoints, runtime backups, downloaded datasets, and outer delivery ZIPs remain outside the published repository.

The intentional evidence and code-release archives inside `task2/` are required publication files. They are distinct from outer delivery ZIPs used to upload packages.

See [repository layout and verification changes](docs/REPOSITORY_LAYOUT.md), [Task 1 source attribution](docs/SOURCE_ATTRIBUTION.md), and [Task 2 attribution](task2/README.md#attribution). Repository maintenance does not change model code, hyperparameters, results, or experimental approvals.
