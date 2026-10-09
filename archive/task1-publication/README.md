# ATML PA2 — Task 1: Direct Preference Optimization

Task 1 experiments and the saved-evidence audit are complete. Five DPO conditions and the untouched SFT baseline were evaluated. **Tasks 2–5 remain course scaffolds, not completed implementations.** The combined assignment report still needs the Task 1 evidence integrated within its eight-page limit.

## Start here

- [Results and interpretation](docs/task1/RESULTS.md): required comparisons, length diagnostics and approved qualitative examples.
- [Audit and requirement coverage](docs/task1/AUDIT.md): what was verified and the limits of that verification.
- [Protocol and decisions](docs/task1/PROTOCOL.md): fixed settings, approvals, TA clarifications, corrections and the failed attempt.
- [Reproduce or inspect](docs/task1/REPRODUCE.md): independent CPU analysis and standalone experiment commands.
- [Evidence index and chronology](docs/task1/EVIDENCE.md): exact sources, traces, approvals, dependencies and Drive handoff.
- [Source attribution](docs/SOURCE_ATTRIBUTION.md).

## Inspect the completed results without a GPU

From this folder, with Python 3.10 or newer:

```bash
python -m task1_dpo.analyze_results --output outputs/task1_recomputed
```

This verifies the exported file hashes and recomputes the saved statistics with the Python standard library. It checks **1,992 held-out pair records, 1,830 generated answers and rewards, five completed training traces, and the preserved failed attempt**. It does not download models or execute training. Choose a new output directory each time.

The checked tables are already in [results/task1/analysis](results/task1/analysis). Raw evidence is in [results/task1/evidence](results/task1/evidence), with an original SHA-256 manifest. Original result files and source snapshots are unchanged.

## Repository layout

| Location | Purpose |
|---|---|
| `task1_dpo/`, `common/`, `configs/` | Standalone implementation and exact experimental configuration |
| `tests/` | Objective, scoring, runtime, evaluation and packaging checks |
| `docs/task1/` | Current Task 1 documentation |
| `results/task1/analysis/` | Independently verified tables and audit result |
| `results/task1/evidence/` | Immutable exported records, executed sources and historical documentation |
| `results/task1/original_summary/` | Original Colab summary export, retained for comparison |
| `archive/atml_pa2_task1.py.txt` | Original chronological Colab export; history only, **do not Run All** |
| `task2_ppo/` through `task5_feedback/` | Unfinished course starter files for subsequent tasks |

The four large preference datasets, public model weights, trained adapters, optimizer states and caches are intentionally absent from this Git package. The ten fixed word-limit prompts were tracked in the course starter and are retained unchanged. Generated evaluation answers and score records are included as required evidence. Pinned downloads and their hashes are documented; the standard adapter remains on Drive for Task 4.

## Manual publication

Extract this package and review/copy its **contents** into your repository, preserving your repository's existing Git history. Review the changes, commit and push yourself. Suggested commit message: `Complete Task 1 DPO experiments, evidence audit and reproducibility documentation`.

Do not upload runtime backups or model folders. The supplied `.gitignore` excludes these. The assistant did not access, commit to, or push to your Git repository. This package does not claim that the final whole-assignment report has been submitted.
