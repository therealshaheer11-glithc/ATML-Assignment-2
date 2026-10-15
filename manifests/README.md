# Verification indexes

- `REPOSITORY_SHA256.json`: current publication files (excluding this index itself).
- `BASELINE_SHA256.json`: unchanged hashes of every tracked file at the reviewed pre-cleanup commit.
- `BASELINE_PATH_MAP.json`: where each of those original bytes is now retained.
- `TASK1_ORIGINAL_SHA256.json`: unchanged original Task 1 publication manifest.
- `TASK1_FIGURES_ORIGINAL_SHA256.json`: unchanged original figure manifest.

Edited guides and verification wrappers are archived before replacement. Scientific
source, labels, results and original experiment seals retain their exact bytes.
Checks validate both the current publication and all retained baseline files.

From the repository root: `python3 -B scripts/verify_repository.py`.
Staging remains a separate action; `--staged` additionally checks the Git index.
