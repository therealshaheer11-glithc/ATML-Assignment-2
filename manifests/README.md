# Verification indexes

- `REPOSITORY_SHA256.json`: current publication files (excluding this index itself).
- `BASELINE_SHA256.json`: unchanged hashes of every tracked file at the reviewed pre-cleanup commit.
- `BASELINE_PATH_MAP.json`: where each baseline file is located.
- `TASK1_ORIGINAL_SHA256.json`: unchanged original Task 1 publication manifest.
- `TASK1_FIGURES_ORIGINAL_SHA256.json`: unchanged original figure manifest.
- `WORDING_REVISIONS.json`: exact before/after checksums for revised documentation
  and audit-description metadata, anchored to the preceding publication commit.

Historical manifests retain their original hashes. The verification helper checks
the exact replacement hash only for paths listed in the revision index. Labels,
IDs, response text, numerical results, original scoring code and sealed source
releases remain unchanged. The current publication index covers the revised files,
the revision index and the helper itself.

From the repository root: `python3 -B scripts/verify_repository.py`.
Staging remains a separate action; `--staged` additionally checks the Git index.
