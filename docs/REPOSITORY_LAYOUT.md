# Repository maintenance — current through 11 October 2026

Task 3 publication retires the six old root `task3_grpo/` starter files into `archive/course-starter/task3_grpo/`, byte for byte, and publishes the completed implementation under `task3/source/`. No root Task 1 helpers/configurations, Task 2 files, Task 4/5 starters, historical package manifests, recorded experiments or scientific settings are changed.

The prior four maintenance files and their manifest are preserved under `archive/repository-maintenance-20261009/`. The active Task 1 verifier adds the six explicit Task 3 path translations to its existing translations. It still verifies the original 988-file manifest unchanged. The combined verifier additionally verifies the original Task 1 figures, original Task 2 package, new Task 3 package/evidence and current maintenance files.

`python3 scripts/verify_repository.py --stage` stages only the full declared repository inventory and the explicit archived-starter removals. It includes the small required code-release ZIP even though outer delivery ZIPs are ignored. `--staged` verifies exact index membership and byte equality to the checked working files. Neither command commits or pushes. Unexpected tracked files stop the stage operation; unrelated local files are not published.

Use the read-only default verification after checkout. The original small export and executed source are immutable. Approved qualitative review and publication analysis are separate additions; historical pending labels are export-time history. The Task 3 manifest excludes superseded local review drafts and the historical resealing builder. The original Colab ZIP still preserves its original contents exactly.

This maintenance resolves the duplicate active Task 3 starter/import path and strict combined-verifier inventory conflicts. It does not auto-merge unknown remote changes. The delivery installer requires the reviewed baseline commit `cddc252d1116de1b26d48823ccfb95dae83b4857` and exact clean baseline bytes; refresh origin/main before application. If main advances or local files conflict, stop for a targeted review rather than resetting or overwriting them.

## Prior maintenance record (9 October)

The complete prior record is preserved at [archive/repository-maintenance-20261009/docs/REPOSITORY_LAYOUT.md](../archive/repository-maintenance-20261009/docs/REPOSITORY_LAYOUT.md). Its relative links describe the older publication layout. The root README is the current navigation page.
