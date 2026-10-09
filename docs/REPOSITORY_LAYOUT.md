# Repository maintenance — 9 October 2026

The user explicitly approved the proposed cleanup with “sure”: archive the six old Task 2 starter files, replace the root README, preserve the original Task 1 publication documents and checksums, adapt the verifier for moved files, restore the previously approved Task 1 figures, and publish the audited Task 2 package including its required archives.

## Preserved original files

The root `PACKAGE_SHA256.json` remains byte for byte unchanged. Its SHA-256 is `1ea18dbe564c95ebb326b062305279c191344036730a896f78b87b5178dd63ca`.

| Original location | Current location |
|---|---|
| `README.md` | `archive/task1-publication/README.md` |
| `scripts/verify_task1_package.py` | `archive/task1-publication/verify_task1_package.py` |
| The six files under `task2_ppo/` | Corresponding paths under `archive/course-starter/task2_ppo/` |
| All other original Task 1 package files | Unchanged |

The active Task 1 verifier checks the same 988 original bytes against the original manifest, using only these explicit path translations. Its success message covers those historical files. The active combined verifier additionally checks the current README and maintenance scripts, the 11 files listed in the original Task 1 figures manifest, and the 133 files listed in the original Task 2 manifest.

The archived verification script is a source snapshot, not a runnable entry point from its new directory. Use `python3 scripts/verify_repository.py` from the repository root.

## Added publication material

The separate, previously approved Task 1 figures package is copied unchanged, including its own manifest. The approved final Task 2 package is preserved unchanged. The completed PPO implementation is `task2/source/task2_ppo/`; the root starter path has been retired.

The combined verifier's `--stage` option stages exactly the expected publication files and the approved removals from their old paths. It explicitly includes required ignored Task 2 archives, and rejects unrelated tracked files. The `--staged` option checks that the index contains exactly that inventory and matches the verified working files. Neither option commits or pushes. Outer upload ZIPs and local runtime files are excluded from this expected inventory.

The maintenance hash manifest checks the four current maintenance files. Like any repository checksum list, it detects accidental changes relative to its recorded values; it is not an external authenticity signature.

No training, inference, optimizer settings, model code, checkpoint selection, recorded observations, or original evidence were changed by this maintenance. Historical documents retain the status they recorded at their original publication time. The root README is the current navigation page.
