# Repository layout — Task 4 integration, 12 October 2026

The active safety evaluation lives under `task4/`, with CPU reproduction at its top level, immutable exported records in `evidence/`, and sealed source releases plus stage history in `provenance/`. Its README is the current entry point. The five unfinished root `task4_safety/*.py` scaffolds move byte for byte into `archive/course-starter/task4_safety/`.

Previous navigation, both maintenance verifiers, this layout document and their manifest are preserved under `archive/repository-maintenance-20261011/`. The active Task 1 verifier adds only a Task 4 path translation. The original 988-file Task 1 manifest, Tasks 2–3 packages, scientific settings, raw outputs and labels remain unchanged. Task 5 remains unfinished. Source snapshots that share content are kept because their manifests and prior provenance refer to them.

The combined verifier now checks Task 4 integrity and recomputes its saved statistics. `python3 scripts/verify_repository.py --stage` stages only declared files and explicit retired-starter paths; `--staged` checks exact inventory and matching bytes. No command commits or pushes. These commands include required provenance ZIPs despite the outer-ZIP ignore rule.

The delivery installer requires reviewed HEAD and refreshed origin/main at `2173e5e833754e7904a2a89b7950e56919b4565e`. Unexpected local changes, unknown files, path collisions or remote drift stop installation. It first copies the retired source into its archive, verifies the copy, then removes the old active path. Ordinary exceptions restore prior file bytes; a process kill/power failure may require reviewing a partial installation. No reset, stash, history rewrite or remote deletion occurs.

The prior layout record is in `../archive/repository-maintenance-20261011/docs/REPOSITORY_LAYOUT.md`. Historical links/statuses describe their original context. Follow the root README for current navigation.
