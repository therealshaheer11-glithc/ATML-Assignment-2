# Task 4 Block 11b: read-only Colab base layout diagnosis

Block 11 stopped at `validate_base_inventory` with `Base distribution paths/duplicates require review`. Its returned output shows the same A100, a valid 84-package isolated environment and clean isolated dependency check, then a fresh base inventory of 723 packages under Python 3.13.15. It stopped before planning repairs or any package mutation. The generic guard message alone cannot identify whether duplicate metadata, an outside-prefix distribution, or an outside-prefix site path triggered it.

The next stage inspects those exact entries before deciding whether the guard needs correction. It does not relax the guard, accept a dependency exception, install/uninstall a package, remove duplicate metadata, restart the kernel, recreate the environment, load models or run evaluation. Read-only inspection falls within the ongoing review and preparation of the already-approved conditional repair; no new repair action is authorized by this diagnostic.

Run the single Block 11b launch cell on the same connected A100 runtime and upload only `ATML_PA2_Task4_A100_Base_Layout_Diagnostic.zip`. Preserve the local isolated environment. Return the entire output before any further repair/restart.

The block verifies task ownership and sealed source identities, the saved failed report and every saved artifact hash, and that no base mutation command started. It prints the exact saved guard-trigger lists. A fresh standard-library-only child process enumerates every distribution's name, version, installation root and metadata path without importing the installed libraries. Duplicate entries and outside-prefix paths are printed for review; the complete inventory remains in the saved JSON. Python isolated mode here sanitizes the base child process's import search; it does not create a virtual environment or modify packages.

Two fresh scans must agree exactly, and current base versions/interpreter must match the failed attempt's saved inventory. The parent's plotting setting and isolated validation contract must remain unchanged. Repeated occurrences of one metadata path, separate metadata directories with matching versions, and conflicting versions remain distinguishable in the evidence; this diagnostic approves none of them automatically. The exact isolated interpreter still retains its existing strict no-duplicate/no-outside-path rule.

Evidence is saved under the existing Drive experiment root in `base_layout_diagnostic_releases/<source SHA-256>/` and `base_layout_diagnostic_attempts/<UTC>/`. Files include the original failure link, saved rejected inventory, commands/results, complete fresh scans, focused path details and final hash-linked report. `BASE_LAYOUT_DIAGNOSIS_SAVED_REVIEW_PENDING` indicates diagnosis completed, not a repaired base or a GPU-ready environment.

The new scanner is checked locally against actual local metadata and with a synthetic distribution fixture that distinguishes repeated locations and conflicting versions. All earlier released sources remain unchanged. Actual Colab path details remain pending the returned output. New inspection code and documentation were prepared with Codex assistance.

Progress remains responses 0/1,800, judgments 0/1,800 and independent human labels 0/240. No training or experimental setting changes.
