# Task 4 Block 11c: base repair with the observed Colab layout preserved

## Verified diagnosis and correction

Block 11b returned `BASE_LAYOUT_DIAGNOSIS_SAVED_REVIEW_PENDING` from `base_layout_diagnostic_attempts/20261011T162510.009372Z/REPORT.json`. It verified the failed Block 11 evidence and no mutation commands, inspected two identical fresh metadata/path scans and saved the exact entries. Sixteen names have duplicate metadata; cryptography has three entries, making seventeen repeated occurrences. All roots lie under `/usr`; outside-prefix distributions and site paths are empty. Some groups have matching versions; others differ, including NumPy 2.1.3 versus 1.26.4, packaging 26.3 versus 24.0, and cryptography 50.0.2 versus 41.0.7.

Our original base checker required a duplicate-free layout and stopped before any repair. That was too strict for the observed base layout. The correction permits only the exact diagnosed entries and search path to remain unchanged in base Python. It preserves every entry, never removes OS metadata, never selects substitute package versions and never waives a failed dependency check. The fully isolated interpreter keeps its original strict no-duplicate/no-outside-prefix rule.

There is a second inventory detail: a dict created by walking all metadata retains the last encountered version. With duplicate metadata, that historical map does not necessarily describe the distribution selected by a named metadata lookup. The new source keeps the original map for comparison to the sealed historical reports, labels its meaning explicitly, and separately records actual `importlib.metadata.distribution(name)` selections and versions. It requires those selections to match the reviewed enumeration/path order. Fresh NumPy and packaging imports must also match their selected versions and installation roots. No installed version is changed by this reporting correction.

## Existing authorization and exact scope

The explicit “approved” response to `BLOCK8_ISOLATED_SETUP_APPROVAL_PROPOSAL.md` already authorizes the precise conditional base repair. This is a correction to its implementation after reviewing the stopped guard; it adds no package/version/removal target or experimental choice. The inherited approval is unchanged. `AUTHORIZATION_CONTEXT.json` documents the correction, and `REVIEWED_LAYOUT_DETAILS.json` freezes the exact user-returned entries. Saved Drive evidence must match these entries directly before changes. The earlier sealed sources remain untouched.

The original package scope is reused unchanged:

- Restore only known modified Hub to 1.33.0, PEFT to 0.21.1, Tokenizers to 0.23.2, and Transformers to 5.18.0 where necessary.
- Add Jedi 0.20.0 only if absent; preserve Parso 0.8.7.
- Remove only the recorded course-added global TRL 0.27.2/bitsandbytes 0.50.2 copies, only if present at those versions and after isolated validation.
- Preserve IPython 7.34.0, Diffusers 0.40.0, Gradio 6.29.0, global Torch 2.11.0+cu130, all unrelated packages and every diagnosed duplicate entry.

The original plan guard and hashed offline installation/removal commands are copied byte-for-byte. Target names must have unique metadata; none of the seven possible repair targets may be in the diagnosed duplicate groups. Unknown versions, a new/changed duplicate, reordered discovery, changed paths or unrelated metadata drift stop before mutation.

## Run one block on the same A100

Run the supplied Block 11c launch cell and upload only `ATML_PA2_Task4_A100_Base_Repair_Recheck.zip`. Keep the same A100 runtime connected; return complete output before any restart. The ZIP and every source member are checksum-verified. A small PID/already-loaded-module observation is saved to support the later restart check.

Before mutations, the worker verifies original ownership, inherited approval, all referenced sealed sources, original failed repair/no-mutation evidence, the successful diagnostic and all nested artifact hashes, successful isolated validation/ownership, same GPU UUID/Python, and the fixed wheelhouse/model/evaluation protocol identities. It rechecks isolated 84-package inventory and a clean full isolated dependency check.

The base is scanned in fresh processes. The saved exact duplicate entries and `sys.path` must match; all unrelated distribution versions, roots and metadata paths must remain unchanged. The original base version plan is then applied. Every selected install wheel is rechecked for SHA-256, size, receipt, METADATA and compatible tags before any mutation. Offline installation still uses `--no-index --no-deps --only-binary=:all: --require-hashes`; no resolver/source build/download/upgrade/forced reinstall is selected. A known partial approved repair resumes only remaining operations.

After the conditional changes, the complete base inventory must match the exact planned result, and the entire duplicate layout plus unrelated metadata selections must remain unchanged. The full base `pip check` must be clean with no accepted exceptions. Fresh IPython, Jedi, Parso, Gradio and Diffusers imports plus Jedi completion must pass, as must NumPy/packaging import-to-selected-metadata checks. The isolated exact inventory and full isolated dependency check must still pass after base repair. The parent notebook backend and original isolated contracts remain unchanged.

## Evidence and restart boundary

Drive root remains `/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1`. New sources live under `base_repair_recheck_releases/<source SHA-256>/`; reports, complete before/after base layout/selected metadata, original evidence links, exact plan, command journal/logs, fresh tool probe and inventories live under `base_repair_recheck_attempts/<UTC>/`.

Successful repair creates `COLAB_BASE_REPAIR_CONTRACT.json` V2. It records the legacy last-seen version map with its meaning, actual selected metadata versions, exact preserved duplicate entries/search path, original diagnostic source link, clean base/tool checks and preserved isolated checks. Earlier pending flags in original isolated contracts remain historical; this later contract supplies the newer base result. It does not certify the still-running notebook kernel or real model loading.

`BASE_REPAIRED_FRESH_VALIDATION_PASSED_RESTART_PENDING` requires an ordinary kernel/session restart retaining local runtime storage after the output is reviewed, followed by a separate both-interpreter recheck. This block never kills/restarts the kernel automatically. `BASE_ALREADY_MATCHED_FRESH_VALIDATION_PASSED_RECHECK_PENDING` means no mutation was needed; both interpreters still require the next reviewed check. Catchable failures preserve commands/state and stop with `BASE_REPAIR_FAILED_PRESERVED_REVIEW_REQUIRED`; no additional repair follows automatically. Repeated successful runs preserve the original restart requirement and refuse differing existing contracts.

Task 4 actual responses 0/1,800; judgments 0/1,800; independent blind human labels 0/240. No pretrained model load, inference, training, tuning or experimental setting change in this stage. CUDA/NF4 and 37 synthetic checks remain verified from Block 10b; this block does not claim new real-model results.

## Local checks and sources

Eight new layout guards cover the exact sixteen groups/differing versions, actual metadata selection versus last-seen versions, new/changed/removed/reordered duplicates, search-path/outside-prefix/interpreter drift, unrelated metadata drift, unique repair targets, and preservation after an approved Jedi addition. The thirteen original conditional repair guards are also rerun. A real local selected-metadata scan is checked separately; it does not establish the Colab repair result. New wrapper/guards/tests/documentation were prepared with Codex assistance.

Python distinguishes distribution metadata from import packages; installed metadata can be discoverable in multiple search-path locations. See [Python 3.13 importlib.metadata documentation](https://docs.python.org/3.13/library/importlib.metadata.html). The named lookup implementation and version API are verified against [CPython 3.13 source](https://raw.githubusercontent.com/python/cpython/3.13/Lib/importlib/metadata/__init__.py). This block records actual named lookup and fresh module evidence rather than inferring which copy is selected from a collapsed dict.
