# Task 4 Block 11: approved conditional Colab base repair

## Verified progress

Block 10b returned `ISOLATED_A100_ENVIRONMENT_VALIDATED_BASE_REPAIR_PENDING` from `isolated_recheck_attempts/20261011T155750.714353Z/REPORT.json`. The installed isolated Python 3.13.15 has all 84 approved packages; its full dependency check, fresh course imports, float16 CUDA and NF4 probes pass, and all 37 preserved synthetic checks pass with zero skips. Base package versions and the parent's notebook plotting selector were unchanged. No pretrained models were loaded and no evaluation was run. This is user-returned console evidence; Block 11 verifies the saved Drive reports/contracts and all their artifact hashes directly before acting.

## Existing explicit approval

The user replied “approved” to `BLOCK8_ISOLATED_SETUP_APPROVAL_PROPOSAL.md`. Its SHA-256 is `d98521501e80f94db741356c6c13e0b76d09824316652ae3b92d06ff65d55074`. That approval explicitly covers the exact wheel cache, isolated setup, and conditional original Colab base restoration/Jedi repair after isolated validation. The unchanged inherited `APPROVAL.json` records the whole scope; its historical `current_block_scope` describes Block 9 only. `AUTHORIZATION_CONTEXT.json` explains this stage's already-approved subset. No new experimental choice or GPU dependency exception is authorized or introduced.

The conditional base targets are:

| Package | Restore/install exactly | Allowed known course-modified version |
|---|---|---|
| huggingface-hub | 1.33.0 | 0.36.2 |
| peft | 0.21.1 | 0.17.1 |
| tokenizers | 0.23.2 | 0.22.1 |
| transformers | 5.18.0 | 4.57.1 |
| jedi | 0.20.0, only if absent | No other version accepted |

Only the recorded course-added global `trl==0.27.2` and `bitsandbytes==0.50.2` copies may be removed, and only after isolated validation. Missing copies are skipped. Versions already at the original targets are skipped. Parso 0.8.7, IPython 7.34.0, Diffusers 0.40.0, Gradio 6.29.0 and global Torch 2.11.0+cu130 are preserved. Exact original before/after inventories and the saved proposal must agree with these boundaries. All other current versions must match the immediately preceding recheck's saved base inventory. Unknown drift or an additional required change stops for review.

## Run on the same A100

Keep the current A100 runtime connected. Run the single supplied Block 11 launcher and upload only `ATML_PA2_Task4_A100_Base_Repair.zip`. Do not restart, disconnect/delete or factory-reset the runtime before returning this block's output. The launcher verifies the ZIP identity/member hashes before extracting it and streams the worker's output. It saves a small notebook observation (PID and versions of relevant modules already loaded, without importing them) so the subsequent restart check can distinguish installed files from the existing kernel's state.

The block:

1. Verifies original task ownership/approval, sealed stage sources, successful Block 10b evidence including nested artifact hashes, unchanged environment/policy contracts, local environment ownership, exact Python, same A100 UUID, wheelhouse/proposal provenance and frozen evaluation/model-input identities.
2. Rechecks isolated interpreter isolation, all 84 exact packages and its full dependency check before base changes. Reuses the successful fresh CUDA/NF4/37-check evidence; no model or new numerical test is run in this stage.
3. Takes a fresh base interpreter inventory. Requires the approved conditional package identities and no unrelated drift. Saves the exact planned installs/removals and final expected complete base inventory before any package mutation. Known partial approved repairs are resumable; completed contracts cannot be rebound or replaced.
4. Revalidates the exact SHA-256, size, receipt, METADATA and compatible tags of every wheel that will actually be installed. Uses the existing Drive wheelhouse only, with no network/resolver/source build. Checks local working space using twice the selected wheels' logical unpacked bytes as conservative operational headroom; this is not a claim about measured installed disk use.
5. Removes only validated approved global additions if present, then installs only necessary exact base versions/Jedi from hashed local file requirements with `--no-index --no-deps --only-binary=:all: --require-hashes`. Torch and the isolated environment are never installation/removal targets. Each command is saved before starting, with log and return code afterward. Catchable failures preserve partial state and stop; no automatic additional repair follows.
6. Verifies the complete fresh base inventory against the planned result, then requires a clean full base `pip check` with no accepted exceptions. Fresh processes import IPython, Jedi, Parso, Gradio and Diffusers and verify a simple Jedi string-method completion.
7. Rechecks the exact isolated inventory and full isolated dependency check after repair. Requires unchanged isolated contracts and parent plotting setting. Saves the base repair contract and final report only after success.

Base processes preserve the notebook's plotting backend. Isolated processes retain the finalized narrow child policy: strip only the recognized inherited notebook selector in addition to the original Python/pip sanitizer, with no forced replacement backend. No parent environment setting changes.

## Restart boundary and saved evidence

Drive experiment root: `/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1`.

- `base_repair_releases/<source SHA-256>/`: sealed code, tests, instructions, inherited approval and local validation.
- `base_repair_attempts/<UTC>/`: start/final reports, validated plan, notebook observation, hardware, fresh base/isolated inventories, selected wheel checks, space assessment, exact command records/logs, installed-file report if applicable, verified changes and fresh tool probe.
- `COLAB_BASE_REPAIR_CONTRACT.json`: complete validated base versions, isolated/protocol provenance, clean dependency/tool checks, and explicit pending restart/recheck/model validation fields. Prior isolated contracts remain historical evidence and retain their original pending flags; this later contract supplies the new base result.

`BASE_REPAIRED_FRESH_VALIDATION_PASSED_RESTART_PENDING` means the installed files and fresh processes validate, while the existing notebook kernel must restart before the next interpreter recheck. This block never kills the kernel automatically. Return the output first; the next reply will give the appropriate restart instruction and exactly one subsequent block. Use an ordinary kernel/session restart retaining local runtime storage, never disconnect/delete or factory-reset.

`BASE_ALREADY_MATCHED_FRESH_VALIDATION_PASSED_RECHECK_PENDING` means no package mutation was necessary; the next block still rechecks both interpreters before model validation. Repeated runs retain any restart requirement recorded by the first successful run.

`BASE_REPAIR_FAILED_PRESERVED_REVIEW_REQUIRED` means saved evidence must be reviewed. No extra installation/removal/version selection or GPU exception follows automatically. Unexpected drift is rejected before mutations; a failed mutation may leave partial approved changes, recorded in the exit inventory and command journal. Drive disconnection or forced runtime termination can interrupt reporting; command-start evidence and earlier artifacts remain available.

The combined GPU-ready contract and actual model loading remain pending. Task 4 responses: 0/1,800; judgments: 0/1,800; independent blind human labels: 0/240. This infrastructure stage makes no training, tuning, checkpoint, prompt, token-limit, metric, generation or judge change.

## Checks and attribution

Thirteen local regression checks cover exact modified/fresh/already-repaired/partial-repair branches; unknown target/unrelated drift; preservation of Colab tools/Torch/Parso; historical mutation evidence; final complete inventory; interpreter boundaries; strictly offline hashed installs; narrow removals; and importing actual preserved cache helpers. They do not establish real Colab package installation, completion, dependency or GPU results; those require the returned Block 11 output. New orchestration/guards/tests/documentation were created with Codex assistance; sealed course helpers, the evaluation protocol and prior releases remain unchanged.
