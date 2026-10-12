# Task 4 Block 10b: isolated A100 validation recheck

## Actual result and cause

Block 10 returned ISOLATED_A100_PREPARATION_FAILED_PRESERVED from isolated_gpu_attempts/20261011T153611.551337Z. All 90 cached package files passed fresh A100 checks. The isolated interpreter was created with Python 3.13.15, pip 24.1.2 was bootstrapped, all 84 exact packages were installed, interpreter isolation/inventory guards passed, and the full isolated dependency check reported “No broken requirements found.” The install completed; the fresh validation probe failed at Matplotlib import, before CUDA/NF4 checks and before the 37 synthetic tests.

Colab supplied `MPLBACKEND=module://matplotlib_inline.backend_inline`. Our child-process wrapper removed inherited Python/pip settings but mistakenly retained that notebook-only plotting selector. Matplotlib 3.10.0 read it during import; its named notebook backend is outside the isolated approved package set. This is our environment inheritance bug. The traceback does not establish a package version conflict, CUDA failure or failed wheel installation.

The isolated environment contract was not finalized. The base package equality check was not reached; source code performed no base package installation/removal, and this recheck will directly compare its actual inventory to the installation attempt's saved baseline. GPU exceptions remain unapproved. Actual model loading, responses, judgments and human labels are all zero.

## Scope under existing approval

The user already approved fully isolating the course environment, keeping its exact 84 versions and validating it on A100. This block corrects the implementation's inheritance omission and reruns that same approved validation on the completed installation. It chooses no additional package/version, no fixed Matplotlib backend, no ML setting and no broader base repair. APPROVAL.json is the inherited approval unchanged; AUTHORIZATION_CONTEXT.json documents this narrow correction.

Only a copy of the child environment loses the recognized notebook-only MPLBACKEND. Matplotlib then uses its existing configuration/default selection. The parent's notebook setting remains unchanged. An absent/empty selector is also allowed; any other nonempty selector stops for review instead of being silently replaced. The original Python/pip sanitizer is reused unchanged. This process rule is saved as an isolated child environment contract so later Task 4 subprocesses preserve the corrected isolation boundary.

## Run one block on the same A100 runtime

Keep the current A100 runtime connected. Download ATML_PA2_Task4_A100_Isolated_Recheck.zip, run the single supplied launch cell and upload only that file. The launcher validates the ZIP/source manifest and streams output. If the local installed environment is missing after a reset, this block stops; it does not reinstall or recreate anything.

The recheck:

1. Verifies original ownership/approval, all referenced sealed sources, the exact failed report and every nested saved artifact hash. Requires successful prior installation/inventory/pip-check command records and the precise observed Matplotlib traceback.
2. Verifies the owned local environment, completed wheelhouse contract, saved successful CPU wheel-cache evidence and unchanged frozen protocol/model-input links. Requires the same A100 hardware/runtime and exact installed Python identity.
3. Requires the base inventory to still match the installation attempt's saved baseline. Builds a separate child environment with only the recognized MPLBACKEND removed beyond the existing sanitation rules. Logs that specific parent/child setting without dumping arbitrary environment variables.
4. Uses the existing isolated interpreter to verify all 84 installed versions, user/system site isolation and distribution paths. Runs a clean full isolated dependency check.
5. Reuses the original sealed A100 probe and preserved 37-test suite unchanged: complete 83-package course chain, fresh library/course imports, tiny float16 CUDA/NF4 checks and all 37 synthetic checks. No pretrained model is loaded or constructed.
6. Rechecks isolated/base inventories and the parent's plotting selector, then finalizes the isolated environment and child environment policy contracts only after full success. The already-approved Colab base repair remains a later block.

No package install, uninstall, download, forced reinstall, environment creation, cache replacement, model load, training or actual evaluation occurs. Previously sealed sources, reports, cached wheels and the installation ownership file remain unchanged.

## Saved evidence and status

Drive root: `/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1`.

- `isolated_recheck_releases/<source manifest SHA-256>/`: sealed source, tests, documentation and inherited approval.
- `isolated_recheck_attempts/<UTC attempt>/`: initial/final report, original failure linkage, observed backend inheritance, child process policy, hardware, base/isolated inventories, exact commands/logs and original GPU probe graph/transcript/results.
- `ISOLATED_CHILD_ENVIRONMENT_CONTRACT.json`: narrow inherited plotting-selector rule for future isolated Task 4 subprocesses; no replacement backend is forced.
- `ISOLATED_GPU_ENVIRONMENT_CONTRACT.json`: isolated validation result, exact versions/interpreter/wheel/protocol/source identities, original installation provenance and corrected child environment policy. It explicitly leaves Colab base validation and actual model loading pending.

ISOLATED_A100_ENVIRONMENT_VALIDATED_BASE_REPAIR_PENDING means all 84 exact installed versions, the clean full isolated check, fresh imports, CUDA/NF4 and 37/37 synthetic checks pass. It also verifies that base versions and the parent's notebook setting remain unchanged. This does not certify Colab's base dependencies or actual model inference.

ISOLATED_A100_RECHECK_FAILED_PRESERVED saves a catchable failure and available evidence; no additional workaround follows automatically. Return the entire output for review. Keep this A100 runtime connected for the next approved base-repair block.

Task 4 progress remains responses 0/1,800; judgments 0/1,800; blind human labels 0/240. No training/tuning or experimental setting change is made.

## Local checks and attribution

Six local regression tests cover the exact inherited backend, preservation of the parent's environment and driver settings, no forced replacement backend, unexpected-selector rejection, preservation of the original Python/pip sanitation, safe nested artifact paths, and acceptance only of the observed saved failure. These are synthetic tests; actual isolated Matplotlib/CUDA/NF4 and the 37 checks remain pending Colab execution. New wrapper, tests and documentation were created with Codex assistance; course helpers, original probe, dependencies and saved releases remain unchanged.

## Primary references

- [Matplotlib 3.10.0 environment variables](https://matplotlib.org/3.10.0/install/environment_variables_faq.html): MPLBACKEND is an optional plotting backend selector.
- [Matplotlib 3.10.0 backend selection](https://matplotlib.org/3.10.0/users/explain/figure/backends.html): MPLBACKEND overrides rc configuration; when unset, existing configuration and automatic usable-backend selection apply.
