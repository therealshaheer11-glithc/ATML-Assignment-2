# Task 4 Block 10: approved isolated A100 environment

## Entered state and approval

The user approved the exact setup in APPROVED_PROPOSAL.md with “approved”. That scope covers the Drive wheel cache, fully isolated A100 environment and conditional Colab base repair. APPROVAL.json reproduces the inherited approval unchanged; its current_block_scope describes the earlier CPU download block. AUTHORIZATION_CONTEXT.json explains that this release implements the approved second stage only. No renewed approval, package-version choice or experiment setting is introduced.

Block 9 returned CPU_WHEEL_CACHE_VERIFIED_NO_INSTALLATION from wheel_cache_attempts/20261011T151647.989109Z. All 90 wheels were downloaded, checksum-verified and metadata-inspected. Course closure: 83 distributions; isolated closure: 84 including pip; Jedi/Parso closure passes. Parso 0.8.7 was observed. The exact 84 isolated versions remain frozen in APPROVED_ISOLATED_VERSIONS.json.

The measured isolated wheel ZIP contents total 5,522,644,346 logical bytes (about 5.143 GiB). The largest isolated wheel contains 1,136,912,993 unpacked bytes. These are archive file sizes, not measured installed disk use. Actual Colab base repair, isolated installation, model loading and evaluation have not occurred at entry. Responses 0/1,800; judgments 0/1,800; blind human labels 0/240.

## Run on A100

Select an A100 runtime in Colab, then run the single supplied launcher and upload ATML_PA2_Task4_A100_Isolated_Environment.zip. Changing hardware may reset local files/packages; the source releases, model references, wheel cache and prior evidence used here are persistent on Drive. The local isolated environment will be recreated from verified files as needed. Do not copy a virtual environment between runtimes.

The launcher stays in Colab's base Python, mounts Drive, validates this ZIP and its source manifest, and streams the installation worker. Task 4's installed packages go only into `/content/task4_isolated_runtime_v1`. Notebook dependencies remain in Colab's base interpreter.

## Operations in this block

1. Verify original Task 4 ownership, exact inherited approval/proposal/version identities and every referenced source release. Query NVIDIA hardware and require A100. Verify the exact returned Block 9 report, all its saved artifact hashes, the wheelhouse contracts and frozen model-input/protocol links.
2. Match the fresh A100 Python implementation/full version, Linux and architecture to the reviewed target. Recheck all 90 cached wheels and receipts against their approved SHA-256 and byte sizes. Inspect actual wheel metadata and fresh A100 compatibility tags and require them to match the frozen CPU inspections. No package download or source build occurs.
3. Check local free space before mutation. The operational guard reserves twice the measured isolated ZIP logical bytes: 11,045,288,692 bytes, about 10.287 GiB, allowing headroom for installed files, compilation and installer temporary extraction. This is a conservative guard, not a prediction of installed disk use. Insufficient space stops the run; no cleanup is automatic.
4. Create the approved local virtual environment without system site packages or bundled pip. Bootstrap only the approved pip 24.1.2 from its verified wheel, then install the exact 84-package set from local file requirements with SHA-256 hashes, binary-only artifacts, no index, no dependency resolution and no cache. Python/pip environment install targets are stripped and pip configuration files disabled. Driver/OS settings are retained. No base package is installed, removed, upgraded or downgraded.
5. Check fresh interpreter isolation, disabled user/system site packages, distribution locations, duplicate distributions and exact package inventory. Require a clean full isolated `pip check`.
6. Reuse the unchanged Block 7 A100 probe in the isolated interpreter: fresh course/model-library imports, complete 83-package course dependency chain, tiny float16 CUDA/NF4 backend checks and all preserved 37 synthetic tests. No pretrained model is constructed or loaded; the synthetic fixtures are not actual XSTest responses or judgments.
7. Verify the base package inventory is unchanged. Save the isolated-environment contract only after all checks pass. Colab base restoration/Jedi repair and clean base validation remain for the next block under the same approval.

## Bootstrap and offline behavior

Using `venv` without pip avoids depending on Colab having ensurepip bundled. The already checksum-verified pip wheel is temporarily the only added Python import path for its bootstrap process. Safe-path/user-site flags exclude the current directory and user packages; bootstrap uses ignore-installed so the installer wheel is actually installed into the new environment. Normal installs and validation run the isolated Python explicitly with Python isolated mode. The installer operates solely on approved local wheel URIs and hashes; it cannot select a newer dependency from an index.

Bootstrap and normal installation logs/reports, exact requirement files and command arguments are saved. The worker does not expose arbitrary inherited environment variables or credentials in logs. Fresh model-library checks use Hugging Face offline settings.

## Resume and saved evidence

Drive root: `/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1`.

- `isolated_gpu_releases/<source manifest SHA-256>/`: source, tests, docs and inherited approval/proposal.
- `isolated_gpu_attempts/<UTC attempt>/`: START/REPORT, hardware, fresh wheel checks, disk headroom, base and isolated inventories, exact offline requirement files, installation commands/logs/JSON reports, full dependency check and the unchanged fresh probe's graph/results/37-test transcript.
- `ISOLATED_GPU_ENVIRONMENT_CONTRACT.json`: exact interpreter/version/approval/wheel/protocol identities and isolated validation result. It explicitly records Colab base repair/validation and actual model loading as pending.
- Local `.task4_isolation_owner.json`: binds the disposable local environment to the exact source, approval and wheelhouse contract.

The existing wheel cache is read only. An existing local environment is reused only if its ownership binding matches and all existing distributions are a subset of the exact approved versions. Unknown versions, packages or an unowned directory stop the run. There is no deletion/recreation of an unexpected environment, no force reinstall of a corrupt environment and no automatic alternate package selection. Interrupted installs preserve logs and the owned local environment; the same block can resume a matching partial installation after reviewing the error. After a runtime reset, the local environment must be recreated and revalidated even though its prior Drive contract remains.

REPORT.json hashes include relative paths inside the fresh_probe subdirectory. Later stages must safely validate those nested artifact paths instead of assuming that all report files are top-level.

## Success meaning

ISOLATED_A100_ENVIRONMENT_VALIDATED_BASE_REPAIR_PENDING means all 90 saved wheels and fresh tags pass; the isolated interpreter has exactly the approved 84 distributions; full isolated dependencies, the 83-package course chain, fresh imports, CUDA/NF4 and 37/37 synthetic tests pass; base package versions are unchanged. It does not certify Colab's base dependency state or actual model inference.

ISOLATED_A100_PREPARATION_FAILED_PRESERVED records a catchable failure and available artifacts. No CPU fallback, GPU exception acceptance, alternate version, base repair, model load or evaluation follows automatically.

Keep this A100 runtime connected after completion so the next reviewed block can repair/validate Colab's base and reuse the isolated interpreter. Return the entire output before proceeding. No new training, tuning, policy selection, generation settings, audit labels or experimental results are introduced.

## Local validation and attribution

Thirteen local tests validate exact/subset inventory guards, user/system site and distribution-path isolation, duplicate packages, environment sanitization, offline hashed commands, exact local file requirements, path/version rejection and disk headroom. Two tests create and remove empty temporary local fixture venvs without pip/packages: one exercises the real interpreter inventory, and the other verifies the bootstrap flags can import a mocked pip module from a local ZIP without running an installer. Local tests do not install the approved package set, download wheels, use CUDA, load models or validate A100. LOCAL_VALIDATION.json and LOCAL_TEST_OUTPUT.txt record the actual local result.

The original course helpers, original dependency inspector, Block 7 GPU probe and preserved 37-test suite remain unchanged. New installation orchestration, guards, tests, docs and packaging were created with Codex assistance under the user's recorded approval.

## Primary technical references

- [Python 3.13 venv](https://docs.python.org/3.13/library/venv.html): isolated environments and explicit interpreter invocation; with_pip=False avoids ensurepip.
- [pip configuration](https://pip.pypa.io/en/stable/topics/configuration/): PIP_CONFIG_FILE set to os.devnull disables configuration-file loading.
- [pip secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/): local SHA-256 hash checking and binary-only requirements.
