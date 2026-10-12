# Task 4 Block 12: recheck both interpreters after the ordinary kernel restart

## Last verified result

Block 11c returned `BASE_REPAIRED_FRESH_VALIDATION_PASSED_RESTART_PENDING` from `base_repair_recheck_attempts/20261011T164024.510450Z/REPORT.json`. Only Jedi 0.20.0 needed installation. No original base restoration or global removal was needed. Base inventory increased from 723 to 724 names, all sixteen observed duplicate metadata groups stayed unchanged, and no outside-prefix paths appeared. The full base dependency check, fresh IPython/Jedi/Parso/Gradio/Diffusers imports, Jedi completion and actual NumPy 2.1.3/packaging 26.3 selected-module checks passed. The isolated 84-package inventory stayed unchanged and its full dependency check passed. No model was loaded and evaluation remained zero. These are user-returned results; this next block verifies saved Drive evidence directly.

## Restart manually before this one block

Use Colab's Runtime menu to perform an ordinary **Restart session** / kernel restart, retaining the current virtual machine and local files. Do not choose **Disconnect and delete runtime**, factory reset, or restart-and-run-all. The next launch cell alone should run after the restart. Keep the A100 runtime type unchanged.

The completed setup approval explicitly covered restarting the base kernel if needed and rechecking both interpreters before actual model validation. No new experimental choice or repeated permission is required. This block does not restart anything automatically. It requires a different notebook kernel PID from the one saved during the successful repair. If local `/content/task4_isolated_runtime_v1` or the same A100 is missing, it stops instead of reinstalling/recreating anything.

Google's [Colab FAQ](https://research.google.com/colaboratory/faq.html) identifies “Disconnect and delete runtime” as a reset of assigned managed virtual machines. This workflow needs the current local installation retained; the ordinary kernel restart is a separate operation. Menu wording can vary by interface.

## Run and scope

Download `ATML_PA2_Task4_A100_Post_Restart.zip`, run the single supplied Block 12 cell after the ordinary restart, and upload only that ZIP. The cell verifies archive/member hashes, records the current PID/interpreter, imports repaired IPython and Jedi in the actual notebook kernel, tests Jedi completion there, and records versions of relevant already-loaded modules. It does not import any pretrained model.

The worker verifies original Task 4 ownership/approval, all referenced sealed source manifests, the successful Jedi-only repair report and every nested artifact hash, the paired base repair V2/isolated/child policy contracts, original local environment ownership, exact Python, frozen evaluation/model-input contracts, and the same A100 UUID. It verifies the current notebook PID differs and the actual kernel's IPython/Jedi/completion and already-loaded package versions match selected base metadata.

Fresh base inventories, metadata entries and named selections must match the successful repair exactly, including all sixteen duplicate groups. A clean full base dependency check and fresh base tool/completion/NumPy/packaging checks are required. The isolated interpreter must retain all 84 exact approved versions, its original no-duplicate/no-outside-prefix strictness and a clean full dependency check. The original sealed GPU probe is rerun unchanged after the restart: full course imports and 83-package required chain, tiny float16 CUDA and NF4 operations, plus all 37 synthetic semantic checks. The restart and base repair justify this new validation; no actual model is constructed or loaded.

Both inventories/layouts and parent plotting setting must remain unchanged afterward. Future isolated subprocesses still use the finalized narrow plotting selector policy. No installation, removal, download, dependency exception, environment creation, data/model extraction, training or real evaluation occurs.

## Saved result and limits

Drive experiment root remains `/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1`.

- `post_restart_releases/<source SHA-256>/`: sealed scripts, tests, inherited approval, instructions and local validation.
- `post_restart_attempts/<UTC>/`: repair links, kernel evidence/restart assessment, hardware, base/isolated before/after inventories and layouts, command logs, fresh tool probe, original GPU dependency graph/probe and 37-check transcript, final report and artifact hashes.
- `GPU_ENVIRONMENT_CONTRACT.json` V2: combined fresh validated isolated course interpreter and repaired Colab base, with original contract/protocol/source identities and successful restart validation. Earlier contracts remain historical evidence with their original pending fields; this combined contract supplies the later result.

`BOTH_INTERPRETERS_VALIDATED_AFTER_RESTART_NO_INFERENCE` means the environment checks pass. It explicitly leaves actual pretrained policy/judge loading validation pending. `POST_RESTART_VALIDATION_FAILED_PRESERVED` saves evidence and stops, with no automatic repair/fallback. Return the complete output before proceeding.

Actual Task 4 progress remains responses 0/1,800; judgments 0/1,800; independent blind human labels 0/240. No new prompt, checkpoint, generation, quantization, audit or metric setting is selected. The tiny CUDA/NF4 operations and synthetic checks are infrastructure validation and do not produce assignment evaluation responses.

Six local guards cover changed/invalid kernel PID, actual repaired kernel tools/completion, interpreter identity, selected versus legacy NumPy versions, and stale removed global packages. No real Colab restart, GPU probe or package operation is performed locally. New orchestration/guards/tests/documentation were created with Codex assistance; prior sealed course helpers/probes/tests/releases remain unchanged.
