# Task 4 Block 7: A100 environment preparation

Block 6 returned CPU_ORCHESTRATION_VALIDATED_NO_INFERENCE from orchestration_attempts/20261011T123747.440376Z: 37/37 synthetic CPU checks pass, with all 450 prompt IDs/order and 113 generation batches verified. CPU preparation is complete. This block verifies that saved report, every linked attempt artifact, the immutable CPU protocol, model-input contracts, fixed CSV/audit identities, and prepared adapter bytes before preparing the A100 environment.

The original Task 4 approval specifies Colab, CPU preparation, and A100 inference. This is the required transition to that approved hardware and the unchanged course software requirements. No model, revision, adapter, checkpoint, decoding setting, hyperparameter, retry strategy, deadline, session limit or disconnect behavior is newly selected. Prior Task 3 settings are not inherited. Course files and all earlier releases stay unchanged. Neither Task 3 base files nor Task 4 judge files are copied or modified.

## Single-block use

Switch Colab's hardware accelerator to A100, then run the supplied single launcher cell. Mount Drive and upload only ATML_PA2_Task4_A100_Environment.zip. The launcher verifies the ZIP SHA-256, safe unique member paths, and every manifest file hash before running the saved script in a separate process. Switching the Colab runtime may reset temporary files; required inputs and previous evidence are preserved on Drive.

## Checks and installation

1. Verify the entered CPU result and all source/input/adapter contracts from Drive. Preserve this new source release and a timestamped attempt.
2. Query NVIDIA hardware before installing anything. A missing GPU or a GPU other than A100 stops; no alternate hardware is selected.
3. Record all installed distribution versions, then run the ordinary pip install command against the byte-identical course requirements.txt. No extra pin, upgrade flag, package removal, CPU Torch substitution or workaround is added. Record the command, complete output, before/after inventories and every version change. The GPU Torch build is validated as a GPU build; the historical CPU Torch version is not imposed on it.
4. Record the global dependency check. In a fresh process, traverse the entire task-required dependency graph, including active markers, Python bounds and requested extras. The GPU graph may contain CUDA distributions absent from the 64-package CPU graph; no fixed GPU package count is invented.
5. Import the course libraries and unchanged helpers offline. Verify actual A100 visibility and compute capability 8.0, a tiny float16 CUDA matrix multiplication and a 256-value NF4 quantize/dequantize fixture with finite float16 output. These are backend checks, not pretrained model loading, inference results or judge validation. No experimental prompt or response is used.
6. Rerun the preserved 37 synthetic orchestration/parser/course-helper checks under the new GPU software environment. The controlled models in that suite are test doubles; no pretrained model constructor runs. Record the actual GPU versions and compare them with the CPU inventory.

Six local standard-library checks cover the hardware gate and global-check approval boundary. CUDA/NF4 and the 37 checks in the actual A100 environment remain pending until the returned Colab output. Local syntax validation does not establish GPU success.

## Dependency approval boundary

The earlier recovery approval explicitly finalized the CPU environment with three exact outside-chain exceptions. This release does not extend that approval to GPU. The new global check is reported accurately. A matching historical CPU error is marked only as diagnostic context, and all GPU exceptions remain unaccepted.

If the required chain, fresh imports, backend checks and synthetic suite pass and the global check is clean, status is A100_ENVIRONMENT_VALIDATED_NO_INFERENCE and GPU_ENVIRONMENT_CONTRACT.json is finalized. Its stable content records exact required GPU package versions, source/CPU/model-input links, course bounds, CUDA version and checked hardware requirement. Individual device UUIDs and driver/memory inventory remain in the timestamped attempt rather than the stable root contract.

If the technical checks pass but the global check fails, status is GPU_ENVIRONMENT_REVIEW_REQUIRED. All error messages and task-chain membership evidence are preserved, no GPU exception is accepted, and no root GPU environment contract is finalized. This is a concrete result for subsequent review under the user's approval requirement. Other catchable failures produce GPU_ENVIRONMENT_FAILED_PRESERVED with the full available evidence. No automatic repair or retry occurs. Hard interruptions can leave an incomplete attempt; a later user-launched rerun creates a new attempt and preserves the old one.

## Saved evidence and remaining work

- gpu_environment_releases/<source manifest SHA-256>: this immutable release.
- gpu_environment_attempts/<UTC timestamp>: START.json, commands/results/logs, hardware and package inventories, changes, dependency graph, global-check assessment, fresh-process probe, synthetic test transcript/result and REPORT.json.
- GPU_ENVIRONMENT_CONTRACT.json: finalized only after the complete clean-check success condition above. A differing existing contract stops and is preserved.

Model weights are not rehashed or loaded by this environment-only stage. Their previous verification contracts are checked by identity; fresh weight integrity and actual policy/judge loading still need validation before execution. Actual policy defaults, unmerged frozen adapters, float16 policy placement and NF4 judge loading remain pending.

Actual experimental progress after this block remains 0/1,800 generated responses, 0/1,800 judgments and 0/240 independent human labels. Model loading/inference, manual audit export, analysis and report are later stages. Wait for this block's output before receiving another executable block. All new scripts and documentation were prepared with Codex assistance.
