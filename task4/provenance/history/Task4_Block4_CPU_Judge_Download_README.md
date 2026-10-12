# Task 4 Block 4: approved CPU judge download

## Approved scope

The user replied “approve” to the documented model-inventory result and judge-download proposal. Reuse the existing pinned base snapshot by read-only reference and prepare Qwen/Qwen2.5-3B-Instruct at commit aa8e72537993ba99e69dfaafa59ed015b17504d1. The approved snapshot has nine files totaling 6,183,451,098 bytes. Remain in the same CPU Colab runtime.

The package includes the exact reviewed proposal, explicit approval record, and user-returned inventory observation. Prior Task 4 source releases and Task 3 files remain unchanged. Task 4 training is neither needed nor performed.

## What the block does

Verify this release and the original preflight, environment-recovery and model-inventory source releases. Check finalized CPU/audit/input/approval identities, unchanged Python and all 64 required package versions, and the successful saved inventory with artifact hashes. The CPU environment retains the three previously approved outside-chain global dependency failures; this block does not claim the whole Colab environment passes a global check.

Read public judge metadata at the approved commit with authentication disabled, compare it exactly with the saved proposal, and enforce the nine-file and byte totals. Save a revision contract and approval before transfer. Download only these files to /content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1/models/judge using the installed Hub library and its local-directory bookkeeping. No model is constructed, and no experimental settings change.

Verify LFS files with the published SHA-256 and ordinary Git files with the Git blob hash, which includes the Git blob header. Check the size and save a local SHA-256 receipt for every complete file. Check the completed file set and weight-index shard references. The base reference points to the verified Task 3 snapshot and preserves its original inventory; its weight bytes are not copied or rehashed by this block. Model loading and inference validation remain pending.

## Persistence and resume

- judge_download_releases/<source-manifest SHA-256>: sealed source, approval, proposal, observation and local validation.
- judge_download_attempts/<UTC timestamp>: entered state, pinned metadata, download-start records, completed-file checkpoints and final report, including catchable failures.
- judge_file_receipts/<judge revision>: immutable per-file identity receipts.
- Root JUDGE_DOWNLOAD_APPROVAL.json, JUDGE_REVISION_CONTRACT.json, BASE_MODEL_REFERENCE.json, and, only after full verification, JUDGE_SNAPSHOT_CONTRACT.json.

Reruns check every existing complete file before making any new download call. Complete verified files are reused. Incomplete transfer files remain in the Hub-managed local cache for the library's normal continuation behavior; continuation of a particular interrupted transfer is not guaranteed. No force download, deletion, alternate revision or model fallback is requested. Differing final files, receipts or contracts stop and preserve evidence. Run one Colab session at a time against this experiment root.

The expected success status is JUDGE_FILES_VERIFIED_NO_INFERENCE. After returning the output, the next preparation stage will be reviewed separately. Generation remains 0/1,800, judging 0/1,800, and human labels 0/240. Stay on CPU after this block.

## Validation and sources

Nine synthetic tests exercise Git blob hashing, LFS and size rejection, checking all complete files before new transfers, interruption and complete reruns, receipt mismatch/missing files, unexpected files and unsafe paths, shard-index coverage, proposal identity and explicit approval scope. These fixtures contain no actual model weights and perform no network requests. Actual Colab execution has not occurred when this source release is prepared.

API and continuation behavior were checked against the installed-version source: https://raw.githubusercontent.com/huggingface/huggingface_hub/v0.36.2/src/huggingface_hub/file_download.py and https://raw.githubusercontent.com/huggingface/huggingface_hub/v0.36.2/src/huggingface_hub/_local_folder.py . Public metadata API: https://huggingface.co/docs/huggingface_hub/v0.36.0/en/package_reference/hf_api . New preparation orchestration was created with Codex assistance.
