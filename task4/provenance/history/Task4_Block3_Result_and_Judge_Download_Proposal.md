# Task 4: model inventory result and next approval

## Verified returned state

The user returned the Colab output for model_inventory_attempts/20261011T113929.803036Z/REPORT.json. Its status is BASE_FILES_VERIFIED_JUDGE_METADATA_READY. This record is based on the returned console output and the locally reviewed inventory source; the saved Drive JSON files have not been independently opened from this computer.

The inventory checked all seven files of Qwen/Qwen2.5-1.5B-Instruct at the previously frozen revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306. Their combined size is 3,098,955,668 bytes. The original Task 3 model manifest and preparation receipt supplied the expected hashes. This verifies saved file integrity; loading and inference are still pending.

Public metadata for the assignment-required judge Qwen/Qwen2.5-3B-Instruct identified commit aa8e72537993ba99e69dfaafa59ed015b17504d1. The proposed standard Transformers snapshot contains nine files totaling 6,183,451,098 bytes (6.183 GB decimal). No snapshot candidates were found in the specific local Hugging Face cache checked by the inventory. This does not claim that every location on Drive was searched.

No judge revision has been selected for the experiment, and no judge files have been downloaded. No models have been loaded, and no Task 4 generation, judging, or training has occurred. Completed: generation 0/1,800; judging 0/1,800; blind human labels 0/240. The fixed 60 audit IDs are already saved.

## Fixed assignment requirements

Use the untouched SFT policy and the three standard final policy checkpoints. Keep the required Qwen/Qwen2.5-3B-Instruct judge, original course judge loader, prompt, parser, and deterministic settings. Preserve the approved evaluation and audit settings. Task 4 involves evaluation only. A successful download will not count as model-loading or inference validation.

## Concrete proposal requiring explicit approval

1. Reuse the verified base snapshot at /content/drive/MyDrive/ATML-Assignment-2/task3/experiment_v1/models/policy as a read-only input. Avoid duplicating its 3.099 GB in Task 4; retain the pinned model identity and checksums in Task 4 provenance.
2. Freeze the required judge at commit aa8e72537993ba99e69dfaafa59ed015b17504d1. This is the exact commit observed by the completed inventory, rather than a branch that can move.
3. Prepare one CPU Colab block to download only the nine proposed snapshot files, totaling 6,183,451,098 bytes, into /content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1/models/judge. Save the explicit approval, pinned metadata, source release, per-file integrity receipts, and attempt report under the Task 4 experiment root.
4. Before downloading, verify the saved inventory report and its artifact hashes, fixed model ID and revision, nine-file list and byte total, and existing source/environment/input contracts. Reuse already complete files only after checking their identities. Check file sizes and Hugging Face metadata identities: SHA-256 for LFS files and Git blob hashes for ordinary files. Record local SHA-256 for every completed file. Stop and preserve evidence if any identity or size differs; do not substitute another model or revision.
5. Remain on CPU. Downloading does not construct a model or run inference. Review the returned result before the next executable block. Later CPU orchestration, tokenizer/adapter preparation, and the GPU readiness check remain separate stages; an A100 is not needed for this download.

The operational impact is approximately 6.183 GB of new persistent model storage plus small bookkeeping and receipts, with network transfer time dependent on Colab/Drive. No speed or completion-time claim has been measured. Preparation does not change experimental hyperparameters, prompts, labels, or prior training artifacts.

## Approval boundary

The user's instruction requires explicit approval when a decision needs to be made. The previous approval covered the documented CPU environment recovery. Block 3 explicitly withheld judge revision selection and model-download authorization. Therefore this proposal is documented but neither the revision selection nor the download has been executed.

The approval-gated academic ML skill reinforces this boundary: “Stop at the approval gate. Resume implementation only after the user explicitly approves the stated plan or a revised subset.” Source: /Users/shaheerahmadmasood/.codex/memories/skills/approval-gated-academic-ml/SKILL.md, line 28.

After explicit approval of this proposal, the next reply will provide exactly one executable CPU preparation block and wait for its output.
