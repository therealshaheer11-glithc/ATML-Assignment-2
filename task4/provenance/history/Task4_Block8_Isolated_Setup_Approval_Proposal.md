# Task 4 Block 8: isolated setup approval proposal

The returned CPU preflight reports ISOLATION_PREFLIGHT_READY_FOR_REVIEW. It verified the saved A100 dependency graph and found published compatible binary files for the entire proposed setup. This is a concrete proposal; package downloads, installations, removals and environment creation have not occurred.

## Evidence and limits

- Saved report: `/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1/isolation_preflight_attempts/20261011T145304.669615Z/REPORT.json`.
- Saved A100 course dependency chain: 83 distributions; the isolated list adds the recorded pip version for 84 pins. IPython, Jedi, Gradio and Diffusers are outside that chain.
- 90 distinct proposed wheel files: 84 isolated packages, four original base versions, Jedi, and the existing compatible Parso version. The exact filenames, publisher URLs, hashes and versions are recorded in the saved ISOLATION_PROPOSAL.json.
- Compressed total: 2,969,739,166 bytes, approximately 2.766 GiB. Metadata-derived size coverage is complete; unresolved artifacts: zero.
- This output establishes a viable candidate, not a completed installation. Actual wheel contents, installed disk space, A100 compatibility and both full dependency checks remain to be validated. Installation time is unmeasured.
- Evidence available here is the user-returned console output. The next approved block must verify saved Drive report/artifact hashes and the precise proposal directly before downloading anything.

## Requested authorization

Approve the following exact infrastructure choices, implemented in separate Colab blocks and reviewed through returned outputs.

1. **CPU package cache.** Download only the 90 frozen binary files from the saved proposal into `/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1/environment_wheelhouse`. Preserve each approved version, URL and SHA-256; verify each complete file, save receipts atomically, and reuse only verified files on resume. Persistent Drive storage is approximately 2.766 GiB plus small receipts; temporary working space may also be needed. Inspect actual ZIP contents, METADATA, target Python/platform tags and the full pinned dependency closure; record uncompressed sizes and required local disk space before A100 installation. No package installation or model download in this CPU block.

2. **A100 isolated environment.** After successful CPU verification, create `/content/task4_isolated_runtime_v1` on local A100 runtime disk with no system site packages. Install the exact saved 84 pins from verified offline wheels. Preserve Torch 2.11.0+cu130, Transformers 4.57.1, Tokenizers 0.22.1, TRL 0.27.2, PEFT 0.17.1 and the entire verified dependency chain. The Colab notebook launcher remains in its base kernel; Task 4 scripts run with the isolated Python executable. Require a clean full dependency check, fresh course imports, CUDA float16/NF4 probes and all 37 preserved synthetic checks before treating the isolated environment as ready.

3. **Conditional Colab base repair.** Preserve Colab's tools. Restore only recorded course-modified base versions if they are still at the known modified versions: Hugging Face Hub 1.33.0, PEFT 0.21.1, Tokenizers 0.23.2 and Transformers 5.18.0. Add Jedi 0.20.0; retain the existing compatible Parso version identified in the saved proposal. Remove only the course-added global TRL and bitsandbytes copies, and only after the isolated environment passes validation and the current inventory confirms they are the recorded additions. A fresh base that already matches needs no restoration/removal. Verify a clean full base dependency check and fresh IPython/Jedi, Gradio and Diffusers imports. Restart the base kernel if needed, then recheck both interpreters before actual model validation.

This approval would cover these specified environment actions and storage, not arbitrary repairs. Unexpected package drift, identity mismatch, missing dependency, incompatible wheel, insufficient storage or an additional required package change stops the dependent stage for review. Do not select substitute versions, source builds, CPU Torch fallback, or new GPU exceptions automatically.

## Assignment requirements remain fixed

This proposal changes environment layout only. The four fixed policies and checkpoints, 450 ordered XSTest prompts, batch size 4, deterministic generation, original course judge, token caps, audit IDs, metric denominators and saved progress protocol remain unchanged. Task 4 performs evaluation only; no training, tuning, checkpoint selection or fabricated human labels.

Generation: 0/1,800 complete, 1,800 remaining. Judging: 0/1,800 complete, 1,800 remaining. Blind human labels: 0/240 complete, 240 remaining. No pretrained models have been loaded. No GPU environment contract is finalized and no GPU dependency exceptions are accepted.

## Approval boundary

The user's original instruction requires explicit approval for discretionary decisions. The latest “sure” authorized reviewing an isolated setup; Block 8's AUTHORIZATION_CONTEXT.json explicitly excludes wheel downloads, package changes and virtual environment creation. Therefore this proposal must be approved before implementation.

The approval-gated academic ML procedure also states: “Stop at the approval gate. Resume implementation only after the user explicitly approves the stated plan or a revised subset.” Source: `/Users/shaheerahmadmasood/.codex/memories/skills/approval-gated-academic-ml/SKILL.md`, Procedure 6. This agrees with the user's direct requirement; there has been no sandbox approval rejection.

After approval, the next reply supplies exactly one CPU executable block for the wheel cache and verification stage, then waits for its actual output. A100 installation and base repair are subsequent blocks.
