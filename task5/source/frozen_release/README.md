# ATML PA2 Task 5 evaluation

This release evaluates the unchanged supplied SFT, RLVR and RLAIF policies. It contains no training code, raw course data, pretrained weights or experimental results. All paths point to existing Google Drive assets.

The assignment-fixed settings and approved choices are in `protocol.json`; the user's approval and exact 84-package installation plan are preserved separately. `bindings.json` pins all data, adapter and model snapshot identities. `MANIFEST.json` pins every packaged source file.

## Execution stages

Run `python -B -m task5_run.preflight` on CPU. It verifies the previous exact-scoring checkpoint, all fixed inputs, 16 model snapshot files, 90 cached wheels, the exact dependency closure and the complete CPU synthetic test suite. It saves `CPU_READINESS_CONTRACT.json` and a separate attempt report on Drive. No packages are installed or changed, and no pretrained models are loaded. Synthetic tests use CPU tensors and model/tokenizer doubles; they do not constitute A100 or pretrained inference validation.

Next, run the already approved isolated A100 environment setup. Fresh A100 platform and driver checks, exact offline installation, pinned tokenizer/context checks and synthetic CUDA/NF4 validation remain necessary on that runtime. Preserve the Colab base packages.

Actual pretrained execution requires `EXECUTION_APPROVAL.json` and `ISOLATED_ENVIRONMENT_CONTRACT.json` for this Task 5 protocol. The release intentionally refuses execution until these are recorded. The latter must bind the CPU readiness contract, exact package inventory and successful synthetic CUDA/NF4 checks.

From the release directory, the isolated interpreter then runs:

- `python -B -m task5_run.run generate` for all 1,200 matched policy answers.
- `python -B -m task5_run.run judge` for 800 policy comparisons and 80 diagnostic contrasts.
- `python -B -m task5_run.run aggregate` for machine-readable quantitative results.

Each phase preserves source row order. Generation resets seed 6304 for each policy/domain, saves each answer and RNG-after state in the same immutable checkpoint, and restores that state after model loading when resuming. Incomplete temporary files are retained. Resume checks the frozen source, input, hardware, settings and committed checkpoint chain. Greedy judgments retain raw output, canonical model identity, deterministic course A/B orientation and unparseable-to-TIE behavior. Identical comparisons may use the fixed course cache; comparison events and actual model calls are counted separately.

The quantitative aggregate leaves qualitative selection and failure classification pending approval. It reports reasoning and outcome sensitivity separately and keeps filler and distractor contrasts separate. The clean-reference aggregate overlaps the four category rows; do not sum them into 160 comparisons.

## Course source attribution and deliberate errors

`common/`, `task5_feedback/`, `configs/` and `requirements.txt` are copied unchanged from the course starter at commit `1d64ac65acd5e45d1e4e1f415edc80455e21274e`, https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining. The original supplied diagnostic arithmetic/final-answer errors are intentional test interventions and must remain unchanged. The assignment's explicitly announced objective defects concern Tasks 1–3.

The approved evaluation preserves the judge helper's effective four-token limit despite `feedback.yaml` specifying 64. The course's exact verifier and judge decision logic are unchanged. Full policy prompts are checked before passing their actual width to the original generation helper, preventing implicit truncation. Checkpoint generation defaults, including inherited settings, are retained and logged.

`vendor/wheels.py` and `vendor/dependencies.py` are unchanged integrity/dependency inspectors reused from the earlier Task 4 preparation. Only their read-only inspection functions are called. Their historical source hashes are recorded in `bindings.json`; their prior Task 4 approval is not used as Task 5 authorization.
