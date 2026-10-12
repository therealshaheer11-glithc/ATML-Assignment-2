# Task 4 Block 6: CPU orchestration validation

## Entered state and scope

Block 5 returned CPU_ADAPTERS_TOKENIZERS_READY from model_inputs_attempts/20261011T121238.845030Z. All three frozen standard adapters pass their 112-tensor header checks, both saved tokenizers pass, and all 450 prompts are checked in 113 fixed batches. Zero prompts exceed the unchanged 256-token cap. SFT has no adapter. This package checks the saved Drive report, artifact hashes and CPU_MODEL_INPUT_CONTRACT.json directly.

The original Task 4 approval covers reproducible scripts with preserved helpers, batch four/sequential policies, checksummed saving/resume, fixed rates and the documented manual audit. This stage implements those approved orchestration and analysis semantics. It selects no new model, checkpoint, decoding setting, hyperparameter, retry policy, confidence threshold, session deadline or disconnect behavior.

## Implemented components

- core.py: contract-bound orchestration, atomic per-generation-batch and per-judgment JSON records, corruption/identity checks, resume, aggregation and manual agreement.
- course_calls.py: thin calls to the unchanged released batch_generate and judge_one. Policy prompts remain one user message. Policy calls preserve the 256/256 prompt/response caps and deterministic decoding. Judge calls preserve the exact course prompt/parser, 64-token cap and EOS/pad convention. The original decoded judge text is recorded without changing its content or parsing.
- test_core.py: 20 tests of persistence, identity, interruption, coverage, EOS lengths, denominators, confidence independence, population SD and manual agreement.
- test_parser.py: eight tests execute the original pure parse_json body extracted directly from the preserved source AST. The parser body is never rewritten. This makes pure-parser testing possible locally without model libraries; the full source is imported in the Colab course checks.
- test_course.py: nine Colab CPU tests import real Torch/Transformers and the preserved course modules. Model loading is mocked, and controlled tensor outputs test actual helper behavior. The tests inspect the policy loader's four fresh-base calls, absent SFT adapter, frozen PEFT adapter flags, NF4/float16-compute judge configuration, fixed generate kwargs and first-EOS accounting. No pretrained model, real quantization or actual GPU execution is performed.
- validate_cpu.py: verifies entered state and source/environment/input/model/adapter identities, runs all 37 checks, and freezes the tested CPU protocol contract.

The local machine lacks Torch and Transformers. Therefore 28 standard-library/core/parser checks can run locally; the nine full course-library checks are syntax checked locally and remain pending until this CPU Colab block runs. This limitation is explicit in LOCAL_VALIDATION.json. No local 37/37 result is claimed.

## Preserved statistical rules

Safe-answer and over-refusal rates divide by all 250 SAFE prompts. Unsafe-compliance and justified-refusal rates divide by all 200 UNSAFE prompts. AMBIGUOUS remains in every relevant denominator. All five labels are shown in overall, class, type and class/type distributions, including cross-class label assignments made by the judge. Confidence is retained for audit and never filters or weights any metric. Response length includes the first generated EOS and excludes later padding; dispersion uses population SD.

Manual agreement requires the independent human labels for every audit ID supplied for that policy and includes ambiguous pairs. Its confusion matrix has five human-label rows and five AI-label columns. Missing human labels remain explicitly pending; synthetic fixture labels are never written as human annotations for the actual experiment. This stage does not export or assign actual manual labels.

## Saved units and resume

A generation unit stores one fixed batch, preserving prompt IDs/order, metadata, response text, token IDs through EOS, response length and termination flags. A judgment unit stores one response's parsed categorical result, confidence/tag and original decoded judge text. Each atomic JSON envelope contains the frozen evaluation-contract hash, input hash and payload hash. Judgment input hashes bind the exact saved generation record. An existing unit is reused only after these identities and its schema pass.

Every existing unit for a policy is checked before any new callback is invoked. A mismatch stops and preserves the saved evidence. Matching completed units are reused; unfinished units require a later user-launched rerun. No automatic model-call retry is added. A hard interruption after a model call but before publishing its artifact can leave that unit unsaved; counts describe verified saved results, and a later resume may repeat that unsaved call. Unpublished .pending-* temporary files are retained, ignored for completion counts, and never treated as valid results. Concurrent writers to the same experiment are unsupported.

Synthetic test caches live in temporary test directories, separate from actual Task 4 results. This block does not create or populate a real evaluation result store.

## Drive artifacts and success meaning

- orchestration_releases/<source manifest SHA-256>: sealed implementation, tests, authorization, entered-state observation and documentation.
- orchestration_attempts/<UTC timestamp>: START.json, CPU_CHECKS.txt, CPU_CHECKS.json and REPORT.json, including catchable failures.
- CPU_ORCHESTRATION_VALIDATION_CONTRACT.json: tested source/helper identities, fixed 450-prompt protocol, 113 batches per policy and planned workload. The contract is deterministic across matching reruns; each validation attempt saves its own timestamped checks.

Expected success: CPU_ORCHESTRATION_VALIDATED_NO_INFERENCE, with 37/37 synthetic semantic checks passing. It confirms CPU semantics and source/input preparation. GPU environment validation, actual model loading, matching loaded generation defaults, precision/quantization and A100 inference validation remain pending. The GPU execution harness must use this tested core and unchanged helpers and will be supplied in a later block.

Stay on CPU until that next block is reviewed. Actual experimental progress remains 0/1,800 generated responses, 0/1,800 judgments and 0/240 human labels. The original 60 audit IDs remain frozen. New orchestration and tests were created with Codex assistance; preserved course helper hashes are saved in the validation contract. No prior Task 3 limits or disconnect settings are inherited.
