# Task 4 — Block 14: deterministic response generation on A100

Block 13 returned FIVE_MODELS_LOADED_AND_VALIDATED_NO_INFERENCE. All four policies and the fixed NF4 judge passed actual loading, placement, eval-mode, adapter and inherited-generation-setting checks. Both base and isolated environments remain unchanged. The saved result will be directly verified by this block.

This is the already-approved response-generation stage: four sequential policies, every one of the same 450 XSTest IDs in source order, batch size four, deterministic decoding and the released 256-token prompt/response caps. There are 113 batches per policy, including the final two-prompt batch, for 452 batches and 1,800 responses overall. Policy order is SFT, DPO, PPO, GRPO. SFT has no adapter; the three other policies use their frozen standard endpoints with a fresh base for each policy. No checkpoint choice, training or safety tuning occurs.

Keep the same A100 runtime connected. Download ATML_PA2_Task4_A100_Generation.zip and upload only that ZIP using the supplied single cell. Run one copy of the cell and return its complete output before continuing. No additional restart, installation or model download is needed.

The block verifies the successful saved model-loading report and all its artifacts, inherited approvals, original helper/source hashes, bound model/data/runtime contracts, same A100, both full dependency checks, strict isolated inventory and exact reviewed Colab base layout. It freshly rehashes all seven policy snapshot files and the six prepared adapter files before generation. Judge weights are not loaded or rehashed here; their immutable identity remains bound to the successful loading contract and is rechecked before judging.

Generation uses the unchanged course load_policy, load_tokenizer and batch_generate helpers through the unchanged, already-validated course_calls wrapper. It preserves float16 policy-base precision and the original PEFT adapter dtype behavior. Before each policy's first new batch, all 450 prompt encodings must match the frozen CPU checks; model placement/eval mode, active adapter tensor values and generation defaults must match the successful real-model validation. It does not select extra decoding parameters or replace inherited defaults.

A frozen evaluation contract binds source, approvals, prompts/order, settings, model inputs, successful GPU/model validation and original course helpers. All existing batch receipts across all four policies are checked before new inference. Every generation batch is saved atomically to Drive, read back and bound to its contract, input and payload hashes. EOS is included in response-token length; padding after EOS is excluded, preserving the original course behavior. Progress output reports exact saved batches/responses, including partial last batches.

Persistence locations under /content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1:

- generation_releases/<source-manifest>/ — immutable source release.
- generation_attempts/<UTC>/REPORT.json — attempt, commands, logs, verified counts and errors.
- EVALUATION_EXECUTION_CONTRACT.json and evaluation_v1/CONTRACT.json — immutable common evaluation identity.
- evaluation_v1/generation/<policy>/<batch-index>.json — atomic checksummed batch envelopes.
- evaluation_v1/exports/generation/<policy>.jsonl — complete ordered policy responses after all four finish.
- GENERATION_COMPLETION_CONTRACT.json — final 1,800-response/452-batch evidence and checksums, finalized only on complete success and unchanged environments.

Failure stops without an automatic retry, batch-size change, cap change, new precision, replacement checkpoint or cache deletion. Return the error output for review. Saved valid batches remain usable; a later run of this exact release and contract verifies all saved units and generates only missing batches. Policies with all 450 responses already saved need no model load. A batch interrupted before its atomic save remains uncompleted and can be regenerated on resume. Existing differing/corrupt evidence is preserved and blocks continuation.

This block performs generation only. Judge loading/calls, safety aggregation and blind manual annotation follow in separate stages after the actual generation output is returned. It creates no human labels or safety-comparison findings. At preparation time actual progress remains responses 0/1,800, judgments 0/1,800 and independent human labels 0/240; the five model-loading inspections are complete. No session time limit or automatic disconnect is introduced.

Seven local synthetic integration checks pass, and source/ZIP integrity is checked. The underlying 37 semantic/helper checks passed previously on the actual isolated A100. Real response generation has not been run locally; its completion must be confirmed by the returned Colab output.

Expected complete status: ALL_FOUR_POLICIES_GENERATED_AND_SAVED.
