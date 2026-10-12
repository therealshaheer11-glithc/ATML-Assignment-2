# Task 4 — Block 13: A100 model loading validation

Block 12 returned BOTH_INTERPRETERS_VALIDATED_AFTER_RESTART_NO_INFERENCE. Both full dependency checks, restarted notebook tools, fresh CUDA/NF4 probes and all 37 synthetic checks pass. Actual pretrained models have not yet been loaded.

This block implements the original approved Task 4 model preparation procedure. Keep the same A100 runtime connected; no additional restart is needed. Upload only ATML_PA2_Task4_A100_Model_Loading.zip using the supplied single launch cell, then return its entire output before running anything else.

The block verifies the saved Block 12 report and every saved artifact, the finalized combined GPU environment contract, inherited approvals, exact base layout, isolated 84-package inventory, same GPU identity and both complete dependency checks. It freshly checks all seven base and nine judge file hashes plus the six prepared adapter files. Files are read in place; no weights are downloaded or copied.

It loads SFT, DPO, PPO, GRPO, then the judge in separate isolated child processes. The four policies use the original course load_policy/load_tokenizer helpers, trainable=False and a fresh base for each policy; SFT has no adapter. The original load_judge helper supplies NF4 4-bit quantization with float16 compute and device_map=auto. Loader arguments, precision behavior, generation defaults and saved adapter configs are unchanged.

For each model it verifies CUDA parameter placement, evaluation mode, tokenizer settings and loaded generation defaults against the frozen CPU preview. Policies retain float16 base parameters; actual inherited adapter dtypes are recorded without changing them. All 112 loaded tensors per adapter must equal their saved checkpoint values after the inherited dtype conversion. The judge must contain initialized CUDA NF4 modules with float16 compute. Peak GPU allocation/reservation is measured and saved as runtime observations, without asserting an expected amount.

PEFT 0.17.1's existing default casts half/bfloat16 adapter parameters to float32; this is preserved rather than overridden. Source inspected: https://raw.githubusercontent.com/huggingface/peft/v0.17.1/src/peft/peft_model.py (constructor and autocast_adapter_dtype).

Each successful model inspection is saved immediately, then its process exits to release model memory. Failure preserves the model report/log and completed-role receipts and stops; no fallback model, dtype, quantization or automatic retry is selected. A rerun creates a new validation attempt and reloads all five models; it does not treat a previous load as a currently loaded model. No forward pass, generation, judgment, optimizer, training or human label is performed. This stage validates construction/loading only; actual inference remains pending.

After all five inspections and unchanged environment checks pass, the block finalizes MODEL_LOADING_VALIDATION_CONTRACT.json. Existing historical contracts remain unchanged, including their original pending flags; the newer validation contract records completed loading. Stable inspection signatures exclude measured peak memory so repeating the same validation does not rewrite the root contract with runtime-dependent measurements.

Saved attempts: /content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1/model_loading_attempts/<UTC>/REPORT.json

Expected complete status: FIVE_MODELS_LOADED_AND_VALIDATED_NO_INFERENCE.

Current actual counts: responses 0/1,800; judgments 0/1,800; blind human labels 0/240. Model inspections remain 0/5 until the Colab output confirms them. Eight local synthetic integrity/scope checks pass; real model loading was not performed locally. No new experimental setting, package change or GPU dependency exception is introduced.
