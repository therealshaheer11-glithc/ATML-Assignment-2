# Approved reward-model configuration compatibility correction

User approval: "okay i approve, make sure to document it, and proceed" (2026-10-04).

Pinned reward: yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback @ f98759a4a1ccdc47a3136748b6bfea9a32ba8fcc. Its original config SHA256 is d04c63bcaf27f49f7fea8c3c1287ba4078056f3b3ed35d2635c6caef7dbad9fb. Saved with Transformers 5.12.1, it declares default RoPE theta=1000000 in the nested rope_parameters field. Pinned Transformers 4.57.1 instead uses a top-level rope_theta, defaulting to 10000. The user-confirmed Drive audit found different rotary frequency vectors. The policy configuration was interpreted correctly.

The approved correction copies the checkpoint-declared theta into the supported top-level field in memory before loading the reward model. Only that configuration field changes. The original downloaded files, weights, model revision and manifest remain untouched. Unknown/conflicting RoPE structures stop for review; there is no general format-conversion fallback.

Preserved protocol: course 8-bit frozen reward model with BitsAndBytesConfig(load_in_8bit=True), device_map="auto", low_cpu_mem_usage=True, and the verified canonical Qwen policy tokenizer. The explicit configuration already has num_labels=1, so that setting is not passed again as an unsupported model-constructor keyword. No new dtype override is added to the quantized course loading branch. All three 4096-token limits, generation ceiling/decoding, optimizer, DPO objective, seed, data and budgets stay unchanged.

All Task 1 conditions must use task1_dpo.reward_loading.load_pinned_reward. It verifies pinned snapshot hashes, applies and records the correction, rejects incomplete weight loading or CPU offload, freezes the model, and checks actual loaded rotary constants. The original common/models.py and scoring helpers stay unchanged. Retain raw/effective configurations and conversion evidence in each evaluator run record. Apply consistent treatment if this course reward checkpoint is later used in other tasks.

CPU validation covers exact configuration differences and rotary constants, rejected unsupported/conflicting settings, immutable fixture files, source-hash guards and stopping before weight loading on CPU. The real pinned Drive configuration is checked separately. Synthetic fixtures are not assignment results. Actual 8-bit GPU loading, scoring and memory remain pending; official training/evaluation have not started.

Confirming evidence: reward_config_audit_20261004_111657_549569 in the original setup_evidence directory. Source: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2/configuration_qwen2.py
