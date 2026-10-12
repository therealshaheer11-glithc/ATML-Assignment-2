# Task 4 Block 5: CPU frozen adapters and tokenizers

## Entered state and authorization

Block 4 returned JUDGE_FILES_VERIFIED_NO_INFERENCE from judge_download_attempts/20261011T115834.218702Z. All nine judge files were verified, totaling 6,183,451,098 bytes, and the approved fixed snapshot contract was saved. The base remains a read-only reference to the previously verified Task 3 snapshot. This entered state is user-returned Colab console evidence; this block checks the saved Drive report and contract identities directly.

The original Task 4 approval includes reproducible scripts, unchanged course helpers, CPU preparation, frozen input verification and documented persistence. The subsequent explicit approval fixed the judge commit and download. This block implements the already planned CPU adapter/tokenizer preparation and introduces no new model, checkpoint, hyperparameter or decoding choice. New file paths and immutable copies are bookkeeping under that preparation scope.

## Work performed

1. Verify sealed source, completed judge-preparation evidence and root contracts, original ownership/approval, unchanged Python and all 64 required package versions. Retain the approved outside-chain global Colab failures as recorded; no packages change.
2. Check snapshot file existence and sizes, and freshly hash the small configuration/tokenizer metadata read by this block. Reuse the already verified weight-file hashes by reference instead of repeatedly hashing roughly 9.282 GB of base/judge weights. This is not a fresh weight-integrity or model-loading claim.
3. Revalidate DPO's standard endpoint and PPO/GRPO standard final archives against their frozen handoffs. Read safetensors adapter headers and compare all q_proj/v_proj LoRA factor shapes, layer coverage and byte ranges with the pinned base configuration. Tensor values are not deserialized into tensors, and no model is constructed.
4. Save only adapter_config.json and adapter_model.safetensors, byte-for-byte, for each of DPO/PPO/GRPO under prepared_adapters/<policy>. Preserve historical base_model_name_or_path fields exactly. Check all destinations before writing any copy; reuse matching files and refuse differing files. SFT has no adapter. Prior task files remain read only.
5. In offline mode, call the unchanged course load_tokenizer helper on the two saved snapshot directories. Test fast tokenization, left padding and chat-template rendering on harmless synthetic text. Load generation configuration metadata with the pinned Transformers version; record original defaults, explicit course call arguments and a preview of their combination. Loading metadata does not construct a model or generate text. The preview must later be checked against each actual GPU model.
6. Reverify all 450 XSTest IDs, order, SAFE/UNSAFE counts and the original 60 audit IDs. Tokenize the prompts in the exact course form and fixed 113 batches of four, saving encoding hashes, widths and length diagnostics. Preserve the 256-token prompt cap and inherited truncation side regardless of diagnostics; no automatic parameter adjustment is made. No response or AI label is used.

The policy generation call remains deterministic, with max_new_tokens=256, do_sample=False, tokenizer pad/EOS IDs, and the unchanged helper's 256-token prompt cap. The judge call remains deterministic with max_new_tokens=64 and its original EOS/pad convention. Other generation defaults remain inherited and recorded. Policy float16 and GPU judge NF4 4-bit/float16-compute settings remain fixed for later inference.

## Saved progress and reruns

- model_inputs_releases/<source manifest SHA-256>: exact source, authorization context, entered-state observation, local tests and documentation.
- model_inputs_attempts/<UTC timestamp>: snapshot metadata checks, adapter source/header receipts, prepared-adapter identities, tokenizer/generation metadata, prompt-encoding hashes and final report, including catchable failures.
- prepared_adapters/dpo, prepared_adapters/ppo, prepared_adapters/grpo: immutable adapter copies only.
- CPU_MODEL_INPUT_CONTRACT.json: deterministic source/input/environment/model/adapter/tokenizer identities for the next stage.

Matching saved files/contracts are reused; mismatches stop and preserve evidence. No model download, package installation, model construction, inference or training occurs. No session time limit or automatic disconnect is introduced.

Expected success: CPU_ADAPTERS_TOKENIZERS_READY. This does not complete CPU orchestration tests or GPU inference validation. The next stage still needs CPU checks of actual generation orchestration, EOS/length accounting, judge parsing, denominator/aggregation semantics, result coverage and resume behavior. Stay on CPU after returning this block's output. Experimental counts remain 0/1,800 responses, 0/1,800 judgments and 0/240 human labels.

## Local validation and attribution

Nine synthetic tests cover valid adapter header shapes, wrong shapes, layer/target coverage, byte overlap/holes/trailing bytes, invalid/duplicate headers, hidden LoRA parameterization changes, refusing all copies before a later destination conflict, exact-byte reruns, and metadata hashing without rehashing weights. No network calls or actual model/adapter weights are used in local tests. Actual Colab tokenizers and adapters remain untested until execution.

Course helpers are preserved under the original preflight release. New orchestration was created with Codex assistance. Safetensors format: https://github.com/safetensors/safetensors#format . Pinned tokenizer API source: https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/tokenization_utils_base.py . Generation metadata API: https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation .
