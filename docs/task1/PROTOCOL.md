# Task 1 protocol, approvals and research decisions

This is the current protocol. Older approval records are preserved unchanged in the evidence archive; they describe the state at their original date. In particular, the original 768/1024 input limits and gradient scale 65,536 were superseded by explicit approvals before the corresponding official results.

## Experimental design

Every DPO condition starts from the original Qwen2.5-1.5B-Instruct checkpoint with fresh LoRA adapters, optimizer, gradient scaler and random state. The reference is the original immutable base, with adapters disabled and gradients off. Seed: 6304. No held-out checkpoint selection, scheduler, warmup or early stopping was used.

| Condition | Training data | Beta | Epochs | Optimizer updates | Final window |
|---|---|---:|---:|---:|---:|
| standard | All 1,500 standard pairs | 0.10 | 1 | 94 | 12 pairs |
| beta_003 | Original first 600 standard rows, before shuffling | 0.03 | 1 | 38 | 8 pairs |
| beta_010 | Same first 600 | 0.10 | 1 | 38 | 8 pairs |
| beta_030 | Same first 600 | 0.30 | 1 | 38 | 8 pairs |
| length_balanced | All 1,500 supplied balanced pairs | 0.10 | 1 | 94 | 12 pairs |

The balanced dataset was supplied by the course, with 500 pairs per stratum. It was not reconstructed from evaluation data. Beta comparisons use the matched short-run group; comparing their losses with the longer standard run does not isolate beta.

The optimizer is AdamW, learning rate 0.00002, weight decay 0, beta moments (0.9, 0.999), epsilon 1e-8, no AMSGrad. Other library defaults are captured in every `run_record.json`. Batch size is 2 pairs, accumulated over 8 microbatches. Each partial final window is weighted by its actual pair count. Gradient norm is clipped to 1 after unscaling.

LoRA uses rank 8, alpha 16, dropout 0.05, `q_proj` and `v_proj`, and no trainable bias. The base is frozen FP16; adapters are FP32. Autocast is off. The final gradient scaler starts at 1,024, with growth factor 2, backoff 0.5 and growth interval 2,000. Nonfinite gradients or skipped optimizer updates stop the run rather than silently reducing its budget.

## Objective and scoring

Let P be policy chosen-minus-rejected response log probability and R the corresponding frozen-reference gap. The DPO logit is beta × (P − R), and the loss is negative log-sigmoid of that logit. Preference accuracy counts P − R > 0 over **all** pairs. Ties are non-wins and stay in the denominator. The historical approval phrase “ties excluded” refers to exclusion from wins; it must not be read as dropping tied examples.

Each sequence score sums teacher-forced, next-token log probabilities over the response only, including EOS and excluding prompt/padding. The four originally empty rejected answers are retained as immediate-EOS responses; three occur in the short subset. No replacement text was fabricated.

## Evaluation

Each DPO model scores all 300 common held-out pairs and generates once for each of their prompts. Standard and balanced models also score the 246 stratified pairs (82 per stratum), and generate on the same 10 fixed word-limit prompts. The untouched SFT baseline generates on the same 300 + 10 prompts; no DPO pair-accuracy result is invented for it.

Generation uses batch size 1, seed 6304 reset per model/pool, sampling, temperature 0.7, top-p 0.9 and at most 256 new tokens. Checkpoint defaults top-k 20 and repetition penalty 1.1 are inherited identically across all six models. The complete effective generation configuration is saved. No decoding choice changed after results were seen.

DPO prompt + response, generation prompt, and complete reward-model input limits are all 4,096. The preprocessing audit found zero prompt/answer truncation across all 7,092 supplied responses. All 310 distinct generation prompts fit. Every actual reward input also fit; the longest was 2,119 tokens. A generated answer stopping at 256 new tokens is a separate output ceiling, not input truncation.

The course reward model is frozen and loaded in 8-bit mode with the canonical policy tokenizer. All 1,830 saved answers were replayed for reward scoring, without regeneration. Reward is a mean per answer; length includes generated EOS and excludes padding. Dispersion is population standard deviation. Word counting uses the released regex `\b\w+\b` and its inclusive `<=` limit, even when the prompt says “under”. Neither approved example falls exactly on that wording boundary.

Sampled KL is the global valid-response-token mean of raw policy-minus-reference token log probabilities. Negative observed values are retained. The generation distribution is modified by temperature, top-p/top-k and repetition processing, whereas scoring uses raw model probabilities. Consequently this signed course diagnostic is not an exact nonnegative KL and is not claimed to be an unbiased estimate of one.

## Corrections and approved departures

| Change | Classification and reason | Approval/evidence |
|---|---|---|
| Subtract the reference margin instead of adding it | Starter objective defect; required by the PA equation | Objective tests and exact executed source |
| Use the adjusted margin for accuracy | Starter diagnostic defect; required by the PA definition | Raw pair recomputation and tie checks |
| Prompt-preserving preprocessing and 4,096-token limits | TA-authorized compute adjustment, **not an error in the manual's formula** | User-supplied TA clarification permits increasing the cap or filtering; explicit approval of all three limits; pinned TA patch and complete encoding audit |
| Keep four empty rejected responses | Preserve released data and full budgets | Data hash and EOS-only checks |
| Translate reward `rope_theta` from nested configuration into the field supported by Transformers 4.57.1 | Checkpoint/library compatibility fix; preserves declared value 1,000,000 rather than library fallback 10,000 | Explicit approval; only that field changed in memory; original configuration unchanged; actual loaded rotary-frequency checks passed for all conditions |
| Initial gradient scale 65,536 → 1,024 | Approved numerical remedy after a training failure | Fresh full restart, same initialization/order, shared setting across all five final conditions; no held-out tuning |
| Optional untouched SFT generation/reward baseline | Approved additional comparison | Same prompts, sampler and reward scoring |
| Two qualitative examples | Approved selection from saved outputs | User: “Approve both examples”; IDs wl05 and wl04 |
| Portable preparation, analysis and beta dispatch commands | Post-experiment publication cleanup | No changes to the executed objective, encoder, trainer, evaluator, settings or results; exact original sources archived |

Initial A–H approvals also cover standalone scripts, frozen-reference handling, full fixed budgets, tail accumulation, saving evidence, and reviewing qualitative choices. Later approval JSON files record the three input limits, reward translation and scale reduction. No further ML choice was made during packaging.

## Numerical failure: hypothesis and outcome

The original standard attempt stopped before update 19: 18 completed updates, 288 completed pairs, and eight adapter tensors with nonfinite gradients. All 152 logged forward microbatch losses and response scores were finite. The final eight microbatches belong to the failed window, not completed optimizer updates.

The proposed explanation was FP16 backward overflow, possibly amplified by the initial scale of 65,536. This remains **unconfirmed**: the exact operation causing the overflow was not isolated. Reducing the initial scale to 1,024 reduces temporary gradient amplification by 64. Unscaling preserves the intended gradient in exact arithmetic, although rounding/underflow behavior can differ.

The user approved the change and a fresh restart. The only difference in the failed versus successful training source is the initial scale and its recorded value. The standard retry completed all 94 updates and all later runs also completed. This supports the remedy operationally; it does not prove the hypothesis. The failed attempt is preserved and excluded from final performance tables.

## Study method and interpretation boundaries

Requirements and fixed choices were separated from open choices before implementation. The objective was checked against the PA equation, then response masking, accumulation, frozen reference and checkpoint reload were checked before full runs. A disposable long-input GPU trial established that the proposed cap fit the A100. Synthetic checks and disposable trials are not official performance results.

All official runs used their fixed final budget, with no selection based on held-out scores. The audit independently recomputed saved results instead of training again. Qualitative selection reviewed all 30 word-limit answers and selected two illustrations jointly with the user; these are not a random sample or an estimate of error prevalence.

One seed, three beta values, a 600-pair ablation budget, a 256-token generation ceiling and ten word-limit prompts limit the strength of general conclusions. Balancing is a diagnostic intervention, not proof that content quality improved or length bias disappeared. Reward is an uncalibrated proxy and does not establish correctness by itself.
