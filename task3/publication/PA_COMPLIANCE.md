# Task 3 publication audit — 11 October 2026

The completed code and saved evidence support the required Task 3 comparisons. No additional objective defect was found in this audit. The executed source is preserved unchanged. Publication is appropriate with the interpretation limits below; this is not certification of the final eight-page assignment report.

Authoritative reference: assignment PDF, pages 1–4, 9–11 and 18–20; PDF SHA-256 `a6c858739a69813701e340e4fcfe01ed4634f9f141a1f7c93a34c825f15e07f7`. Course commit `1d64ac65acd5e45d1e4e1f415edc80455e21274e`. The PDF and raw course datasets are not redistributed in this package.

## Requirements and evidence

| PA requirement | Audit finding | Evidence |
|---|---|---|
| Correct within-prompt group-relative advantages | Corrected global-statistics starter defect; population std plus 1e-6 within each group. Multi-group tests cover a defect hidden by one prompt per update. | `source/task3_grpo/grpo.py`, source tests |
| Critic-free, fixed reference, released midpoint and LoRA | Each run loads the same supplied midpoint; only 1,089,536 LoRA parameters train. Reference evaluation disables adapters on the frozen base. No critic is instantiated. | `runtime.py`, `training_core.py`, saved loading/session records |
| Standard continuation: 20 updates, K=4 | Exactly 20 saved updates; fixed learning rate, clipping, KL, prompt schedule and generation protocol. | `results.json`, `schedule.json`, current evidence audit |
| Max-length loss masking | Truncated responses contribute zero training loss. They remain in generation counts and evaluation. | Loss masks and per-response derivative records; source tests |
| Reward, KL, group std, uninformative fraction, policy loss, gradient norm, entropy and length trajectories | All eight diagnostics logged; complete eight-panel figure and 36-row CSV generated without rerunning models. | `derived/complete_training_diagnostics.png`, `derived/training_diagnostics.csv` |
| Peak VRAM and standard continuation time | 17.542 GiB peak allocated, 18.799 GiB reserved; 1,069.879 active seconds after model loading, including checkpoint persistence. | Original session records; `RESULTS.md` |
| Equal-generation K=2/4/8 cache study and at least two difficulty bins | Same 192 cached completions per K; fixed median reward proxy splits 24 prompts into 12/12. All required statistics provided overall and within both bins. Raw cache reread and all analysis fields matched exactly. | `cache_recheck.json`, `derived/GROUP_SIZE.md`, `derived/group_size_all_bins.csv` |
| Independent canonical and course Dr.-style short runs | Eight updates each from the same midpoint; reward, beta, epsilon, prompts and decoding unchanged. Canonical's first eight updates equal standard's first eight exactly. The supplied course variant changes the denominator while retaining standardized advantages. | Saved contracts, histories, corrected objective |
| Fixed generated-token budget | Approved decision J fixes the allocated allowance at 16,384 response tokens per fork. Realized totals are 8,483 versus 8,628. This is an explicit interpretation of the PA wording, not exact realized-token matching. | Approval J, original totals |
| Common held-out reward, KL, length and quality evidence | All four conditions evaluated on the same 200 prompts and per-prompt seeds. Length dispersion is saved. Five selected pairs reviewed using the approved rubric. | Four original JSONL files, evaluation CSV, approved `SCORES.json` |
| Direct length-conditioned normalization diagnostic | Per-completion derivative norms, length/sign bins, and a same-rollout probe with zero optimizer steps. Probe has one unmasked response, so its empirical coverage is narrow. | Original length-gradient CSV and common-probe records |
| Fixed endpoints; no Task 4 selection leakage | Every required endpoint retained. Standard final adapter is the Task 4 GRPO candidate, independently of ablation results. | `TASK4_HANDOFF.json`, saved selection flags |
| Reproducibility and public code | Pinned course/model/assets identities, schedules, approved choices, exact executed source, original small export, CPU tests and dependency-free audit provided. | README, source seal, original release ZIP, manifests |
| No raw datasets or large weights in Git | New publication contains code, aggregate/per-response experimental evidence and the small original code release; downloaded training/evaluation datasets, raw completion cache, public weights and optimizer checkpoints stay outside Git. Prompt text is retained as experimental context. | Publication inventory |

## Decisions and their consequences

Decisions A–M were explicitly approved before execution. Isolation, pinned identities, per-unit checksums, recovery of optimizer/RNG state and deterministic schedules make interrupted runs reviewable. They do not change the scientific endpoints. Both real GPU sessions used the A100 configuration recorded in the export.

The reward-model compatibility bridge is narrowly justified by the pinned checkpoint: Transformers 4.57.1 otherwise uses theta=10,000 instead of the checkpoint's declared 1,000,000. Only the in-memory field was translated; saved model/config bytes were not rewritten. CPU tests and actual GPU loading records verify the effective positional frequencies. This is an approved compatibility choice, not another deliberate TA objective error.

The approved forward mode is `eval()` with gradients enabled. Thus the recorded LoRA dropout setting is 0.05 but stochastic dropout is inactive, and the training-only gradient-checkpoint branch is inactive. Trainable weights and optimizer moments are FP32; frozen policy weights are float16 and the reward model uses 8-bit loading. These operational choices must remain explicit rather than being described as active dropout or fully float16 training. No scheduler, AMP scaler, extra clipping, reward shaping or new stabilization was introduced.

The released one-policy-epoch update recomputes likelihoods before the optimizer step with deterministic forward behavior. Consequently every saved pre-update ratio is one and clipping is inactive. This is consistent with the setup; the run cannot demonstrate clipping's effect. Signed sampled KL can be negative and is distinct from the nonnegative exponential KL term in the loss. Reported categorical entropy is also distinct from sampled negative log probability. Decoding retains inherited top-k=20 and repetition penalty=1.1; log-probability diagnostics use raw logits, as in the release.

The difficulty bins use mean cached reward as an approved proxy; they are not independently annotated difficulty. Fixed consecutive partitions retain all 192 generations. Small K differences are descriptive and depend on that partition. Normalized-signal variance partly tracks informativeness by construction.

Equal allowance preserves released update counts, response caps and natural EOS stopping. It does **not** establish literal equality of realized generated tokens. If the marker requires exact realized-token matching, instructor clarification is needed before changing the frozen experiment. This audit does not silently reinterpret the observed totals or start an unapproved rerun.

## Do the results make sense?

Yes, the saved metrics are arithmetically and procedurally consistent. All 800 evaluation aggregates were recomputed from their prompt records, token lengths and sampled log probabilities were checked, and endpoint summary hashes match the corresponding completed training histories. All 36 optimizer records match the approved settings. The tiny standard reward increase (+0.010797) and constant-denominator difference (−0.007721 versus canonical) do not establish a repeatable improvement or a quality winner.

Truncation is the strongest directly observed source of sample loss: standard discards 11,264 of 24,213 generated response tokens (46.52%); each fork discards 5,632 tokens. All training groups have nonzero reward variation, yet updates 5 and 7 mask all four responses in every run. Their current loss gradients are zero, but AdamW still steps and can move parameters through momentum/weight decay. These are not skipped updates.

The common normalization probe's one unmasked response is 452 tokens. The measured derivative-norm ratio is 452/512, matching the denominator change. Other length bins are sparse and fork rollouts later diverge, so the result supports the mechanism at the observed lengths rather than a broad causal claim about quality.

The same 35/200 policy inputs are truncated at 256 tokens and lose the final assistant marker. Reward-input truncation affects 9/9/11/8 records across midpoint/standard/canonical/constant-denominator. Four full prompts already exceed the 1,024-token reward cap. Four of the five reviewed prompts are affected by policy-input truncation. These inherited caps help explain continuation-like or off-task answers and limit attribution to normalization. They were not changed after examining outcomes.

The approved qualitative table uses the exact selected outputs and explicit criteria. Six fixed extreme-difference slots produce five unique prompts. These cases explain failures; they are not a random quality sample, independent blinded ratings or an aggregate quality benchmark. The approved scores and their provenance remain separate from immutable export-time pending flags.

## Verification boundary and remaining work

The sealed source includes the original ten-prompt `word_limit_prompts.jsonl` starter fixture, which was already present in the repository's course code. It is preserved for source identity and asset-validation compatibility; the downloaded Task 3 train/evaluation datasets and 192-completion cache are not redistributed. The four evaluation JSONL files are experimental records, including prompts as context, generated answers and measured metrics.

The publication audit reran all 52 synthetic CPU tests with no failures/errors/skips. This local run used Python 3.12.10, Torch 2.13.0 and NumPy 2.4.6; it is additional portability validation, not a claim to rerun the Colab environment. The recorded Colab environment remains authoritative for the actual experiment.

The original small export's 11 payload hashes pass. The independent Drive ZIP receipt was not supplied. Large adapters, raw rollout tensors and optimizer archives were not downloaded for this audit; local checks cannot independently prove those remote bytes or recompute reward inference. The handoff records the adapter identity observed in the saved evaluation contract and requires verifying the Drive archive when Task 4 loads it.

The PA's required evidence still needs integration into the combined eight-page report. Publication verification in Codespaces and verification of the resulting remote commit are separate final steps. No new training or checkpoint selection is needed to publish this evidence.
