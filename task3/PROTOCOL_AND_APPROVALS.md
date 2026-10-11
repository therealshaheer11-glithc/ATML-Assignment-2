# ATML PA2 Task 3: requirements, objective audit, and approval plan

> Historical approved plan, preserved from preparation. Execution is now complete; see [current results](RESULTS.md) and [publication audit](publication/PA_COMPLIANCE.md). Future-tense statements below record the original plan.

Review and approval date: 10 October 2026. Status: user approved decisions A-M; isolated implementation and local CPU validation prepared. Actual Colab preparation, course-model training and endpoint evaluation remain pending execution. Local tests use tiny synthetic models.

## Target and inspected sources

- User-confirmed repository: https://github.com/therealshaheer11-glithc/ATML-Assignment-2 ; inspected commit `cddc252d1116de1b26d48823ccfb95dae83b4857`.
- User-confirmed execution environment: **Google Colab**, superseding the earlier Kaggle answer.
- Authoritative assignment: `sources/ATML_PA_2_PDF (1).pdf`, SHA-256 `a6c858739a69813701e340e4fcfe01ed4634f9f141a1f7c93a34c825f15e07f7`. All 20 pages were read. Common requirements on pages 1-4 and Task 3 on pages 9-11 were rendered and visually checked.
- Course starter: https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining ; inspected commit `1d64ac65acd5e45d1e4e1f415edc80455e21274e`.
- Course assets: `AbDu11aHHH/ATML-PA2-assets`, pinned release revision `0b350481fb03f5525a35bcdec4131bd4fe487f98`.
- The user's Task 3 objective, GRPO configuration, shared generation helper, and shared metric helper were compared with the course copies and are unchanged. Task 2 has its own source snapshot and approval record. Its scientific and operational approvals are evidence of prior choices, not automatic Task 3 approval.
- Read-only review fetched the small GRPO cache, the training/evaluation prompt files, midpoint metadata, adapter configuration, and checksum manifest. No model weight files were fetched. The five substantive files match their manifest entries. The manifest's self-entry does not match its current bytes; it was preserved unchanged. This is a limited integrity check, not full course installation validation.

## Assignment-fixed rules

These are requirements, not choices to tune after seeing results.

| Item | Fixed requirement and source |
|---|---|
| Initialization | Use the supplied `checkpoints/grpo_midpoint_policy` adapter. Independently reload it for standard and both normalization forks. Do not start from the Task 1 DPO or Task 2 PPO endpoint. PA pp. 10-11. |
| Policy/reference | `Qwen/Qwen2.5-1.5B-Instruct`; policy LoRA only. Frozen reference is the unchanged base policy through the release's adapter-disabled reference helper. No learned critic. |
| Reward | Same frozen learned reward model as PPO: `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`, with the canonical Qwen policy tokenizer and released 8-bit reward loading. Do not import Task 5 binary/pairwise rewards. |
| LoRA | Rank 8, alpha 16, configured dropout 0.05, target modules `q_proj` and `v_proj`. Downloaded midpoint adapter configuration confirms these settings. |
| Data | Fixed UltraFeedback-derived training/evaluation files; inspected files contain 1,200 training prompts and 200 evaluation prompts. Preserve exact IDs, source indices, prompts, and file bytes. |
| Standard run | 20 updates, 1 prompt/update, 4 completions/prompt, 1 policy optimization epoch/update. |
| Normalization forks | Two independent 8-update continuations from the same supplied midpoint; canonical `grpo` versus supplied `dr_grpo` normalization. Both use K=4, the same reward, beta, epsilon, prompt schedule, generation settings and budget contract. |
| Group-size study | Supplied completion/reward cache only; K=2,4,8 at equal total cached generations. No extra policy training or replacement rollout collection for this study. |
| Objective | Prompt-specific reward mean and population standard deviation; response-token clipped likelihood-ratio objective with `minimum`; frozen, detached rollout probabilities; frozen reference. PA p. 9. |
| Optimizer | Released `AdamW(trainable_parameters(policy), lr=5e-6)` constructor. Its inherited weight decay is 0.01, betas (0.9,0.999), epsilon 1e-8; record these explicitly rather than silently changing them. |
| Other training settings | Clip epsilon 0.20, KL beta 0.10, gradient norm limit 1.0, seed 6304, released float16 backbone loading and gradient checkpointing. |
| Decoding | Sampling enabled, temperature 0.7, top_p 0.9, prompt cap 256, continuation response cap 512. Preserve and record inherited model generation defaults as well as explicitly supplied settings. |
| Truncation | Completions reaching the cap without EOS have zero training-loss token masks. Preserve their generated lengths and termination information separately. EOS-terminated responses and padding need correct distinct masks. |
| Course normalization contrast | Canonical divides each completion's surrogate token sum by its valid response length. The supplied `dr_grpo` branch divides by the constant 512. Both retain the same corrected group advantages, clipping and KL term. |
| Later tasks | Preserve the final **standard 20-update** GRPO adapter for Task 4. Do not choose a normalization fork using safety outcomes. Task 4 labels and Task 5 evaluation/diagnostic outcomes cannot tune Task 3. PA pp. 12-13,18. |
| Reproducibility | Runnable task-specific Python scripts, exact commands, settings and versions, IDs/seeds, machine-readable logs, generations, source attribution, and a frozen final code identity. A notebook may launch scripts; it must not be the hidden-state experiment pipeline. |
| Submission | Public GitHub code and small result evidence; no raw course datasets, public model weights or unnecessary large checkpoints. Required evidence belongs in the main 8-page NeurIPS report; references are excluded from that page limit. PA2 explicitly permits generative AI assistance with report prose. PA pp. 1,18. |

## Confirmed deliberate TA error

`task3_grpo/grpo.py:8-17` accepts `group_ids` but never uses them. It computes the reward mean and standard deviation across **all completions in the batch**, so one prompt's rewards affect another prompt's advantages.

Read-only execution of the unmodified course function reproduced the defect with CPU PyTorch 2.13.0:

| Input | Released result | Required behavior |
|---|---|---|
| Prompt A rewards [1,1]; prompt B rewards [10,10] | A advantages [-1,-1]; B advantages [1,1] | Both groups are constant, so every relative advantage must be zero. |
| Prompt A rewards [0,2], evaluated alone | [-1,1] | Its relative ordering must be independent of other prompts. |
| The same A alongside B rewards [100,102] | A becomes approximately [-1.0198,-0.9798] | A must retain one negative and one positive advantage. |

The mandatory correction is to compute each completion's advantage using only rewards with the same prompt/group ID. The PA denominator is `sigma_group + epsilon`; retain the helper's epsilon value 1e-6 and population-standard-deviation convention while implementing that written formula. The starter currently uses a denominator floor instead of addition; document that numerical discrepancy separately from the deliberate grouping defect. No additional whitening or reward rescaling is implied.

The released run has one prompt per update. In that special case batch statistics happen to equal prompt-group statistics, so a single-prompt smoke test can miss the defect. Multi-prompt checks are necessary even if production updates retain the required one-prompt setting.

## Other traps to keep explicit

- The PA's supplied Dr.-GRPO-style condition isolates **length normalization**. The full paper also removes reward-standard-deviation normalization. Importing that extra change would confound the required course comparison. See Liu et al., https://arxiv.org/html/2503.20783v2 .
- The objective's returned `sampled_kl` field is actually its nonnegative exponential KL-penalty estimator. The common reporting helper is the **signed response-token mean of policy_logp - reference_logp**. Save them under distinct names; use the common helper for reported reference drift.
- `sample_entropy` is the sampled negative log-probability estimate. It is not a direct computation of full categorical entropy, particularly under transformed sampling. Keep both names and definitions explicit if both are collected.
- The downloaded group cache has **192 rows: 24 prompts x 8 completions**, with generation indices 0-7. Its recorded cap is 768, whereas continuation cap is 512. Do not regenerate, recrop, or rescore that cache to make it resemble training rollouts.
- No informative-group tolerance helper exists in the inspected GRPO/common metric implementation, despite the PA referring to one. Its numerical tolerance is therefore an approval item below.
- Root `common/` helpers also support published Task 1. Task 2 already uses an isolated source snapshot. Changes to shared root files could invalidate historical package hashes and must not happen as an incidental Task 3 edit.
- GRPO does not define PPO's explicit frozen-evaluation cap 768 or reward input cap 1280. Its shared reward scorer currently defaults to 1024. Do not copy PPO-only settings into Task 3 without an explicit reason and approval.

## Proposed discretionary decisions: explicit approval required

The user approved all decisions A-M with "All approved go ahead" on 2026-10-10. These are the approved proposals; new consequential choices remain outside this authorization.

| ID | Proposal | Reason and boundary |
|---|---|---|
| A | Build Task 3 as an isolated `task3/` package following the Task 2 pattern: its own course-derived source/config snapshot, tests, approval/provenance record, thin Colab launch notebooks, and a reconstructed runtime checkout. Pin the inspected upstream and asset identities. Pin public policy/tokenizer revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` and reward revision `f98759a4a1ccdc47a3136748b6bfea9a32ba8fcc`, matching the recorded Task 2 artifacts. | Keeps completed Task 1/2 evidence and source bytes intact. Confirm public snapshot identities/configuration at preparation. Publication layout/README updates can be reviewed with the finished package. No replacement of an existing run or branch history. |
| B | Reuse the narrowly scoped Task 2 reward-model RoPE compatibility translation **in memory**, auditing original/effective fields and file hashes. Keep the same weights, tokenizer, architecture, 8-bit loading, and pinned Transformers version. | The prior source records nested versus legacy RoPE configuration incompatibility. This requires a new Task 3 approval; it is separate from the deliberate TA error. Validate on CPU before real reward inference. |
| C | Use one seed-6304 shuffled training schedule; standard uses its first 20 prompts and both forks use its first 8. Use a recorded per-update/per-completion generation seed schedule derived deterministically from 6304, reset identically for each condition. | Matches prompts and randomness without selecting favorable examples. Freeze IDs and actual seed values before training. Future resumed runs must restore the same contract and RNG state. |
| D | During rollout scoring and policy optimization, disable stochastic dropout operationally by evaluation mode **with gradients enabled for the update**. Retain configured LoRA dropout 0.05 in the adapter metadata. Keep released loading; compute probabilities, advantage statistics, losses and KL in float32. Add no autocast, GradScaler, scheduler, entropy bonus, extra clipping, reward whitening or optimizer changes. Stop and preserve evidence on nonfinite losses/gradients. | Dropout must not make old/new ratios differ before any optimizer update. This is a consequential operational/numerical choice, not a PA-prescribed setting. Verify trainable PEFT dtypes and reference isolation. The released norm clipping at 1.0 still applies. |
| E | Sort each cached prompt's eight completions by generation_index. Partition into consecutive disjoint groups: four pairs for K=2, two quartets for K=4, one octet for K=8. Include all 192 cached rows, including flagged truncated rows, in the reward diagnostic. | Every K uses the same 24 prompts and same 192 generations exactly once: 96,48,24 groups respectively. Never group different prompts together. Retain cap/termination flags; a cache-only reward diagnostic is distinct from training-loss truncation masking. Reject missing/duplicate indices instead of silently dropping/replacing rows. |
| F | Define a prompt's difficulty proxy as the mean reward of all its eight fixed cached completions. Use the median of these 24 prompt means: at/below median = lower-reward/harder proxy; above median = higher-reward/easier proxy. Freeze the mapping once and reuse across K. | Gives at least two intended bins without using K-specific outcomes or evaluation labels. Call it a reward-based difficulty proxy, not verified intrinsic difficulty. Keep ties together; if a bin is empty, return for approval rather than inventing another rule. |
| G | Define informative groups by population reward std >1e-6; classify std <=1e-6 as numerically uninformative. Report raw mean std and rates. Define relative-signal variance as population variance of the normalized scalar completion advantages across all completions, including zeros. Additionally report variance of raw group-centered rewards. | The starter has no prescribed tolerance implementation. State the tolerance explicitly. Standardized advantage variance can be close to the informative-group fraction by construction; raw centered-reward variance helps interpret that fact. Reporting tolerance does not silently zero small nonzero training advantages. |
| H | Report a direct normalization statistic: per-completion L2 norm of the policy-surrogate derivative with respect to valid response-token log probabilities, grouped into lengths 1-128,129-256,257-512 and positive/negative/zero advantage. Record bin counts and truncation. Apply both loss formulas to one saved shared midpoint rollout probe, and collect the statistic over both forks. | Measures the objective's actual gradient allocation on identical tokens; it is explicitly a **log-probability gradient statistic**, not a LoRA parameter-gradient norm. The latter is logged separately for every update. Empty bins stay empty. The shared probe has no optimizer update and is excluded from continuation budget accounting. |
| I | Freeze and evaluate the supplied midpoint, standard endpoint and both 8-update endpoints on all 200 fixed held-out prompts, one response/prompt, common recorded per-prompt seeds, released sampling and cap 512. Preserve shared reward-scorer cap 1024. Report raw mean reward, course sampled KL, mean/population-std response length, and termination/truncation. Record exact categorical token entropy and the sampled entropy estimate separately. | The midpoint adds an improvement baseline without checkpoint selection. All conditions use the same evaluator and masks. Raw length/statistics use the generation mask, not the training mask that zeros truncated responses. Note any reward-input truncation. No Task 4/5 judge or labels are used. |
| J | Interpret matched short-run token **budgets** as equal allocated allowance: 8 updates x 1 prompt x 4 completions x at most 512 response tokens =16,384 per fork. Keep natural EOS. Save actual valid generated tokens and loss-bearing tokens separately. | This follows the released fixed updates/K/cap contract and parallels the documented Task 2 interpretation, but requires fresh Task 3 approval. It is not equal actual token consumption. If exact realized-token equality is required, resolve that interpretation before training; do not silently change update counts, prompts, K, EOS or caps to force equality. |
| K | Prepare and test code, packages, small-data analysis and model downloads on CPU Colab first. Save verified preparation receipts and downloaded models/assets to Drive. Use GPU only for prepared probe/training/evaluation jobs. Save recoverable state after each completed update and each evaluated prompt, with source/config/data identities, optimizer/RNG/schedule state, and completed-unit records. | Makes limited GPU time useful and supports interrupted sessions without repeating completed experiments. Proposed Drive root is `ATML-Assignment-2/task3/experiment_v1`, created only if absent; an existing root triggers review. No receipts, model availability or Drive verification will be fabricated. |
| L | Use a 120-minute GPU-session scheduling limit, and automatically disconnect only after the model process stops, all completed work is verified on Drive, and flushing succeeds. | This is a proposal, not an inherited Task 2 authorization. The current unit plus persistence may overrun the scheduling boundary. A persistence/flush failure keeps the runtime available. The user may approve another limit or decline automatic disconnection. |
| M | Prepare assistant-assisted qualitative comparisons after all endpoints are frozen, then submit exact examples and interpretations for user review. Show cached easy/hard or informative/uninformative groups where available, and paired normalization outputs selected by recorded reward/length differences. | Supports the PA's required qualitative evidence without pretending the reward model establishes quality. Record the selection rule and assisted review provenance. Do not claim human-only judgments or general quality improvement from selected examples. |

## Required result inventory

| Experiment | Machine-readable evidence to preserve |
|---|---|
| Standard 20 updates | Per-update raw reward, reported sampled KL and separately named loss KL penalty; mean within-group reward std; uninformative-group fraction; policy term and total loss; pre-clipping parameter gradient norm; entropy fields; response mean/std and termination/truncation; generated and loss-bearing token counts. Also peak CUDA VRAM and synchronized wall-clock continuation time. |
| K cache diagnostic | For K=2,4,8: exact row/group membership and conserved generation count; informative/zero-variation rates; mean reward std; defined signal variances; the same quantities within each frozen difficulty bin; prompt/group counts and cache cap/termination information. |
| Two normalization forks | Eight updates each with identical run contract except loss normalization; trajectories; final held-out reward/KL/length/entropy; actual/allotted tokens; direct length-conditioned gradient statistic, its bin counts, and the shared-probe records. |
| Common held-out evaluation | All 200 prompt IDs and responses per endpoint, per-prompt scores/statistics and aggregate summaries; exact endpoint/source/model identities and decoding settings. |
| Qualitative and final handoff | Reviewed minimal examples, explicit method/selection limits, reproducible figures/tables, attribution and commands; final standard adapter identity/location for Task 4. |

## Implementation sequence after approval

1. **CPU setup and objective validation.** Freeze approval records and identities; build isolated source package; validate small assets and public configurations; implement the mandatory group fix. Meaningful checks cover constant groups, independence from unrelated groups, group-ID permutation, population std/epsilon, clipped-surrogate signs, masking/EOS/padding, both length denominators, reference isolation, and differentiability.
2. **Complete and test the entire pipeline before GPU use.** Implement continuation, detached rollout statistics, training masks, optimizer updates, pre-clipping logging, verified atomic saves/recovery, held-out evaluation, cache regrouping, gradient diagnostics and evidence aggregation. Test small CPU models and interrupted-save/resume cases. Prepare Colab launchers and Drive receipts.
3. **Freeze the experimental contract.** Save every prompt/seed schedule, partition, bin rule, budget interpretation, output root and model/config identity. Run the fixed cache diagnostic. Prepare all model files on CPU. Any newly discovered consequential choice returns for explicit approval.
4. **Prepared GPU execution.** Validate actual Colab/CUDA environment and resource feasibility, run the shared midpoint probe, standard 20 updates, and both independent eight-update forks; evaluate the frozen endpoints using the common evaluator. No Task 3 run has occurred yet.
5. **Evidence audit and review.** Recompute summaries from saved records, verify counts/identities and required evidence, prepare readable figures/tables and qualitative examples, then make the Task 3 package reviewable for repository publication. Keep large weights outside Git and preserve completed Task 1/2 material.

## Approved implementation status

The user approved A-M with **“All approved go ahead”** on 2026-10-10. The isolated implementation, tests, cache diagnostic, and Colab launchers are being prepared under that approval. The exact machine-readable authorization is `source/docs/approval.json`.

Local CPU checks use tiny synthetic Qwen/LoRA models, not the course checkpoint. Actual Colab CPU preparation, real reward/model loading, GPU continuations, endpoint evaluation and qualitative judgments must have their own execution receipts. Instructor midpoint diagnostics are never reported as student results. Large course assets and model weights stay on Drive, outside the submission package.
