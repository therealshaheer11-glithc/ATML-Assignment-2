# ATML PA2 Task 2: coding plan and approval record

Status: read-only audit completed on 8 October 2026. The user explicitly approved decisions A-K with "Approved" on 8 October 2026. Chunk 1 implementation is proceeding; no PPO training has been run.

## Sources checked

- User-supplied assignment: `ATML_PA_2_PDF (1).pdf`, all 20 pages extracted; Task 2 pages 7-9 rendered and visually inspected. Common requirements are on pages 1-4; later-task checkpoint and evaluation boundaries were also reviewed.
- Current public course starter: `AbDu11aHHH/ATML-PA2-LLM-PostTraining`, commit `1d64ac65acd5e45d1e4e1f415edc80455e21274e`.
- Course assets pinned by the release: revision `0b350481fb03f5525a35bcdec4131bd4fe487f98`.
- Downloaded only the small PPO cache, RL prompt files, configuration files, and manifests for this audit. No model weights were downloaded or evaluated.
- The seven inspected substantive asset files match their entries in the supplied SHA-256 manifest. The manifest's own self-entry does not match its current bytes; it was left unchanged. This audit is not a full installation validation.

## Assignment-fixed requirements

| Item | Required behavior |
|---|---|
| Initialization | Supplied PPO midpoint policy adapter and matched merged critic checkpoint. Every fork independently reloads both. Do not use the Task 1 DPO adapter. |
| Policy/reference | Qwen2.5-1.5B-Instruct policy; frozen reference represented by the unchanged base policy with the policy adapter disabled, as implemented by the release helper. |
| Reward | Frozen `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`, canonical Qwen2.5-1.5B-Instruct reward tokenizer, released 8-bit frozen-model loading. |
| Standard continuation | 20 rollout updates; 1 prompt per update; 2 PPO optimization epochs per rollout. |
| Clipping study | Supplied fixed cache at epsilon 0.05, 0.20, 0.50; corresponding 8-update forks with beta_KL=0.10. |
| KL study | 8-update forks at beta_KL 0, 0.10, 0.20, with epsilon=0.20. |
| Learning rates | Policy 3e-6; critic LoRA 1e-4; critic scalar head 3e-4. Critic mode `lora_head`. |
| Optimizers | Released separate AdamW optimizers. Policy inherits AdamW weight decay 0.01; critic explicitly uses 0.0. Betas (0.9, 0.999), optimizer epsilon 1e-8. |
| Credit assignment | gamma=1; GAE lambda=0.95; final valid response position bootstraps with zero, per the supplied helper. Value loss is the supplied masked MSE with coefficient 0.50. |
| Termination | Training cap 512; missing-EOS penalty 1.0. Cache confirms subtraction from raw terminal reward. Do not exclude truncated PPO responses using Task 3's different rule. |
| Length limits | Prompt cap 256; continuation response cap 512; frozen evaluation response cap 768; reward input cap 1280. |
| Other release settings | Base seed 6304; temperature 0.7; top_p 0.9; sampling enabled; maximum gradient norm 1.0; model dtype float16. |
| LoRA | Policy and critic rank 8, alpha 16, configured dropout 0.05, target modules q_proj/v_proj; preserve supplied loader/checkpoint structure. |
| Data | Fixed 1,200-prompt training pool and 200-prompt evaluation pool; save exact prompt IDs and source indices. |
| Logs | Reward, reference KL, actor/critic losses, entropy, gradient norms, clip fraction, response length; standard-run peak VRAM and wall-clock time. Save machine-readable logs and generations. |
| Other evidence | Clipping cache surrogate and affected-token fraction; fork held-out metrics and defined stability statistic; reward-quality agreement and disagreement examples. |
| Final checkpoint | Preserve the final standard 20-update policy for Task 4. Do not substitute a selected ablation. No held-out or Task 4/5 tuning. |
| Code delivery | Standalone task-specific Python entry points, with any notebook cells only installing/invoking those scripts. Attribute the course starter and any materially reused external code. |

## Confirmed defect and implementation traps

1. **Deliberate objective defect:** `task2_ppo/ppo.py` takes `torch.maximum(surr1, surr2)`. The manual requires the negative masked mean of `torch.minimum(surr1, surr2)`. Four hand-checkable sign/ratio cases were run against the unmodified starter; all disagree with the manual. For example, A=1, ratio=1.5, epsilon=0.2 produces loss -1.5 in the starter versus the required -1.2.
2. **Old policy is distinct from the reference.** Store and detach rollout-policy log probabilities before optimization; keep them fixed across both optimization epochs. The reference enters reward shaping and reported KL, not the importance-ratio denominator.
3. **State/action alignment matters.** The critic value for an action is taken from the state before that response token; response log probabilities require the corresponding one-token shift. Mask prompt and padding positions; include the terminal EOS position consistently.
4. **The cache is diagnostic data.** All 32 cache rows belong to the held-out pool. They contain `old_logprobs`, `ref_logprobs`, `values`, raw/effective terminal rewards, response text and lengths, and termination flags, but no new-policy log probabilities or raw response token IDs. Never optimize on this cache. Reconstruct with the release tokenizer and verify token alignment; do not force mismatches to fit by silently trimming or padding token sequences.
5. **Affected fraction and active clipping differ.** The manual's reported clip/affected fraction counts all valid tokens whose ratio lies outside the interval. The fraction whose active surrogate branch changes is a different diagnostic.
6. **Critic imperfections are intentional.** Preserve its supplied state. The continuation loader adds the specified initially zero-output LoRA adaptation and trains its scalar head; do not replace or recalibrate the midpoint critic.
7. **Reward configuration compatibility is a separate issue.** The current reward-model revision `f98759a4a1ccdc47a3136748b6bfea9a32ba8fcc` declares default RoPE theta 1,000,000 in the newer nested `rope_parameters` format. The pinned Transformers 4.57.1 Qwen2 configuration defaults its older top-level theta to 10,000. The Task 1 package already contains a narrowly scoped, previously validated in-memory translation. Task 2 must explicitly address this rather than silently accepting a different positional configuration. No reward weights were loaded in this audit.

## Discretionary choices approved by the user

| ID | Recommendation | Reason / boundary |
|---|---|---|
| A | Freeze the inspected starter and asset revisions. Pin the same public policy/tokenizer and reward revisions recorded for Task 1: policy/tokenizer `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; reward `f98759a4a1ccdc47a3136748b6bfea9a32ba8fcc`. Use the release-pinned packages, record every remaining package/GPU version, and adapt installation commands to the user's runtime. | Prevent moving public artifacts from changing results. Hardware/CUDA-specific setup remains pending the runtime answer. |
| B | Reuse the narrow reward RoPE translation in memory, with an explicit field-change audit and original-file hash check. Keep the same reward weights, tokenizer, architecture, and quantization. | Interpret the checkpoint's declared positional configuration correctly under the assignment's pinned Transformers version. Do not upgrade Transformers or replace the reward model as an unapproved workaround. |
| C | Use one seed-6304 shuffled training-prompt schedule; all runs share its prefix. Record separate deterministic generation seed schedules derived from base seed 6304, identically across conditions. | Match prompt order and sampling randomness across comparisons. Record the exact IDs, order, and derived seeds before training. |
| D | Normalize detached GAE advantages once per rollout using the supplied masked normalization helper. Keep unnormalized returns as critic targets; use no reward whitening. | Uses the provided PPO variance-control helper without changing the critic target. The incomplete scaffold does not prescribe whether to call it. |
| E | Disable stochastic dropout during PPO rollout scoring and actor/critic optimization forwards by using evaluation mode with gradients enabled. Preserve the configured LoRA dropout value in the saved configuration and explicitly record the operational mode. | Prevent random dropout masks from creating an apparent likelihood-ratio change before any optimizer step. This operational choice requires approval. |
| F | Preserve released float16 model loading and 8-bit reward loading; calculate log probabilities, KL shaping, GAE, returns and losses in float32. Start with no additional autocast, GradScaler, scheduler, reward rescaling, value clipping, entropy bonus, or extra clipping. Stop on nonfinite losses/gradients instead of silently skipping updates or changing settings. | Keep arithmetic and failures auditable without adding an unapproved stabilization method. The release's gradient norm limit of 1.0 still applies. |
| G | Use the final standard 20-update policy as the single fixed candidate supplying new log probabilities on the supplied cache. All three epsilon values see identical tokens, old/new probabilities, cached values, effective terminal rewards and reconstructed advantages, with beta_KL=0.10. | The cache does not specify a new policy. This gives a non-adaptive, clearly documented geometric diagnostic without training on held-out cache rows. If exact reconstruction fails, stop for review. |
| H | Define fork stability as population standard deviation and maximum of actor gradient norms measured before clipping, across the 16 actor optimizer steps in each 8-update fork. Also save the corresponding critic statistics and nonfinite counts. | Provides a concrete comparable stability measure; the fixed gradient clip would obscure this if norms were measured only afterward. |
| I | Evaluate the supplied midpoint and every final policy on all 200 fixed held-out prompts, using the released sampling settings, cap 768, and identical per-prompt seeds. Report mean and standard deviation of response length; raw learned reward is the held-out reward metric, with termination/penalty information separately identified. Use the course sampled-entropy helper consistently, and additionally log full categorical token entropy from logits under a distinct name. | The midpoint gives a baseline for improvement/disagreement; two entropy fields distinguish the supplied sampled estimate from exact model-distribution entropy. No checkpoint selection follows evaluation. |
| J | Independently run one 8-update central fork at epsilon=0.20, beta_KL=0.10 and reuse that exact saved result in both sweeps. | This is the same condition in both studies. It yields 1 standard run plus 5 unique short forks: 60 rollout updates total. Clearly disclose baseline reuse; never reuse the 20-update standard endpoint as the 8-update condition. |
| K | Match the release's allocated budgets: 8 updates x 1 prompt x at most 512 tokens per fork. Record both the 4,096-token allowance and actual valid generated tokens, retaining natural EOS termination. | Actual consumed tokens can differ. This is a proposed interpretation of matched token budgets, not a claim of equal realized token counts. Exact realized-token equality cannot be silently forced while preserving all update/prompt/EOS constraints. Seek clarification if that stronger interpretation is required. |

All other consequential choices that emerge during implementation will be presented with reasoning before use. Approval applies to decisions A-K and does not authorize unreviewed later experimental changes.

## Four code chunks after approval

1. **Setup and objective validation:** runtime/package/asset preflight, immutable-source records, the minimal mandatory objective correction, and meaningful sign/gradient/mask/GAE/reference checks.
2. **Continuation implementation:** rollout collection, correctly aligned critic values, detached old-policy statistics, KL-shaped rewards, fixed GAE targets, the two PPO epochs, logs and final policy/critic saving; run the standard continuation only after preflight checks pass.
3. **Held-out evaluation:** shared generation/scoring, midpoint and standard evaluation, detailed per-prompt records, summary metrics, and safe resource cleanup between policies.
4. **Ablations and evidence:** cache reconstruction/diagnostic, matched clipping/KL forks, frozen evaluation, stability summaries, plots/tables and qualitative-review records.

Each chunk will be cohesive and runnable without hidden notebook state. Large checkpoints and assets stay outside Git. No experimental outcome is claimed by this plan.
