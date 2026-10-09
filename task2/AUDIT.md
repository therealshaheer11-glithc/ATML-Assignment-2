# Task 2 audit — 9 October 2026

## Conclusion

**The technical audit passed. No additional GPU run is needed.** All saved budgets, source/asset contracts, prompt schedules, optimizer settings, state dtypes and quantitative evidence checked below are consistent with the approved protocol. The user explicitly approved the required qualitative examples on 9 October 2026 after their explanation. No Task 2 code/evidence approval remains outstanding. The interpretation limits below remain documented.

The audit preserves the original source, original result export and its historical checksums. A separate postprocessing script corrects the clipping table's presentation fields. Model inference was not rerun.

## Evidence verified

| Check | Result |
|---|---|
| Original summary export | All 14 files match its manifest |
| Additional audit export | All 2,968 files match its manifest |
| Training | Six runs; 60 rollout updates; 120 PPO optimization epochs |
| Standard run | 20 updates, two epochs each |
| Short forks | Five independent eight-update runs; central_8 reused in both sweeps |
| Held-out evaluation | Seven conditions × 200 prompts = 1,400 records |
| Diagnostic cache | 32 fixed rows, 8,814 valid tokens, never used for training |
| Training/evaluation separation | IDs disjoint; recorded schedule matches the approved seeded schedule |
| Frozen execution | Source/install/asset hashes and configuration match the approved contracts |
| Optimizers | Approved learning rates, decay, betas, epsilon, and FP32 parameters/moments in all 60 update records |
| Numerical failures | All completed diagnostics finite; zero recorded nonfinite failure events |
| Core implementation tests | 50 passed on CPU in a fresh restored checkout |
| Resume compatibility regressions | 7 passed on CPU |
| Source restoration | Offline pinned course bundle plus exact hashed overlay verified |
| Task 4 PPO policy | Fixed standard 20-update adapter, never selected using held-out scores |

Held-out raw reward means/standard deviations were independently reproduced from the response worksheet. All KL, entropy, length, termination and reward aggregates were recomputed from the underlying 1,400 records. Sampled KL and sampled entropy were also recalculated from saved token log-probabilities: the largest absolute difference was 3.5763e-7. Categorical entropy was re-aggregated from the saved per-prompt values; full logits were not exported, so its per-token calculation was not independently rerun.

Cached advantages, returns and reconstructed response tokens match the hash-verified release cache. Affected-token and active-branch counts agree exactly. Cached surrogate values agree within an absolute tolerance of 2e-8 across the recorded Colab and local CPU PyTorch environments. Original reported values are retained.

All 1,400 prompt token sequences match the pinned tokenizer under the released 256-token cap, and every saved response decodes correctly. Exactly 35 of the 200 unique held-out prompts exceed that prompt cap. The saved tokenizer JSON differs from the original public JSON because `save_pretrained` serializes runtime padding/truncation, modernizes the merge representation and moves the chat template to its own file. Template hashes and actual token sequences were checked; byte equality of these differently serialized files was not assumed.

The three required figures were visually reviewed, then rebuilt from the underlying saved JSON; all three PNGs were pixel-identical to the uploaded figures. Machine-readable values behind all training trajectories are now included in `evidence/reviewed/training_trajectories_all_runs.csv`, with all optimization steps, per-prompt evaluation metrics, budgets and stability statistics alongside it.

## Presentation correction

The original `aggregate_cpu.py` merged cached geometry and held-out dictionaries that both used `valid_tokens` and `aggregation`. The held-out fields overwrote those cache labels in `clipping_study.csv`. The numerical clipped objectives, token counts used internally, affected fractions, rewards and figures were unaffected.

Use **`evidence/reviewed/clipping_study_corrected.csv`** for the final table. It separates `cached_valid_tokens=8814`, `cache_aggregation`, `held_out_valid_tokens` and `held_out_aggregation`. Historical code and the original export remain unchanged for provenance. `analysis/audit_export.py` reproduces the correction.

## Findings that constrain the interpretation

**Training clipping was inactive in all 120 recorded PPO steps**, including epsilon 0.05. Its short fork's largest recorded absolute ratio change was about 0.02918, below the 0.05 boundary. The cached batch still shows the required geometric effect: affected fractions decrease from approximately 10.7102% to 0.8396% to 0.5559% as epsilon increases from 0.05 to 0.20 to 0.50.

Do not attribute the small differences between trained clipping forks to an active clipping intervention. The records already show small differences in first-step actor gradient norms while first responses and old value means are identical, before epsilon could constrain a step. The precise source of that numerical variation was not isolated. The release uses seeded sampling but does not guarantee bitwise deterministic GPU execution. No new stabilization or rerun was introduced to hide this result.

Short forks have equal **allocated** budgets of 4,096 response tokens and eight updates. Realized valid token counts are 2,113 (clip_005), 2,113 (central_8), 2,178 (clip_050), 2,189 (kl_000), and 2,178 (kl_020), retaining natural EOS termination. This follows user-approved decision K and must not be described as exact token-consumption equality. If the instructor means equality of realized tokens, that interpretation needs clarification before changing the experiment; user approval alone does not establish the instructor's interpretation.

Reference KL is the release's signed sampled statistic using raw logits on responses generated with temperature/top-p/top-k/repetition transforms. It can be negative and is not an exact nonnegative KL. Prompt and reward-input truncation are retained from the release. Single-seed, small-budget reward differences do not establish significance, general superiority, reward hacking or safety. Selected qualitative cases illustrate an explicit constraint; they are not a population quality score. Task 4 remains separate.

## Standard timing, memory and environment

Standard active continuation: **305.336372142 seconds**. Peak PyTorch allocated memory: **8.630565166 GiB**; peak reserved memory: **9.167968750 GiB**. The timer includes generation, scoring, optimization and per-update checkpoint persistence after model loading; it excludes setup, model reload, final adapter export and offline analysis. These are not whole-session billing time or total process VRAM.

The saved standard environment records A100-SXM4-40GB, Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 4.57.1, Tokenizers 0.22.1, PEFT 0.17.1, TRL 0.27.2, Accelerate 1.15.0 and bitsandbytes 0.50.2. Full environment snapshots are retained in the records archive. The local CPU tests/audit used the separately recorded local environment; no claim of byte-identical cross-platform inference is made.

The approved critic-head exception is explicitly documented: the active trainable scalar value head and its AdamW moments are FP32; the frozen critic and original head copy remain FP16. Recorded actor/critic loads have no missing, unexpected or mismatched keys. The reward model is frozen, 8-bit and uses the approved positional-configuration translation.

## Archive provenance and public-package scope

- Original result ZIP SHA-256: `65aca40d2770631d18ba479da56ce64404f3019143bb73183030b1c59a95a5a0`.
- Additional audit ZIP SHA-256: `19683f9d2990ce97de45e36d6f00f8cd8f4b5feb2eac5debf8384fdea1f7d908`.
- Released fixed-cache SHA-256: `6d9c28c1cc534b60410640f2f23694c227d4d653bf8088e9b1b2d6bf39be8b6d`.
- Standard final adapter SHA-256: `0611abe4991b033dda4352b34a72b0f499a50f27536799a13eeee7de56b41bf0`.

`evidence/raw_records_verified.zip` retains all experimental records and metadata byte for byte. It omits only 30 duplicate public tokenizer JSON files, reducing the uploaded audit archive from 45.3 MB to approximately 25.9 MB. It has a new subset manifest, preserves the original full export manifest and lists every omitted file/hash in `PUBLICATION_SUBSET.json`. No dataset pool files, model weights, optimizer tensor checkpoints, credentials or nested working `.git` directory are included. The tiny course Git bundle preserves upstream source history.

The full GPU checkpoint remains on Drive. Its archive receipt and final adapter hashes are recorded in `TASK4_PPO_HANDOFF.json`. Hashes/provenance and the read-back checks performed during CPU export were verified; the audit did not reload the large checkpoint weights or independently rerun GPU logits. The original cache lacks raw token IDs, so text/count reconstruction cannot prove uniqueness of its historical tokenization.

## Approved qualitative evidence

The same prompt requests 20 titles, each 6–10 words. At beta 0.20, compliant titles improve from 18/20 to 20/20 while reward rises from 3.298828125 to 3.515625 (agreement). At beta 0, compliance declines to 17/20 while reward rises to 3.3828125 (disagreement). Both candidates still produce 20 titles. These assessments cover this explicit instruction only, not overall response quality. Full responses, title counts, the two-row review CSV and the explicit user approval are included. The original blank worksheet and pre-approval audit log are retained as historical snapshots.
