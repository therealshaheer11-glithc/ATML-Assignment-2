# Task 3 results and reviewed evidence

The standard 20-update continuation, both independent 8-update normalization forks, and all four 200-prompt evaluations are complete. The five-pair qualitative score table was approved on 11 October 2026.

## Frozen evaluation

| Condition | Mean raw reward | Signed sampled KL/token | Mean response tokens | Length std (population) | Categorical entropy/token | Output truncation |
|---|---:|---:|---:|---:|---:|---:|
| midpoint | 1.026841 | 0.00002828 | 271.800 | 191.172 | 1.046091 | 26.0% |
| standard | 1.037638 | 0.00007640 | 270.645 | 191.534 | 1.052233 | 24.5% |
| canonical_8 | 1.002466 | 0.00002506 | 272.660 | 191.619 | 1.061594 | 25.0% |
| dr_grpo_8 | 0.994745 | 0.00007220 | 270.425 | 190.268 | 1.068035 | 24.5% |

Standard reward is 0.010797 above midpoint. Constant-denominator reward is 0.007721 below canonical, with responses 2.235 tokens shorter on average. These are descriptive, single-seed differences; they do not establish a reliable improvement or normalization winner.

## Equal-generation cache study

| K | Generations | Informative-group rate | Mean group reward std | Relative-signal variance | Harder-proxy informative rate | Easier-proxy informative rate |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 192 | 93.75% | 0.384567 | 0.937481 | 89.58% | 97.92% |
| 4 | 192 | 95.83% | 0.541004 | 0.958328 | 91.67% | 100.00% |
| 8 | 192 | 95.83% | 0.619054 | 0.958329 | 91.67% | 100.00% |

The frozen difficulty proxy is the median split of mean reward across each prompt’s eight cached completions. Rates increase from K=2 to K=4 and then plateau at K=8. Standardization makes relative-signal variance track informative-group rate partly by construction. These cached diagnostics do not measure the effect of training with different K.

## Continuation cost and signal

| Condition | Updates | Allocated tokens | Generated tokens | Loss-bearing tokens | Active continuation seconds | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| standard | 20 | 40960 | 24213 | 12949 | 1069.879 | 17.542 |
| canonical_8 | 8 | 16384 | 8483 | 2851 | 373.797 | 16.177 |
| dr_grpo_8 | 8 | 16384 | 8628 | 2996 | 376.746 | 16.177 |

Time includes active rollout/scoring/update/checkpoint persistence after loading; it excludes setup, evaluation and final export. Both forks have equal allocated allowance, with different realized token counts due to natural EOS.

All recorded training reward groups have nonzero variation. However, updates 5 and 7 in each continuation mask all four responses after they hit the output cap, giving zero current loss gradients. AdamW still steps, so optimizer momentum and weight decay can change parameters. Standard masks 22/80 completions; each short fork masks 11/32. Loss of usable response tokens is an observed source of sample inefficiency.

## Normalization diagnostic

The same-rollout probe has only one unmasked completion, of length 452. Its policy-log-probability derivative norm is 0.013369502 under canonical normalization and 0.011802763 under the constant denominator; the ratio is 452/512. At likelihood ratio one, the declared norms scale as |A|/(4 sqrt(L)) and |A| sqrt(L)/(4*512), respectively. This verifies the mechanism at one observed length. Sparse length-bin counts, differing advantages and diverging fork rollouts limit broader empirical claims about length bias. These derivatives exclude KL and are distinct from parameter gradient norms.

## Reviewed selected examples

Each answer was assessed for answering the question, support in the supplied text (technical correctness for C), and clarity/focus, each 0–2.

| Example | Canonical /6 | Constant denominator /6 |
|---|---:|---:|
| Poem/video | 2 | 1 |
| C static keyword | 4 | 4 |
| Polish sentiment | 1 | 3 |
| MidJourney instructions | 2 | 2 |
| SafeSpace passage | 2 | 2 |

The five unique prompts come from six frozen selection slots ranked by extreme reward or length differences. One prompt occurs in both lists and is counted once. The scores illustrate failures and tradeoffs, including unsupported elaboration, inconsistent labels and poor instruction following. They are not a representative quality benchmark.

## Input and evaluation limits

The same 35/200 prompts exceed the 256-token policy-input cap in every condition, and their saved inputs lack the final assistant-generation marker. Several selected responses continue the supplied passage. These cases limit attribution of failure to normalization alone. Reward-input truncation affects 9 midpoint, 9 standard, 11 canonical and 8 constant-denominator records at the 1024-token cap. Four original prompts already exceed that reward cap. All 200 prompts remain in each frozen evaluation.

## Evidence and handoff

- [Local audit](results/actual_run_20261011/review/AUDIT.md) and [machine-readable checks](results/actual_run_20261011/review/AUDIT.json).
- [Approved score table and reasons](results/actual_run_20261011/review/SCORES.md) and [machine-readable approved scores](results/actual_run_20261011/review/SCORES.json).
- Exact questions and outputs remain in `results/actual_run_20261011/qualitative_review.json` and the four evaluation JSONL files. Original pending flags describe the export-time state; the approved sidecar records the subsequent review.
- [Complete required training trajectories](publication/derived/complete_training_diagnostics.png), [their exact CSV](publication/derived/training_diagnostics.csv), and [all difficulty-bin statistics](publication/derived/GROUP_SIZE.md).
- [PA compliance and decision audit](publication/PA_COMPLIANCE.md).
- Source manifest SHA-256: `f87021b8ee1057d2afbae29d12c5bd0e904d10b1e9888d9476f99f2959a05c07`.
- Received results ZIP SHA-256: `4ee4ad4b317c308be910b1dc5d4d3f54751f58a5fa12bf217bcd5ba23572267d`; all 11 internal payload hashes pass. The independent Drive archive receipt was not attached.
- Large adapters, optimizer tensors and raw per-unit archives remain on Drive; they were not independently inspected during the local small-export audit.
- Task 4 uses the final standard adapter: `MyDrive/ATML-Assignment-2/task3/experiment_v1/training/standard/final.zip`, inside `policy/`.
- Main-report integration and GitHub publication remain outstanding.
