# Task 3 saved-results audit

**Quantitative export audit passed. The qualitative score table was approved by the user on 11 October 2026.**

Received archive: 6,902,613 bytes; SHA-256 `4ee4ad4b317c308be910b1dc5d4d3f54751f58a5fa12bf217bcd5ba23572267d`. All 11 payload hashes and ZIP CRCs pass. The independent Drive ZIP receipt was not supplied. The local sealed source and approval match the experiment contracts; sealed source files were not changed.

All 36 update records and all 800 evaluation records satisfy the saved schedules and contracts. Evaluation aggregates exactly reproduce results.json and evaluation.csv. Token counts, EOS/truncation flags, saved-token sampled KL/entropy, and analytical log-probability gradient norms agree. Canonical eight-update history exactly matches the first eight standard updates. The raw 192-row cached reward file matches its frozen checksum; recomputed overall and difficulty-bin metrics agree within float32 rounding. Adapter weights, optimizer tensors and raw per-unit ZIPs remain on Drive and were not independently inspected in this local audit.

| Evaluation | Prompts | Mean raw reward | Signed sampled KL/token | Mean response tokens | Categorical entropy/token | Output truncation |
|---|---:|---:|---:|---:|---:|---:|
| midpoint | 200 | 1.026841 | 0.00002828 | 271.800 | 1.046091 | 26.0% |
| standard | 200 | 1.037638 | 0.00007640 | 270.645 | 1.052233 | 24.5% |
| canonical_8 | 200 | 1.002466 | 0.00002506 | 272.660 | 1.061594 | 25.0% |
| dr_grpo_8 | 200 | 0.994745 | 0.00007220 | 270.425 | 1.068035 | 24.5% |

Standard versus midpoint reward difference: +0.010797. Constant denominator versus canonical reward difference: -0.007721; mean response length difference: -2.235 tokens. These single-seed descriptive differences do not establish repeatable improvement or a reliable normalization winner. The signed sampled KL is the course reporting statistic and may be negative; the nonnegative exponential loss penalty is separately logged.

| Continuation | Updates | Allocated response tokens | Generated response tokens | Loss-bearing tokens | Active continuation seconds | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| standard | 20 | 40960 | 24213 | 12949 | 1069.879 | 17.542 |
| canonical_8 | 8 | 16384 | 8483 | 2851 | 373.797 | 16.177 |
| dr_grpo_8 | 8 | 16384 | 8628 | 2996 | 376.746 | 16.177 |

Time covers continuation after model loading, including rollout/scoring/updates/checkpoint persistence; it excludes setup, endpoint evaluation and final export. Forks match allocated allowance (16,384), with natural EOS causing different realized token counts.

| K | Fixed generations | Informative groups | Mean group reward std | Relative-signal population variance | Harder-proxy informative rate | Easier-proxy informative rate |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 192 | 93.75% | 0.384567 | 0.937481 | 89.58% | 97.92% |
| 4 | 192 | 95.83% | 0.541004 | 0.958328 | 91.67% | 100.00% |
| 8 | 192 | 95.83% | 0.619054 | 0.958329 | 91.67% | 100.00% |

Difficulty is the frozen median split of each prompt’s mean eight cached rewards (median 2.19873046875; 12 prompts per bin), a reward-based proxy. Informative groups increase from K=2 to K=4 and then plateau at K=8. Standardized advantage variance tracks informative-group rate partly by construction; it is not independent proof of greater learning effectiveness.

## Interpretation limits

- Input cap: the same 35/200 prompts were cut to 256 tokens in every condition. Their stored inputs lack the final assistant-generation marker. Several selected outputs continue the input passage, consistent with this loss of context; normalization alone cannot be blamed for those failures.
- Reward-input cap: 9 midpoint, 9 standard, 11 canonical and 8 constant-denominator inputs exceeded 1024 tokens. Four original prompts already exceed that cap, so response-sensitive scoring is limited for those cases. These records remain included in the frozen 200-prompt evaluation.
- Output masking: updates 5 and 7 in all three runs had all four completions truncated and zero current loss gradients. All reward groups still had nonzero within-group variation. Reward variation therefore did not guarantee usable loss tokens. AdamW still steps, so momentum/weight decay can change parameters despite zero current gradients.
- Standard loses 11,264 of 24,213 generated tokens to masking (22/80 completions). Forks mask 11/32 completions each: canonical retains 2,851/8,483 loss tokens and constant denominator 2,996/8,628. This is a directly observed source of sample inefficiency.
- Pre-update ratios are exactly one and clip fractions zero throughout, consistent with one epoch and deterministic evaluation-mode forwards; these logs do not show active ratio clipping across repeated policy epochs.
- Common probe: three of four completions are masked; the one valid completion has length 452 and positive advantage. Its gradient norm is 0.013369502 canonical versus 0.011802763 constant denominator, ratio 452/512. This verifies the declared length scaling at one length, not a broad empirical short-versus-long trend.
- Fork length-bin tables have sparse cells and differing rollout trajectories/advantages. They are log-probability derivative norms excluding KL, not parameter gradient norms. At ratio one, canonical scales as |A|/(4 sqrt(L)); constant denominator scales as |A| sqrt(L)/(4*512). Distinguish these formulas from claims about answer quality.
- Qualitative examples: six frozen selection slots contain five unique prompts. Exact outputs and selection rules pass the audit; the five unique pairs were subsequently reviewed and their score table approved by the user. See SCORES.md.
- Final Task 4 handoff stays training/standard/final.zip::policy/. No checkpoint/fork selection was performed. No GitHub publication or new training occurred in this audit.

Files: AUDIT.json contains the machine-readable local checks; ASSISTANT_QUALITATIVE_REVIEW.json and QUALITATIVE_REVIEW.md contain proposed interpretations and exact saved examples. The original received export remains unchanged.


Current review outcome: [approved score table](SCORES.md). The originally exported pending-review flags are preserved as historical run outputs.
