# Task 1 results and report evidence

All values below are recomputed from the saved records. Full-precision values remain in the JSON/CSV files. These are descriptive results from one seed, not significance claims.

## Common 300-prompt evaluation

Length counts generated tokens including EOS; ± is population standard deviation. SFT is the untouched starting model and has no DPO loss/accuracy entry. Standard and balanced have larger training budgets than the three short beta runs.

| Condition | Train pairs | Beta | Held-out loss | Accuracy | Sampled KL | Reward mean ± SD | Tokens mean ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| sft | 0 | — | — | — | 0.00000000 | 1.081303 ± 1.384992 | 156.68 ± 104.20 |
| standard | 1500 | 0.1 | 0.672752 | 187/300 (62.33%) | 0.00016604 | 1.208806 ± 1.386690 | 158.99 ± 103.64 |
| beta_003 | 600 | 0.03 | 0.691439 | 191/300 (63.67%) | -0.00043214 | 1.035452 ± 1.484218 | 151.91 ± 106.54 |
| beta_010 | 600 | 0.1 | 0.687744 | 186/300 (62.00%) | -0.00044671 | 1.096526 ± 1.464286 | 153.46 ± 105.05 |
| beta_030 | 600 | 0.3 | 0.679951 | 194/300 (64.67%) | -0.00047494 | 1.144778 ± 1.426365 | 156.68 ± 103.64 |
| length_balanced | 1500 | 0.1 | 0.673079 | 187/300 (62.33%) | 0.00224995 | 1.122151 ± 1.498092 | 155.01 ± 106.50 |

Standard DPO has higher mean reward than SFT (1.208806 versus 1.081303), with little mean-length change under the shared output cap. It correctly ranks 187/300 pairs by the reference-adjusted margin. This does not establish overall response correctness.

Across beta 0.03, 0.10 and 0.30, reward and mean generated length increase in these runs. Accuracy is not monotonic: 63.67%, 62.00%, 64.67%. Loss changes with beta scaling and is not directly a quality ranking. All three sampled-KL values are small and negative; preserve them as the signed course diagnostic, not “negative true KL”.

## Length-confounding study

| Held-out stratum | Standard correct / total | Balanced correct / total |
|---|---:|---:|
| preferred_longer | 35/82 (42.68%) | 34/82 (41.46%) |
| length_matched | 46/82 (56.10%) | 51/82 (62.20%) |
| rejected_longer | 58/82 (70.73%) | 65/82 (79.27%) |
| All strata | 139/246 (56.50%) | 150/246 (60.98%) |

Balanced training improves overall stratified accuracy by 11/246 pairs, mainly in matched-length and rejected-longer strata. Preferred-longer accuracy falls from 35/82 to 34/82. Thus the intervention changes which strata are learned well; it does not demonstrate that length bias has been eliminated. The balanced and standard datasets differ in supplied composition, so this is not a pure manipulation of answer length while holding semantic content fixed.

## Same ten word-limit prompts

| Condition | Compliant | Words mean ± SD | Tokens mean ± SD | Reward mean |
|---|---:|---:|---:|---:|
| sft | 4/10 | 52.20 ± 32.43 | 60.60 ± 36.35 | 3.891992 |
| standard | 5/10 | 39.90 ± 15.28 | 47.10 ± 17.74 | 3.789844 |
| length_balanced | 5/10 | 38.30 ± 11.30 | 45.70 ± 13.90 | 3.706641 |

Both DPO conditions are shorter on this set and comply on 5/10 prompts, compared with 4/10 for SFT. The small set supports counts and examples, not broad claims. All 30 answers terminate with EOS before 256 tokens; no compliance failure here is caused by hitting the generation ceiling.

## Output ceiling and input limits

| Condition | General answers reaching 256 tokens without EOS |
|---|---:|
| sft | 130/300 |
| standard | 139/300 |
| beta_003 | 131/300 |
| beta_010 | 132/300 |
| beta_030 | 132/300 |
| length_balanced | 130/300 |

Reported general-response lengths are capped observations, and some content incompleteness may reflect that fixed ceiling. Raising the input limit to 4,096 preserved prompts and supplied responses; it did not increase the approved 256-new-token output ceiling. No actual reward input was shortened; the longest was 2,119 tokens.

## Approved qualitative evidence

Selection was from saved answers, reviewed jointly with the user. The user approved both examples. They illustrate particular failure modes and are not a random sample or a measured error rate. Full responses and the approval are in [approved_examples.json](../../results/task1/approved_examples.json).

**wl05 — higher reward with a misplaced limitation of attention.** The prompt asks for one advantage and one limitation of attention in at most 45 words. SFT identifies attention as “computationally expensive” for long sequences (reward 3.539063). Standard and balanced instead invoke the “filter bubble” effect and individuals seeing information that confirms their beliefs (reward 4.246094). In this machine-learning context, that social-media claim does not explain a technical limitation of neural attention. This is a content-relevance judgment: SFT also violates the word limit, and DPO is much shorter (110 versus 48 words). Neither is endorsed as a perfect answer.

**wl04 — instruction-compliance regression.** Asked to explain gradient descent in under 35 words, SFT uses 31 words and complies. Standard and balanced produce the same 43-word response and fail the limit; all stop naturally. This is a per-prompt regression despite aggregate compliance rising from 4/10 to 5/10. Here reward falls from 3.949219 to 3.750000, so this example is not itself a reward-increase claim.

## What goes into the main assignment report

Include the common-results table with training budgets, the per-stratum comparison, common word-limit length/compliance statistics, and brief versions of both qualitative examples. State seed, fixed decoding, averaging conventions, the TA-approved 4,096 limits, zero input truncation, output-ceiling effects, reward compatibility translation and approved gradient-scale restart. Explain the beta and length findings with the limits above. Keep the required evidence in the main eight-page report; archival audit documents do not substitute for it.

The standard adapter remains the prescribed DPO model for Task 4. No “best” beta model was selected from these held-out results.
