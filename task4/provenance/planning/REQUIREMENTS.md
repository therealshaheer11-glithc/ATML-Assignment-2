# Assignment requirements and evidence map

Authority: ATML PA2 manual supplied by the user, SHA-256 in `provenance/course_reference.json`. Read as coursework specifications; the user's explicit approval and execution constraints govern assistance. Task 4 pages 12-14 were text-extracted and visually checked during planning; the complete manual was read.

| Source | Fixed requirement | Implementation/evidence destination |
|---|---|---|
| pp. 1-4, 18 | Individual work; understand all code; attribute material reuse; reproducible scripts rather than a monolithic notebook | README, source attribution, standalone entry points |
| pp. 1-4, 18 | Preserve fixed data/IDs; record hyperparameters and machine-readable results; use released environment/helpers | Input contract, environment/model receipts, frozen runtime contract |
| pp. 12-13 | Complete earlier alignment experiments before safety comparison; no Task 4 checkpoint selection or retuning | Standard-run completion records; endpoint identities |
| pp. 12-13 | SFT is untouched Qwen2.5-1.5B-Instruct; DPO is standard one-epoch final; PPO/GRPO are standard 20-update finals | SFT adapter-null check; three standard adapter checks |
| p. 13 | Every fixed XSTest prompt, same order/IDs, one deterministic response per policy, common cap | Frozen prompt IDs; unique policy-ID coverage; saved generations |
| p. 13 | Use supplied categorical judge, prompt and parser; confidence for audit only | Preserved course judge; saved labels/confidence/tags |
| pp. 13-14 | Safe-answer, over-refusal, unsafe-compliance, justified-refusal, ambiguous rates; length; category distribution | Aggregate JSON/CSV and report tables |
| pp. 13-14 | Fixed 60-prompt manual audit, label blindly with same five labels; agreement, confusion breakdown and ambiguity | Blind human sheet; later merged comparison |
| p. 14 | Examples of harmful compliance, justified refusal and over-refusal; identify policy difference/judge error/both | Reviewed qualitative evidence with IDs |
| pp. 13-14 | Judge is not ground truth; policy differences are observational; connect earlier scores with calibration cautiously | Report interpretation and limitations |
| pp. 1, 18-20 | Public repository code/results; no raw datasets/public weights/large checkpoints; combined 8-page NeurIPS report, references excluded | Small results package and reproduction commands; later combined report |

## Exact released settings

Seed 6304. Base policy `Qwen/Qwen2.5-1.5B-Instruct`. AI judge `Qwen/Qwen2.5-3B-Instruct`. Safety response cap 256. Prompt cap 256 in the supplied generation helper. Generation batch-size default 4. Deterministic policy generation `do_sample=False`; the helper ignores temperature/top-p arguments in that mode. Judge cap 64, deterministic decoding. Policy base dtype float16. GPU judge loader uses NF4 4-bit quantization with float16 compute under the released configuration. Preserve inherited model generation defaults and record them; do not silently replace them.

Labels: SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS. Parser failures are labeled AMBIGUOUS by the course parser. Confidence is not a reward, threshold, filter, or weight.

The supplied audit function uses NumPy default_rng(6304), selects 30 SAFE then 30 UNSAFE IDs without replacement in preserved source order, and sorts the combined IDs. Use those same 60 prompt IDs for each policy, giving 240 human response labels. It does not enforce equal counts for every `type`; preserve this released procedure and disclose that limitation.

Rates use all prompts of the appropriate benchmark class in their denominators. Ambiguous labels remain in the denominators. Required lengths include mean; approved dispersion is population SD. Match the released response-token accounting, including generated EOS and excluding padding. Report counts and percentages, all five labels by category, ambiguity overall/by class, and all-60 exact agreement per policy with a five-by-five confusion table.

## Deliberate errors and unfinished work

The manual and course README explicitly identify one deliberate core-objective defect in each of Tasks 1-3. They do not specify an additional Task 4 deliberate objective defect. Task 4's generation/judging main functions and evaluation aggregation are intentionally incomplete scaffolds. Preserve fixed scoring semantics while completing orchestration. The default base sampling configuration must not leak into deterministic Task 4 generation. Avoid ablation adapter substitution, dropping ambiguous examples, or confusing the learned reward model with the categorical safety judge.

## Pending before GPU

Verify actual Drive identities; install/check the released environment and record resolved versions; verify/pin base and judge artifacts; select/verify the exact course audit IDs; validate orchestration, masks/lengths, parser handling, denominators, resumability and output coverage on CPU. Do not use held-out safety behavior to choose settings.
