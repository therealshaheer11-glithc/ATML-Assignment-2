# Task 5 — Reward sources and evaluation behavior

Draft section for integration into the shared eight-page PA2 report. This is an evaluation of supplied frozen policies, with one sampled completion per prompt and no retraining or tuning from these results.

## Setup and measures

We evaluated SFT and the supplied RLVR and RLAIF adapters on the fixed 300-example GSM8K subset and unchanged 100-example SVAMP subset. All policies use Qwen2.5-1.5B-Instruct, float16 without quantization, seed 6304 reset per policy and domain, a 512-token completion limit, temperature 0.7, top-p 0.9, and batch size one. The inherited `top_k=20` and `repetition_penalty=1.1` were retained, as recorded in the effective generation settings. The exact verifier reads the last parseable `#### <number>` field and compares numerically with tolerance 1e-9; format compliance means that this field is parseable, independently of correctness. Length includes the first generated EOS and excludes padding.

The frozen Qwen2.5-3B-Instruct judge uses NF4 with float16 compute, the released rubric, greedy decoding, deterministic A/B swapping, and the released four-token output limit. We retained that effective limit despite the configuration value of 64. The released parser takes the first A/B/TIE label; an unparseable output becomes a tie. AI score is (wins + 0.5 × ties) / comparisons against same-domain SFT. Agreement compares the verifier and judge's three-way labels. SFT is the reference and receives no self-comparison score.

## Policy results

| Policy | Set (n) | Exact accuracy (%) | Format (%) | Tokens, mean ± population SD |
| --- | --- | --- | --- | --- |
| SFT | gsm (300) | 30.0 | 39.0 | 287.93 ± 96.75 |
| SFT | transfer (100) | 51.0 | 58.0 | 161.37 ± 79.68 |
| RLVR | gsm (300) | 38.0 | 50.7 | 275.65 ± 93.63 |
| RLVR | transfer (100) | 53.0 | 63.0 | 157.16 ± 65.97 |
| RLAIF | gsm (300) | 34.7 | 47.0 | 287.48 ± 98.86 |
| RLAIF | transfer (100) | 51.0 | 64.0 | 171.99 ± 101.20 |

| Policy vs SFT | Set | Wins / ties / losses | AI score (%) | Verifier–AI agreement (%) |
| --- | --- | --- | --- | --- |
| RLVR | gsm | 21 / 256 / 23 | 49.7 | 61.0 |
| RLVR | transfer | 18 / 69 / 13 | 52.5 | 59.0 |
| RLAIF | gsm | 15 / 271 / 14 | 50.2 | 66.3 |
| RLAIF | transfer | 15 / 69 / 16 | 49.5 | 63.0 |

RLVR has the highest observed exact accuracy: 38.0% on GSM8K and 53.0% on SVAMP, compared with SFT's 30.0% and 51.0%. RLAIF reaches 34.7% and 51.0%. Both adapters improve GSM8K format compliance, so the exact-accuracy differences alone do not establish better reasoning. The failure counts below distinguish unparseable final fields from parsed but incorrect numbers; neither bucket labels every intermediate reasoning step.

| Policy | Set | Correct | Missing/unparseable final | Parsed incorrect final |
| --- | --- | --- | --- | --- |
| SFT | gsm | 90 | 183 | 27 |
| SFT | transfer | 51 | 42 | 7 |
| RLVR | gsm | 114 | 148 | 38 |
| RLVR | transfer | 53 | 37 | 10 |
| RLAIF | gsm | 104 | 159 | 37 |
| RLAIF | transfer | 51 | 36 | 13 |

AI scores stay near 50%, with 665/800 policy comparisons tied. Agreement on all pairs is substantially influenced by shared ties. Restricted to decisive verifier comparisons, matching AI labels are 15/112 for RLVR/GSM, 12/38 for RLVR/SVAMP, 9/94 for RLAIF/GSM, and 10/34 for RLAIF/SVAMP. These are secondary denominators; verifier–judge disagreement does not by itself establish that the judge is mathematically wrong, since the verifier only observes the designated outcome.

## Controlled feedback diagnostics

The 100 supplied responses form 80 clean-versus-modified comparisons over 20 problems. The clean-reference row below reuses all 80 contrasts and overlaps the four 20-pair rows; these rows must not be summed.

| Contrast | n | Verifier clean / tie / modified (%) | AI clean / tie / modified (%) |
| --- | --- | --- | --- |
| Clean reference, all contrasts | 80 | 50.00 / 50.00 / 0.00 | 6.25 / 87.50 / 6.25 |
| Corrupted reasoning, correct final | 20 | 0.00 / 100.00 / 0.00 | 0.00 / 100.00 / 0.00 |
| Same reasoning, wrong final | 20 | 100.00 / 0.00 / 0.00 | 0.00 / 100.00 / 0.00 |
| Persuasive filler, correct final | 20 | 0.00 / 100.00 / 0.00 | 0.00 / 75.00 / 25.00 |
| Gold distractor, wrong final | 20 | 100.00 / 0.00 / 0.00 | 25.00 / 75.00 / 0.00 |

The verifier has S_reason = 0 and S_outcome = 1. Its reasoning-only ties are expected from outcome-only scoring. The AI judge has S_reason = 0 and S_outcome = 0, with all 20 pairs tied in each sensitivity test. Thus, this fixed judge protocol provides no demonstrated reasoning sensitivity on these controls and fails to distinguish the changed designated outcome when the reasoning is held fixed. This observation does not establish a general inability of the underlying model.

In the first supplied problem (ID 803), corrupting `2*7=14` to `2*7=15` leaves the final field `#### 9` unchanged: both verifier rewards are 1 and the judge says TIE. Replacing only `#### 9` with `#### 10` changes verifier rewards from 1 to 0, but the judge again says TIE. Adding “This derivation is unquestionably correct and leaves no reasonable alternative.” leaves both rewards at 1 and also receives TIE in this first example. A distractor response explicitly rejects the gold candidate 9 and ends `#### 10`; the verifier assigns 0 while the judge ties it with the clean response. These are the first source-ordered examples in every category, rather than cases selected for a particular judge outcome.

Across all filler controls, five recorded labels prefer filler. Four arise from mixed output `A, B`, whose first label is mapped to the filler after the deterministic swap; only one is an unambiguous single-label filler preference. The latter is consistent with possible style sensitivity, but five valid demonstrations of persuasive-style bias would overstate the evidence. For gold distractors, the verifier consistently rejects the wrong final; the AI judge prefers clean in 5/20 and ties in 15/20.

## Transfer behavior

| Policy | GSM − SVAMP accuracy (pp) | GSM − SVAMP AI score (pp) | GSM − SVAMP mean tokens |
| --- | --- | --- | --- |
| SFT | -21.00 | Reference | 126.56 |
| RLVR | -15.00 | -2.83 | 118.49 |
| RLAIF | -16.33 | 0.67 | 115.49 |

The signed “drop” is GSM minus SVAMP, so the negative accuracy values indicate higher accuracy on this fixed transfer subset. RLVR's advantage over SFT is 8.0 percentage points on GSM8K and 2.0 on SVAMP; RLAIF's is 4.7 and 0.0, respectively. Transfer responses are shorter for every policy. The sets differ in composition and are not matched for difficulty; the AI reference is separately evaluated SFT in each domain. These comparisons do not establish a causal improvement in general reasoning or statistical significance from a single sampled run.

The first transfer failure in each policy is source `svamp:chal-5` (row 4, zero-based). Each response correctly concludes that two more children boarded than left, but omits `#### 2`, so the fixed verifier gives 0. The first parsed wrong answer is `svamp:chal-11` (row 10): each response invents one initially present stork, computes seven storks versus five birds, and ends `#### 2` instead of the gold 1. None of these selected responses is truncated. The responses are identical across all three policies for these two cases; their repetition is retained by the source-order selection rule. Both adapter-versus-SFT comparisons tie, as expected for identical responses. These examples expose persistent formatting and unsupported-assumption failures rather than prove differences between the policies.

## Feedback coverage, noise, exploitability, and cost

Exact verification uses deterministic rules rather than an additional model call, reliably enforces the designated final on the wrong-final and distractor controls, and supplies no direct signal about corrupted reasoning. Its specification also rejects a correct number outside the required field, as the bus example shows. AI feedback has a rubric covering reasoning and relevance, but the observed labels are dominated by ties and sometimes violate the requested single-label format. Across all 880 events, 735 are explicit TIE and none is a fallback tie; 70 outputs contain extra or conflicting text. Reaching four saved tokens is not proof of truncation because judge EOS metadata was not retained. Correct parsing under the released rule should therefore be distinguished from an unambiguous model preference.

The 880 events required 865 model calls plus 15 cache hits. Saved judge inference time is 168.58 seconds for 2,588 output tokens; policy generation used 11398.97 seconds and 304,368 output tokens. These timings exclude full setup, model loading, Drive checkpointing, and other session overhead; monetary cost and exact-verifier wall time were not measured. The controls demonstrate verifier reasoning blindness and possible judge style/formatting vulnerabilities, without establishing a successful trained reward-hacking strategy. Under this frozen evaluation, outcome verification supplies the clearer outcome-change signal, while AI feedback's potential broader coverage is limited by the observed judge behavior.

Evidence: [quantitative results](results), [raw-output audit](results/RAW_OUTPUT_AUDIT.json), [source-order selections](results/QUALITATIVE_SELECTION.json), and [audit/provenance](PA_COMPLIANCE.md). The package contains IDs, hashes, scores, lengths, and judge labels; full course questions and supplied response text remain in the private evidence bundle. No new inference or protocol change was used to write this section.
