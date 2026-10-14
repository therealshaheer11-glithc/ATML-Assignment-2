# Assignment coverage and process audit

The final audit found no uncorrected experiment implementation error in the
available notebook, frozen source, and saved evidence. This is an evidence-based
audit conclusion, not a guarantee about every possible model/runtime behavior.

| Requirement | Evidence and result |
| --- | --- |
| Fixed policies, evaluation only | Unchanged supplied SFT base and RLVR/RLAIF adapters; model/adapter revisions and hashes in bindings; actual loading code verifies adapter tensors. No training phase was executed. |
| Fixed evaluation sets | 300 GSM8K and 100 unchanged SVAMP rows per policy; all 1,200 output IDs/order/hash bindings verified. Split files contain IDs and hashes, not raw datasets. |
| Matched generation | Full original prompts, batch one, seed 6304 per policy/domain, 512 tokens, temperature 0.7, top-p 0.9; checkpoint top-k 20 and repetition penalty 1.1 retained. Base weights float16; released adapter loading retained; no policy quantization. |
| Fixed feedback mechanisms | Last designated-final parser with tolerance 1e-9; fixed 3B judge/rubric, NF4/float16, greedy four-token helper, canonical model identity, deterministic A/B swap and cache. |
| In-domain results | Accuracy, format, response mean/population SD, AI win/tie/loss score and three-way verifier agreement for the prescribed policies. |
| Controlled diagnostics | 100 responses, 80 unique pairs, all five category rows; clean-reference row overlaps four contrasts; reason/outcome sensitivities and tie/opposite rates retained. |
| Transfer | Same three policies, unchanged 100-example transfer set; exact/AI metrics, signed GSM-minus-SVAMP change, lengths and failure examples. |
| Qualitative evidence | User-approved first-in-source-order examples: four diagnostic pairs and two transfer failure categories per policy. Repeated cases preserved. |
| Feedback-source discussion | Coverage, noise, exploitability and recorded inference costs in RESULTS.md, with judge ambiguity and outcome-only verifier limits. |
| Required deliverables | Modular source, fixed IDs, small logs, machine-readable records/results and exact README commands. Full combined NeurIPS report remains separate. |

## Recovered procedural errors

The notebook records `KeyboardInterrupt` during post-stop generation checkpoint
counting after 1,000 saved answers. The CPU resume audit verified exactly those
1,000 records, then the resumed generation saved the remaining 200. The final
CPU audit verified all 1,200 generation chains and all 880 judgment chains.

The notebook's first inspection loader split `WRAPPER_SOURCE_SHA256` across two
lines, causing `AttributeError` before its `main()` ran. The recovery cell checked
the uploaded source hash, restored the correct attribute and completed inspection.
The failed historical source remains under `provenance/notebook_sources/`; the
correct supplied helper remains under `source/stage_helpers/`.

Notebook/source differences elsewhere were formatting, printed wording, or
equivalent assignments. Model-execution cells and loaded helper hashes matched
the saved releases. Initial base-package version mismatches were identified
before inference and resolved through the approved isolated 84-package runtime.
The base Colab environment was left intact. The final local test command first
omitted the package-root flag, produced import errors, and was corrected to `-t .`;
all 44 tests then passed. This was an audit-command issue, not an experiment run.

## Limitations that affect interpretation

- The preserved helper uses four tokens despite the config's value of 64. This
  approved mismatch is disclosed; it was not treated as an instructor-announced
  deliberate Task 5 defect. The explicitly announced objective defects concern
  Tasks 1–3. Diagnostic arithmetic/final-answer errors are intentional controls.
- Across 880 events, 735 outputs explicitly say TIE; no tie is a parser fallback.
  Seventy outputs have extra/conflicting labels accepted by the first-label rule.
  Four of five recorded filler preferences arise from mixed `A, B` output.
- Judge EOS metadata was not retained, so four output tokens alone do not prove
  truncation. The unchanged single-prompt judge helper emitted generation-setting
  and attention-mask warnings; their occurrence is not evidence of a new policy
  training or dataset change.
- Correct math without a parseable designated final receives zero. Higher exact
  accuracy is therefore not sufficient evidence of higher reasoning quality.
- The local private bundle contains payloads and checkpoint digests, not full
  RNG-bearing journals or weights. The original CPU audit report/source verifies
  those Drive chains; the final local audit rechecks the available payloads.
- Seed/hardware/settings were matched, but deterministic CUDA algorithms were
  disabled. One sampled run and unmatched domain difficulty do not support a
  statistical significance or general-reasoning claim.
- Cost tables measure model-generate elapsed time and output tokens. They exclude
  setup/loading/Drive overhead and do not claim a measured monetary cost.

## Publication boundary

The repository payload excludes full raw course datasets, model weights, caches,
and the private notebook/bundle. Public records retain score/length/label metadata
and hashes. Full exact-score recomputation requires the privately held bundle;
public-only reproduction verifies aggregates from those published score records.
Historical pending-qualitative flags are unchanged for hash integrity; STATUS.json
and the separate approved selection record describe the completed analysis stage.

GitHub application, staging, commit, push, and LMS submission are separate actions.
Package preparation does not claim that any of them has occurred.
