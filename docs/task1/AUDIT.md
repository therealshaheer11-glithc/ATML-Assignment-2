# Final Task 1 audit

**Status: saved-evidence audit passed; Task 1 package prepared for manual Git publication.** No new GPU experiment or access to the user's Git repository was needed. The missing raw records and three configuration/provenance files were supplied in `Task1_final_audit_20261004_221141_263984.zip`; that evidence gap is resolved.

## Scope and findings

The original review ZIP, chronological Colab Python export, final evidence ZIP, PA manual, user approvals and user-supplied TA clarifications were inspected. The PA's Task 1 formulas and required-evidence list were checked against the implementation and saved results. The final export passed ZIP integrity checks and all **855 payload SHA-256 checks**.

The independent CPU audit verified:

- All five full training budgets, 4,800 processed training pairs across conditions, exact short-subset membership/order, accumulation tails, finite completed updates, and pair-weighted training diagnostics.
- Identical fresh initialization, frozen-base hashes before/after each successful run, and saved fresh-process reload proofs. The standard retry used the same first 304 pair order as the failed attempt.
- All **1,992 raw held-out pair records**: FP32 reference-adjusted margins, strict accuracy including ties in the denominator, scalar DPO loss and aggregate/stratum metrics.
- All **1,830 saved generated answers and reward scores**: ordered common prompts, identical decoding settings, per-token differences, EOS and ceiling flags, token-weighted sampled KL, mean/population-SD reward and length, and all 30 word-limit cases.
- All **23 executed evaluation source/configuration/provenance files**, with identical snapshots across the six evaluation conditions. All training source snapshots match the final source apart from the documented failed-run scale change.
- The reward correction changed only the declared rotary-position theta in memory. Every saved GPU loading record reports matching actual frequencies, no missing/unexpected/mismatched weights, and a frozen reward model. No recorded reward input was truncated.

The audit found **no additional calculation or protocol defect requiring another experiment**. It verifies saved artifacts and their internal consistency; it is not a new replay of the model computations. Binary model/optimizer files were deliberately omitted from the export. Their hashes and historical reload/backup proofs are retained; the actual files remain on Drive.

## Required-evidence coverage

| PA requirement | Location |
|---|---|
| Correct DPO objective, response-only scoring, fixed reference | Executed source snapshots; objective/scoring tests; raw-score audit |
| Standard one-epoch run | `training/standard` records and summary table |
| Three matched short beta runs | `training/beta_*`, full common evaluation and summary table |
| Loss, accuracy, sampled KL, reward, token length and dispersion | [Results](RESULTS.md), `analysis/summary.csv` |
| Standard vs balanced per-stratum accuracy | `analysis/length_strata.csv` and results table |
| Common-prompt generation length and word compliance | `analysis/word_limits.csv`, saved metrics and results table |
| Reward/quality disagreement | Approved wl05 illustration |
| Length/instruction behavior | Approved wl04 illustration |
| Scripts independent of hidden notebook state | Original train/evaluate CLI; new preparation/dispatch/analysis interface |
| Traceable settings, deviations and attribution | [Protocol](PROTOCOL.md), [evidence](EVIDENCE.md), source attribution |
| Main eight-page report | Evidence ready; integrate into the combined report before final submission |

## Publication cleanup after the experiments

The original notebook is preserved as `archive/atml_pa2_task1.py.txt`; it is a chronological record with old settings, failed attempts and launch cells, not a clean executable pipeline. The exact exported project and per-run source snapshots are retained unchanged under `results/task1/evidence`.

Two unused starter placeholders were replaced in the public working code: `ablate_beta.py` now dispatches the existing fixed named conditions; `analyze_length.py` calls the saved-results analysis. **Neither new wrapper produced the historical results.** The experiments used `train.py` and `evaluate.py` directly. Added `prepare_assets.py` makes the original pinned preparation portable; added `analyze_results.py` independently verifies evidence and emits tables. Other additions are documentation, manifests, dependency constraints and focused packaging tests.

The objective, encoder, trainer, runtime, model/reference/reward loaders, evaluator and experimental configurations have not been altered during publication cleanup. No model setting, data subset, generation, metric, checkpoint selection or qualitative choice was changed. Tasks 2–5 remain explicitly unfinished.

The final checks comprised the saved-record audit, four focused publication tests, Python syntax checks, offline command plans, internal documentation links, file inclusion/exclusion review and archive integrity. GPU training, model downloads and the historical test suites were not needlessly repeated.

## Limits that must accompany the results

The 256-token output ceiling affects 130–139 of the 300 general responses per model; it is separate from zero input truncation at 4,096. The signed sampled-KL diagnostic uses raw scores on responses sampled with decoding transformations. One seed and ten word-limit prompts do not support broad significance or population claims. Balancing improved aggregate stratified accuracy but slightly worsened preferred-longer accuracy. The overflow explanation remains a hypothesis. Higher reward is not proof of correctness.

The full assignment is not yet complete: the student must manually publish/review the repository, preserve the standard adapter for Task 4, complete the later tasks, and integrate the required Task 1 evidence into the main report.

## Evidence packaging details

The historical `evidence/project/.gitignore` is stored as `.gitignore.archived.txt`, with identical bytes and hash. This prevents its old ignore rules from silently excluding evidence during publication. The audit maps that one archived filename back to the unchanged export manifest. All experimental source and result paths/bytes remain unchanged. Six redundant general-response copies from the first review bundle are omitted from `original_summary`; complete records remain in the canonical final evidence. Its local README explains the historical receipt.
