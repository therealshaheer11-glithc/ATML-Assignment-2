# Task 5 — RLVR versus RLAIF

Frozen-policy evaluation is complete: **1,200 policy answers, 880 comparison events,
865 actual judge calls, and 100 controlled diagnostic responses**. This folder
provides modular source, fixed IDs, execution evidence, quantitative results,
approved qualitative analysis, and CPU reproduction of the saved statistics.

From the repository root, Python 3.10+ and its standard library are sufficient:

```bash
python3 -B task5/verify_package.py
python3 -B task5/verify_package.py --output outputs/task5_recomputed
```

The first command is read-only. The second requires a new output directory.
Both verify all package/source hashes, 1,200 ordered metric records, 880 joined
judgments, 100 diagnostic score records, parser/cache conventions, and nine
byte-identical result files. They load no models and use no GPU.

- [Results and report-section draft](RESULTS.md)
- [Reproduction commands and original A100 workflow](REPRODUCE.md)
- [Assignment coverage, recovered errors and limitations](PA_COMPLIANCE.md)
- [Frozen experiment source](source/frozen_release/README.md)
- [Machine-readable results](results), [record-level metrics](records),
  [fixed source IDs](splits), [execution logs](logs), and [provenance](provenance/README.md)

Public-mode reproduction calculates statistics from the saved score/length
records. The optional private-bundle audit also recomputes exact scores from full
responses and gold finals. Full course datasets, supplied diagnostic response
text, generated answer text, weights, wheel caches, and the raw notebook are
excluded from this publication payload. Short qualitative excerpts and authored
case explanations appear in the report section as required evidence.

The judge's high tie rate and mixed-label outputs are disclosed. The course's
effective four-token limit and first-label parser were preserved under the
approved protocol. No training or evaluation-driven policy tuning was performed.
The full combined PA2 report and submission are separate artifacts.

## Publication and attribution

The experiment package was published at `7e7099d99180ce7e180b09da22fd9ee8fc835566`.
Current repository-wide status is in [PUBLICATION_STATUS.json](../docs/PUBLICATION_STATUS.json).
Course helpers, models, fixed data identities and Codex-assisted additions are
attributed in [source provenance](provenance/README.md). Earlier reports retain
their original pending flags as historical evidence; they do not override current status.
