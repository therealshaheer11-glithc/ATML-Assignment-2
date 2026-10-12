# Decisions and approval boundaries

Approved by the user on 11 October 2026: “Sure I approve” followed by “Yes I approve but document everything”. The approved recommendations are reproduced in `approval.json`; these are Task 4 approvals, separate from historical Task 1-3 approvals.

| Decision | Reason and impact | Status |
|---|---|---|
| Separate Task 4 script pipeline, preserved course helpers, Colab launch cells | Avoid hidden notebook state and accidental changes from earlier task-specific helpers | Approved recommendation 1 |
| Batch 4, sequential policies; preserve precision/quantization | Retain released behavior and keep memory requirements controlled | Approved recommendation 2 |
| Approved Drive root; save each generation batch/judgment; checksummed source/data/model contract | Survive Colab interruptions and prevent mixing conditions or stale records | Approved recommendation 3 |
| Mean plus population SD, exact all-60 agreement including ambiguity, counts plus percentages | Consistent denominators and transparent audit reliability | Approved recommendation 4 |
| Exactly one executable block followed by returned output | User controls staged Colab execution and progress is based on verified outputs | Direct user requirement |
| Human labels made without AI-label exposure | Manual audit remains independent of the AI evaluation instrument | Assignment requirement |
| Document all choices, checks, changes and actual progress | Reproducibility and reviewability | Direct user requirement |

No Task 3 session budget or automatic disconnect is adopted. No training optimizer, scheduler, clipping, reward normalization, AMP, safety retuning, alternate judge, parser change, retry policy, or checkpoint selection is introduced. If an unexpected condition requires a consequential change, explain it and seek explicit approval first.

The first block's standard-library-only verifier is an implementation of the approved input audit. It uses recorded paths, preserves prior-task files, saves immutable source/report files with readback, and stops for missing/mismatched inputs. It does not silently search for or substitute another adapter, repair a receipt, or fetch replacements. File layout, UTC attempt timestamps and checksums are bookkeeping supporting the approved persistence approach, not additional experimental settings.

Approval of this plan is not a fabricated verification result. Colab, Drive and model checks remain pending until their actual output is returned. Future stages will extend documentation with exact executed commands, versions, artifact hashes, tests, counts and failure/resume records.
