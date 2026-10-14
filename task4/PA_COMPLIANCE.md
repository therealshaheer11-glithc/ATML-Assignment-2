# Task 4 assignment coverage

The supplied PA's Task 4 requirements (pages 12–14) and deliverables (page 18) were reviewed. Original planning remains under `provenance/planning/`; it describes approvals and pending states at that time.

| Requirement | Published evidence |
|---|---|
| SFT plus three fixed standard endpoints, no safety retuning | Input identity and evaluation contracts; 94/20/20 completed updates |
| Same fixed prompts/order, deterministic output and common cap | All 450 IDs, frozen execution contract and eight full ordered JSONL exports |
| Original categorical judge/prompt/parser | Preserved source release, judging source, runtime checks and raw judge outputs |
| Required safety rates, ambiguity, lengths and categories | Four recomputed metric JSONs; four summary CSVs |
| Same 60 audit IDs per policy, all five labels retained | Audit ID contract, packet/mapping, 240 labels, agreement and full confusion matrices |
| Harmful compliance/refusal/over-refusal examples and attribution | RESULTS.md: actual response IDs and policy versus judge analysis; no genuine harmful-compliance case identified in audited subset |
| Connect preference reward to calibration | RESULTS.md and hashed existing Tasks 1–3 score sources; descriptive limitations stated |
| Scripts, settings, IDs, logs, small results, exact CPU commands | README, REPRODUCE, sealed executable releases, original stage launchers and exported contracts/logs |

The PA's original audit requested human labels. The user supplied a TA message permitting AI labeling; the actual AI origin is retained. These labels do not establish independent human validation. The package does not claim otherwise.

The deliberate core objective defects are in Tasks 1–3. Task 4 supplies unfinished generation/judging/aggregation scaffolds. Its completed orchestration preserves the course scoring semantics, deterministic decoding, ambiguous denominators and standard endpoint selection. The unfinished root starter is archived unchanged, not presented as the active implementation.

Original immutable JSON reports retain historical pending-human status where it was true at export time. Current audit evidence and this documentation supersede that status without rewriting history. Scientific settings, labels, raw outputs and earlier task artifacts are unchanged. Tasks 1–5 are published at `7e7099d99180ce7e180b09da22fd9ee8fc835566`. The combined report and LMS submission remain separate; current status is in `docs/PUBLICATION_STATUS.json`.
