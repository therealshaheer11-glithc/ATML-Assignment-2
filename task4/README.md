# Task 4 — safety calibration and judge audit

Completed saved-record evaluation of SFT, standard DPO, standard PPO and standard GRPO: **1,800 responses, 1,800 judgments and 240 audit labels**. No additional training, generation or judging is needed to reproduce the published statistics.

## Start here

From the repository root, Python 3.10+ with no third-party packages:

```bash
python3 task4/verify_package.py
python3 task4/reproduce.py --output outputs/task4_recomputed
```

The first command is read-only. The second requires a fresh output directory and writes four metric JSONs, four agreement JSONs, six byte-identical CSV tables and a verification receipt. Both validate original export hashes, all 450 ordered responses per policy, token lengths, response/judgment joins, the fixed 60 audit IDs and the frozen labels. Neither loads models or uses a GPU.

- [Results and qualitative interpretation](RESULTS.md)
- [CPU reproduction and original Colab workflow](REPRODUCE.md)
- [Assignment coverage and limitations](PA_COMPLIANCE.md)
- [Original summaries](evidence/summaries) and [audit comparisons](evidence/audit)
- [All response/judgment records](evidence/experiment/evaluation_v1/exports)
- [Frozen settings, IDs and endpoint contracts](evidence/experiment)
- [Executed orchestration source](evidence/source/orchestration), [generation logs](evidence/execution/generation) and [judging source/logs](evidence/execution/judging)
- [Historical source releases and stage helpers](provenance/README.md)

## Evidence boundaries

The 160 exported payloads remain under `evidence/`. Their original manifest is unchanged; exact before/after hashes for publication wording and audit-description revisions are recorded in `../manifests/WORDING_REVISIONS.json`. Four worker receipt paths are renamed to avoid case-only collisions on macOS; `provenance/EXPORT_PATH_MAP.json` maps original names to package paths. Response text, labels, settings, scoring code, numerical results and sealed source releases are unchanged. Individual batch envelopes, model files, installed wheel caches and downloaded raw CSV datasets remain on Drive. Prompts and responses are included as required evaluation records; no separate raw benchmark dataset is bundled.

The additional audit applies the original five-label rubric to the fixed 60-prompt subset for each policy using an automated procedure. Decision notes record the reasoning for each label, and exact duplicate prompt-response pairs share a label. The analysis uses the term “audit labels.”

The original AI judge is not ground truth. Its zero reported over-refusal contrasts with 44 over-refusals in the audit. See the results for denominators, ambiguity, category counts and limitations. The combined assignment report and Task 5 remain separate work.

## Attribution

The PA's released policy loading/generation helpers, categorical judge prompt/parser and audit-ID procedure were preserved. Their original references are included in the preflight source release and bound by the execution contracts. Qwen models and XSTest are identified in the frozen input/model contracts. Earlier standard adapters come from the verified Tasks 1–3 experiments. Task 4 orchestration, recovery helpers, packaging, analysis and documentation were developed with Codex assistance.
