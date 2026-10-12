# Reproduction

## Recompute the completed experiment on CPU

From the repository root:

```bash
python3 scripts/verify_repository.py
python3 task4/reproduce.py --output outputs/task4_recomputed
```

Use a fresh output directory. Verification checks the entire frozen publication, and recomputes each of four policy summaries and four 5×5 confusion matrices from 1,800 saved response–judgment pairs. It also compares all six exported CSV tables byte for byte. This does not rerun inference or regenerate audit labels.

The fixed decoding settings are seed 6304, batch four, prompt cap 256, response cap 256 and deterministic generation. The original judge uses deterministic decoding, a 64-token output cap and NF4 with float16 compute. All policies use the same 450 prompt IDs/order (250 SAFE, 200 UNSAFE). Standard endpoints are final one-epoch DPO (94 updates), final 20-update PPO, and final 20-update GRPO; SFT has no adapter. Contracts preserve model revisions, hashes, package versions, inherited defaults and precise endpoint identities. Confidence never filters or weights rates; ambiguous pairs stay in the denominators; token-length dispersion is population SD.

## Inspect the original runnable source

```bash
python3 task4/restore_releases.py --destination outputs/task4_original_releases
```

This verifies and extracts all 19 immutable source releases into a fresh directory, without executing them. Their original entry points, manifests, tests, requirements and stage READMEs are preserved. The generation release contains `generate_responses.py`, `generate_policy.py`, `generation_io.py` and runtime checks. The exported orchestration implements resumable units, aggregation and helper calls; the judging execution directory preserves `JUDGING_SOURCE.py`. These are Python scripts rather than a monolithic notebook pipeline.

`provenance/ORIGINAL_HELPERS.json` maps the historical filenames to their canonical paths. For example, the original `Task4_A100_Judging.py` maps to the executed `JUDGING_SOURCE.py`; CPU summary/audit sources likewise reuse their exact exported copies. The launchers are Colab cells that ask for the matching sealed ZIP or Python helper and enforce its checksum. Execute one stage at a time and check its returned report before continuing.

## Original model execution in Colab

Model execution requires the earlier Tasks 1–3 Drive artifacts, the course assets and the fixed model snapshots. Codespaces is for CPU verification and Git integration. The staged original workflow was:

1. CPU: input preflight; environment diagnostics/recovery; model inventory, judge download, adapter/tokenizer inputs, orchestration validation, isolation preflight and wheel cache.
2. A100: environment and isolated setup; base repair/rechecks when their reports require them; ordinary session restart; post-restart validation; actual model-loading validation; deterministic generation.
3. After the break: CPU resume check, A100 preflight, isolated restoration/Jedi repair, ordinary restart, post-restart resume and judge loading. These are historical recovery branches with exact recorded preconditions.
4. A100: sequential judging of saved responses with atomic progress. CPU: judgment verification, summaries, blind audit packet, import of the frozen permitted audit labels, comparison and final evidence export.

The stage index below links each original launcher or script. Read its contemporaneous README and required state; do not run repairs merely because they appear in the history. The launchers can be pasted as individual Colab cells, or uploaded and invoked one at a time with `runpy.run_path(filename, run_name="__main__")`. Their sealed guards deliberately stop on changed inputs or environment.

Fresh GPU reproduction was not attempted during packaging. Historical scripts are bound to their recorded Drive/runtime state; a new unrelated runtime may require reviewed preparation or recovery before they will proceed. This publication validates the complete saved experiment and preserves execution source; it does not claim that an unprepared Codespace can reproduce GPU inference. Never change hashes or fixed settings just to bypass a guard.

## Source stage index

- [Task4_A100_Judge_Load_Resume.py](provenance/launchers/Task4_A100_Judge_Load_Resume.py)
- [Task4_A100_Judge_Load_Resume_Launcher.py](provenance/launchers/Task4_A100_Judge_Load_Resume_Launcher.py)
- [Task4_A100_Judging.py](evidence/execution/judging/JUDGING_SOURCE.py)
- [Task4_A100_Judging_Launcher.py](provenance/launchers/Task4_A100_Judging_Launcher.py)
- [Task4_A100_Post_Restart_Resume.py](provenance/launchers/Task4_A100_Post_Restart_Resume.py)
- [Task4_A100_Post_Restart_Resume_Launcher.py](provenance/launchers/Task4_A100_Post_Restart_Resume_Launcher.py)
- [Task4_A100_Repair_Jedi.py](provenance/launchers/Task4_A100_Repair_Jedi.py)
- [Task4_A100_Repair_Jedi_Launcher.py](provenance/launchers/Task4_A100_Repair_Jedi_Launcher.py)
- [Task4_A100_Restore_Isolated.py](provenance/launchers/Task4_A100_Restore_Isolated.py)
- [Task4_A100_Restore_Isolated_Launcher.py](provenance/launchers/Task4_A100_Restore_Isolated_Launcher.py)
- [Task4_A100_Resume_Preflight.py](provenance/launchers/Task4_A100_Resume_Preflight.py)
- [Task4_A100_Resume_Preflight_Launcher.py](provenance/launchers/Task4_A100_Resume_Preflight_Launcher.py)
- [Task4_Block10_A100_Isolated_Environment.py](provenance/launchers/Task4_Block10_A100_Isolated_Environment.py)
- [Task4_Block10b_A100_Isolated_Recheck.py](provenance/launchers/Task4_Block10b_A100_Isolated_Recheck.py)
- [Task4_Block11_A100_Base_Repair.py](provenance/launchers/Task4_Block11_A100_Base_Repair.py)
- [Task4_Block11b_A100_Base_Layout_Diagnostic.py](provenance/launchers/Task4_Block11b_A100_Base_Layout_Diagnostic.py)
- [Task4_Block11c_A100_Base_Repair_Recheck.py](provenance/launchers/Task4_Block11c_A100_Base_Repair_Recheck.py)
- [Task4_Block12_A100_Post_Restart.py](provenance/launchers/Task4_Block12_A100_Post_Restart.py)
- [Task4_Block13_A100_Model_Loading.py](provenance/launchers/Task4_Block13_A100_Model_Loading.py)
- [Task4_Block14_A100_Generation.py](provenance/launchers/Task4_Block14_A100_Generation.py)
- [Task4_Block1_CPU.py](provenance/launchers/Task4_Block1_CPU.py)
- [Task4_Block2_CPU.py](provenance/launchers/Task4_Block2_CPU.py)
- [Task4_Block2_Diagnostic_CPU.py](provenance/launchers/Task4_Block2_Diagnostic_CPU.py)
- [Task4_Block2_Recovery_CPU.py](provenance/launchers/Task4_Block2_Recovery_CPU.py)
- [Task4_Block3_CPU_Model_Inventory.py](provenance/launchers/Task4_Block3_CPU_Model_Inventory.py)
- [Task4_Block4_CPU_Judge_Download.py](provenance/launchers/Task4_Block4_CPU_Judge_Download.py)
- [Task4_Block5_CPU_Model_Inputs.py](provenance/launchers/Task4_Block5_CPU_Model_Inputs.py)
- [Task4_Block6_CPU_Orchestration.py](provenance/launchers/Task4_Block6_CPU_Orchestration.py)
- [Task4_Block7_A100_Environment.py](provenance/launchers/Task4_Block7_A100_Environment.py)
- [Task4_Block8_CPU_Isolation_Preflight.py](provenance/launchers/Task4_Block8_CPU_Isolation_Preflight.py)
- [Task4_Block9_CPU_Wheel_Cache.py](provenance/launchers/Task4_Block9_CPU_Wheel_Cache.py)
- [Task4_CPU_Audit_Analysis.py](evidence/audit/AUDIT_ANALYSIS_SOURCE.py)
- [Task4_CPU_Audit_Analysis_Launcher.py](provenance/launchers/Task4_CPU_Audit_Analysis_Launcher.py)
- [Task4_CPU_Audit_Form.py](evidence/audit_preparation/AUDIT_PREPARATION_SOURCE.py)
- [Task4_CPU_Audit_Form_Launcher.py](provenance/launchers/Task4_CPU_Audit_Form_Launcher.py)
- [Task4_CPU_Summaries.py](evidence/summaries/AGGREGATION_SOURCE.py)
- [Task4_CPU_Summaries_Launcher.py](provenance/launchers/Task4_CPU_Summaries_Launcher.py)
- [Task4_Resume_Check.py](provenance/launchers/Task4_Resume_Check.py)
- [Task4_Resume_Check_Launcher.py](provenance/launchers/Task4_Resume_Check_Launcher.py)
- [Task4_Save_Before_Break.py](provenance/launchers/Task4_Save_Before_Break.py)
- [Task4_Verify_Judgments.py](provenance/launchers/Task4_Verify_Judgments.py)
