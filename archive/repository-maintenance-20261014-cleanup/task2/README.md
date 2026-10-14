# ATML PA2 — Task 2 PPO

**Technical audit passed.** All quantitative evidence and the restored code have passed the checks in `AUDIT.md`. The user explicitly approved both qualitative examples on 9 October 2026; their approval and the assistant-assisted review method are recorded. No additional GPU work is needed.

This Task 2 addition belongs in the existing PA2 repository as a single `task2/` directory. It does not replace Task 1 files or shared helpers in that repository. The complete Task 2 implementation is in `source/task2_ppo/`; older starter scaffolds elsewhere in the PA2 repository are not the implementation used for these experiments.

## Contents

- `source/`: exact executed PPO implementation, its pinned course helpers/configuration, tests and approval/install manifests. Training and evaluation files are preserved byte for byte.
- `evidence/original/`: all 14 uploaded result files and their original checksum manifest, unchanged. Its blank worksheet/incomplete qualitative status describe the earlier export. The approved examples and current evidence inventory are in `evidence/reviewed/`.
- `evidence/reviewed/`: independently checked tables, all training trajectories/optimizer steps, 1,400 per-prompt metrics, the Task 4 adapter handoff and the separately corrected clipping table.
- `evidence/raw_records_verified.zip`: all original experimental JSON records and metadata, with duplicate public tokenizer copies omitted and every omission documented. No weights are included.
- `PROTOCOL_AND_APPROVALS.md`: assignment-fixed bounds, approved discretionary choices A–L, precision exception, resource limits and resume compatibility repair.
- `QUALITATIVE_EXAMPLES.md`: approved reward/quality agreement and disagreement comparisons based on explicit instruction compliance.
- `provenance/`: sealed code releases, course asset manifest, source hashes, exact upstream Git bundle, compatibility fix and CPU test logs.
- `notebooks/`: historical staged setup notebooks, corrected GPU resume launcher, CPU result collection and final audit export. No GPU rerun is needed for this audit.
- `analysis/`: CPU-only verification and presentation correction. These files are post-run analysis; they did not affect training or checkpoint selection.

## Read or verify without GPU

Inspect the code directly in `source/`. Recheck the uploaded export with:

```bash
python3 task2/analysis/audit_export.py \
  --original task2/evidence/original \
  --output task2/.runtime/export-check
```

Extract all underlying records with `python3 task2/analysis/unpack_records.py`. Rebuild the figures on CPU with `python task2/analysis/rebuild_figures.py`; the recorded Python dependencies are required.

Recreate the exact original course checkout plus the approved Task 2 overlay, offline:

```bash
python3 task2/restore_source.py
```

This creates `task2/.runtime/course` and leaves its Git HEAD at the original course commit. That is deliberate: the historical preflight checks the original course commit and the independently hashed applied changes. Committing the copied code into your public PA2 repository does not require weakening those source guards. Run experiments or source tests in the isolated checkout, not directly inside `source/`.

With the approved Python environment installed:

```bash
cd task2/.runtime/course
python -m unittest discover -s tests -p 'test_task2_*.py' -v
cd ../../..
python -m unittest discover -s task2/analysis/tests -p 'test_resume_portable.py' -v
```

Core tests: 50. Resume compatibility tests: 7. The latter use portable import paths in `analysis/tests/`; their historical originals remain in `provenance/resume-fix/`. Tests use tiny CPU models and mock persistence; no full course model is trained during testing.

## Reproduction and checkpoints

The staged Colab notebooks are the executed preparation and launch procedure. Chunk 1–3 archives remain in `provenance/releases/`. The corrected GPU launcher is **Task2_GPU_Resume.ipynb**; use it for already prepared Drive experiments. Do not rerun training to resolve this packaging audit.

The notebooks deliberately reference the user's verified Drive receipts and checkpoint locations. A new user must obtain the pinned public assets/models and create their own valid preparation receipts; an existing receipt must never be fabricated. Full frozen weights and training checkpoints remain outside Git, as the assignment requires.

The standard **20-update** policy is the Task 4 PPO candidate. Its existing Drive archive is `ATML-Assignment-2/task2/experiment_v1/training/standard/final.zip`; use the `policy/` adapter within that verified archive. Do not substitute whichever ablation has the highest held-out reward. Optimizer resume uses the verified per-update checkpoint and promotes the scalar critic head before restoring its FP32 state.

## Metric and interpretation notes

- Primary held-out means weight each of the 200 prompts equally; dispersion uses population standard deviation. Token-weighted fields have separate names.
- Learned reward is the raw reward-model score. Effective reward subtracts the released missing-EOS penalty and is reported separately.
- Reference KL is the released signed sampled log-probability difference. It is not an exact nonnegative KL, and it is evaluated on samples drawn using the released decoding transforms.
- Full categorical entropy and the course sampled entropy estimate have separate fields.
- Short forks use eight updates, one prompt per update, two PPO epochs, and an allowance of 4,096 response tokens. Natural EOS termination produces unequal actual token counts. This is explicitly approved decision K; it is not a claim of equal realized token counts. Exact realized-token matching would require clarification of the manual's budget language before changing the experiment.
- Prompt truncation and response caps are retained from the release. Some long prompts lose later instructions. The reward scorer independently truncates the full prompt-plus-response to its released cap. These are interpretation limits, not post-hoc changes to the experiment.
- The cached diagnostic uses the same 32 rows and 8,814 valid tokens for every epsilon, with the standard final policy as the fixed new-policy candidate. The original cache lacks raw token IDs; count/text round-trip checks cannot prove unique original tokenization.
- Training clipping was inactive in every recorded PPO step, including epsilon 0.05. The cache shows clipping geometry, but trained-fork differences cannot be attributed to active clipping.
- Single-seed, small-budget differences and selected qualitative cases do not establish a general reward/quality improvement or safety. Task 4 remains separate.

## Attribution

Materially reused course code: [AbDu11aHHH/ATML-PA2-LLM-PostTraining](https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining), commit `1d64ac65acd5e45d1e4e1f415edc80455e21274e`. The pinned upstream Git bundle preserves its history. The PPO objective was corrected from the deliberate `maximum` defect to the manual's `minimum`; the surrounding helpers, configuration, masks and GAE are course-derived. The continuation, strict persistence, ablation orchestration, diagnostics, tests and audit tooling were completed with Codex assistance. The student is responsible for understanding the submitted code.

Assets: `AbDu11aHHH/ATML-PA2-assets` revision `0b350481fb03f5525a35bcdec4131bd4fe487f98`. Policy/tokenizer: `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Reward model: `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback` revision `f98759a4a1ccdc47a3136748b6bfea9a32ba8fcc`. Their weights and raw dataset files are not redistributed here. Prompt/response text is retained as experimental output evidence.
