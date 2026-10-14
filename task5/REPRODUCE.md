# Reproduce Task 5

## Saved-result reproduction on CPU

Run from the repository root:

```bash
python3 -B task5/verify_package.py
python3 -B task5/verify_package.py --output outputs/task5_recomputed
```

The second command writes the original aggregate, four CSV tables, two cost/
sensitivity JSONs, the raw-label audit, and the approved failure-count CSV, plus
a verification receipt. All nine result files must be byte-identical to their
committed counterparts. Output directories must be new. Python 3.10+ is enough;
no dependency installation, model download, or CUDA runtime is needed.

## Full response/score audit with the private evidence bundle

Keep the private bundle outside the repository, then run:

```bash
python3 -B task5/verify_package.py --private-bundle /path/to/Task5_CPU_Inspection_Private.zip
```

The expected bundle SHA-256 is
`8bace113620a0080808a50df9b9ca296e881cb7ac430c118b6c73f339c030067`.
This additionally verifies its 44 manifest entries, recomputes all 1,200 policy
exact scores and 100 diagnostic scores with the unchanged released verifier,
checks full response/comparison joins and hash-based A/B orientation, and checks
saved token lengths/EOS accounting. It does not reconstruct the complete
RNG-bearing Drive journals. Their full-chain verification was performed by the
preserved CPU inspection, whose source and original report are included.

## Original model execution and resume

The immutable `source/frozen_release/` contains the exact modular evaluation
code used in Colab; `source/a100_setup/` contains the original approved setup.
`source/stage_helpers/` preserves inventory, checks, resume, export and release
wrappers. These historical sources retain their original manifests and bindings.
The notebook is execution history; the modular Python release is the pipeline.

Original execution prerequisites are the privately retained Google Drive project
`/content/drive/MyDrive/ATML-Assignment-2`, its fixed course datasets/adapters,
model snapshots, verified 90-wheel cache, prior CPU diagnostic/readiness records,
and the protocol-bound execution/environment approvals. Exact paths, revisions
and checksums are in `source/frozen_release/bindings.json`; the original runtime
contract is represented in `evidence/BINDINGS.json`. These prerequisites are not
created or fabricated by the public package. Historical preflight is intentionally
specific to this approved project; an empty machine cannot regenerate its prior
approval and checkpoint records from public result metadata.

On the original mounted CPU Colab runtime, from the frozen release directory:

```bash
python -B -m task5_run.preflight
```

On the approved A100 runtime, run `prepare_a100.py` from the preserved setup
directory to restore the isolated environment. It requires the original verified
Drive prerequisites and installs the approved cached wheels offline. It runs the
44 core tests plus tokenizer/context and small CUDA/NF4 checks before creating
the environment contract. The runtime used Python 3.13.15, Torch 2.11.0+cu130,
CUDA 13.0, driver 580.82.07 and NVIDIA A100-SXM4-40GB; all 84 package versions
are in `source/frozen_release/package_versions.json`.

Once the original contracts pass, from the original frozen release directory:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /content/task5_isolated_runtime_v1/bin/python -B -m task5_run.run generate --project /content/drive/MyDrive/ATML-Assignment-2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /content/task5_isolated_runtime_v1/bin/python -B -m task5_run.run judge --project /content/drive/MyDrive/ATML-Assignment-2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /content/task5_isolated_runtime_v1/bin/python -B -m task5_run.run aggregate --project /content/drive/MyDrive/ATML-Assignment-2
```

These are separately executed phases. They use the already approved private
project and resume committed results; they do not request new experiments or
authorize a paid rerun. Never erase the saved results to force regeneration.
Generation resets seed 6304 per policy/domain and restores the last saved RNG
state after loading models. New GPU execution must pass the hardware/settings
bindings. Deterministic algorithms were disabled in the recorded runtime, so
the package promises exact CPU reproduction of saved statistics, not guaranteed
bit-identical fresh GPU completions on different environments.

## Test the unchanged evaluation implementation

In an environment with the recorded ML dependencies, from
`task5/source/frozen_release/`:

```bash
python -B -m unittest discover -s task5_run -t . -p 'test_*.py' -v
```

The `-t .` flag is required for package-relative test imports. The 44 tests cover
immutable chains, interruption/resume/RNG, schedules, verifier cases, full prompts,
EOS length accounting, inherited generation defaults, judge parsing/swaps/cache,
and aggregation. They are synthetic checks; the saved A100 run supplies actual
model-execution evidence. The final package audit does not rerun model inference.
