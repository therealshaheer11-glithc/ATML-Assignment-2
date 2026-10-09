# Task 2 — Chunk 2: continuation code and CPU preparation

**Use a CPU runtime for this chunk.** The GPU remains off while the remaining evaluation and analysis jobs are prepared. This package installs the training code, runs 37 small CPU checks, restores only the necessary Task 2 course files, verifies/reuses the existing pinned Task 1 public model files where available, and saves everything to Drive. It does not start assignment training or load course-model weights into models.

## Run now

1. Download `task2-chunk2.zip` and open `Task2_Chunk2_CPU.ipynb` in Colab.
2. Select **Runtime > Change runtime type > Hardware accelerator: None** and connect to the CPU runtime.
3. Run the notebook. Mount the same Drive and upload `task2-chunk2.zip` when prompted. The notebook uses the already verified backup at:
   `/content/drive/MyDrive/ATML-Assignment-2/task2/runtime_backups/20261008T161047_398047Z/project.zip`.
4. Let CPU preparation finish. It verifies the saved Task 1 policy/RM snapshot before reusing it. If that directory is absent, it downloads exactly the approved public revisions on CPU and saves verified files to Drive. No login/token is requested for these public files.
5. The final output must include `CPU PREPARATION VERIFIED` and a `preparation.json` path. Keep that path and the test summary. Drive is flushed and unmounted at successful completion.

The expected saved-model source is `/content/drive/MyDrive/ATML-Assignment-2/model_snapshots/20261004_002851_903477`. This is a historical pointer to check, not a claim that its current files have already been verified. A checksum mismatch stops preparation rather than substituting other weights.

## What the training code does

- Independently loads the supplied policy midpoint and merged critic for each run. Keeps one frozen 8-bit reward model resident across consecutive runs.
- Generates one training prompt response per update with the release caps/sampler and the approved seed schedule. Uses only the 1,200 training prompts; the held-out pool and cached diagnostic rollouts never enter an optimization step.
- Copies generation tensors out of inference mode; computes/detaches old policy log probabilities, base-reference probabilities, critic values, KL-shaped rewards, GAE advantages and returns once. The missing-EOS penalty is applied once to the terminal reward.
- Normalizes advantages once and keeps returns raw. Runs exactly two PPO epochs using the corrected minimum surrogate and `0.5 × masked value MSE`. Logs raw and weighted value losses separately.
- Uses approved evaluation forward mode with gradients enabled for optimization. Checks both sets of gradients before either optimizer steps; checks parameters and optimizer state afterward. Stops on nonfinite arithmetic.
- Applies and audits **decision L**: only the active trainable critic scalar head is promoted to float32; critic LoRA stays float32 and frozen critic weights stay float16. Restores saved head state only after promotion. See `precision_approval.md` and `.json` for the explicit approval and diagnostic limits.
- Saves prompts, response token IDs, masks, old/reference probabilities, GAE/returns, generation text, termination flags, losses, entropy, clip fraction, pre-clipping actor/critic gradient norms, and actual token counts. Exact categorical entropy is named separately from the course sampled-entropy estimate.
- After each complete rollout update, saves trainable parameters, AdamW moments, RNG state and logs atomically, then copies a hash-verified archive to Drive. Frozen multi-gigabyte model weights are not duplicated in these update checkpoints. Interrupted partial updates are not advertised as completed. Resume is explicit and rejects changed settings or source files.
- Exports the final policy adapter and critic adapter/head. The final standard 20-update policy remains the Task 4 policy; no held-out selection occurs.

## Fixed run table

| Run | Updates | PPO epochs per update | Clip epsilon | KL beta |
|---|---:|---:|---:|---:|
| standard | 20 | 2 | 0.20 | 0.10 |
| clip_005 | 8 | 2 | 0.05 | 0.10 |
| central_8 | 8 | 2 | 0.20 | 0.10 |
| clip_050 | 8 | 2 | 0.50 | 0.10 |
| kl_000 | 8 | 2 | 0.20 | 0.00 |
| kl_020 | 8 | 2 | 0.20 | 0.20 |

The independent `central_8` result is shared between both sweeps as approved in J. Each short fork has a 4,096-response-token allowance; actual EOS-dependent consumption is recorded separately, as approved in K. The CLI accepts these six names, with no arbitrary budget/hyperparameter overrides.

## Timing and recovery

Standard-run wall time measures active continuation sessions after model loading, including generation, scoring, optimization and per-update persistence. Setup/model loading, final adapter export and offline analysis are excluded explicitly. Peak allocated and reserved VRAM are both saved. A hard runtime loss can leave the last timing interval incomplete; the result flags this rather than inventing an exact total. CUDA memory/time are unverified until the real prepared GPU job runs.

Local code tests include a simulated Drive-upload interruption after a completed update. Resume repairs that upload, continues from the committed checkpoint, and matches uninterrupted training exactly on tiny CPU models. This is algorithm/recovery evidence, not a CUDA determinism or full-checkpoint guarantee.

## Files and provenance

`task2_ppo/continuation.py` is the entry point; `training_core.py`, `runtime.py`, `checkpoint.py`, `storage.py`, and `critic_precision.py` supply its implementation. `scripts/task2_cpu_prepare.py` prepares files without loading weights. `setup_chunk2.py` installs without replacing modified files. `LOCAL_VALIDATION.json` and `tests.log` record the local checks; `PACKAGE_SHA256.json` seals this package.

The code reuses the course's generation, model configuration, parameter grouping, masks, GAE, metrics, and corrected PPO objective from commit `1d64ac65acd5e45d1e4e1f415edc80455e21274e`. Public-file preparation adapts the earlier approved Task 1 workflow. New continuation/checkpoint orchestration was written for this task. The original Chunk 1 archive, objective patch, approval A–K and their checksums remain unchanged.

The next chunk supplies held-out evaluation, fixed-cache clipping analysis, evidence aggregation and the bounded GPU launch. No new experimental setting is introduced here, and no experimental results are claimed.
