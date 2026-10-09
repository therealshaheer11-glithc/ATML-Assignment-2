# Task 2 — remaining code and bounded GPU sessions

This completes the remaining coding chunks together: frozen evaluation, fixed-cache clipping analysis, CPU aggregation, and the resumable GPU launcher. No full course model was loaded and no experimental training result was produced locally.

## Run order

1. Keep Colab on **CPU / Hardware accelerator: None**. Open `Task2_Chunk3_CPU.ipynb`, run its one code cell, and upload `task2-chunk3.zip`. It finds the verified Chunk 2 receipt on Drive, restores the code/assets if needed, checks the sealed Chunk 2 files, installs the remaining code, runs the CPU tests and all 32 real cache reconstructions, then saves a final project snapshot. Wait for **GPU JOBS READY** and **Drive flushed**.
2. Open `Task2_GPU_Run.ipynb`, select **A100**, and run its one code cell. It requires the saved GPU-ready receipt; there is no code upload or model download. It copies verified weights from Drive to local disk, runs the prepared jobs, and saves a session log. It disconnects automatically only after the model process stops, persistence is verified, and Drive flush succeeds. If automatic disconnection is unavailable, the notebook prints a manual disconnect instruction after saving.
3. If the session prints **SESSION_LIMIT**, rerun the same GPU notebook in a new A100 runtime. Completed updates and prompts resume without repetition. Do not rerun CPU preparation. If it prints **FAILED**, read the saved error before retrying. If saving or flushing fails, the runtime stays available for recovery.
4. After **GPU_WORK_COMPLETE**, use `Task2_Results_CPU.ipynb` on CPU. It makes verified tables, plots and a qualitative-review worksheet on Drive. The worksheet needs actual reasoned assessments; blank fields do not count as agreement/disagreement evidence.

The CPU/GPU/results notebooks contain their launchers directly. Keep the ZIP only for step 1. Latest matching receipts are selected automatically; a path override is available at the top of each launcher when needed. Keep one experiment active at a time.

## Explicit resource approvals

The user approved **120 minutes** and **Approve automatic disconnection** on 2026-10-08. Exact replies and shutdown conditions are recorded in `resource_approval.json` and installed as `docs/task2_resource_approval.json`.

The 120-minute scheduling limit includes setup from the start of the GPU launcher. It checks before the next rollout or held-out/cache prompt. A current work unit plus required copying and flushing can extend beyond 120 minutes. Every required update/prompt remains scheduled for resumed sessions. This is not a claim that the whole assignment fits in two hours. Full-model runtime and VRAM remain unmeasured until the actual GPU run; session logs record measured duration and standard-run peak allocated/reserved memory.

Google Colab's supported `google.colab.runtime.unassign()` is called only after verified persistence and successful `drive.flush_and_unmount()`. API reference: https://raw.githubusercontent.com/googlecolab/colabtools/main/google/colab/runtime.py

## Fixed experiments and approved choices

The original manual/release bounds and approved A–L decisions remain in force. Six independent midpoint starts: standard 20 updates; clip_005, central_8, clip_050, kl_000, kl_020 each 8. Two PPO epochs per update; 60 total rollouts and 120 actor/critic optimizer steps each. The independent central_8 fork is reused in both sweeps. No ablation is selected for Task 4: the standard final policy is the Task 4 endpoint.

All six endpoints are frozen before the midpoint and six finals are evaluated on every one of the 200 held-out prompts, with matched per-prompt seeds and the required 768-response-token cap. Raw RM reward is primary; EOS-penalized reward, termination, truncation, length mean/std, sampled reference log-probability difference, sampled entropy and categorical entropy are separate. Prompt-weighted means and token-weighted KL/entropy are named distinctly. Signed sampled KL can be negative.

The original deliberate PPO `maximum` defect is corrected to `minimum` in the sealed Chunk 1 implementation. The original manifest's stale self-entry is documented without changing the manifest. The approved reward RoPE compatibility translation changes the loaded configuration only, preserving the pinned weight files. The active critic scalar head uses explicitly approved float32 parameters and optimizer moments (L); no reinitialization or optimizer/loss/budget change. Frozen backbones and original head copy stay float16. Any nonfinite arithmetic stops the job.

The pinned policy's `generation_config.json` also supplies **top_k=20**, **repetition_penalty=1.1**, and inherited **use_cache=True** for generation. The released helper preserves those defaults while overriding temperature=.7, top_p=.9, sampling, canonical EOS/PAD and token cap. They are recorded and guarded, not newly selected tuning options. Scoring/optimization explicitly use no KV cache. Raw logits supply loss probabilities and statistics, as in the release. Forward eval mode disables dropout and the training-only gradient-checkpoint branch; the released LoRA configuration remains unchanged.

## Fixed cache and evidence

All 32 cached responses are evaluation-only. Their old/reference log-probabilities, values and terminal rewards stay fixed. GAE uses beta=.1; advantages are normalized once per cached response, matching the one-prompt rollout unit, and reused for every epsilon. The new log-probabilities come only from the final standard policy. Affected-token fraction counts ratios outside the interval; active clipped-branch fraction is reported separately.

CPU checks with the pinned tokenizer verified all 32 recorded lengths and decoded texts. Reconstruction refuses mismatches and never trims/pads. Original token IDs are absent, so matching length/text alone cannot establish uniqueness of the original tokenization. `CACHE_RECONSTRUCTION_CPU.json` records this limitation. The CPU preparer repeats the check on the user's verified tokenizer files.

Each completed update/prompt is committed locally and persisted with archive checksums. Interrupted uploads are recovered before shutdown. A lost in-flight rollout/prompt can be repeated from its last saved boundary; required completed work is retained. Training timing excludes setup/model loading and final export, and includes per-update persistence; the total GPU session clock includes setup. The Chunk 2 timing code records a planned budget exception as a closed `FAILED` interval with `error.type=SessionBudgetReached`; the overall pipeline labels it `SAVED_AT_SESSION_LIMIT`, and it is not a nonfinite failure. Hard disconnection may leave timing incomplete, explicitly flagged.

CPU aggregation verifies saved archives before calculating quantitative evidence and requires complete run/evaluation counts. It prepares plots and all midpoint/final response pairs without inventing quality judgments. Qualitative reward/quality agreement and disagreement remain pending until supported by actual response review. Task 4 safety evaluation remains separate.

## Package checks

`LOCAL_VALIDATION.json` and `tests.log` record the local CPU checks, including objective signs/causal alignment, critic precision, exact checkpoint continuation, prompt resume, shutdown gates and complete-job orchestration. The orchestration check uses tiny/mocked fixtures, not full-model GPU evidence. `PACKAGE_SHA256.json` seals this small code package. Existing sealed Chunk 1 and Chunk 2 packages are unchanged.
