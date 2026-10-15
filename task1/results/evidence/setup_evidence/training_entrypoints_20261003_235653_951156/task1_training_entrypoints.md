# Task 1 training entry points

The original released train.py is archived in this installation's evidence.
The completed train CLI uses the approved numerical runtime and support loader.
No model weights are loaded and no official training occurs in this CPU block.

Five independent conditions use fresh base weights, LoRA, AdamW, GradScaler
and seed6304. One full epoch includes every selected pair and the partial tail.
There is no scheduler, warmup, early stopping or held-out checkpoint selection.
The frozen reference disables adapters under evaluation mode and no gradients.
Training data only provide the initial agreement and final reload probes.

Each GPU run records the effective configuration, source/config snapshot,
package versions, original and shuffled IDs, optimizer defaults, per-batch
loss/scores, per-update gradients/scales/timing, memory, frozen-base hashes,
initial/final adapter hashes, and initial agreement with the reference.
A completed run saves final LoRA adapters, tokenizer, optimizer/scaler/RNG
state and reload probe. Each output path must be new; no overwrites occur.

A separate process reloads the final adapter against the pinned frozen base,
checks parameter hashes and reproduces saved policy/reference probe scores.
Only a full budget plus successful reload earns COMPLETE_RELOAD_VERIFIED.
An exception marks the run FAILED and backs up the available evidence and
checkpoint files. No automatic retry, resumption or settings changes occur.
A verified Drive copy is required; Drive flushing and runtime release are
handled by the later launch cell. The standard adapter stays at the released
outputs/task1_dpo/standard path for Task4 compatibility.

The new CPU tests check plan execution without loading weights, rejection of
unapproved overrides, rejection of CPU training before creating output, and
rejection of FAILED checkpoints. Actual GPU training/reload remain pending.

Runnable examples, once CPU model preparation and GPU launch are ready:
python -m task1_dpo.train --run-name standard --plan
python -m task1_dpo.train --run-name standard --models-dir /content/ATML_PA2_models --backup-root /content/drive/MyDrive/ATML-Assignment-2/task1_runs
python -m task1_dpo.verify_checkpoint --checkpoint outputs/task1_dpo/standard

Do not commit raw datasets, model weights, optimizer states or large
checkpoints to Git. Preserve them separately on Drive for reproducibility.
