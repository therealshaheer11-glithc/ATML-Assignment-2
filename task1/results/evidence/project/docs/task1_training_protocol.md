# Task 1 training support and saving protocol

Five approved independent runs: standard1500/beta0.10, original first600 at
beta0.03/0.10/0.30, and balanced1500/beta0.10; one complete epoch each. Full
runs have94 updates with12 pairs in the tail; short runs have38 with8 in the
tail. Original file hashes and all other approved settings are checked.

The Task1 loader mirrors the released policy loader, with pinned revisions,
the audited tokenizer fingerprint, FP16 base, FP32 trainable LoRA, and
non-reentrant gradient checkpointing. Downloading is a separate CPU step.
Official model loading uses only local files; missing or modified snapshots
stop the job. GPU package versions must match the verified environment.
Adapter metadata names the public base model and exact revision for portability.
The shared course common/models.py file is not modified.

Artifact copying uses a new destination, compares every file hash, and refuses
to overwrite an older run or recursively copy into its own source. CPU tests
use dummy bytes and synthetic row identifiers; they produce no ML results.
The committed tokenizer manifest replaces a dependency on notebook variables.

Next install the completed training and fresh-process verification CLI scripts.
They preserve source/config snapshots, exact IDs/order, optimizer/scaler/RNG
state, loss/gradient/timing traces and final adapters. A failed run stays marked
FAILED, stops without automatic retries, and is backed up for review. Only a
full-budget adapter passing reload checks can be reported as complete. Save the
standard adapter at outputs/task1_dpo/standard for the course Task4 loader.
A launch cell will flush Drive and release GPU after verified saving, keeping
GPU off during code generation, debugging and discussion. Large model weights,
raw data and checkpoint files must remain outside the final Git submission.
