# Task 1 numerical routines and CPU validation

The shared routines implement online adapter-disabled reference scoring,
response-only DPO, pair-weighted gradient accumulation with a complete tail,
unscale-before-clipping, and stopping on invalid losses, gradients, weights,
or skipped updates. They also implement strict accuracy with ties retained
in the denominator and globally response-token-weighted sampled KL.

The 13 CPU tests use a tiny random Qwen2 model with rank-2 zero-dropout LoRA
and SGD solely as deterministic numerical fixtures. These choices are not
assignment runs, hyperparameter trials, or substitutes for approved settings.
The official settings remain Qwen1.5B, rank8/alpha16/dropout0.05, FP16 base,
FP32 adapters, AdamW LR2e-5, batch2/accum8, clipping1 and one complete epoch.
The adapter reload check uses a new Python process and verifies both policy
scores and unchanged frozen-reference scores. No official model weights are
loaded and no original training data is used by these CPU fixtures.

The course CLI entrypoints will be completed next, before any GPU launch.
Official checkpoints, their final-budget manifests, durable Drive saving,
and real-model save/reload verification remain required. The standard adapter
must be retained for Task 4. This cell does not start official training.
