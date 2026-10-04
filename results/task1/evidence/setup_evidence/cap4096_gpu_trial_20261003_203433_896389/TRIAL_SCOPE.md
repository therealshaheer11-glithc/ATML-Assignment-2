# Approved 4096-token GPU feasibility trial

User approved the four-item feasibility proposal by replying "sure".
The completed data audit retained every original example, with zero answer
truncation or length-order changes. Official training configuration remains
pending review. The separate reward-scoring cap remains unchanged at 1024.

Diagnostic membership: 16 longest retained pairs across both training sets,
ranked by maximum chosen/rejected sequence length, then dataset name and
original row index. Shuffle with course seed. Held-out scores are not used.
This selection changes no official dataset, budget, or training order.

Fresh pinned Qwen initialization; course LoRA configuration; FP16 frozen
base and FP32 trainable adapters. Nonreentrant gradient checkpointing,
input gradients enabled, use_cache false. Course AdamW settings and defaults.
Beta0.10; eight batches of two pairs; loss weighted by pair_count/16;
one accumulated optimizer update. GradScaler initial65536, growth2,
backoff0.5, interval2000; no autocast. Unscale before norm1 clipping.
Stop on nonfinite loss/gradients or a skipped update without changing settings.

The candidate prompt-preserving 4096-token encoding runs inside this trial
only. Full responses are checked against the completed zero-truncation audit.
Active source/config files and original datasets remain unchanged.
Reference scoring uses no_grad, evaluation mode, and disabled adapters.
Check initial loss log(2), updated adapters, identical frozen-weight hashes,
unchanged reference scores, and peak allocated/reserved memory.
Reference diagnostic tolerance: absolute1e-4, relative0.

The trial covers the longest real training examples, without artificial
padding to 4096. Generation and reward-model memory are separate checks.
No generation or reward scoring is performed here. Run in a subprocess;
discard its model, optimizer, scaler, and adapters on exit.
No official checkpoint or save/reload validation is implied by a pass.
Preserve all evidence, including failures. No student Git access.
