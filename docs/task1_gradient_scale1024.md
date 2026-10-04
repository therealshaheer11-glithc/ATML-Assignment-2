# Approved Task 1 numerical change: initial gradient scale 1024

The first standard attempt completed 18 updates / 288 pairs. Update 19
(eight microbatches / 16 pairs) produced nonfinite gradients in eight adapter
tensors despite finite logged forward losses and scores. The strict guard
stopped before applying that update. No completed checkpoint is claimed.
The saved objective and preference accuracy match the PA reference-adjusted margin.

Theory: FP16 backward overflow, potentially amplified by scale 65536.
This is an unconfirmed hypothesis; the trace does not locate the exact operation.
FP32 adapters still receive gradients through FP16 model calculations.

The user explicitly approved lowering initial GradScaler from 65536 to 1024
and a fresh restart, conditional on recording the theory and change.
This reduces temporary amplification 64-fold. Unscaling preserves the intended
update in exact arithmetic; numerical rounding can differ, and lower scaling
can increase underflow susceptibility. Successful full-budget training remains unverified.

The shared training entrypoint applies 1024 to all five fresh DPO conditions.
Only the constructor value and its matching run-record value changed.
All data, 4096 limits, beta values, seed, LoRA, optimizer, accumulation,
clipping, precision and generation settings remain as previously approved.
The strict finite-gradient guard remains: no skipped pairs or automatic precision changes.

The failed attempt and original script are preserved. Standard retries from
the pinned original base with fresh adapters/optimizer/scaler and seed6304,
for all 1500 pairs / 94 updates. Later evaluation must use the completed retry's
actual adapter path, recorded in the retry job, rather than the failed output.
The paired JSON records approval, hashes, theory, tradeoff and evidence paths.
