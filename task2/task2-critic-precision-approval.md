# Task 2 approval addendum L: critic scalar-head precision

**Approved by the user on 8 October 2026:** “Approved but document it explicitly”. This approves the previously proposed float32 parameters and optimizer state for only the critic's trainable scalar value head. It supplements decisions A–K; it does not retroactively change the sealed Chunk 1 package or its validation record.

## Classification and reason

This is an explicitly approved numerical implementation correction, not an assignment-prescribed setting or a confirmed deliberate TA error. The confirmed deliberate PPO defect remains the use of `maximum` where the manual requires `minimum`.

The released critic loader loads the scalar head in float16. PEFT promotes the trainable LoRA adapters to float32, but leaves the trainable saved scalar head in float16. In the tested PyTorch AdamW implementation, that head's first and second moment tensors also remain float16. The released AdamW epsilon, 1e-8, rounds to zero in float16. Small squared gradients can also underflow to zero, leaving an invalid update denominator.

**Evidence and limits:** a controlled CPU AdamW test with initial parameter 0.1, gradient 1e-5, learning rate 3e-4, weight decay 0, betas (0.9, 0.999), and epsilon 1e-8 produced `-inf` for a float16 parameter. The corresponding float32 parameter remained finite at approximately 0.0997003. The deterministic seed-6304 tiny critic test remained finite in float16. These are numerical diagnostics, not training results; no failure of the full supplied critic checkpoint has been established. The saved diagnostic JSON records both cases.

## Exact approved change

1. Load the supplied merged critic checkpoint in float16 and add the released value LoRA configuration as before.
2. Before constructing the critic optimizer, convert only the **active trainable scalar head** parameters to float32. Preserve their numeric values exactly during conversion; do not reinitialize them. Leave the frozen original head copy and frozen transformer parameters in float16. Keep the existing trainable LoRA parameters in float32.
3. Obtain token hidden states directly from the underlying critic transformer. Cast those hidden states to float32 when applying the active scalar head. This cast stays differentiable, so critic gradients reach the LoRA adapters. Avoid invoking the sequence-classification wrapper before this cast.
4. Construct the same AdamW optimizer after promotion. Its head moment tensors will then be float32. Keep head LR 3e-4, critic LoRA LR 1e-4, betas (0.9, 0.999), epsilon 1e-8, and critic weight decay 0.
5. Record the approval ID and parameter/optimizer-state dtypes with each training run. Apply the same correction to every continuation and fork.
6. On resumption, promote the reconstructed head **before** loading its saved float32 training state and optimizer state. Loading a trained float32 head into a float16 destination first would silently round it. Preserve the head's float32 state in saved checkpoints.

This supersedes decision F only for the trainable critic scalar head and its input arithmetic. The frozen backbone remains float16; reward quantization, policy parameters, losses, GAE, advantage normalization, clipping, seeds, schedules, and token/update budgets retain their approved settings. No new gradient scaler, optimizer epsilon change, or extra stabilization is authorized.

Converting the stored head values is exact, but using float32 for the head multiplication changes numerical rounding. Report this correction when describing the implementation; do not claim bitwise equality of the critic outputs or optimizer trajectory with the released float16-head path.

## Verification and runtime boundary

The companion tests use randomly initialized tiny models and controlled scalar examples on CPU. They check parameter preservation, frozen-parameter dtypes, head-only promotion, token-value alignment, gradient flow, float32 optimizer moments, checkpoint round-trip, and the small-gradient failure mechanism. Actual course-model CUDA memory use and training behavior remain unverified until a prepared GPU job runs. Keep the Colab GPU disconnected while preparing the remaining code.

No GPU training or assignment result is implied by this approval or these tests.
