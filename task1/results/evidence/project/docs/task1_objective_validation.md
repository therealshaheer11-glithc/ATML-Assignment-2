# Task 1 DPO objective correction and validation

Course starter: commit 05543bec4fba95eba708eae9d6e15730d53ca317.
Basis: PA2 DPO equation and approved implementation plan.

- Correct margin: (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected).
- Loss: mean negative log-sigmoid of beta times that margin.
- Accuracy: fraction of pairs with a strictly positive adjusted margin.
  Zero-margin ties remain in the denominator and count as zero.
- Reference inputs are detached to prevent gradients entering the reference.
- Beta must be positive.

Seven independent checks cover equality with the reference, a hand-calculated
scalar loss, reference-adjusted accuracy, the analytical gradient and frozen
reference, common-offset cancellation, swapped responses, and invalid beta.

These checks validate the objective only. Response-token scoring, real-model
training, adapter saving/loading, and evaluation require separate verification.
