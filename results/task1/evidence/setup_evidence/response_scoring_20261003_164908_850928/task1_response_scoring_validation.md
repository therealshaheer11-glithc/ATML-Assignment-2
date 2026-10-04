# Task 1 response-scoring validation

Uses unchanged course helpers from commit 05543bec4fba95eba708eae9d6e15730d53ca317.
Nine tests use explicitly specified probabilities and token IDs to check
response sums, next-token alignment, masks, gradients, EOS, and truncation.

The first response token is predicted from the last prompt position.
EOS is included in the response score and length. Positions after the first
EOS are excluded from the generation response mask.

Released truncation first removes prompt context. If the response itself
exceeds the cap, the helper retains its tail and may remove all prompt context.
The first retained token then has no preceding position and is not scored.
These tests document that boundary without changing the helper.
Actual-data truncation counts and real-model reference checks remain pending.
