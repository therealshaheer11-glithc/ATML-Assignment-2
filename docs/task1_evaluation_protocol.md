# Task 1 evaluation conventions and safeguards

This CPU stage installs evaluation support. It produces no model scores or
experimental results. The runnable evaluation entrypoint remains to be installed.

All five adapters receive the same original300 standard held-out pairs and
300 generation prompts. Their held-out DPO loss uses their own training beta.
Standard and balanced adapters also receive all246 supplied stratified pairs,
82 per supplied group. Group labels are not recomputed or rebalanced.
Standard, balanced and untouched SFT each receive the same10 word-limit prompts.
Untouched SFT receives the same300 general generation prompts without training.
Generation keeps original row order and resets seed6304 per model/prompt pool.
Batch1, sampling, temperature0.7, top-p0.9 and256 new-token ceiling are fixed.

Generation prompt rendering/tokenization matches the released batch helper.
Complete reward inputs include the entire prompt, answer and chat template.
Both are checked before calling helpers, so their truncation option cannot
silently crop an oversized input. The4096 limits remain unchanged. A failed
length check stops for review instead of filtering or altering the limit.
Actual generated reward inputs and GPU reward memory remain to be validated.

The released EOS response mask controls which generated tokens count. EOS is
included; subsequent padding is excluded. Reaching256 tokens without EOS is
recorded separately from ending at EOS. This generation ceiling is separate
from DPO preprocessing truncation, which the completed active audit found zero.

Response length uses the mean and population standard deviation over individual
responses. Reward is the mean scalar per response. Existing runtime routines
provide response-token-weighted sampled KL and strict reference-adjusted pair
accuracy with ties kept in the denominator. Finite negative sampled KL is kept.
The word counter, parser and <= compliance rule are the released helpers,
including for prompts phrased as 'under N words'. Missing parsed limits stop
rather than being omitted from the fixed10-prompt denominator. Qualitative
examples will be selected with the user after outputs exist.

Original metric/generation helpers are hash-checked and unchanged. All CPU
fixtures are synthetic and are not assignment results. Preserve all original
rows, including repeated IDs, by recording row index as well as prompt ID.
