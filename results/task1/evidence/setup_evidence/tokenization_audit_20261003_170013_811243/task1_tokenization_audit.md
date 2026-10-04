# Task 1 tokenization audit

Uses the course Qwen tokenizer and unchanged prompt/response encoding helper.
Records original/retained/scored token counts, including appended EOS, for
both responses of every fixed training/evaluation pair. The short-run summary
uses the approved first 600 standard-training rows. The 768-token cap remains
fixed. Audit counts document behavior and are not used for held-out tuning.

Prompt shortening means some prompt tokens were removed. Response shortening
means some response tokens were removed. A fully removed prompt is the boundary
where the retained response occupies the whole cap and its first token has no
preceding position to predict it. Per-response records preserve exact prompt
IDs, source row order, and supplied length strata.

Tokenizer source revision and content hashes are recorded. Real-model checks,
training, and reward-input truncation audits remain separate stages.
