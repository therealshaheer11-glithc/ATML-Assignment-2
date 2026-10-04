# Approved Task 1 4096-token feasibility trial

User approval: "sure", 2026-10-03, for the revised four-item trial proposal.
Official training configuration remains pending review of trial evidence.

TA clarification supplied by the user: "Yes, some of the prompts are > 768 tokens. The reason for this cap is purely compute based. You can try increasing the token limit as much as your compute allows, or filter out those particular prompts."

Trial encoding: complete formatted prompt + response prefix + reserved EOS, within 4096 combined tokens. Truncated answers receive EOS despite being incomplete.
No filtering or replacement. Short runs retain the original first 600 rows.
Supplied length-group labels remain unchanged; ordering changes include new ties.
Pinned tokenizer and original data hashes are checked. Active source/config files are checked before and after this audit.
Generation audit covers all 300 standard evaluation questions and 10 word-limit prompts. The existing reward limit of 1024 is inspected, not changed. Actual generated answers, complete reward inputs, and reward-model memory remain to be checked. The generation ceiling remains 256 new tokens.
The longest 16 retained standard-training pairs are recorded for a disposable GPU check, with ties resolved by original row index. This does not change official membership or training order. Held-out scores are not used to choose the cap.
No model weights loaded and no training performed by this audit. GPU feasibility remains untested. Pin/attribute the TA patch and validate active response scoring before official training. Preserve earlier audits as historical evidence.
