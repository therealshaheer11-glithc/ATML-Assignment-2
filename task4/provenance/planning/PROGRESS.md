# Task 4 progress

## Verified locally

- Complete assignment reviewed; Task 4 pages 12-14 visually inspected during planning.
- Four Task 4 course scripts, common generation/model/data helpers, configs and requirements available from the clean course checkout at recorded commit.
- Prior preserved Task 4 files/configs/generation helper compared across two local copies and matched.
- DPO standard completion metadata and backup identity available; PPO and GRPO standard handoff identities available.
- User explicitly approved recommendations 1-4 and requested documentation.
- First CPU verifier and documents prepared. Actual local validation is recorded separately in `validation/local_tests.json` and its transcript; consult those records for results rather than treating this document as a test receipt.

## Pending actual Colab output

First CPU block: verify XSTest and all three standard adapter inputs on Drive, and save input identities/work counts. Four input groups are planned. No actual Drive checks are claimed in this local release.

The number of course prompts N remains unverified here. Planned generation is 4N responses and judging is 4N calls. The fixed audit covers 60 IDs across four policies, or 240 manual response labels. No Task 4 generation, judging, training or manual annotation has been performed by this work.

## Subsequent stages

1. Finish CPU environment/model preparation and exact audit-ID freezing after the first block's output.
2. Implement and validate CPU orchestration/analysis in reviewable stages.
3. A100 deterministic generation with per-batch persistence.
4. A100 fixed categorical judging with per-response persistence.
5. Blind user manual audit followed by agreement analysis on CPU.
6. CPU evidence/reproduction export and requirement coverage review.

For every returned Colab output, append observed completed/remaining units, successful saved receipts, unresolved issues and the next authorized block. Never infer progress from a launched cell alone. A failed block preserves its successful checks and error report; it does not mark the stage complete.

Latest local outcome: 20/20 synthetic CPU checks passed; zero failures, errors or skips. Source syntax and unchanged reference-file hashes checked. Actual Colab input verification remains 0/4 groups verified by this chat until the first returned output.
