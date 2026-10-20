# Budget interpretation and audit-label provenance

## Equal allocated response-token allowances

The assignment asks for matched generated-token budgets in Tasks 2–3. The
executed experiments fixed update counts, prompts, response caps and allocated
token allowances, while preserving natural EOS stopping. On 14 October 2026 the
student reaffirmed this as a justified and disclosed experimental choice; no TA
confirmation was claimed or required by the student. Historical approvals and
original measurements remain unchanged.

Why this choice is defensible: changing update counts to spend identical realized
tokens would also change optimization exposure; forcing generation beyond EOS
would change response semantics; cutting the final update short would change the
released schedule. Equal allowances preserve those controls. The cost is that
comparisons are not exactly matched in realized generated-token consumption.

| Task / condition | Updates | Allocated tokens | Actual generated tokens |
|---|---:|---:|---:|
| PPO clip 0.05 | 8 | 4,096 | 2,113 |
| PPO central (clip 0.20, KL 0.10) | 8 | 4,096 | 2,113 |
| PPO clip 0.50 | 8 | 4,096 | 2,178 |
| PPO KL 0 | 8 | 4,096 | 2,189 |
| PPO KL 0.20 | 8 | 4,096 | 2,178 |
| GRPO canonical | 8 | 16,384 | 8,483 |
| GRPO constant denominator | 8 | 16,384 | 8,628 |

Sources: [PPO budgets](../task2/evidence/reviewed/training_budgets_and_stability.csv)
and [GRPO results](../task3/RESULTS.md). Report both allocated and actual counts.
Do not describe these conditions as exact token-consumption matches, infer
instructor endorsement, or silently rerun selected conditions after evaluation.

## Task 4 audit labels

The additional audit applies the original five-label rubric to the fixed packet
using an automated labeling procedure. Exact duplicate prompt-response pairs
receive identical labels, and ambiguous labels remain in the agreement
denominators. The saved decision notes record the criteria and reasoning for
each label. No new labels were collected during repository cleanup.
[Audit record](../task4/evidence/audit/AUDIT_ANALYSIS_PROVENANCE.json).
