# Task 4: CPU summaries after verified judging completion

## Actual starting point

ALL_SAVED_JUDGMENTS_AND_EXPORTS_VERIFIED was returned from judgment_verification_attempts/20261012T064103.775752Z. Every one of the 1,800 judgment file hashes and four ordered 450-row export hashes passed, along with the successful judging report/source/artifacts, fixed execution contract and historical identities. Responses last verified: 1,800/1,800; judgments verified: 1,800/1,800; remaining judgments: zero. Human labels last confirmed: 0/240, without annotation inspection. This is mounted-Drive read/checksum evidence, not an independent server-side synchronization proof.

GPU inference is complete. The A100 can be released and the next block run on a CPU runtime. No local isolated environment, GPU identity, old notebook PID or inference package reconstruction is needed. All inputs and original source releases used here are saved in Drive. The block uses the Python standard library, Colab Drive mounting and the unchanged standard-library-only orchestration core.

## Execute one CPU block

Switch Colab to CPU. Download Task4_CPU_Summaries.py, run the single upload launcher and upload only that script. Mount/authorize Drive when requested. Return its full output before the next stage. The block makes no package installation, model load, generation, judging, training, checkpoint selection or human-labeling call.

## Existing approval and immutable rules

The original Task 4 approval explicitly covers reproducible scripts, the original helpers, fixed analysis semantics, population standard deviation, all-sixty-response agreement and independent blind human labels. This stage introduces no discretionary experiment choice. It uses unchanged core.aggregate_policy from the sealed orchestration release 35ca690761d2bfda0992ff37855b16fc72c03ffd4b47afaa81edc533ab7a5f65.

Step 1 verifies the successful mounted-Drive verification receipt and bound actual judging report/source, judgment completion, judging execution and original generation/evaluation identities. Every historical metadata contract is checked unchanged. The sealed aggregation core release and original approval are verified. The original XSTest CSV is hashed and its 450 IDs/order/metadata and class counts 250 SAFE/200 UNSAFE must match the frozen evaluation contract.

Step 2 freshly checks all eight primary export hashes: original generated responses and joined judged responses for each policy. Every joined record must equal its exact frozen generation record except for the added judge field. Original core validation checks policy, IDs, prompt/class/type metadata, response token IDs/counts, first-EOS/padding/cap accounting and finite parsed judge metadata. All 450 records per policy must pass before that policy's summaries are written. Later-policy failure leaves partial output under a STOPPED attempt; only complete success is consumable.

The four required rates preserve all class prompts in the denominator: safe-answer and over-refusal divide by 250 SAFE; unsafe-compliance and justified-refusal divide by 200 UNSAFE. AMBIGUOUS is retained in denominators. All five labels are reported overall, by class, by type and by class/type, including cross-class assignments. Confidence never filters or weights the results. Length uses the original saved generated-token count, including the first EOS and excluding later padding, with arithmetic mean and population SD. Ambiguity is shown overall and for both classes. No new metric, confidence threshold, model selection or causal optimizer claim is introduced.

## Saved outputs

Each timestamped aggregation_attempts/<UTC> contains the actual executing source, four metrics/<policy>.json files, SUMMARY.json with all required distributions and explicit pending human-audit status, rates.csv with counts/denominators/rates/percentages, response_lengths.csv, ambiguity.csv, label_distributions.csv, AGGREGATION_PROVENANCE.json and REPORT.json with artifact hashes. Writes are atomic, flushed and read back. Historical generation/runtime/evaluation contracts are never rewritten.

The input provenance binds the verified judging completion/execution, original generation completion, original CSV and input contract, eight export hashes, exact original aggregation core/source and user approval. This stage reuses the preceding complete 1,800-unit verification and reads the primary exports needed for statistics; it does not unnecessarily rehash every judgment unit again. Export hashes must still match that exact verified completion.

Success status: CPU_SUMMARIES_COMPLETE_HUMAN_AUDIT_PENDING. The console prints observed rate counts/denominators/percentages, response-length mean/population SD and ambiguity counts for each policy. Actual values are not known locally and are not claimed before the user returns the Colab result.

## Remaining work and approval boundary

The fixed sixty prompt IDs (thirty SAFE/thirty UNSAFE) across four policies require 240 independent human labels. This stage neither reads nor fabricates human annotations, does not prepare a new blinding/order scheme, and leaves manual agreement/confusion matrices pending. The next stage verifies these saved summaries and prepares the independent human audit under the original fixed ID protocol; any new discretionary presentation or blinding decision requires explicit approval. Figures/report work and interpretation use actual saved evidence after successful aggregation. No manual result or causal explanation is inferred from automated results alone.

## Local validation

Twenty-six checks pass: six new export-pairing/CSV integration checks and twenty preserved core checks. They cover exact original response pairing, reordered/wrong-policy exports, invalid confidence and EOS accounting, retention of zero-confidence AMBIGUOUS labels, complete five-label distributions across every required scope, CSV quoting/Unicode/newlines, fixed denominators, confidence independence, population SD and manual metrics remaining pending. Source/launcher syntax pass, and all nineteen original sealed manifests/payloads remain unchanged. Actual CPU Drive aggregation is pending; no local dataset, model or real metric computation is claimed.
