# Evidence index, chronology and handoff

The immutable evidence root is [results/task1/evidence](../../results/task1/evidence). Its original `SHA256SUMS.json` verifies 855 exported payload files. Historical “pending”/“not started” statements describe the time they were written; current completion status is in the README and final audit.

## Main records

| Folder | Contents |
|---|---|
| `training/standard`, `training/beta_*`, `training/length_balanced` | Original run records, traces, package freezes, saved adapter configuration, fresh reload proofs and executed source snapshots |
| `training/standard_failed_scale65536` | Preserved first attempt, finite forward trace and failed gradient event |
| `evaluation/<condition>/policy_results` | All raw pair scores and generation token arrays, policy metrics and sources |
| `evaluation/<condition>` | Reward replay scores, complete-input audit, reward correction/loading record and final metrics |
| `project` | Exact exported pre-retry project used by final runs; some starter/historical files intentionally incomplete |
| `model_metadata` | Pinned public model manifest and original model configurations |
| `setup_evidence` | Chronological approvals, validation, preparation and launch logs |
| `export_metadata.json` | Explicit list of omitted binary/tokenizer artifacts and hashes |

The archived chronological Colab code export is [atml_pa2_task1.py.txt](../../archive/atml_pa2_task1.py.txt). It has no cell outputs. Do not execute it as a pipeline; it includes failed attempts and superseded setup. Inputs and source-archive hashes are in [INPUTS.json](INPUTS.json).

## Chronology

Dates/timestamps in records are preserved verbatim. Colab library log timestamps sometimes use UTC while evidence folder names use the recorded session time; do not infer a new ordering from mixed display time zones.

1. Course source/assets were pinned; discretionary A–H decisions were explicitly approved.
2. The objective/reference-sign and accuracy defects were corrected and response-only scoring checked.
3. The 768-token audit exposed lost prompts. User-supplied TA clarifications allowed prompt preservation, filtering or a larger compute-dependent cap. The approved 4,096 proposal preserved every supplied pair and response.
4. An initial disposable trial stopped on the four original empty rejected answers; the validation was corrected to retain/score EOS. The subsequent memory trial passed; it was not official training.
5. CPU runtime/evaluation checks, prepared public model snapshots and durable backups supported safe GPU shutdowns.
6. The reward checkpoint/library rotary-position mismatch was explained, explicitly approved and corrected in memory; actual GPU reward loading later verified it.
7. The first official standard run at scale 65,536 failed before update 19. Evidence was preserved; an overflow hypothesis was recorded as unconfirmed.
8. After approval, scale 1,024 was applied consistently. The standard run restarted fresh and completed its full budget, then standard/SFT evaluations and reward replay completed.
9. The three independent short beta runs and the balanced run completed with fresh initialization; their full fixed evaluation pools and reward replay completed.
10. Quantitative summaries were exported. The user approved wl05 and wl04 qualitative examples.
11. The final CPU-only export supplied previously missing raw pair rows, training traces and three source/provenance files. The independent audit passed and publication utilities/documentation were prepared without changing experimental computation.

## Historical setup folders

- [active_encoding_20261003_222414_346767](../../results/task1/evidence/setup_evidence/active_encoding_20261003_222414_346767/)
- [cap4096_audit_20261003_201859_200185](../../results/task1/evidence/setup_evidence/cap4096_audit_20261003_201859_200185/)
- [cap4096_gpu_retry_20261003_210335_215061](../../results/task1/evidence/setup_evidence/cap4096_gpu_retry_20261003_210335_215061/)
- [cap4096_gpu_trial_20261003_203433_896389](../../results/task1/evidence/setup_evidence/cap4096_gpu_trial_20261003_203433_896389/)
- [cap4096_preflight_diagnostic_20261003_205053_195045](../../results/task1/evidence/setup_evidence/cap4096_preflight_diagnostic_20261003_205053_195045/)
- [comparison_complete_evaluation_jobs_20261004_175600_525886](../../results/task1/evidence/setup_evidence/comparison_complete_evaluation_jobs_20261004_175600_525886/)
- [comparison_policy_complete_reward_jobs_20261004_204232_870665](../../results/task1/evidence/setup_evidence/comparison_policy_complete_reward_jobs_20261004_204232_870665/)
- [cpu_environment_20261003_220629_970425](../../results/task1/evidence/setup_evidence/cpu_environment_20261003_220629_970425/)
- [cpu_model_preparation_20261004_002851_903477](../../results/task1/evidence/setup_evidence/cpu_model_preparation_20261004_002851_903477/)
- [cpu_package_resume_20261004_105818_429305](../../results/task1/evidence/setup_evidence/cpu_package_resume_20261004_105818_429305/)
- [cpu_restore_20261003_215920_761391](../../results/task1/evidence/setup_evidence/cpu_restore_20261003_215920_761391/)
- [cpu_resume_20261004_103347_602560](../../results/task1/evidence/setup_evidence/cpu_resume_20261004_103347_602560/)
- [dpo_objective_20261003_164034_472116](../../results/task1/evidence/setup_evidence/dpo_objective_20261003_164034_472116/)
- [evaluation_core_20261004_130619_867609](../../results/task1/evidence/setup_evidence/evaluation_core_20261004_130619_867609/)
- [evaluation_entrypoint_20261004_132847_178907](../../results/task1/evidence/setup_evidence/evaluation_entrypoint_20261004_132847_178907/)
- [evaluation_support_20261004_001519_971472](../../results/task1/evidence/setup_evidence/evaluation_support_20261004_001519_971472/)
- [final_settings_20261003_221559_208443](../../results/task1/evidence/setup_evidence/final_settings_20261003_221559_208443/)
- [gpu_four_comparison_policy_20261004_180508_319409](../../results/task1/evidence/setup_evidence/gpu_four_comparison_policy_20261004_180508_319409/)
- [gpu_four_comparison_reward_20261004_204919_086345](../../results/task1/evidence/setup_evidence/gpu_four_comparison_reward_20261004_204919_086345/)
- [gpu_four_comparison_training_20261004_172413_595406](../../results/task1/evidence/setup_evidence/gpu_four_comparison_training_20261004_172413_595406/)
- [gpu_standard_launch_20261004_135013_361644](../../results/task1/evidence/setup_evidence/gpu_standard_launch_20261004_135013_361644/)
- [gpu_standard_retry_scale1024_20261004_143416_467965](../../results/task1/evidence/setup_evidence/gpu_standard_retry_scale1024_20261004_143416_467965/)
- [gpu_standard_sft_policy_20261004_151459_318156](../../results/task1/evidence/setup_evidence/gpu_standard_sft_policy_20261004_151459_318156/)
- [gpu_standard_sft_reward_20261004_165852_548773](../../results/task1/evidence/setup_evidence/gpu_standard_sft_reward_20261004_165852_548773/)
- [gradient_scale1024_20261004_142324_585826](../../results/task1/evidence/setup_evidence/gradient_scale1024_20261004_142324_585826/)
- [policy_complete_reward_jobs_20261004_164606_103611](../../results/task1/evidence/setup_evidence/policy_complete_reward_jobs_20261004_164606_103611/)
- [response_scoring_20261003_164908_850928](../../results/task1/evidence/setup_evidence/response_scoring_20261003_164908_850928/)
- [reward_complete_comparison_jobs_20261004_171524_857140](../../results/task1/evidence/setup_evidence/reward_complete_comparison_jobs_20261004_171524_857140/)
- [reward_config_audit_20261004_111657_549569](../../results/task1/evidence/setup_evidence/reward_config_audit_20261004_111657_549569/)
- [reward_loader_20261004_124522_353637](../../results/task1/evidence/setup_evidence/reward_loader_20261004_124522_353637/)
- [runtime_validation_20261003_224707_061389](../../results/task1/evidence/setup_evidence/runtime_validation_20261003_224707_061389/)
- [standard_complete_evaluation_jobs_20261004_145929_395368](../../results/task1/evidence/setup_evidence/standard_complete_evaluation_jobs_20261004_145929_395368/)
- [standard_failure_diagnostic_20261004_140644_711479](../../results/task1/evidence/setup_evidence/standard_failure_diagnostic_20261004_140644_711479/)
- [tokenization_audit_20261003_170013_811243](../../results/task1/evidence/setup_evidence/tokenization_audit_20261003_170013_811243/)
- [training_entrypoints_20261003_235653_951156](../../results/task1/evidence/setup_evidence/training_entrypoints_20261003_235653_951156/)
- [training_support_20261003_234453_795861](../../results/task1/evidence/setup_evidence/training_support_20261003_234453_795861/)

## Standard adapter for Task 4

Keep the complete verified Drive folder:

`/content/drive/MyDrive/ATML-Assignment-2/training_runs/standard_20261004_143635_293670_COMPLETE_RELOAD_VERIFIED`

Its `adapter_model.safetensors` SHA-256 is `803e05a9355eb24a1889474350d63377e97c067432d55f3b8ce0f173c17d63c0`. This is a file hash, distinct from the tensor-content digest recorded in training/evaluation. The public Git package contains its receipt/configuration and reload proof, not the binary adapter. Retain it for Task 4; do not substitute a beta fork.

Prepared public model backup:

`/content/drive/MyDrive/ATML-Assignment-2/model_snapshots/20261004_002851_903477`

Final pre-training project backup:

`/content/drive/MyDrive/ATML-Assignment-2/runtime_backups/20261004_142324_585826`

Each training/evaluation folder also contains its original backup receipt with the full Drive location. These locations are historical provenance, not prerequisites for another reader: standalone preparation downloads pinned public assets. The binary adapters and optimizer states remain outside Git.

## Remaining work

Manually review/copy/commit/push this package; integrate Task 1 evidence into the final combined report; then begin Task 2 with its own requirements and decision review. No further Task 1 GPU work is indicated by this audit.

Packaging note: the original project ignore file is archived as `.gitignore.archived.txt` without changing its bytes; the audit resolves this one filename. See the audit for duplicate-summary omissions.
