# Inspect saved results or reproduce Task 1

## Saved-results verification: CPU only

Run commands from the repository root. Python 3.10+ and the standard library are sufficient for this step. No course assets, model weights, account credentials or GPU are needed:

```bash
python -m task1_dpo.analyze_results --output outputs/task1_recomputed
```

The command refuses an existing output directory. It verifies the original export hashes, original source snapshots, training budgets/trace calculations, checkpoint reload records, common prompts/settings, reward-model correction records and all raw metric calculations. It emits six files: `audit.json`, `training.csv`, `summary.csv`, `length_strata.csv`, `word_limits.csv`, and `generation_ceiling.csv`. The length-analysis entrypoint is an alias to this same complete audit:

```bash
python -m task1_dpo.analyze_length --output outputs/task1_length_check
```

The verifier checks this completed experiment archive. It intentionally expects the five fixed completed runs and preserved failure; it is not a general audit of arbitrary new runs. New training/evaluation commands save their own metrics and verified backups.

## Environment for a fresh experiment

This is a reproduction procedure, **not work that must be repeated to accept the existing results**. The original GPU runtime was Python 3.13.15, PyTorch 2.11.0+cu130, CUDA 13.0 on an NVIDIA A100-SXM4-40GB. CPU sessions used PyTorch 2.11.0+cpu. Full package captures are stored with each training run. GPU loaders deliberately reject incompatible versions instead of silently changing behavior.

In a compatible environment with working CUDA PyTorch 2.11.0 already installed:

```bash
python -m pip install -r requirements.txt -c requirements-task1-lock.txt
```

The lock file constrains observed Task 1 dependencies; the original course requirements remain unchanged. A newer default Colab image may need these packages restored. Run GPU commands in a fresh process after installation. Do not replace working CUDA PyTorch with CPU PyTorch on a GPU runtime. Exact bitwise reproduction across hardware/library changes is not promised; even the original runtime did not enable deterministic algorithms globally. Saved outputs remain the auditable record.

## Public assets: prepare on CPU before enabling GPU

```bash
python -m task1_dpo.prepare_assets --models-dir /content/ATML_PA2_models --plan
python -m task1_dpo.prepare_assets --models-dir /content/ATML_PA2_models
```

This downloads only the four Task 1 preference files and the pinned policy/reward model snapshots, approximately 5.77 GiB for models. It verifies every recorded SHA-256 and copies the exact model manifest required by the reward loader. The ten word-limit prompts are already tracked; if that file is missing or modified, preparation stops. It never obtains a replacement word file from an unrelated source.

Use `--scope data` or `--scope models` to prepare only one group. Use `--verify-only` to check an existing preparation without network access. Existing mismatched files cause a stop; they are not overwritten. No model weights are instantiated during preparation. A fresh reader does not need the owner's private Drive: public repositories/revisions/hashes are in `docs/task1_assets_manifest.json` and `docs/task1_model_manifest.json`.

## Standalone training and evaluation

The following are shell commands. In Colab, put a command in a cell with `!` or use a `%%bash` cell. Do not run the historical notebook export. Pick an unused local output tree and a separate durable backup folder. Mount Drive before using a Drive backup path; each command requires that backup folder to exist.

Examples below use `/content/ATML_PA2_models` for prepared models and `/content/drive/MyDrive/ATML-Assignment-2/reproduction_backups` for durable backups. Create that backup folder first. The course `train --plan` and `evaluate --plan` flags validate data/configuration without loading model weights.

Standard training:

```bash
python -m task1_dpo.train --run-name standard --models-dir /content/ATML_PA2_models --output outputs/reproduce/standard/checkpoint --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
```

Standard policy evaluation and reward replay:

```bash
python -m task1_dpo.evaluate --name standard --phase policy --adapter outputs/reproduce/standard/checkpoint --models-dir /content/ATML_PA2_models --output outputs/reproduce/standard/policy --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
python -m task1_dpo.evaluate --name standard --phase reward --policy-results outputs/reproduce/standard/policy --models-dir /content/ATML_PA2_models --output outputs/reproduce/standard/reward --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
```

The same three commands with `length_balanced` replacing every `standard` argument/path run the balanced condition. The fixed configuration selects its supplied 1,500 pairs and all required evaluation pools automatically. Do not edit beta or dataset sizes.

The three beta conditions can use individual `train`/`evaluate` commands with their respective names, or the publication dispatcher below. First append `--plan` to preview each command. Run the phases in this order; each model receives a fresh training process:

```bash
python -m task1_dpo.ablate_beta --phase train --run-root outputs/reproduce --models-dir /content/ATML_PA2_models --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
python -m task1_dpo.ablate_beta --phase policy --run-root outputs/reproduce --models-dir /content/ATML_PA2_models --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
python -m task1_dpo.ablate_beta --phase reward --run-root outputs/reproduce --models-dir /content/ATML_PA2_models --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
```

Untouched SFT baseline, with no adapter or training:

```bash
python -m task1_dpo.evaluate --name sft --phase policy --models-dir /content/ATML_PA2_models --output outputs/reproduce/sft/policy --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
python -m task1_dpo.evaluate --name sft --phase reward --policy-results outputs/reproduce/sft/policy --models-dir /content/ATML_PA2_models --output outputs/reproduce/sft/reward --backup-root /content/drive/MyDrive/ATML-Assignment-2/reproduction_backups
```

Each successful training command performs a fresh-process checkpoint reload check and verifies a durable backup. Evaluation saves all pairs/generations before reward replay. Existing output folders are preserved by refusal to overwrite. Catchable failures are marked failed and backed up; a sudden runtime loss is not guaranteed to execute cleanup. Do not treat partial jobs as completed experiments or reuse partial adapters as a fresh run.

After a job reports completion and `BACKUP_VERIFIED`, save the notebook and finish/flush Drive writes before deleting its GPU runtime. Keep copies of outputs and prepared models in durable storage before switching runtime type. Use CPU for analysis, code changes and discussion. Do not expect `/content` to survive a runtime change.

## Validation boundaries

Historical objective/scoring/runtime/evaluation tests and successful full GPU runs are retained in evidence. The final publication audit ran the independent saved-results audit and four focused CPU packaging checks. It did **not** retrain or re-evaluate the models, install a new GPU environment, or freshly download 5.77 GiB of public weights. New download paths/dispatchers were checked with offline plans and refusal tests. The original computational core and configurations remain byte-identical to the successful runs.
