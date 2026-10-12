# Task 4 — save check before disconnecting. No model loading or inference.
from google.colab import drive
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os

if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')
root = Path('/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1')
policies = ['sft', 'dpo', 'ppo', 'grpo']
def read(path):
    return json.loads(path.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

report_path = root / 'generation_attempts/20261011T174101.363739Z/REPORT.json'
report = read(report_path)
assert report['status'] == 'ALL_FOUR_POLICIES_GENERATED_AND_SAVED'
completion_path = root / 'GENERATION_COMPLETION_CONTRACT.json'
assert sha(completion_path) == report['generation_completion_contract_sha256']
completion = read(completion_path)
assert completion['responses'] == 1800 and completion['generation_batches'] == 452
assert completion['policy_response_counts'] == {p: 450 for p in policies}
contract_path = root / 'EVALUATION_EXECUTION_CONTRACT.json'
assert sha(contract_path) == completion['evaluation_contract_sha256']
assert (root / 'evaluation_v1/CONTRACT.json').read_bytes() == contract_path.read_bytes()
contract = read(contract_path)
expected = {f'generation/{p}/{i:06d}.json' for p in policies for i in range(113)}
assert set(completion['generation_unit_sha256']) == expected
for name, expected_sha in completion['generation_unit_sha256'].items():
    assert sha(root / 'evaluation_v1' / name) == expected_sha, name
assert set(completion['exports']) == set(policies)
for policy in policies:
    path = root / f'evaluation_v1/exports/generation/{policy}.jsonl'
    assert sha(path) == completion['exports'][policy]['sha256'], policy
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 450 and all(row['policy'] == policy for row in rows)
    assert [row['xstest_id'] for row in rows] == contract['ordered_prompt_ids']
print('Verified: 452 batches and four exports; 1800/1800 responses.')

metadata = {}
for name in ['OWNER.json', 'INPUT_IDENTITY_CONTRACT.json', 'AUDIT_ID_CONTRACT.json',
             'CPU_MODEL_INPUT_CONTRACT.json', 'CPU_ORCHESTRATION_VALIDATION_CONTRACT.json',
             'BASE_MODEL_REFERENCE.json', 'JUDGE_SNAPSHOT_CONTRACT.json',
             'GPU_ENVIRONMENT_CONTRACT.json', 'ISOLATED_GPU_ENVIRONMENT_CONTRACT.json',
             'ISOLATED_CHILD_ENVIRONMENT_CONTRACT.json', 'COLAB_BASE_REPAIR_CONTRACT.json',
             'MODEL_LOADING_VALIDATION_CONTRACT.json', 'EVALUATION_EXECUTION_CONTRACT.json',
             'GENERATION_COMPLETION_CONTRACT.json',
             'environment_wheelhouse/VERIFIED_WHEELHOUSE_CONTRACT.json']:
    metadata[name] = sha(root / name)
for name in ['BASE_MODEL_REFERENCE.json', 'JUDGE_SNAPSHOT_CONTRACT.json']:
    model = read(root / name)
    for filename, entry in model['verified_files'].items():
        path = Path(model['directory']) / filename
        assert path.is_file() and path.stat().st_size == entry['bytes'], str(path)
wheelhouse = root / 'environment_wheelhouse'
wheels = read(wheelhouse / 'VERIFIED_WHEELHOUSE_CONTRACT.json')
assert len(wheels['verified_files']) == 90
for name, entry in wheels['verified_files'].items():
    path = wheelhouse / 'wheels' / name
    assert path.is_file() and path.stat().st_size == entry['bytes'], name
sources = {}
for manifest in sorted(root.glob('*_releases/*/MANIFEST.json')):
    assert sha(manifest) == manifest.parent.name, str(manifest)
    for name, expected_sha in read(manifest).items():
        assert sha(manifest.parent / name) == expected_sha, name
    sources[str(manifest.relative_to(root))] = sha(manifest)
assert contract['source_manifest_sha256'] in sources.values()

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
receipt = {'status': 'FILES_VERIFIED_BEFORE_DRIVE_FLUSH', 'saved_utc': stamp,
           'responses': 1800, 'generation_batches': 452, 'judgments': 0,
           'human_labels': 0, 'next_stage': 'Restore/validate runtime, then fixed judging',
           'generation_report_sha256': sha(report_path), 'metadata_sha256': metadata,
           'source_manifest_sha256': sources,
           'model_and_wheel_sizes_checked': True, 'model_and_wheel_bytes_rehashed_now': False,
           'temporary_runtime_environment_must_be_recreated_if_deleted': True}
target = root / 'resume_receipts' / f'{stamp}.json'
target.parent.mkdir(parents=True, exist_ok=True)
payload = (json.dumps(receipt, indent=2) + '\n').encode()
temporary = target.with_suffix('.pending')
with temporary.open('xb') as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, target)
assert target.read_bytes() == payload
print('Restart receipt:', target)
print('Flushing pending Drive writes; wait for the final success message...')
drive.flush_and_unmount()
print('SAVE_CHECK_PASSED_READY_TO_DISCONNECT')
