from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os
from google.colab import drive

if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')
root = Path('/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1')
attempt = root / 'judging_attempts/20261012T053854.082678Z'
policies = ['sft', 'dpo', 'ppo', 'grpo']
source_sha = 'b7a51de411113a6a3412176f03b33a58206115b9a60ecaa83e5b21b3d07405b8'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), f'Missing/linked file: {path}'
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

def safe(folder, name):
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in name
    path = folder / relative
    assert path.resolve().is_relative_to(folder.resolve()), 'Evidence path escaped folder'
    return path

print('Verifying saved judging report, source, artifacts and completion contracts.', flush=True)
report = read(attempt / 'REPORT.json')
assert report['status'] == 'ALL_FOUR_POLICIES_JUDGED_AND_SAVED'
assert report['source_sha256'] == sha(attempt / 'JUDGING_SOURCE.py') == source_sha
assert report['completed_judgments'] == report['new_judgments_saved'] == 1800
assert report['reused_judgments'] == report['remaining_judgments'] == 0
assert report['policy_judgment_counts'] == {p: 450 for p in policies}
assert report['base_and_isolated_environment_unchanged'] is True
for name, expected in report['artifact_sha256'].items():
    assert sha(safe(attempt, name)) == expected, f'Attempt artifact differs: {name}'
execution_path = root / 'JUDGING_EXECUTION_CONTRACT.json'
assert sha(execution_path) == report['judging_execution_contract_sha256']
execution = read(execution_path)
assert execution['source_sha256'] == source_sha and execution['policy_order'] == policies
for name, expected in execution['historical_metadata_sha256'].items():
    assert sha(safe(root, name)) == expected, f'Historical contract differs: {name}'
assert sha(root / 'EVALUATION_EXECUTION_CONTRACT.json') == execution['evaluation_contract_sha256']
assert sha(root / 'evaluation_v1/CONTRACT.json') == execution['evaluation_contract_sha256']
assert sha(root / 'GENERATION_COMPLETION_CONTRACT.json') == execution['generation_completion_contract_sha256']
completion_path = root / 'JUDGING_COMPLETION_CONTRACT.json'
assert sha(completion_path) == report['judging_completion_contract_sha256']
completion = read(completion_path)
assert completion['source_sha256'] == source_sha
assert completion['judging_execution_contract_sha256'] == sha(execution_path)
assert completion['evaluation_contract_sha256'] == execution['evaluation_contract_sha256']
assert completion['generation_completion_contract_sha256'] == execution['generation_completion_contract_sha256']
assert completion['judgments'] == 1800 and completion['policy_judgment_counts'] == {p: 450 for p in policies}
frozen = read(root / 'EVALUATION_EXECUTION_CONTRACT.json')
ids = frozen['ordered_prompt_ids']
assert frozen['policies'] == policies and len(ids) == len(set(ids)) == 450
expected_units = {f'judgment/{p}/{identifier:06d}.json' for p in policies for identifier in ids}
assert set(completion['judgment_unit_sha256']) == expected_units
for number, (name, expected) in enumerate(sorted(completion['judgment_unit_sha256'].items()), 1):
    assert sha(safe(root / 'evaluation_v1', name)) == expected, f'Judgment differs: {name}'
    if number % 100 == 0:
        print(f'Judgment files verified: {number}/1800; remaining: {1800-number}', flush=True)
assert set(completion['exports']) == set(policies)
for policy in policies:
    entry = completion['exports'][policy]
    path = root / 'evaluation_v1/exports/judgment' / f'{policy}.jsonl'
    assert entry['path'] == str(path) and entry['judgments'] == 450 and sha(path) == entry['sha256']
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(records) == 450 and [row['xstest_id'] for row in records] == ids
    assert all(row['policy'] == policy for row in records)
    print(f'{policy}: ordered export verified, 450/450 judgments.', flush=True)
receipt = {'status': 'ALL_SAVED_JUDGMENTS_AND_EXPORTS_VERIFIED', 'judgments': 1800,
    'policy_judgment_counts': {p: 450 for p in policies}, 'remaining_judgments': 0,
    'judging_report_sha256': sha(attempt / 'REPORT.json'),
    'judging_completion_contract_sha256': sha(completion_path), 'exports_verified': 4,
    'human_labels_last_confirmed': 0, 'human_annotations_inspected': False,
    'inference_or_training_performed': False,
    'verification_scope': 'Files read through mounted Drive; no independent remote synchronization check.'}
folder = root / 'judgment_verification_attempts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
folder.mkdir(parents=True, exist_ok=False)
payload = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
temporary = folder / '.pending-REPORT.json'
with temporary.open('xb') as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
destination = folder / 'REPORT.json'
os.replace(temporary, destination)
assert destination.read_bytes() == payload, 'Verification report readback failed'
print('Saved report:', destination, flush=True)
print('Status:', receipt['status'], flush=True)
print('Responses last verified: 1800/1800; judgments verified now: 1800/1800; human labels last confirmed: 0/240.', flush=True)
print('GPU inference is complete. Next: CPU summaries and the independent human audit.', flush=True)
