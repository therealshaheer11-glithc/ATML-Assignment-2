# Task 5 - Block 3: verify saved model files and inspect cached-package metadata.
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, platform, shutil, subprocess, uuid

project = Path('/content/drive/MyDrive/ATML-Assignment-2')
root = project / 'task5/experiment_v1'
previous = root / 'asset_inventory_attempts/20261012T164833.926502Z_55de9866/REPORT.json'
smi = shutil.which('nvidia-smi')
gpu = subprocess.run([smi, '--query-gpu=name', '--format=csv,noheader'],
                     capture_output=True, text=True, timeout=15) if smi else None
if (gpu and gpu.returncode == 0 and gpu.stdout.strip()) or os.environ.get('COLAB_TPU_ADDR') or os.environ.get('TPU_NAME'):
    raise RuntimeError('Use the same CPU runtime for this block.')

def sha(path):
    h = hashlib.sha256()
    size = path.stat().st_size
    completed = 0
    threshold = 512 * 2**20
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(chunk)
            completed += len(chunk)
            if completed >= threshold:
                print(f'  Read {completed / 2**30:.2f}/{size / 2**30:.2f} GiB', flush=True)
                threshold += 512 * 2**20
    return h.hexdigest()

def checked_json(path, expected):
    if not path.is_file() or path.is_symlink() or sha(path) != expected:
        raise RuntimeError('Saved record identity differs: ' + str(path))
    return json.loads(path.read_text())

prior = checked_json(previous, 'aa5dc0c28e6e5ac82caf84bf1313d60797f967a659e9c8c9be2300d91a4da6b5')
if prior['status'] != 'TARGETED_CPU_INVENTORY_SAVED_REVIEW_PENDING':
    raise RuntimeError('Previous inventory status differs.')
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid.uuid4().hex[:8]
attempt = root / 'model_integrity_attempts' / stamp
attempt.mkdir(parents=True, exist_ok=False)
report = {'status': 'RUNNING', 'snapshots': {}, 'verified_snapshots': 0,
          'verified_model_files': 0, 'errors': [], 'installs': 0, 'downloads': 0,
          'model_loads': 0, 'policy_responses': 0, 'ai_judgments': 0,
          'previous_files_modified': False, 'previous_report_sha256': sha(previous)}

def save():
    payload = (json.dumps(report, indent=2, sort_keys=True) + '\n').encode()
    pending = attempt / 'REPORT.pending'
    pending.write_bytes(payload)
    os.replace(pending, attempt / 'REPORT.json')
    if (attempt / 'REPORT.json').read_bytes() != payload:
        raise RuntimeError('Report readback failed.')

try:
    save()
    base = checked_json(project / 'task3/experiment_v1/models/model_manifest.json',
                        'b00f63237dda95ad175983bf231ac52a4133242d831e13f07629fa247b353aa8')
    judge = checked_json(project / 'task4/experiment_v1/JUDGE_SNAPSHOT_CONTRACT.json',
                         '46d53c18648022ba1b01310b4940313629ab246e31a1b722c01efe00cb7c9228')
    specs = [
        ('policy', project / 'task3/experiment_v1/models/policy', base['policy'],
         base['policy']['sha256'], 'Qwen/Qwen2.5-1.5B-Instruct', '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'),
        ('judge', project / 'task4/experiment_v1/models/judge', judge,
         {k: v['sha256'] for k, v in judge['verified_files'].items()},
         'Qwen/Qwen2.5-3B-Instruct', 'aa8e72537993ba99e69dfaafa59ed015b17504d1'),
    ]
    for label, folder, metadata, files, model_id, revision in specs:
        if metadata['model_id'] != model_id or metadata['revision'] != revision:
            raise RuntimeError('Model identity differs: ' + label)
        actual_files = {p.relative_to(folder).as_posix() for p in folder.rglob('*')
                        if p.is_file() and '.cache' not in p.relative_to(folder).parts}
        if actual_files != set(files):
            raise RuntimeError('Snapshot file list differs: ' + label)
        entry = {'model_id': model_id, 'revision': revision,
                 'expected_files': len(files), 'verified_files': {}, 'complete': False}
        report['snapshots'][label] = entry
        for number, (relative, expected) in enumerate(sorted(files.items()), 1):
            rel = Path(relative)
            if rel.is_absolute() or '..' in rel.parts:
                raise RuntimeError('Unsafe snapshot filename.')
            path = folder / rel
            print(f'{label}: checking file {number}/{len(files)}: {relative}', flush=True)
            if not path.is_file() or path.is_symlink():
                raise RuntimeError('Missing snapshot file: ' + str(path))
            actual = sha(path)
            if actual != expected:
                raise RuntimeError('Snapshot checksum differs: ' + str(path))
            entry['verified_files'][relative] = {'sha256': actual, 'bytes': path.stat().st_size}
            report['verified_model_files'] += 1
            save()
        entry['complete'] = True
        report['verified_snapshots'] += 1
        save()

    transfer = Path(prior['datasets']['math_transfer_eval.jsonl']['path'])
    expected_transfer = 'f24163301035ccd4b66d12b687887fe1e167dd31ef62d6bc62a2fa152dcdcb76'
    if sha(transfer) != expected_transfer:
        raise RuntimeError('Transfer file differs from the independently reconstructed course file.')
    report['transfer_reference'] = {'matches_official_first_100_under_course_helper': True,
        'sha256': expected_transfer, 'official_source': 'https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json',
        'course_helper_sha256': 'b1c19a59e6dc0672336a306a23fc349e57a10400072b1716e9439d30ff9a0720',
        'reference_comparison_performed_by_assistant': True, 'network_access_in_this_block': False}

    cache = checked_json(project / 'task4/experiment_v1/environment_wheelhouse/VERIFIED_WHEELHOUSE_CONTRACT.json',
                         '37bb9bd34543234252ed2973222a30870a86ce83d2f2bc9449eac492e621cb52')
    selected = ['torch', 'transformers', 'tokenizers', 'trl', 'peft', 'huggingface-hub', 'bitsandbytes']
    report['cached_package_candidate'] = {'recorded_wheels': len(cache['files']),
        'recorded_compressed_bytes': cache['total_compressed_bytes'],
        'selected_versions': {name: cache['isolated_package_versions'].get(name) for name in selected},
        'current_python': platform.python_version(), 'historical_marker_environment': cache['saved_gpu_marker_environment'],
        'wheel_payloads_freshly_verified': False, 'current_compatibility_verified': False,
        'reuse_or_installation_approved_for_task5': False}
    report['status'] = 'CPU_MODEL_INTEGRITY_COMPLETE_ENVIRONMENT_APPROVAL_PENDING'
except Exception as exc:
    report['status'] = 'CPU_MODEL_INTEGRITY_STOPPED_REVIEW_PENDING'
    report['errors'].append(type(exc).__name__ + ': ' + str(exc))
finally:
    save()
    print('Status:', report['status'])
    print('Verified snapshots:', report['verified_snapshots'], '/ 2')
    print('Verified model files:', report['verified_model_files'])
    print('Transfer reference:', json.dumps(report.get('transfer_reference', {}), indent=2))
    print('Cached package candidate:', json.dumps(report.get('cached_package_candidate', {}), indent=2))
    print('Errors:', report['errors'])
    print('Saved report:', attempt / 'REPORT.json')
    print('0 installs; 0 downloads; 0 model loads; 0 responses; 0 AI judgments.')
