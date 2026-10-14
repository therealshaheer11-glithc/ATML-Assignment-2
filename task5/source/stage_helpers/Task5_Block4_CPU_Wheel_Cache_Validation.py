# Task 5 - Block 4: approved CPU-only cache validation; no installation.
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.metadata as metadata, json, os, platform
import shutil, subprocess, types, uuid

project = Path('/content/drive/MyDrive/ATML-Assignment-2')
root = project / 'task5/experiment_v1'
task4 = project / 'task4/experiment_v1'
prior_path = root / 'model_integrity_attempts/20261012T170126.140651Z_ebfc96ec/REPORT.json'
cache_path = task4 / 'environment_wheelhouse/VERIFIED_WHEELHOUSE_CONTRACT.json'
cache_sha = '37bb9bd34543234252ed2973222a30870a86ce83d2f2bc9449eac492e621cb52'

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            value.update(chunk)
    return value.hexdigest()

def read_json(path, expected=None):
    require(path.is_file() and not path.is_symlink(), 'Missing or linked record: ' + str(path))
    if expected is not None:
        require(digest(path) == expected, 'Record checksum differs: ' + str(path))
    return json.loads(path.read_text())

def package_inventory():
    return sorted((d.metadata.get('Name', ''), d.version, str(d.locate_file('')))
                  for d in metadata.distributions())

def load_helper(folder, manifest_sha, filename):
    manifest = read_json(folder / 'MANIFEST.json', manifest_sha)
    source = folder / filename
    require(source.is_file() and not source.is_symlink(), 'Missing helper: ' + filename)
    payload = source.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == manifest[filename], 'Helper checksum differs: ' + filename)
    module = types.ModuleType('task5_checked_' + filename.replace('.', '_'))
    module.__file__ = str(source)
    exec(compile(payload, str(source), 'exec'), module.__dict__)
    return module, manifest[filename]

smi = shutil.which('nvidia-smi')
gpu = subprocess.run([smi, '--query-gpu=name', '--format=csv,noheader'],
                     capture_output=True, text=True, timeout=15) if smi else None
require(not ((gpu and gpu.returncode == 0 and gpu.stdout.strip()) or
             os.environ.get('COLAB_TPU_ADDR') or os.environ.get('TPU_NAME')),
        'Keep Colab on CPU / None for this block.')
require(root.is_dir() and not root.is_symlink(), 'The approved Task 5 Drive folder is unavailable.')
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid.uuid4().hex[:8]
attempt = root / 'wheel_validation_attempts' / stamp
attempt.mkdir(parents=True, exist_ok=False)
report = {'status': 'RUNNING', 'approval': {'user_message': 'sure',
    'scope': 'One CPU-only cache integrity, current wheel compatibility and dependency validation block'},
    'verified_files': {}, 'inspected_files': {}, 'verified_bytes': 0, 'errors': [],
    'installs': 0, 'downloads': 0, 'environments_created': 0, 'model_loads': 0,
    'policy_responses': 0, 'ai_judgments': 0, 'network_access_in_this_block': False,
    'installation_approved_for_task5': False, 'gpu_runtime_validation_performed': False,
    'previous_files_modified_by_this_block': False}

def save_json(name, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    pending = attempt / (name + '.pending')
    pending.write_bytes(payload)
    os.replace(pending, attempt / name)
    require((attempt / name).read_bytes() == payload, 'Saved evidence readback failed: ' + name)

before = None
try:
    save_json('REPORT.json', report)
    from packaging.markers import default_environment
    from packaging.tags import sys_tags
    from packaging.utils import canonicalize_name
    from packaging.version import Version

    before = package_inventory()
    save_json('BASE_PACKAGES_BEFORE.json', before)
    prior = read_json(prior_path)
    require(prior['status'] == 'CPU_MODEL_INTEGRITY_COMPLETE_ENVIRONMENT_APPROVAL_PENDING' and
            prior['verified_snapshots'] == 2 and prior['verified_model_files'] == 16 and not prior['errors'],
            'Previous returned model-integrity result differs.')
    report['previous_report_sha256'] = digest(prior_path)
    cache = read_json(cache_path, cache_sha)
    require(cache['format'] == 'ATML_TASK4_VERIFIED_WHEELHOUSE_V1' and
            len(cache['files']) == 90 and len(cache['isolated_package_versions']) == 84 and
            cache['total_compressed_bytes'] == 2969739166 and
            cache['relative_wheel_directory'] == 'wheels' and
            cache['relative_receipt_directory'] == 'receipts', 'Recorded cache identity differs.')
    report['wheelhouse_contract_sha256'] = cache_sha
    wheel_release = 'df469ec50f8fc5e493f2f818eb490858c713eb88e3216309aeb43e5fb49a7aab'
    diagnostic_release = '71bdb7fdbe130c5adaef8328437371bcd89621d0a93e849bab61b7db8491c540'
    wheels, wheel_helper_sha = load_helper(task4 / 'wheel_cache_releases' / wheel_release,
                                         wheel_release, 'wheels.py')
    dependencies, dependency_helper_sha = load_helper(task4 / 'diagnostic_releases' / diagnostic_release,
                                                     diagnostic_release, 'dependencies.py')
    report['checked_helper_sha256'] = {'wheels.py': wheel_helper_sha,
                                      'dependencies.py': dependency_helper_sha}
    course_path = task4 / 'preflight_releases/9a121f29750b5b09c5feafeb491630ba4b88b6c91dc2ed5ee94ebdb989793b65/course_reference/requirements.txt'
    require(course_path.is_file() and not course_path.is_symlink() and
            digest(course_path) == 'f13c0e6952c39ee99b8996ecee9cf5a426e4f798bc646c2aade007bec803603f',
            'Pinned course requirements differ.')
    course_lines = course_path.read_text().splitlines()
    environment, tags = default_environment(), list(sys_tags())
    require(environment['implementation_name'] == 'cpython' and environment['python_version'] == '3.13' and
            environment['platform_system'] == 'Linux' and environment['platform_machine'] == 'x86_64',
            'Unexpected inspection platform; review before proceeding.')
    report['current_marker_environment'] = environment
    report['historical_marker_environment'] = cache['saved_gpu_marker_environment']
    save_json('CURRENT_PLATFORM.json', {'python': platform.python_version(),
        'marker_environment': environment, 'compatible_tags': [str(t) for t in tags]})
    pins = cache['isolated_package_versions']
    course_pins = {'transformers': '4.57.1', 'tokenizers': '0.22.1', 'trl': '0.27.2', 'peft': '0.17.1'}
    require(all(pins.get(k) == v for k, v in course_pins.items()), 'The candidate changes a course pin.')
    require(all(canonicalize_name(k) == k for k in pins), 'Candidate package names are ambiguous.')
    folder, receipts = cache_path.parent / 'wheels', cache_path.parent / 'receipts'
    require(all(p.is_dir() and not p.is_symlink() for p in [cache_path.parent, folder, receipts]),
            'Cache directory is missing or linked.')
    require({p.name for p in folder.iterdir()} == set(cache['files']) and
            {p.name for p in receipts.iterdir()} == {n + '.json' for n in cache['files']},
            'Cached file or receipt coverage differs.')
    require(sum(v['bytes'] for v in cache['files'].values()) == cache['total_compressed_bytes'],
            'Recorded byte total differs.')
    nodes_by_identity = {}
    for number, (name, record) in enumerate(sorted(cache['files'].items()), 1):
        require(wheels.safe_name(name) == record['filename'], 'Wheel filename binding differs.')
        print(f'Checking cached file {number}/90: {name}', flush=True)
        verified = wheels.verify_file(folder / name, record)
        require(verified == cache['verified_files'][name] == read_json(receipts / (name + '.json')),
                'Wheel receipt differs: ' + name)
        report['verified_files'][name] = verified
        report['verified_bytes'] += verified['bytes']
        inspected = wheels.inspect_wheel(folder / name, record, tags, environment['python_full_version'])
        require(inspected == cache['wheel_inspections'][name], 'Wheel metadata differs: ' + name)
        identity = (canonicalize_name(record['package']), Version(record['version']))
        require(identity not in nodes_by_identity, 'Duplicate wheel package/version.')
        nodes_by_identity[identity] = inspected['metadata']
        report['inspected_files'][name] = inspected
        save_json('REPORT.json', report)

    nodes = {}
    for name, version in pins.items():
        key = (name, Version(version))
        require(key in nodes_by_identity, 'Candidate wheel metadata missing: ' + name)
        nodes[name] = nodes_by_identity[key]
    print('Checking course and full isolated dependencies using the current Python markers.', flush=True)
    course = dependencies.inspect_chain(course_lines, nodes.get, environment)
    full = dependencies.inspect_chain([f'{n}=={v}' for n, v in sorted(pins.items())], nodes.get, environment)
    save_json('COURSE_DEPENDENCY_CHECK.json', course)
    save_json('ISOLATED_DEPENDENCY_CHECK.json', full)
    report['course_dependencies'] = {'packages': len(course['nodes']), 'passed': course['required_chain_passed'],
                                     'issues': course['issues']}
    report['isolated_dependencies'] = {'packages': len(full['nodes']), 'passed': full['required_chain_passed'],
                                       'issues': full['issues']}
    require(course['required_chain_passed'] and full['required_chain_passed'],
            'Current dependency closure failed; inspect the saved dependency reports.')
    require({k: v['version'] for k, v in course['nodes'].items()} ==
            {k: v for k, v in pins.items() if k != 'pip'}, 'Current course dependency set differs from the candidate.')
    require({k: v['version'] for k, v in full['nodes'].items()} == pins,
            'Full isolated dependency set differs from the candidate.')
    save_json('CANDIDATE_ISOLATED_VERSIONS.json', pins)
    report['course_pins'] = course_pins
    report['candidate_versions'] = {n: pins[n] for n in
        ['torch', 'transformers', 'tokenizers', 'trl', 'peft', 'huggingface-hub', 'bitsandbytes']}
    report['nonisolated_cached_files_checked'] = [n for n, v in cache['files'].items()
        if pins.get(canonicalize_name(v['package'])) != v['version']]
    require(digest(prior_path) == report['previous_report_sha256'] and digest(cache_path) == cache_sha,
            'An input record changed during validation.')
    report['status'] = 'CPU_WHEEL_CACHE_VALIDATED_INSTALLATION_APPROVAL_PENDING'
except Exception as exc:
    report['status'] = 'CPU_WHEEL_CACHE_STOPPED_REVIEW_PENDING'
    report['errors'].append(type(exc).__name__ + ': ' + str(exc))
finally:
    if before is not None:
        after = package_inventory()
        save_json('BASE_PACKAGES_AFTER.json', after)
        report['base_package_inventory_unchanged'] = before == after
        if before != after:
            report['status'] = 'CPU_WHEEL_CACHE_STOPPED_REVIEW_PENDING'
            report['errors'].append('Base package inventory changed during validation.')
    save_json('REPORT.json', report)
    print('Status:', report['status'])
    print('Verified files:', len(report['verified_files']), '/ 90')
    print('Inspected metadata:', len(report['inspected_files']), '/ 90')
    print('Verified bytes:', report['verified_bytes'])
    print('Course dependencies:', json.dumps(report.get('course_dependencies', {})))
    print('Isolated dependencies:', json.dumps(report.get('isolated_dependencies', {})))
    print('Candidate versions:', json.dumps(report.get('candidate_versions', {}), indent=2))
    print('Base package inventory unchanged:', report.get('base_package_inventory_unchanged'))
    print('Errors:', report['errors'])
    print('Saved report:', attempt / 'REPORT.json')
    print('0 installs; 0 downloads; 0 environments created; 0 model loads; 0 responses; 0 AI judgments.')
    print('A100 validation and environment installation still require review and approval.')
