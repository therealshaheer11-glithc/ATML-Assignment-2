# Task 5 - Block 5: record setup approval and inspect existing diagnostic text.
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, json, os, shutil, subprocess, uuid

project = Path('/content/drive/MyDrive/ATML-Assignment-2')
root = project / 'task5/experiment_v1'
previous = root / 'wheel_validation_attempts/20261012T173139.703528Z_6024d7df'
diagnostics = project / 'task3/experiment_v1/course_assets/data/task5_controlled_reward_diagnostics.jsonl'
diagnostic_sha = '9008b96595bb1419f1326bf02beefd689f09028c2aa6598d7c427394602f036c'
pins_sha = '9f695dba56dfdbc4d3597586fe8ab6c90d38fa20a8bc3bcfccf50422bfe030ab'
variants = ['clean_correct', 'corrupt_reasoning_correct_final', 'good_reasoning_wrong_final',
            'persuasive_filler_correct', 'gold_distractor_wrong_final']

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()

def read_json(path, expected=None):
    require(path.is_file() and not path.is_symlink(), 'Missing or linked input: ' + str(path))
    if expected is not None:
        require(digest(path) == expected, 'Input checksum differs: ' + str(path))
    return json.loads(path.read_text())

def inspect_groups(rows):
    require(len(rows) == 100, 'Expected the fixed 100 diagnostic responses.')
    groups = {}
    for row in rows:
        require(all(k in row for k in ['problem_id', 'variant_type', 'question', 'gold_final', 'response']),
                'Required diagnostic field missing.')
        require(isinstance(row['response'], str), 'Diagnostic response is not text.')
        pid, variant = str(row['problem_id']), row['variant_type']
        require(variant in variants, 'Unexpected diagnostic variant: ' + str(variant))
        group = groups.setdefault(pid, {})
        require(variant not in group, 'Duplicate problem/variant: ' + pid + '/' + variant)
        group[variant] = row
    require(len(groups) == 20, 'Expected the fixed 20 diagnostic problems.')
    structure = []
    for pid, group in groups.items():
        require(set(group) == set(variants), 'Incomplete diagnostic problem: ' + pid)
        require(len({json.dumps(r['question'], sort_keys=True) for r in group.values()}) == 1,
                'Question differs across variants: ' + pid)
        require(len({json.dumps(r['gold_final'], sort_keys=True) for r in group.values()}) == 1,
                'Gold answer differs across variants: ' + pid)
        clean = group['clean_correct']['response']
        wrong = group['good_reasoning_wrong_final']['response']
        same_prefix = (clean.rsplit('####', 1)[0].strip() == wrong.rsplit('####', 1)[0].strip()
                       if '####' in clean and '####' in wrong else None)
        structure.append({'problem_id': pid,
            'clean_vs_wrong_final_text_before_last_marker_identical': same_prefix,
            'marker_counts': {v: group[v]['response'].count('####') for v in variants},
            'course_metadata': {v: {k: group[v].get(k) for k in
                ['reasoning_quality', 'style', 'expected_exact_reward', 'manual_validation', 'staff_edit_required']}
                for v in variants}})
    return groups, structure

smi = shutil.which('nvidia-smi')
gpu = subprocess.run([smi, '--query-gpu=name', '--format=csv,noheader'],
                     capture_output=True, text=True, timeout=15) if smi else None
require(not ((gpu and gpu.returncode == 0 and gpu.stdout.strip()) or
             os.environ.get('COLAB_TPU_ADDR') or os.environ.get('TPU_NAME')),
        'Keep the same CPU / None runtime for this block.')
require(root.is_dir() and not root.is_symlink(), 'Task 5 Drive storage is unavailable.')
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid.uuid4().hex[:8]
attempt = root / 'diagnostic_structure_attempts' / stamp
attempt.mkdir(parents=True, exist_ok=False)
report = {'status': 'RUNNING', 'errors': [], 'installs': 0, 'downloads': 0,
    'environments_created': 0, 'model_loads': 0, 'policy_responses': 0, 'ai_judgments': 0,
    'diagnostic_scores_computed': 0, 'diagnostic_comparison_protocol_frozen': False,
    'source_data_modified': False, 'network_access_in_this_block': False}

def save_json(name, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    pending = attempt / (name + '.pending')
    pending.write_bytes(payload)
    os.replace(pending, attempt / name)
    require((attempt / name).read_bytes() == payload, 'Saved evidence readback failed: ' + name)

try:
    save_json('REPORT.json', report)
    prior = read_json(previous / 'REPORT.json')
    require(prior['status'] == 'CPU_WHEEL_CACHE_VALIDATED_INSTALLATION_APPROVAL_PENDING' and
            len(prior['verified_files']) == len(prior['inspected_files']) == 90 and
            prior['course_dependencies'] == {'packages': 83, 'passed': True, 'issues': []} and
            prior['isolated_dependencies'] == {'packages': 84, 'passed': True, 'issues': []} and
            prior['base_package_inventory_unchanged'] is True and not prior['errors'],
            'The returned cache-validation checkpoint differs.')
    report['previous_report_sha256'] = digest(previous / 'REPORT.json')
    pins = read_json(previous / 'CANDIDATE_ISOLATED_VERSIONS.json', pins_sha)
    approval = {'user_message': 'okay go ahead', 'approved': True,
        'scope': 'Exact 84-package isolated environment plan; offline cached installation and synthetic A100 checks',
        'candidate_versions_sha256': pins_sha, 'package_versions': pins,
        'local_environment': '/content/task5_isolated_runtime_v1',
        'minimum_free_bytes_before_installation': 11045288692,
        'a100_setup_deferred_until_cpu_readiness': True, 'base_package_changes_approved': False,
        'downloads_approved': False, 'pretrained_model_loading_approved': False,
        'policy_generation_approved': False, 'ai_judging_approved': False,
        'experimental_protocol_choices_approved': False}
    save_json('ENVIRONMENT_PLAN_APPROVAL.json', approval)
    report['environment_approval_sha256'] = digest(attempt / 'ENVIRONMENT_PLAN_APPROVAL.json')
    print('1/3: environment-plan approval saved; setup remains deferred.', flush=True)
    require(diagnostics.is_file() and not diagnostics.is_symlink() and digest(diagnostics) == diagnostic_sha,
            'Fixed diagnostic file identity differs.')
    rows = [json.loads(line) for line in diagnostics.read_text().splitlines() if line.strip()]
    groups, structure = inspect_groups(rows)
    save_json('DIAGNOSTIC_STRUCTURE.json', structure)
    report.update({'diagnostic_sha256': diagnostic_sha, 'rows': len(rows), 'problems': len(groups),
        'variant_counts': dict(Counter(r['variant_type'] for r in rows)),
        'identical_reasoning_prefix_clean_vs_wrong_final':
            sum(s['clean_vs_wrong_final_text_before_last_marker_identical'] is True for s in structure),
        'final_marker_count_histogram': dict(Counter(str(r['response'].count('####')) for r in rows)),
        'all_course_manual_validation_flags_true': all(r.get('manual_validation') is True for r in rows)})
    first_id = next(iter(groups))
    first_rows = [groups[first_id][v] for v in variants]
    save_json('FIRST_PROBLEM_VARIANTS_FOR_REVIEW.json', first_rows)
    require(digest(diagnostics) == diagnostic_sha, 'Diagnostic source changed during inspection.')
    print('2/3: diagnostic structure inspected; no reward scores or judgments computed.', flush=True)
    report['status'] = 'CPU_DIAGNOSTIC_STRUCTURE_SAVED_PROTOCOL_APPROVAL_PENDING'
except Exception as exc:
    report['status'] = 'CPU_DIAGNOSTIC_STRUCTURE_STOPPED_REVIEW_PENDING'
    report['errors'].append(type(exc).__name__ + ': ' + str(exc))
finally:
    save_json('REPORT.json', report)
    print('3/3: inspection checkpoint saved.')
    print('Status:', report['status'])
    print('Responses / problems:', report.get('rows'), '/', report.get('problems'))
    print('Variant counts:', json.dumps(report.get('variant_counts', {})))
    print('Identical reasoning text before final-answer marker (clean vs wrong final):',
          report.get('identical_reasoning_prefix_clean_vs_wrong_final'), '/ 20')
    print('Final-answer marker counts:', json.dumps(report.get('final_marker_count_histogram', {})))
    print('Course manual-validation flags all true:', report.get('all_course_manual_validation_flags_true'))
    if report['status'] == 'CPU_DIAGNOSTIC_STRUCTURE_SAVED_PROTOCOL_APPROVAL_PENDING':
        print('First problem in source order — existing course responses, not newly generated outputs:')
        print(json.dumps(first_rows, indent=2, ensure_ascii=False))
    print('Errors:', report['errors'])
    print('Saved report:', attempt / 'REPORT.json')
    print('0 installs; 0 downloads; 0 environments created; 0 model loads; 0 responses generated; 0 scores; 0 AI judgments.')
