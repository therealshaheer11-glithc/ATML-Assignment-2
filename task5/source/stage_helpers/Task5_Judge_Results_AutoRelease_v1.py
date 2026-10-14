"""Approved Task 5 setup, 880 comparison events, quantitative exports and safe release."""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import types
import uuid

PROJECT = Path('/content/drive/MyDrive/ATML-Assignment-2')
PYTHON = Path('/content/task5_isolated_runtime_v1/bin/python')
RELEASE_SHA = 'd6dd999e7fdf241c7136ee13f46babdf13338fa6dc12ff9046d2ea6a2e89fd87'
SETUP_SHA = 'cace5df894e95d9e7d4d7064da06f32edc183500b0df6fb64f7919b36085ff9d'
PROTOCOL_SHA = '58632138b81533416b3bb46979e03ca508323dd56dbbba572e0249d1b1d8665a'
GENERATION_PATH = 'generation_attempts/20261013T185651.698351Z_3acc1fba/REPORT.json'
GENERATION_SHA = '70e94700d334fc67febeb54be36edb8a767e06f70132b47166da13e334bc98e4'
AUDIT_PATH = 'resume_audit_attempts/20261013T181642.449932Z_ac16e163/REPORT.json'
AUDIT_SHA = '3c4956879b74cc7b7dbdaaa2df4c564377bbd390758853c7aced813c9e1918e2'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    require(path.is_file() and not path.is_symlink(), 'Missing/linked file: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def checked(path, expected):
    require(digest(path) == expected, 'File checksum differs: ' + str(path))
    return json.loads(path.read_text())


def verify_source(folder, expected):
    require(folder.is_dir() and not folder.is_symlink(), 'Missing/linked source directory.')
    manifest = checked(folder / 'MANIFEST.json', expected)
    for name, value in manifest.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source path.')
        require(digest(folder / name) == value, 'Saved source differs: ' + name)


def write_blob(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), 'Refusing to replace existing evidence.')
    temporary = path.parent / ('.pending_' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(temporary.read_bytes() == data, 'Temporary evidence readback failed.')
    os.replace(temporary, path)
    require(path.read_bytes() == data, 'Saved evidence readback failed.')


def write_csv(path, rows):
    require(bool(rows), 'Empty result table.')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write_blob(path, stream.getvalue().encode('utf-8'))


def export_results(project, release, attempt, aggregate_path, core):
    result = core.read_json(aggregate_path)
    require(result['release_sha256'] == RELEASE_SHA and result['ai_comparison_events'] == 880
            and aggregate_path.name == core.checksum(result) + '.json', 'Canonical aggregate identity differs.')
    policy_rows, transfer_rows, agreement_rows, diagnostic_rows = [], [], [], []
    for policy in core.POLICIES:
        for domain in core.DOMAINS:
            value = result['policies'][policy][domain]
            require(value['responses'] == core.COUNTS[domain], 'Policy result denominator differs.')
            pair = value['ai_comparison']
            require((policy == 'sft' and pair is None) or
                    (policy != 'sft' and pair is not None and pair['pairs'] == core.COUNTS[domain]),
                    'Pairwise result denominator differs.')
            row = {'policy': policy, 'domain': domain, 'responses': value['responses'],
                   'exact_accuracy': value['exact_accuracy'], 'format_compliance': value['format_compliance'],
                   'response_tokens_mean': value['length_mean'], 'response_tokens_population_sd': value['length_population_sd'],
                   'truncated_responses': value['truncated_responses'], 'generation_seconds': value['generation_seconds'],
                   'generated_tokens': value['generated_tokens'],
                   'generated_tokens_per_generation_second': value['generated_tokens_per_generation_second'],
                   'ai_reference': 'SFT reference baseline' if policy == 'sft' else 'same-domain SFT',
                   'ai_pairs': pair['pairs'] if pair else None, 'ai_win_rate_half_ties': pair['ai_win_rate'] if pair else None,
                   'ai_wins': pair['wins'] if pair else None, 'ai_ties': pair['ties'] if pair else None,
                   'ai_losses': pair['losses'] if pair else None,
                   'verifier_judge_agreement': pair['verifier_judge_agreement'] if pair else None,
                   'unparseable_ai_outputs': pair['unparseable_ai_outputs'] if pair else None,
                   'decisive_verifier_pairs': pair['decisive_verifier_pairs'] if pair else None,
                   'secondary_agreement_decisive_verifier': pair['secondary_agreement_decisive_verifier'] if pair else None}
            policy_rows.append(row)
            if pair:
                for verifier_label, cells in pair['verifier_by_ai_confusion_matrix'].items():
                    for ai_label, count in cells.items():
                        agreement_rows.append({'policy': policy, 'domain': domain,
                            'verifier_preference': verifier_label, 'ai_preference': ai_label, 'count': count})
        transfer_rows.append({'policy': policy, **result['policies'][policy]['transfer_drop']})
    b = core.read_json(release / 'bindings.json')
    prior = checked(project / b['block6_report']['path'], b['block6_report']['sha256'])['summary']
    require(prior['responses_scored'] == 100 and prior['unique_controlled_pairs'] == 80,
            'Prior controlled-verifier denominators differ.')
    diagnostics = result['diagnostics']
    require(diagnostics['unique_pairs'] == 80, 'Controlled AI diagnostic denominator differs.')
    for mechanism in ('exact_verifier', 'ai_judge'):
        source = prior if mechanism == 'exact_verifier' else diagnostics
        categories = {'clean_correct': source['clean_reference_all_contrasts'], **source['contrasts']}
        for category, value in categories.items():
            require(value['pairs'] == (80 if category == 'clean_correct' else 20), 'Diagnostic category size differs.')
            diagnostic_rows.append({'mechanism': mechanism, 'category': category, 'pairs': value['pairs'],
                'clean_preference_rate': value['clean_preference_rate'] if mechanism == 'exact_verifier' else value['win_fraction'],
                'tie_rate': value['tie_rate'] if mechanism == 'exact_verifier' else value['tie_fraction'],
                'modified_preference_rate': value['modified_preference_rate'] if mechanism == 'exact_verifier' else value['loss_fraction'],
                'clean_pairwise_score_half_ties': value['clean_pairwise_score_with_half_ties'] if mechanism == 'exact_verifier' else value['ai_win_rate'],
                'category_denominators_overlap': True})
    paths = []
    for filename, rows in (('POLICY_RESULTS.csv', policy_rows), ('TRANSFER_RESULTS.csv', transfer_rows),
                           ('VERIFIER_AI_CONFUSION.csv', agreement_rows), ('DIAGNOSTIC_RESULTS.csv', diagnostic_rows)):
        path = attempt / filename
        write_csv(path, rows)
        paths.append(path)
    sensitivity = {'exact_verifier': {'S_reason': prior['verifier_S_reason'], 'S_outcome': prior['verifier_S_outcome']},
                   'ai_judge': {'S_reason': diagnostics['S_reason'], 'S_outcome': diagnostics['S_outcome']},
                   'pairs_per_sensitivity': 20, 'category_denominators_overlap': True,
                   'interpretation': 'Reasoning-only verifier ties are expected; filler ties are acceptable. Modified filler preference may indicate style bias.'}
    cost = {k: result[k] for k in ('ai_comparison_events', 'ai_model_generate_calls',
                                  'ai_inference_seconds', 'ai_inference_output_tokens')}
    cost['policy_generation_seconds'] = sum(row['generation_seconds'] for row in policy_rows)
    cost['policy_generated_tokens'] = sum(row['generated_tokens'] for row in policy_rows)
    cost['monetary_cost_measured'] = False
    for filename, value in (('DIAGNOSTIC_SENSITIVITY.json', sensitivity), ('INFERENCE_COUNTS_AND_TIME.json', cost)):
        path = attempt / filename
        core.write_once(path, value)
        paths.append(path)
    index = {'format': 'ATML_TASK5_QUANTITATIVE_RESULTS_V1', 'release_sha256': RELEASE_SHA,
             'policy_answers': 1200, 'ai_comparison_events': 880, 'qualitative_evidence_pending': True,
             'aggregate': {'path': str(aggregate_path), 'sha256': digest(aggregate_path)},
             'exports': {p.name: {'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size} for p in paths}}
    core.write_once(attempt / 'RESULTS_INDEX.json', index)
    return index


def worker(phase, project, release, attempt):
    sys.path.insert(0, str(release))
    from task5_run import core
    from task5_run.run import judge, aggregate, verify_source as verify_evaluation
    require(verify_evaluation(release) == RELEASE_SHA, 'Worker evaluation source differs.')
    if phase == 'judge':
        from task5_run.runtime import execution_gate
        audit = checked(project / 'task5/experiment_v1' / AUDIT_PATH, AUDIT_SHA)
        require(execution_gate(project, release, 'ai_judging') == audit['saved_hardware'],
                'Full runtime identity differs; no judge model loaded.')
        judge(project, release)
    elif phase == 'aggregate':
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            aggregate(project, release)
        lines = capture.getvalue().splitlines()
        print(capture.getvalue(), end='', flush=True)
        require(len(lines) == 1 and lines[0].startswith('Saved aggregate: '), 'Unexpected aggregate output.')
        aggregate_path = Path(lines[0][len('Saved aggregate: '):])
        expected_folder = project / 'task5/experiment_v1/evaluation_v1/aggregate'
        require(aggregate_path.parent == expected_folder, 'Aggregate destination differs.')
        export_results(project, release, attempt, aggregate_path, core)
        print('Saved policy, transfer, agreement and diagnostic result tables.', flush=True)
    else:
        raise RuntimeError('Unexpected worker phase.')


def clean_environment(original):
    require(original.get('MPLBACKEND') in {None, '', 'module://matplotlib_inline.backend_inline'}, 'Unexpected plotting backend.')
    env = {k: v for k, v in original.items() if not k.startswith(('PYTHON', 'PIP_')) and k != 'VIRTUAL_ENV'}
    env.pop('MPLBACKEND', None)
    env.update({'PIP_CONFIG_FILE': os.devnull, 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    return env


def stop_owned(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=30)
        return True
    return False


def saved_judgments(root, audit, core):
    folder = root / 'evaluation_v1/judgments'
    if not folder.exists():
        return 0
    require(folder.is_dir() and not folder.is_symlink() and (folder / 'units').is_dir(), 'Unbound/linked judgment store.')
    binding = core.read_json(folder / 'BINDING.json')
    require(binding['phase'] == 'ai_judging' and binding['release_sha256'] == RELEASE_SHA
            and binding['hardware'] == audit['saved_hardware'] and binding['canonical_judge'] == 'Qwen/Qwen2.5-3B-Instruct'
            and binding['max_new_tokens'] == 4 and len(binding['ordered_spec_sha256']) == 880,
            'Saved judgment protocol/runtime differs.')
    records = core.Journal(folder, binding).records()
    require(len(records) <= 880 and all(r['rng_after'] is not None for r in records), 'Saved judgment count/RNG differs.')
    return len(records)


def main():
    from google.colab import drive, runtime
    smi = shutil.which('nvidia-smi')
    require(smi is not None, 'Select and connect an A100 runtime first.')
    gpu = subprocess.run([smi, '--query-gpu=name,driver_version', '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
    hardware_rows = list(csv.reader(io.StringIO(gpu.stdout.strip())))
    require(gpu.returncode == 0 and len(hardware_rows) == 1 and len(hardware_rows[0]) == 2
            and 'A100' in hardware_rows[0][0], 'Exactly one A100 is required.')
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    root = PROJECT / 'task5/experiment_v1'
    release = root / 'evaluation_releases' / RELEASE_SHA
    setup = root / 'a100_setup_releases' / SETUP_SHA
    verify_source(release, RELEASE_SHA)
    core = types.ModuleType('verified_task5_feedback_persistence')
    exec(compile((release / 'task5_run/core.py').read_bytes(), 'verified_task5_core.py', 'exec'), core.__dict__)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid.uuid4().hex[:8]
    attempt = root / 'feedback_attempts' / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    source = globals().get('SOURCE_BYTES')
    if source is None:
        source = Path(__file__).read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    require(source_sha == globals().get('WRAPPER_SOURCE_SHA256', source_sha), 'Pipeline source differs.')
    write_blob(attempt / 'PIPELINE_SOURCE.py', source)
    report = {'status': 'TASK5_FEEDBACK_RUNNING', 'errors': [], 'commands': [], 'results_complete': False,
              'pipeline_source_sha256': source_sha, 'release_sha256': RELEASE_SHA,
              'generation_report_sha256': GENERATION_SHA, 'planned_ai_comparison_events': 880,
              'last_reported_ai_comparisons': None, 'policy_answers_generated': 0, 'downloads': 0,
              'training': False, 'protocol_changes': False, 'qualitative_evidence_pending': True}
    core.write_once(attempt / 'START.json', report)
    process = None
    def command(argv, name, env):
        nonlocal process
        report['current_phase'] = name
        core.write_once(attempt / (name + '_COMMAND.json'), {'argv': list(map(str, argv))})
        print('Stage:', name, flush=True)
        with (attempt / (name + '.log')).open('x') as log:
            process = subprocess.Popen(list(map(str, argv)), cwd='/content', env=env, start_new_session=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                event = re.fullmatch(r'AI comparisons: (\d+)/880 saved\s*', line)
                if event:
                    count = int(event[1])
                    require(count == report['last_reported_ai_comparisons'] + 1 and count <= 880, 'Unexpected judge progress.')
                    report['last_reported_ai_comparisons'] = count
                    if count % 25 == 0 or count == 880:
                        core.write_once(attempt / 'progress' / f'judged_{count:06d}.json', {'comparison_events_reported_saved': count})
                if not event or int(event[1]) == 1 or int(event[1]) % 25 == 0 or int(event[1]) == 880:
                    print(line, end='', flush=True)
            returncode = process.wait()
        event = {'phase': name, 'returncode': returncode, 'log_sha256': digest(attempt / (name + '.log'))}
        core.write_once(attempt / (name + '_RESULT.json'), event)
        report['commands'].append(event)
        require(returncode == 0, name + ' stopped; checkpoint evidence preserved.')
    try:
        audit = checked(root / AUDIT_PATH, AUDIT_SHA)
        generation = checked(root / GENERATION_PATH, GENERATION_SHA)
        require(generation['status'] == 'POLICY_GENERATION_COMPLETE_AI_JUDGING_PENDING'
                and not generation['errors'] and generation['worker_returncode'] == 0
                and generation['exact_final_count_checked'] is True and generation['completed_answers'] == 1200
                and generation['remaining_answers'] == 0 and generation['evaluation_release_sha256'] == RELEASE_SHA,
                'Completed generation evidence differs.')
        require(digest((root / GENERATION_PATH).parent / 'GENERATION.log') == generation['generation_log_sha256'],
                'Generation log differs.')
        require(digest(release / 'protocol.json') == PROTOCOL_SHA, 'Frozen protocol differs.')
        approval = core.read_json(root / 'EXECUTION_APPROVAL.json')
        require(approval.get('approved') is True and approval.get('ai_judging') is True
                and approval.get('quantitative_aggregation') is True and approval.get('evaluation_release_sha256') == RELEASE_SHA
                and approval.get('protocol_sha256') == PROTOCOL_SHA, 'Existing execution approval does not cover this run.')
        observed = {'python': platform.python_version(), 'gpu': hardware_rows[0][0].strip(), 'driver': hardware_rows[0][1].strip()}
        report['observed_runtime'] = observed
        require(all(observed[k] == audit['saved_hardware'][k] for k in observed),
                'New Python/GPU/driver differs from the saved run; installation not started.')
        contract_path = root / 'ISOLATED_ENVIRONMENT_CONTRACT.json'
        contract_sha = digest(contract_path)
        require(contract_sha == audit['prior_environment_contract_sha256'], 'Saved environment contract differs.')
        for name, total in generation['counts_after'].items():
            policy, domain = name.split('/')
            folder = root / 'evaluation_v1/generations' / policy / domain
            binding = core.read_json(folder / 'BINDING.json')
            paths = sorted((folder / 'units').glob('*.json'))
            require(total == core.COUNTS[domain] and len(paths) == total
                    and all(p.name == f'{i:06d}.json' for i, p in enumerate(paths))
                    and binding['release_sha256'] == RELEASE_SHA and binding['hardware'] == audit['saved_hardware'],
                    'Saved policy inventory differs: ' + name)
        report['previously_saved_ai_comparisons'] = saved_judgments(root, audit, core)
        report['last_reported_ai_comparisons'] = report['previously_saved_ai_comparisons']
        verify_source(setup, SETUP_SHA)
        before_setup = set((root / 'a100_setup_attempts').iterdir())
        env = clean_environment(os.environ)
        print('1,200 policy answers preserved. AI comparisons already saved:', report['previously_saved_ai_comparisons'], '/ 880', flush=True)
        command([sys.executable, '-s', '-B', setup / 'prepare_a100.py'], 'restore_verified_a100_environment', env)
        new_setups = set((root / 'a100_setup_attempts').iterdir()) - before_setup
        require(len(new_setups) == 1, 'Unexpected fresh setup evidence coverage.')
        setup_report_path = new_setups.pop() / 'REPORT.json'
        fresh = core.read_json(setup_report_path)
        require(fresh['status'] == 'A100_ISOLATED_ENVIRONMENT_READY_PRETRAINED_EXECUTION_APPROVAL_PENDING'
                and not fresh['errors'] and fresh['isolated_packages_verified'] == 84
                and fresh['base_package_inventory_unchanged'] is True
                and fresh['gpu_probe']['tests_passed'] is True and fresh['gpu_probe']['tests_run'] == 44
                and fresh['environment_contract_sha256'] == contract_sha == digest(contract_path),
                'Fresh A100 setup did not reproduce the approved environment.')
        report['fresh_setup_report'] = str(setup_report_path)
        report['fresh_setup_report_sha256'] = digest(setup_report_path)
        boot = r'''
import hashlib, sys, types
from pathlib import Path
source_path, expected, phase, project, release, attempt = sys.argv[1:]
payload = Path(source_path).read_bytes()
if hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError('Worker pipeline source checksum differs.')
module = types.ModuleType('verified_task5_feedback_worker')
exec(compile(payload, 'verified_feedback_pipeline.py', 'exec'), module.__dict__)
module.worker(phase, Path(project), Path(release), Path(attempt))
'''
        def worker_argv(phase):
            return [PYTHON, '-I', '-B', '-u', '-c', boot, attempt / 'PIPELINE_SOURCE.py', source_sha,
                    phase, PROJECT, release, attempt]
        if report['previously_saved_ai_comparisons'] < 880:
            command(worker_argv('judge'), 'judge_880_comparisons', env)
        report['verified_saved_ai_comparisons'] = saved_judgments(root, audit, core)
        require(report['verified_saved_ai_comparisons'] == 880, 'AI comparison schedule is incomplete.')
        cpu_env = dict(env, CUDA_VISIBLE_DEVICES='')
        command(worker_argv('aggregate'), 'calculate_and_export_results_on_cpu', cpu_env)
        index_path = attempt / 'RESULTS_INDEX.json'
        index = core.read_json(index_path)
        require(index['release_sha256'] == RELEASE_SHA and index['policy_answers'] == 1200
                and index['ai_comparison_events'] == 880, 'Result index totals differ.')
        for value in [index['aggregate'], *index['exports'].values()]:
            require(digest(Path(value['path'])) == value['sha256'], 'Saved result readback checksum differs.')
        report.update({'status': 'TASK5_QUANTITATIVE_RESULTS_SAVED_READY_TO_RELEASE', 'results_complete': True,
                       'results_index': str(index_path), 'results_index_sha256': digest(index_path),
                       'aggregate': index['aggregate'], 'remaining_ai_comparison_events': 0,
                       'runtime_release_not_yet_confirmed': True})
    except KeyboardInterrupt:
        report['status'] = 'TASK5_FEEDBACK_INTERRUPTED_CHECKPOINTS_PRESERVED'
        report['errors'].append('Interrupted; committed judgment checkpoints retained. No automatic release on incomplete results.')
    except Exception as exc:
        report['status'] = 'TASK5_FEEDBACK_STOPPED_REVIEW_PENDING'
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    finally:
        report['owned_worker_group_stopped'] = stop_owned(process)
        core.write_once(attempt / 'REPORT.json', report)
        print('Status:', report['status'])
        print('AI comparisons verified saved:', report.get('verified_saved_ai_comparisons', 'not fully checked'), '/ 880')
        print('Errors:', report['errors'])
        print('Saved report:', attempt / 'REPORT.json')
        print('Report SHA-256:', digest(attempt / 'REPORT.json'))
    if not report['results_complete']:
        print('Runtime kept connected for review. Send this output; existing checkpoints remain saved.')
        return
    result = core.read_json(Path(report['aggregate']['path']))
    print('Results index:', report['results_index'], flush=True)
    print('Results folder:', attempt, flush=True)
    for policy in core.POLICIES:
        for domain in core.DOMAINS:
            value = result['policies'][policy][domain]
            pair = value['ai_comparison']
            pair_text = f"{pair['ai_win_rate']:.4f}" if pair else 'SFT reference'
            print(f"{policy}/{domain}: accuracy={value['exact_accuracy']:.4f}; AI score={pair_text}; mean tokens={value['length_mean']:.2f}", flush=True)
    print('Diagnostic AI sensitivity:', result['diagnostics']['S_reason'], result['diagnostics']['S_outcome'], flush=True)
    receipt = attempt / 'DISCONNECT_READY.json'
    core.write_once(receipt, {'status': 'RESULTS_VERIFIED_READY_TO_FLUSH', 'report_sha256': digest(attempt / 'REPORT.json'),
                             'results_index_sha256': report['results_index_sha256'], 'ai_comparison_events': 880,
                             'runtime_release_success_not_yet_confirmed': True})
    print('All 880 comparisons and quantitative results saved. Flushing Drive...', flush=True)
    drive.flush_and_unmount(timeout_ms=60000)
    print('Drive flush succeeded. Requesting GPU runtime release now.', flush=True)
    runtime.unassign()


if __name__ == '__main__':
    main()
