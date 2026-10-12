"""Verify the resumed A100 and inspect only the original judge; no inference."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path('/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1')
LOCAL = Path('/content/task4_isolated_runtime_v1')
RUNTIME_ATTEMPT = '20261012T051358.122710Z'
RUNTIME_SOURCE_SHA = '0a4fbc85656b74e01c145a2476ae058f9687e6bcf3cc21b3815716a9da67115d'
RESUME_SOURCE_SHA = '818224dbe9f5084839db920aceeb0de785db6bb77adecc8fb76039057637706d'
RESTORE_SOURCE_SHA = 'a3025083ed0094652e39e306c7d2bdaf226d025a651e705cf3f8f37b17ac130e'
JUDGE_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
SUCCESS = 'RESUMED_JUDGE_LOADED_AND_VALIDATED_NO_INFERENCE'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or linked evidence: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def verify_runtime(root, helper):
    folder = root / 'resume_post_restart_attempts' / RUNTIME_ATTEMPT
    report = read(folder / 'REPORT.json')
    require(report['status'] == 'RESUMED_BOTH_INTERPRETERS_VALIDATED_AFTER_RESTART_NO_INFERENCE'
            and report['source_sha256'] == RUNTIME_SOURCE_SHA
            and report['kernel_restart_verified'] is True
            and report['base_full_dependency_check_passed'] is True
            and report['isolated_full_dependency_check_passed'] is True
            and report['fresh_imports_cuda_nf4_passed'] is True
            and report['synthetic_checks_passed'] == 37
            and report['base_package_names'] == 724
            and report['base_duplicate_groups'] == 16
            and report['isolated_package_count'] == 84
            and report['gpu_exceptions_accepted'] == []
            and report['package_changes_performed'] is False
            and report['historical_contracts_rewritten'] is False
            and report['pretrained_model_loads'] == report['generation_calls'] == report['judge_calls'] == 0,
            'Saved post-restart success differs from the returned result')
    for index, (relative, expected) in enumerate(report['artifact_sha256'].items(), 1):
        require(sha(helper.safe_file(folder, relative)) == expected,
                'Saved current-runtime artifact differs: ' + relative)
        if index % 10 == 0:
            print('  Runtime artifacts verified:', index, '/', len(report['artifact_sha256']), flush=True)
    require(sha(folder / 'POST_RESTART_SOURCE.py') == RUNTIME_SOURCE_SHA,
            'Saved post-restart source differs')
    path = folder / 'CURRENT_RUNTIME_VALIDATION_CONTRACT.json'
    require(sha(path) == report['current_runtime_validation_contract_sha256'],
            'Current runtime validation contract differs')
    current = read(path)
    require(current['format'] == 'ATML_TASK4_RESUMED_CURRENT_RUNTIME_V1'
            and current['source_sha256'] == RUNTIME_SOURCE_SHA
            and current['kernel_restart_verified'] is True
            and current['actual_notebook_tools_and_jedi_completion_passed'] is True
            and current['base_full_dependency_check_passed'] is True
            and current['isolated_full_dependency_check_passed'] is True
            and current['fresh_imports_cuda_nf4_passed'] is True
            and current['synthetic_checks_passed'] == 37
            and current['runtime_ready_for_judge_loader_validation'] is True
            and current['judge_model_loading_validation_pending'] is True
            and current['gpu_exceptions_accepted'] == []
            and current['generation_complete'] is True,
            'Saved current-runtime validation is incomplete')
    for filename, field in [('BASE_AFTER_INVENTORY.json', 'base_inventory_sha256'),
                            ('BASE_LAYOUT.json', 'base_layout_sha256'),
                            ('ISOLATED_AFTER.json', 'isolated_inventory_sha256')]:
        require(sha(folder / filename) == current[field], 'Bound inventory differs: ' + filename)
    for relative, expected in current['historical_metadata_sha256'].items():
        require(sha(helper.safe_file(root, relative)) == expected,
                'Historical generation/preparation evidence differs: ' + relative)
    return folder, path, current


def validate_current_identity(current, owner, notebook_pid, executable, python_version, local):
    require(notebook_pid == current['notebook_pid'],
            'Notebook restarted or changed after the successful current-runtime check; return output for review')
    require(executable == current['base_python_executable']
            and python_version == current['python']
            and current['isolated_python_executable'] == str(local / 'bin/python'),
            'Current interpreter identity differs')
    require(owner['source_sha256'] == RESTORE_SOURCE_SHA
            and owner['directory'] == str(local)
            and owner['system_site_packages'] is False
            and owner['approval_sha256'] == current['approval_sha256']
            and owner['wheelhouse_contract_sha256'] == current['wheelhouse_contract_sha256']
            and owner['target_python_full_version'] == current['python']
            and owner['base_python_executable'] == str(Path(executable).resolve())
            and owner['hardware_gpu_uuids'] == [item['uuid'] for item in current['hardware']],
            'Resumed local environment ownership differs')


def validate_judge_result(result, historical):
    require(result['role'] == 'judge'
            and result['status'] == 'MODEL_LOADED_AND_INSPECTED_NO_INFERENCE'
            and result['model_load_completed'] is True
            and result['forward_passes'] == result['generate_calls']
                == result['responses_generated'] == result['judgments_produced'] == 0
            and result['training_performed'] is False
            and result['loader_settings_changed'] is False,
            'Original judge load/inspection was incomplete or performed inference')
    signature = {key: value for key, value in result.items() if not key.startswith('cuda_peak_')}
    require(signature == historical, 'Judge settings or inspection differ from original validated signature')
    return signature


def main():
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    helper_path = ROOT / 'resume_preflight_attempts/20261012T040418.708090Z/RESUME_CHECK_SOURCE.py'
    require(sha(helper_path) == RESUME_SOURCE_SHA, 'Saved resume helper differs')
    helper = load('task4_judge_resume_helpers', helper_path)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    attempt = ROOT / 'resume_judge_loading_attempts' / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    report = {'format': 'ATML_TASK4_RESUMED_JUDGE_LOADING_V1', 'attempt_utc': stamp,
              'source_sha256': sha(__file__), 'status': 'RUNNING', 'artifact_sha256': {},
              'commands': [], 'package_changes_performed': False, 'downloads': 0,
              'environment_created': False, 'model_load_attempts_started': [],
              'model_load_calls_returned': 0, 'forward_passes': 0, 'generation_calls': 0,
              'judge_calls': 0, 'training_performed': False, 'historical_contracts_rewritten': False,
              'gpu_exceptions_accepted': [], 'responses_last_verified': 1800,
              'judgments_last_verified': 0, 'human_labels_last_confirmed': 0}

    def save(name, value):
        path = attempt / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = value if isinstance(value, bytes) else (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        require(not path.exists() and not path.is_symlink(), 'Refusing evidence overwrite')
        temporary = path.parent / ('.pending-' + path.name)
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        require(not path.exists(), 'Concurrent writer created evidence')
        os.replace(temporary, path)
        require(path.read_bytes() == data, 'Evidence readback failed')
        if name != 'REPORT.json':
            report['artifact_sha256'][name] = sha(path)

    def command(args, label, environment):
        event = {'stage': label, 'argv': [str(x) for x in args]}
        save(label + '_COMMAND.json', event)
        print('Judge loading validation:', label, flush=True)
        log_path = attempt / (label + '.log')
        with log_path.open('x') as log:
            process = subprocess.Popen(event['argv'], cwd='/content', env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            log.flush()
            os.fsync(log.fileno())
        event.update(returncode=code, log_sha256=sha(log_path))
        report['commands'].append(event)
        save(label + '_RESULT.json', event)
        require(code == 0, label + ' failed; no automatic fallback or package repair')

    try:
        save('JUDGE_LOADING_SOURCE.py', Path(__file__).read_bytes())
        print('Step 1/4: verifying saved post-restart success and original source releases.', flush=True)
        runtime_folder, runtime_path, current = verify_runtime(ROOT, helper)
        runtime_report_sha = sha(runtime_folder / 'REPORT.json')
        runtime_contract_sha = sha(runtime_path)
        folders = helper.verify_sources(ROOT)
        print('All 19 original source releases verified.', flush=True)
        for kind, expected in current['sealed_sources_used'].items():
            require(helper.SOURCES[kind] == expected, 'Runtime helper release differs: ' + kind)
        first = folders['preflight']
        task_approval = first / 'docs/approval.json'
        require(read(task_approval)['approved'] is True
                and read(ROOT / 'OWNER.json')['approval_sha256'] == sha(task_approval)
                and sha(folders['model_loading'] / 'TASK4_APPROVAL.json') == sha(task_approval),
                'Original model-preparation approval differs')
        require(sha(folders['post_restart'] / 'APPROVAL.json') == current['approval_sha256']
                and read(folders['post_restart'] / 'APPROVAL.json')['approved'] is True,
                'Existing runtime setup approval differs')
        guards = load('task4_judge_resume_isolation_guards', folders['isolated_gpu'] / 'guards.py')
        policy = load('task4_judge_resume_child_policy', folders['isolated_recheck'] / 'isolation.py')
        gates = load('task4_judge_resume_gpu_gates', folders['gpu_environment'] / 'gates.py')
        loading = load('task4_judge_resume_model_guards', folders['model_loading'] / 'loading_guards.py')
        require(LOCAL.is_dir() and not LOCAL.is_symlink()
                and sha(LOCAL / '.task4_isolation_owner.json') == current['local_owner_sha256'],
                'Owned local runtime missing or changed; retain this A100')
        owner = read(LOCAL / '.task4_isolation_owner.json')
        validate_current_identity(current, owner, os.getpid(), sys.executable, platform.python_version(), LOCAL)
        pins = current['isolated_package_versions']
        require(len(pins) == 84 and pins == read(folders['isolated_gpu'] / 'APPROVED_ISOLATED_VERSIONS.json'),
                'Approved exact isolated versions differ')
        base_env = guards.clean_environment(os.environ)
        child_env = policy.child_environment(os.environ, guards.clean_environment)
        parent_backend = os.environ.get('MPLBACKEND')
        python = LOCAL / 'bin/python'
        inspect = folders['isolated_gpu'] / 'inspect_python.py'
        print('Step 2/4: confirming the same A100 and unchanged validated environments.', flush=True)
        gpu = helper.command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,uuid',
                              '--format=csv,noheader,nounits'])
        save('CURRENT_GPU_QUERY.json', gpu)
        require(gpu.get('returncode') == 0
                and gates.parse_a100_inventory(gpu['stdout']) == current['hardware'],
                'A100 assignment changed after current-runtime validation')
        base_before = helper.inventory(sys.executable)
        save('BASE_BEFORE_INVENTORY.json', base_before)
        require(base_before == read(runtime_folder / 'BASE_AFTER_INVENTORY.json'),
                'Base inventory changed after successful current-runtime validation')
        command([sys.executable, '-I', '-B', folders['base_repair_recheck'] / 'scan_base_layout.py',
                 folders['base_layout_diagnostic'] / 'scan_layout.py', attempt / 'BASE_LAYOUT.json'],
                'base_layout', base_env)
        require(read(attempt / 'BASE_LAYOUT.json') == read(runtime_folder / 'BASE_LAYOUT.json'),
                'Base metadata layout or duplicate selection changed')
        command([python, '-I', '-B', inspect, attempt / 'ISOLATED_BEFORE.json'],
                'isolated_inventory_before', child_env)
        iso_before = read(attempt / 'ISOLATED_BEFORE.json')
        guards.validate_inventory(iso_before, pins, LOCAL, complete=True)
        require(iso_before == read(runtime_folder / 'ISOLATED_AFTER.json'),
                'Isolated inventory changed after current-runtime validation')
        inputs_path = ROOT / 'CPU_MODEL_INPUT_CONTRACT.json'
        inputs = read(inputs_path)
        require(inputs['source_manifest_sha256'] == helper.SOURCES['model_inputs']
                and inputs['policy_order'] == ['sft', 'dpo', 'ppo', 'grpo'] and inputs['sft_adapter'] is None,
                'Frozen model input contract differs')
        judge_path = ROOT / 'JUDGE_SNAPSHOT_CONTRACT.json'
        require(sha(judge_path) == inputs['judge_snapshot_contract_sha256'], 'Bound judge snapshot differs')
        judge = read(judge_path)
        require(judge['revision'] == JUDGE_REVISION
                and judge['directory'] == str(ROOT / 'models/judge')
                and len(judge['verified_files']) == 9, 'Pinned judge revision/directory/file coverage differs')
        original_models = read(ROOT / 'MODEL_LOADING_VALIDATION_CONTRACT.json')
        require(original_models['source_manifest_sha256'] == helper.SOURCES['model_loading']
                and original_models['all_five_models_loaded_and_inspected'] is True
                and original_models['gpu_exceptions_accepted'] == []
                and original_models['task4_approval_sha256'] == sha(task_approval)
                and original_models['model_input_contract_sha256'] == sha(inputs_path),
                'Historical original model-validation identity differs')
        print('Step 3/4: freshly hashing all 9 cached judge files, then loading only the original judge.', flush=True)
        verified_count = 0

        def snapshot_digest(path):
            nonlocal verified_count
            print('  Hashing judge file:', Path(path).name, flush=True)
            value = sha(path)
            verified_count += 1
            print('  Judge file hashes completed:', verified_count, '/9', flush=True)
            return value

        snapshot = loading.verify_snapshot(judge['directory'], judge['verified_files'], snapshot_digest)
        save('VERIFIED_JUDGE_FILES.json', snapshot)
        require(sum(entry['bytes'] for entry in snapshot.values()) == 6183451098,
                'Pinned judge total byte count differs')
        report['model_load_attempts_started'].append('judge')
        save('JUDGE_LOAD_STARTED.json', {'role': 'judge', 'source_sha256': report['source_sha256']})
        output = attempt / 'JUDGE_MODEL.json'
        try:
            command([python, '-I', '-B', '-u', folders['model_loading'] / 'probe_models.py',
                     first / 'course_reference', inputs_path, 'judge', output], 'original_judge_loading', child_env)
        finally:
            if output.exists() and read(output).get('model_load_completed') is True:
                report['model_load_calls_returned'] = 1
        result = read(output)
        signature = validate_judge_result(result, original_models['validated_models']['judge'])
        print('Original NF4 judge loaded, inspected and matched to the frozen signature. No inference.', flush=True)
        print('Step 4/4: confirming unchanged environments and saving separate judge-load evidence.', flush=True)
        base_after = helper.inventory(sys.executable)
        save('BASE_AFTER_INVENTORY.json', base_after)
        command([python, '-I', '-B', inspect, attempt / 'ISOLATED_AFTER.json'],
                'isolated_inventory_after', child_env)
        iso_after = read(attempt / 'ISOLATED_AFTER.json')
        guards.validate_inventory(iso_after, pins, LOCAL, complete=True)
        require(base_after == base_before and iso_after == iso_before
                and os.environ.get('MPLBACKEND') == parent_backend
                and sha(LOCAL / '.task4_isolation_owner.json') == current['local_owner_sha256'],
                'Environment ownership/inventory or parent plotting setting changed during judge loading')
        for relative, expected in current['historical_metadata_sha256'].items():
            require(sha(helper.safe_file(ROOT, relative)) == expected,
                    'Historical generation/preparation evidence changed: ' + relative)
        require(sha(runtime_path) == runtime_contract_sha
                and sha(runtime_folder / 'REPORT.json') == runtime_report_sha,
                'Saved current-runtime evidence changed')
        contract = {'format': 'ATML_TASK4_RESUMED_JUDGE_LOADING_VALIDATION_V1',
            'source_sha256': report['source_sha256'], 'task4_approval_sha256': sha(task_approval),
            'current_runtime_report_sha256': runtime_report_sha,
            'current_runtime_validation_contract_sha256': runtime_contract_sha,
            'current_runtime_attempt': RUNTIME_ATTEMPT,
            'model_input_contract_sha256': sha(inputs_path), 'judge_snapshot_contract_sha256': sha(judge_path),
            'original_model_loading_contract_sha256': sha(ROOT / 'MODEL_LOADING_VALIDATION_CONTRACT.json'),
            'judge_probe_sha256': sha(output), 'judge_model_loading_source_manifest_sha256': helper.SOURCES['model_loading'],
            'validated_judge_signature': signature, 'judge_revision': JUDGE_REVISION,
            'judge_files_verified': 9, 'judge_file_bytes': 6183451098,
            'snapshot_checksums_freshly_verified': True, 'original_nf4_float16_judge_load_inspection_passed': True,
            'model_signature_matches_historical_validation': True, 'hardware': current['hardware'],
            'notebook_pid': os.getpid(), 'local_owner_sha256': current['local_owner_sha256'],
            'base_python_executable': sys.executable, 'isolated_python_executable': str(python),
            'base_inventory_sha256': sha(attempt / 'BASE_AFTER_INVENTORY.json'),
            'isolated_inventory_sha256': sha(attempt / 'ISOLATED_AFTER.json'),
            'isolated_package_versions': pins, 'base_and_isolated_environment_unchanged': True,
            'historical_metadata_sha256': current['historical_metadata_sha256'], 'gpu_exceptions_accepted': [],
            'forward_passes_performed': False, 'generation_or_judging_performed': False,
            'judge_process_exited_after_inspection': True, 'ready_for_judging_execution': True,
            'judging_execution_pending': True}
        save('JUDGE_LOADING_VALIDATION_CONTRACT.json', contract)
        report.update(status=SUCCESS, judge_loading_validation_contract_sha256=sha(attempt / 'JUDGE_LOADING_VALIDATION_CONTRACT.json'),
            current_runtime_report_sha256=runtime_report_sha, current_runtime_validation_contract_sha256=runtime_contract_sha,
            judge_snapshot_freshly_verified=True, original_judge_load_inspection_passed=True,
            model_signature_matches_historical_validation=True, base_and_isolated_environment_unchanged=True)
    except BaseException as error:
        import traceback
        report['status'] = 'RESUMED_JUDGE_LOADING_STOPPED_EVIDENCE_PRESERVED'
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    for path in sorted(attempt.rglob('*')):
        if path.is_file() and not path.name.startswith('.pending-') and path.name != 'REPORT.json':
            report['artifact_sha256'][str(path.relative_to(attempt))] = sha(path)
    save('REPORT.json', report)
    print('Saved report:', attempt / 'REPORT.json', flush=True)
    print('Status:', report['status'], flush=True)
    if report['status'] == SUCCESS:
        print('Judge files 9/9 freshly verified; original NF4/float16 judge load and inspection PASS.', flush=True)
        print('The inspection process exited and released the model. Actual judging remains pending.', flush=True)
    else:
        print(json.dumps(report['error'], indent=2), flush=True)
    print('Last verified responses: 1800/1800; judgments: 0/1800; human labels last confirmed: 0/240.', flush=True)
    print('No installations, downloads, forward passes, response generation, judging or training.', flush=True)
    print('Keep this A100 and return the full output.', flush=True)


if __name__ == '__main__':
    main()
