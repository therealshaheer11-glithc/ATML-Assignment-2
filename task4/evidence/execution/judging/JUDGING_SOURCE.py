"""Task 4 fixed categorical judging with atomic per-response Drive persistence."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import queue
import subprocess
import sys
import threading

ROOT = Path('/content/drive/MyDrive/ATML-Assignment-2/task4/experiment_v1')
LOCAL = Path('/content/task4_isolated_runtime_v1')
LOAD_ATTEMPT = '20261012T052212.342979Z'
LOAD_SOURCE_SHA = '255b62f2e701444bcf05f850b1028c60289f218b6aa38323bfc520a435919904'
RESUME_SOURCE_SHA = '818224dbe9f5084839db920aceeb0de785db6bb77adecc8fb76039057637706d'
POLICIES = ['sft', 'dpo', 'ppo', 'grpo']


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or linked file: ' + str(path))
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


def immutable(path, value):
    path = Path(path)
    data = value if isinstance(value, bytes) else (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    require(not path.is_symlink(), 'Refusing linked evidence')
    if path.exists():
        require(path.is_file() and path.read_bytes() == data, 'Existing evidence differs; preserved: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / ('.pending-' + path.name + '-' + str(os.getpid()))
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(not path.exists(), 'Concurrent writer created evidence')
    os.replace(temporary, path)
    require(path.read_bytes() == data, 'Evidence readback failed')


def get_helpers():
    helper_path = ROOT / 'resume_preflight_attempts/20261012T040418.708090Z/RESUME_CHECK_SOURCE.py'
    require(sha(helper_path) == RESUME_SOURCE_SHA, 'Saved CPU resume helper differs')
    helper = load('task4_judging_resume_helpers', helper_path)
    loader_path = ROOT / 'resume_judge_loading_attempts' / LOAD_ATTEMPT / 'JUDGE_LOADING_SOURCE.py'
    require(sha(loader_path) == LOAD_SOURCE_SHA, 'Saved judge-loading runner differs')
    loader = load('task4_judging_loader_evidence', loader_path)
    return helper, loader


def stream_process(args, environment, log_path, label, working_dir='/content', heartbeat_seconds=30):
    with Path(log_path).open('x') as log:
        process = subprocess.Popen([str(x) for x in args], cwd=working_dir, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        channel = queue.Queue()

        def drain():
            try:
                for line in process.stdout:
                    channel.put(line)
            except BaseException as error:
                channel.put({'reader_error': str(error)})
            finally:
                channel.put(None)

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            while True:
                try:
                    line = channel.get(timeout=heartbeat_seconds)
                except queue.Empty:
                    line = datetime.now(timezone.utc).isoformat() + ' Still running: ' + label + '; waiting for the next stage/result update.\n'
                if line is None:
                    break
                require(isinstance(line, str), 'Child output reader failed: ' + str(line))
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            reader.join(timeout=1)
            process.stdout.close()
            log.flush()
            os.fsync(log.fileno())


def verify_load_result(root, helper, loader):
    folder = root / 'resume_judge_loading_attempts' / LOAD_ATTEMPT
    report = read(folder / 'REPORT.json')
    require(report['status'] == 'RESUMED_JUDGE_LOADED_AND_VALIDATED_NO_INFERENCE'
            and report['source_sha256'] == LOAD_SOURCE_SHA
            and report['model_load_attempts_started'] == ['judge']
            and report['model_load_calls_returned'] == 1
            and report['judge_snapshot_freshly_verified'] is True
            and report['original_judge_load_inspection_passed'] is True
            and report['model_signature_matches_historical_validation'] is True
            and report['base_and_isolated_environment_unchanged'] is True
            and report['gpu_exceptions_accepted'] == []
            and report['package_changes_performed'] is False
            and report['historical_contracts_rewritten'] is False
            and report['forward_passes'] == report['generation_calls'] == report['judge_calls'] == 0,
            'Saved judge-loading success differs')
    for relative, expected in report['artifact_sha256'].items():
        require(sha(helper.safe_file(folder, relative)) == expected, 'Saved judge-loading artifact differs: ' + relative)
    path = folder / 'JUDGE_LOADING_VALIDATION_CONTRACT.json'
    require(sha(path) == report['judge_loading_validation_contract_sha256'], 'Fresh judge validation contract differs')
    proof = read(path)
    require(proof['format'] == 'ATML_TASK4_RESUMED_JUDGE_LOADING_VALIDATION_V1'
            and proof['source_sha256'] == LOAD_SOURCE_SHA
            and proof['snapshot_checksums_freshly_verified'] is True
            and proof['original_nf4_float16_judge_load_inspection_passed'] is True
            and proof['model_signature_matches_historical_validation'] is True
            and proof['base_and_isolated_environment_unchanged'] is True
            and proof['ready_for_judging_execution'] is True
            and proof['gpu_exceptions_accepted'] == []
            and proof['forward_passes_performed'] is False
            and proof['generation_or_judging_performed'] is False
            and proof['judge_files_verified'] == 9 and proof['judge_file_bytes'] == 6183451098,
            'Saved judge validation is incomplete')
    for filename, field in [('JUDGE_MODEL.json', 'judge_probe_sha256'),
                            ('BASE_AFTER_INVENTORY.json', 'base_inventory_sha256'),
                            ('ISOLATED_AFTER.json', 'isolated_inventory_sha256')]:
        require(sha(folder / filename) == proof[field], 'Judge-load bound artifact differs: ' + filename)
    for filename, field in [('CPU_MODEL_INPUT_CONTRACT.json', 'model_input_contract_sha256'),
                            ('JUDGE_SNAPSHOT_CONTRACT.json', 'judge_snapshot_contract_sha256'),
                            ('MODEL_LOADING_VALIDATION_CONTRACT.json', 'original_model_loading_contract_sha256')]:
        require(sha(root / filename) == proof[field], 'Bound judge/model input differs: ' + filename)
    historical = read(root / 'MODEL_LOADING_VALIDATION_CONTRACT.json')['validated_models']['judge']
    require(loader.validate_judge_result(read(folder / 'JUDGE_MODEL.json'), historical)
            == proof['validated_judge_signature'], 'Saved judge signature differs')
    runtime_folder, runtime_path, current = loader.verify_runtime(root, helper)
    require(sha(runtime_folder / 'REPORT.json') == proof['current_runtime_report_sha256']
            == report['current_runtime_report_sha256']
            and sha(runtime_path) == proof['current_runtime_validation_contract_sha256']
            == report['current_runtime_validation_contract_sha256']
            and proof['historical_metadata_sha256'] == current['historical_metadata_sha256']
            and proof['hardware'] == current['hardware']
            and proof['notebook_pid'] == current['notebook_pid']
            and proof['local_owner_sha256'] == current['local_owner_sha256']
            and proof['isolated_package_versions'] == current['isolated_package_versions'],
            'Judge validation and current runtime proofs disagree')
    return folder, path, proof, current


def collect_cached(store, rows, core):
    """Validate every policy and every cached judgment before any new call."""
    generated, joined, counts = {}, {}, {}
    for policy in core.POLICIES:
        batches = core.saved_generations(store, policy, rows, require_complete=True)
        records = [record for unit in range(len(core.chunks(rows))) for record in batches[unit]]
        store.check_paths('judgment', policy, [record['xstest_id'] for record in records])
        cached = []
        for record in records:
            unit, digest = record['xstest_id'], core.checksum(record)
            payload = store.read('judgment', policy, unit, digest)
            if payload is not None:
                require(set(payload) == {'xstest_id', 'policy', 'generation_sha256', 'judge'}
                        and payload['xstest_id'] == unit and payload['policy'] == policy
                        and payload['generation_sha256'] == digest, 'Cached judgment identity differs')
                core.validate_judge_output(payload['judge'])
                cached.append({**record, 'judge': payload['judge']})
        generated[policy], joined[policy], counts[policy] = records, cached, len(cached)
    return generated, joined, counts


def guard_initial_judgments(root, execution_exists):
    if not execution_exists:
        folder = Path(root) / 'evaluation_v1/judgment'
        require(not folder.is_symlink(), 'Linked judgment directory')
        require(not folder.exists() or not any(folder.rglob('*.json')),
                'Judgments exist without this fixed judging execution contract; preserved for review')


def validate_snapshot_receipt(files, receipt):
    # Snapshot records also contain the original remote identity and filename.
    # The preceding loading probe records only these two freshly verified fields.
    projected = {name: {'bytes': entry['bytes'], 'sha256': entry['sha256']}
                 for name, entry in files.items()}
    require(projected == receipt, 'Cached judge snapshot differs from freshly verified file identities')


def inspect_loaded_judge(tokenizer, model, inputs, expected, guards):
    import bitsandbytes as bnb
    require(not model.training and all(not module.training for module in model.modules()), 'Judge is not in evaluation mode')
    metadata = inputs['tokenizer_and_generation_metadata']['judge']
    tok = metadata['tokenizer']
    require(tokenizer.is_fast and tokenizer.padding_side == 'left'
            and tokenizer.pad_token_id == tok['pad_token_id'] and tokenizer.eos_token_id == tok['eos_token_id'],
            'Judge tokenizer settings differ')
    defaults = model.generation_config.to_dict()
    guards.validate_defaults(defaults, metadata['generation_defaults_from_pinned_transformers'])
    groups = {}
    for name, parameter in model.named_parameters():
        key = (str(parameter.dtype), str(parameter.device), 'lora_' in name, bool(parameter.requires_grad))
        entry = groups.setdefault(key, {'dtype': key[0], 'device': key[1], 'is_lora': key[2],
            'requires_grad': key[3], 'tensors': 0, 'count': 0})
        entry['tensors'] += 1
        entry['count'] += parameter.numel()
    summary = sorted(groups.values(), key=lambda item: (item['is_lora'], item['dtype'], item['device'], item['requires_grad']))
    guards.validate_parameter_summary(summary, 'judge')
    require(bool(getattr(model, 'is_loaded_in_4bit', False)), 'Judge is not actually loaded in 4-bit')
    quantization = model.config.quantization_config.to_dict()
    modules = [{'name': name, 'compute_dtype': str(module.compute_dtype),
        'quant_type': module.weight.quant_type, 'device': str(module.weight.device),
        'quant_state_present': module.weight.quant_state is not None}
        for name, module in model.named_modules() if isinstance(module, bnb.nn.Linear4bit)]
    guards.validate_judge_quantization(quantization, modules)
    actual = {'parameter_summary': summary, 'loaded_generation_defaults': defaults,
        'fixed_course_generate_kwargs': metadata['fixed_course_generate_kwargs'],
        'quantization_config': quantization, 'quantized_modules': modules,
        'device_map': getattr(model, 'hf_device_map', None)}
    require(actual == {key: expected[key] for key in actual}, 'Loaded inference judge differs from validated signature')
    return actual


def execution_contract(source_sha, proof_path, proof, helper):
    return {'format': 'ATML_TASK4_JUDGING_EXECUTION_V1', 'source_sha256': source_sha,
        'evaluation_contract_sha256': sha(ROOT / 'EVALUATION_EXECUTION_CONTRACT.json'),
        'generation_completion_contract_sha256': sha(ROOT / 'GENERATION_COMPLETION_CONTRACT.json'),
        'judge_loading_report_sha256': sha(proof_path.parent / 'REPORT.json'),
        'judge_loading_validation_contract_sha256': sha(proof_path),
        'current_runtime_validation_contract_sha256': proof['current_runtime_validation_contract_sha256'],
        'task4_approval_sha256': proof['task4_approval_sha256'],
        'judge_snapshot_contract_sha256': proof['judge_snapshot_contract_sha256'],
        'model_input_contract_sha256': proof['model_input_contract_sha256'],
        'historical_metadata_sha256': proof['historical_metadata_sha256'],
        'hardware': proof['hardware'], 'notebook_pid': proof['notebook_pid'],
        'local_owner_sha256': proof['local_owner_sha256'], 'isolated_package_versions': proof['isolated_package_versions'],
        'original_source_manifest_sha256': helper.SOURCES,
        'policy_order': POLICIES, 'prompts_per_policy': 450, 'total_judgments': 1800,
        'original_judge_one_and_prompt_and_parser': True, 'judge_max_new_tokens': 64, 'do_sample': False,
        'confidence_used_for_filtering_or_weighting': False, 'human_labels_assigned': False,
        'response_generation_or_training_performed': False, 'per_judgment_atomic_persistence': True}


def worker(attempt, expected_source_sha, expected_execution_sha):
    require(sha(__file__) == expected_source_sha, 'Executing worker source differs')
    helper, loader = get_helpers()
    # -I starts with no notebook/course imports. Verify every executed release first.
    folders = {}
    for kind in ['preflight', 'orchestration', 'generation', 'model_loading']:
        identity = helper.SOURCES[kind]
        folder = ROOT / (kind + '_releases') / identity
        require(sha(folder / 'MANIFEST.json') == identity, 'Worker source identity differs: ' + kind)
        for relative, expected in read(folder / 'MANIFEST.json').items():
            require(sha(helper.safe_file(folder, relative)) == expected, 'Worker source payload differs: ' + relative)
        folders[kind] = folder
    execution_path = ROOT / 'JUDGING_EXECUTION_CONTRACT.json'
    require(sha(execution_path) == expected_execution_sha, 'Worker judging execution contract differs')
    execution = read(execution_path)
    require(execution['source_sha256'] == expected_source_sha and execution['original_source_manifest_sha256'] == helper.SOURCES,
            'Worker execution/source binding differs')
    for relative, expected in execution['historical_metadata_sha256'].items():
        require(sha(helper.safe_file(ROOT, relative)) == expected, 'Historical identity changed before judging: ' + relative)
    for filename, field in [('EVALUATION_EXECUTION_CONTRACT.json', 'evaluation_contract_sha256'),
                            ('GENERATION_COMPLETION_CONTRACT.json', 'generation_completion_contract_sha256')]:
        require(sha(ROOT / filename) == execution[field], 'Bound generation evidence differs')
    print('Worker: verifying all 452 saved batches, four exports and all cached judgments before model loading.', flush=True)
    counts, core, inputs = helper.check_saved(ROOT, folders)
    sys.modules['core'] = core
    sys.path.insert(0, str(folders['orchestration']))
    sys.path.insert(0, str(folders['preflight'] / 'course_reference'))
    calls = load('task4_fixed_judge_calls', folders['orchestration'] / 'course_calls.py')
    io = load('task4_fixed_judge_io', folders['generation'] / 'generation_io.py')
    guard = load('task4_fixed_judge_model_guard', folders['model_loading'] / 'loading_guards.py')
    data = read(ROOT / 'INPUT_IDENTITY_CONTRACT.json')['inputs']['xstest']
    rows = io.load_rows(data['path'], core)
    store = core.UnitStore(ROOT / 'evaluation_v1', read(ROOT / 'EVALUATION_EXECUTION_CONTRACT.json'))
    generated, cached_joined, completed = collect_cached(store, rows, core)
    require(all(len(generated[p]) == 450 and counts[p]['judgments'] == completed[p] for p in POLICIES),
            'Full generation or initial judgment coverage differs')
    initial = dict(completed)
    result = {'format': 'ATML_TASK4_JUDGING_WORKER_V1', 'status': 'RUNNING',
        'source_sha256': expected_source_sha, 'judging_execution_contract_sha256': expected_execution_sha,
        'initial_judgments_by_policy': initial, 'completed_judgments_by_policy': dict(completed),
        'new_judge_call_attempts': 0, 'judge_calls_returned': 0, 'new_judgments_saved': 0,
        'reused_judgments': sum(initial.values()), 'model_loaded': False,
        'response_generation_calls': 0, 'training_performed': False, 'human_labels_assigned': 0,
        'confidence_filtering_or_weighting': False}
    immutable(attempt / 'INITIAL_JUDGING_COUNTS.json', result)
    model = tokenizer = None
    try:
        remaining = 1800 - sum(completed.values())
        print('Verified responses: 1800/1800; cached judgments:', str(sum(completed.values())) + '/1800;',
              'remaining:', remaining, flush=True)
        if remaining:
            import torch
            from task4_safety.judge_responses import load_judge
            require(torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0)
                    and torch.cuda.get_device_capability(0) == (8, 0), 'A100 required for fixed NF4 judging')
            proof = read(ROOT / 'resume_judge_loading_attempts' / LOAD_ATTEMPT / 'JUDGE_LOADING_VALIDATION_CONTRACT.json')
            require(sha(ROOT / 'resume_judge_loading_attempts' / LOAD_ATTEMPT / 'JUDGE_LOADING_VALIDATION_CONTRACT.json')
                    == execution['judge_loading_validation_contract_sha256'], 'Saved judge proof changed')
            cfg = dict(inputs['course_configuration'])
            cfg['ai_judge_model'] = read(ROOT / 'JUDGE_SNAPSHOT_CONTRACT.json')['directory']
            print('Loading the original fixed judge once for all four policies.', flush=True)
            tokenizer, model = load_judge(cfg)
            result['model_loaded'] = True
            inspection = inspect_loaded_judge(tokenizer, model, inputs, proof['validated_judge_signature'], guard)
            immutable(attempt / 'INFERENCE_JUDGE_CHECK.json', inspection)
            print('Inference judge eval/placement/NF4/float16/defaults matched. Starting saved-response judging.', flush=True)
        else:
            print('All judgments already saved and verified; no model load or judge calls needed.', flush=True)
        exports = {}
        for policy in POLICIES:
            print('Policy:', policy, '; cached:', str(completed[policy]) + '/450', flush=True)

            def callback(prompt, response):
                result['new_judge_call_attempts'] += 1
                value = calls.judge_response(tokenizer, model, prompt, response)
                result['judge_calls_returned'] += 1
                return value

            def progress(done, total):
                completed[policy] = done
                result['completed_judgments_by_policy'] = dict(completed)
                result['new_judgments_saved'] += 1
                print(datetime.now(timezone.utc).isoformat(), policy, 'saved judgments', str(done) + '/' + str(total),
                      '; overall', str(sum(completed.values())) + '/1800;',
                      'remaining', 1800 - sum(completed.values()), flush=True)

            judged = core.run_judging(store, policy, rows, callback, on_progress=progress)
            require(judged['completed_judgments'] == 450 and len(judged['records']) == 450,
                    'Policy judgment coverage incomplete')
            export = ROOT / 'evaluation_v1/exports/judgment' / (policy + '.jsonl')
            immutable(export, io.jsonl_bytes(judged['records']))
            exports[policy] = {'path': str(export), 'sha256': sha(export), 'judgments': 450}
            completed[policy] = 450
            result['completed_judgments_by_policy'] = dict(completed)
            immutable(attempt / (policy.upper() + '_JUDGING_RESULT.json'), {
                'policy': policy, 'completed_judgments': 450, 'new_judgments': judged['new_judgments'],
                'reused_judgments': judged['reused_judgments'], 'export': exports[policy]})
        _, joined_after, after_counts = collect_cached(store, rows, core)
        require(after_counts == {p: 450 for p in POLICIES}, 'Final judgment coverage incomplete')
        hashes = {}
        for policy in POLICIES:
            require(io.jsonl_bytes(joined_after[policy]) == (ROOT / 'evaluation_v1/exports/judgment' / (policy + '.jsonl')).read_bytes(),
                    'Final export differs from saved judgments')
            for record in generated[policy]:
                path = store.path('judgment', policy, record['xstest_id'])
                hashes[path.relative_to(store.root).as_posix()] = sha(path)
        immutable(attempt / 'JUDGING_COMPLETION_CANDIDATE.json', {
            'format': 'ATML_TASK4_JUDGING_COMPLETE_V1', 'source_sha256': expected_source_sha,
            'judging_execution_contract_sha256': expected_execution_sha,
            'evaluation_contract_sha256': store.identity,
            'generation_completion_contract_sha256': execution['generation_completion_contract_sha256'],
            'judgments': 1800, 'policy_judgment_counts': after_counts,
            'judgment_unit_sha256': hashes, 'exports': exports,
            'confidence_used_for_filtering_or_weighting': False, 'human_audit_pending': True})
        result.update(status='ALL_FOUR_POLICIES_JUDGED_AND_SAVED', completed_judgments=1800,
                      completed_judgments_by_policy=after_counts, remaining_judgments=0,
                      completion_candidate_sha256=sha(attempt / 'JUDGING_COMPLETION_CANDIDATE.json'))
    except BaseException as error:
        import traceback
        result.update(status='JUDGING_WORKER_STOPPED_EVIDENCE_PRESERVED',
            error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
    finally:
        # The unchanged core saves each judgment before invoking its progress callback.
        immutable(attempt / 'WORKER_RESULT.json', result)
        if model is not None:
            del model, tokenizer
        print('Worker status:', result['status'], flush=True)
    require(result['status'] == 'ALL_FOUR_POLICIES_JUDGED_AND_SAVED', 'Judging stopped; saved units are preserved')


def main():
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    helper, loader = get_helpers()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    attempt = ROOT / 'judging_attempts' / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    source_sha = sha(__file__)
    report = {'format': 'ATML_TASK4_JUDGING_ATTEMPT_V1', 'attempt_utc': stamp, 'source_sha256': source_sha,
        'status': 'RUNNING', 'commands': [], 'package_changes_performed': False, 'downloads': 0,
        'response_generation_performed': False, 'training_performed': False, 'human_labels_assigned': 0,
        'historical_generation_contracts_rewritten': False, 'gpu_exceptions_accepted': [],
        'new_judge_call_attempts': None, 'new_judgments_saved': None,
        'completed_judgments': None, 'artifact_sha256': {}}

    def save(name, value):
        immutable(attempt / name, value)

    def command(args, label, environment):
        event = {'stage': label, 'argv': [str(x) for x in args]}
        save(label + '_COMMAND.json', event)
        print('Task 4 judging:', label, flush=True)
        log_path = attempt / (label + '.log')
        code = stream_process(event['argv'], environment, log_path, label)
        event.update(returncode=code, log_sha256=sha(log_path))
        report['commands'].append(event)
        save(label + '_RESULT.json', event)
        require(code == 0, label + ' stopped; no automatic fallback or change to judge settings')

    try:
        save('JUDGING_SOURCE.py', Path(__file__).read_bytes())
        print('Step 1/4: verifying saved successful judge loading, runtime evidence and original sources.', flush=True)
        load_folder, proof_path, proof, current = verify_load_result(ROOT, helper, loader)
        folders = helper.verify_sources(ROOT)
        print('All 19 original source releases and saved judge/current-runtime evidence verified.', flush=True)
        require(LOCAL.is_dir() and not LOCAL.is_symlink()
                and sha(LOCAL / '.task4_isolation_owner.json') == current['local_owner_sha256'],
                'Owned isolated environment missing or changed')
        loader.validate_current_identity(current, read(LOCAL / '.task4_isolation_owner.json'),
            os.getpid(), sys.executable, platform.python_version(), LOCAL)
        require(read(folders['preflight'] / 'docs/approval.json')['approved'] is True
                and sha(folders['preflight'] / 'docs/approval.json') == proof['task4_approval_sha256']
                == read(ROOT / 'OWNER.json')['approval_sha256'], 'Original Task 4 approval differs')
        guards = load('task4_judging_isolation_guards', folders['isolated_gpu'] / 'guards.py')
        policy = load('task4_judging_child_policy', folders['isolated_recheck'] / 'isolation.py')
        gates = load('task4_judging_hardware_gates', folders['gpu_environment'] / 'gates.py')
        child_env = policy.child_environment(os.environ, guards.clean_environment)
        parent_backend = os.environ.get('MPLBACKEND')
        python = LOCAL / 'bin/python'
        inspect = folders['isolated_gpu'] / 'inspect_python.py'
        pins = current['isolated_package_versions']
        require(pins == read(folders['isolated_gpu'] / 'APPROVED_ISOLATED_VERSIONS.json') and len(pins) == 84,
                'Approved exact isolated pins differ')
        print('Step 2/4: confirming the current A100 and unchanged validated inventories.', flush=True)
        gpu = helper.command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,uuid', '--format=csv,noheader,nounits'])
        save('CURRENT_GPU_QUERY.json', gpu)
        require(gpu.get('returncode') == 0 and gates.parse_a100_inventory(gpu['stdout']) == current['hardware'],
                'Current A100 assignment changed')
        base_before = helper.inventory(sys.executable)
        save('BASE_BEFORE_INVENTORY.json', base_before)
        require(base_before == read(load_folder / 'BASE_AFTER_INVENTORY.json'), 'Base inventory changed after judge validation')
        command([python, '-I', '-B', inspect, attempt / 'ISOLATED_BEFORE.json'], 'isolated_before', child_env)
        iso_before = read(attempt / 'ISOLATED_BEFORE.json')
        guards.validate_inventory(iso_before, pins, LOCAL, complete=True)
        require(iso_before == read(load_folder / 'ISOLATED_AFTER.json'), 'Isolated inventory changed after judge validation')
        # Reuse the freshly verified weight hashes from the preceding load block.
        # This block checks availability/size; it does not claim another full weight rehash.
        judge = read(ROOT / 'JUDGE_SNAPSHOT_CONTRACT.json')
        require(judge['revision'] == loader.JUDGE_REVISION and judge['directory'] == str(ROOT / 'models/judge')
                and Path(judge['directory']).is_dir() and not Path(judge['directory']).is_symlink(),
                'Cached judge snapshot revision or directory differs')
        validate_snapshot_receipt(judge['verified_files'], read(load_folder / 'VERIFIED_JUDGE_FILES.json'))
        require(len(judge['verified_files']) == 9, 'Judge file coverage differs')
        actual_files = {p.relative_to(Path(judge['directory'])).as_posix() for p in Path(judge['directory']).rglob('*')
                        if p.is_file() and '.cache' not in p.relative_to(Path(judge['directory'])).parts}
        require(actual_files == set(judge['verified_files']), 'Judge directory file coverage changed')
        for name, entry in judge['verified_files'].items():
            require(helper.safe_file(Path(judge['directory']), name).stat().st_size == entry['bytes'], 'Judge file size changed: ' + name)
        save('JUDGE_FILES_CURRENT_SIZE_CHECK.json', {'files': 9, 'fresh_sha256_this_block': False,
            'prior_fresh_sha256_evidence': sha(load_folder / 'VERIFIED_JUDGE_FILES.json'), 'sizes_match': True})
        execution_path = ROOT / 'JUDGING_EXECUTION_CONTRACT.json'
        guard_initial_judgments(ROOT, execution_path.exists())
        execution = execution_contract(source_sha, proof_path, proof, helper)
        immutable(execution_path, execution)
        execution_sha = sha(execution_path)
        report['judging_execution_contract_sha256'] = execution_sha
        print('Step 3/4: checking all saved responses/caches, then judging missing responses only.', flush=True)
        command([python, '-I', '-B', '-u', attempt / 'JUDGING_SOURCE.py', '--worker', attempt,
                 source_sha, execution_sha], 'fixed_judging_worker', child_env)
        worker_result = read(attempt / 'WORKER_RESULT.json')
        require(worker_result['status'] == 'ALL_FOUR_POLICIES_JUDGED_AND_SAVED'
                and worker_result['completed_judgments'] == 1800
                and worker_result['completed_judgments_by_policy'] == {p: 450 for p in POLICIES},
                'Saved worker completion differs')
        report.update(new_judge_call_attempts=worker_result['new_judge_call_attempts'],
            new_judgments_saved=worker_result['new_judgments_saved'],
            reused_judgments=worker_result['reused_judgments'], completed_judgments=1800)
        print('Step 4/4: confirming preserved environments/contracts and finalizing judgment evidence.', flush=True)
        base_after = helper.inventory(sys.executable)
        save('BASE_AFTER_INVENTORY.json', base_after)
        command([python, '-I', '-B', inspect, attempt / 'ISOLATED_AFTER.json'], 'isolated_after', child_env)
        iso_after = read(attempt / 'ISOLATED_AFTER.json')
        guards.validate_inventory(iso_after, pins, LOCAL, complete=True)
        require(base_before == base_after and iso_before == iso_after
                and os.environ.get('MPLBACKEND') == parent_backend
                and sha(LOCAL / '.task4_isolation_owner.json') == current['local_owner_sha256'],
                'Environments or owner changed during judging')
        for relative, expected in current['historical_metadata_sha256'].items():
            require(sha(helper.safe_file(ROOT, relative)) == expected, 'Historical generation evidence changed: ' + relative)
        require(sha(execution_path) == execution_sha, 'Fixed judging execution contract changed')
        candidate = attempt / 'JUDGING_COMPLETION_CANDIDATE.json'
        require(sha(candidate) == worker_result['completion_candidate_sha256'], 'Worker completion candidate differs')
        immutable(ROOT / 'JUDGING_COMPLETION_CONTRACT.json', candidate.read_bytes())
        report.update(status='ALL_FOUR_POLICIES_JUDGED_AND_SAVED', remaining_judgments=0,
            judging_completion_contract_sha256=sha(ROOT / 'JUDGING_COMPLETION_CONTRACT.json'),
            base_and_isolated_environment_unchanged=True, responses_retained=1800,
            policy_judgment_counts={p: 450 for p in POLICIES}, human_audit_pending=True)
    except BaseException as error:
        import traceback
        report.update(status='JUDGING_STOPPED_EVIDENCE_PRESERVED',
            error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        worker_result_path = attempt / 'WORKER_RESULT.json'
        if worker_result_path.is_file():
            partial = read(worker_result_path)
            report.update(new_judge_call_attempts=partial['new_judge_call_attempts'],
                new_judgments_saved=partial['new_judgments_saved'],
                completed_judgments=sum(partial['completed_judgments_by_policy'].values()),
                policy_judgment_counts=partial['completed_judgments_by_policy'])
    for path in sorted(attempt.rglob('*')):
        if path.is_file() and not path.name.startswith('.pending-') and path.name != 'REPORT.json':
            report['artifact_sha256'][path.relative_to(attempt).as_posix()] = sha(path)
    save('REPORT.json', report)
    print('Saved report:', attempt / 'REPORT.json', flush=True)
    print('Status:', report['status'], flush=True)
    if report['status'] == 'ALL_FOUR_POLICIES_JUDGED_AND_SAVED':
        print('Responses 1800/1800; judgments 1800/1800; remaining judgments 0.', flush=True)
        print('New saved judgments:', report['new_judgments_saved'], '; reused:', report['reused_judgments'], flush=True)
        print('Four ordered judgment exports saved. Human labels last confirmed 0/240.', flush=True)
    else:
        print(json.dumps(report['error'], indent=2), flush=True)
        print('Judgments last accounted for in this attempt:', report['completed_judgments'],
              '; any fully saved unit remains resumable. Return full output before continuing.', flush=True)
    print('No policy regeneration, package changes, downloads, training or human labeling.', flush=True)
    print('Return the full output. Keep this runtime until its saved completion is reviewed.', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        require(len(sys.argv) == 5, 'Worker arguments differ')
        worker(Path(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        main()
