"""CPU resume audit. Existing evaluation files are read only; new audit evidence only."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import subprocess
import types
import uuid

RELEASE_SHA = 'd6dd999e7fdf241c7136ee13f46babdf13338fa6dc12ff9046d2ea6a2e89fd87'
PROTOCOL_SHA = '58632138b81533416b3bb46979e03ca508323dd56dbbba572e0249d1b1d8665a'
CPU_REPORT = 'combined_cpu_preflight_attempts/20261013T090737.971920Z_38745111/REPORT.json'
CPU_REPORT_SHA = '80b7e94d72a248c606f75d255e1315debdf9262c679455a5d69fc5def88388a1'
GPU_REPORT = 'a100_setup_attempts/20261013T093754.042876Z_44eefae6/REPORT.json'
GPU_REPORT_SHA = '440f9c23ddc59291a96730b99cc5d9c6c6df00ee8d90996f7cac40f911b5af06'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 2**20), b''):
            value.update(part)
    return value.hexdigest()


def read(path):
    require(path.is_file() and not path.is_symlink(), 'Missing/linked file: ' + str(path))
    return json.loads(path.read_text())


def checked(path, expected):
    value = read(path)
    require(file_sha(path) == expected, 'File checksum differs: ' + str(path))
    return value


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), 'Refusing to replace audit evidence.')
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()
    pending = path.parent / ('.pending_' + uuid.uuid4().hex)
    with pending.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    require(pending.read_bytes() == payload, 'Audit write readback failed.')
    os.replace(pending, path)
    require(path.read_bytes() == payload, 'Saved audit readback failed.')


def verify_rng(state):
    require(isinstance(state, dict) and set(state) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'},
            'Saved RNG fields differ.')
    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value, list) else value
    random.Random(0).setstate(tuples(state['python']))  # Independent local object only.
    n = state['numpy']
    require(isinstance(n, list) and len(n) == 5 and n[0] == 'MT19937'
            and isinstance(n[1], list) and len(n[1]) == 624
            and all(type(x) is int and 0 <= x <= 2**32 - 1 for x in n[1])
            and type(n[2]) is int and 0 <= n[2] <= 624 and n[3] in (0, 1)
            and isinstance(n[4], (int, float)) and math.isfinite(n[4]), 'Saved NumPy RNG structure differs.')
    require(isinstance(state['torch_cuda'], list) and len(state['torch_cuda']) == 1,
            'Saved CUDA RNG device count differs.')
    for value in [state['torch_cpu']] + state['torch_cuda']:
        require(isinstance(value, list) and bool(value)
                and all(type(x) is int and 0 <= x <= 255 for x in value), 'Invalid Torch RNG bytes.')


def pending_files(folder, root):
    result = []
    for path in sorted(folder.glob('.pending_*')):
        require(path.is_file() and not path.is_symlink(), 'Unexpected linked/non-file temporary evidence.')
        result.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                       'sha256': file_sha(path)})
    return result


def audit_generation(folder, policy, domain, rows, core, verifier, expected_hardware,
                     reference_settings, asset_sha, prompt_checks, root, attempt, report, context_limit):
    name = policy + '/' + domain
    print('Checking ' + name + '...', flush=True)
    if not folder.exists():
        return 0, reference_settings
    require(folder.is_dir() and not folder.is_symlink(), 'Linked/non-directory result folder.')
    binding = read(folder / 'BINDING.json')
    require(set(binding) == {'phase', 'policy', 'domain', 'release_sha256', 'hardware', 'settings',
                            'dataset_sha256', 'row_sha256_in_order', 'seed'}, 'Generation binding fields differ.')
    require(binding['phase'] == 'generation' and binding['policy'] == policy and binding['domain'] == domain
            and binding['release_sha256'] == RELEASE_SHA and binding['seed'] == 6304
            and binding['dataset_sha256'] == asset_sha
            and binding['row_sha256_in_order'] == [core.checksum(row) for row in rows],
            'Frozen dataset/order/policy identity differs: ' + name)
    hardware = binding['hardware']
    require(all(hardware.get(k) == v for k, v in expected_hardware.items()),
            'Saved A100 environment identity differs: ' + name)
    require(all(type(hardware.get(k)) is bool for k in ('deterministic_algorithms', 'cudnn_deterministic',
            'cudnn_benchmark', 'cuda_matmul_allow_tf32', 'cudnn_allow_tf32')), 'Missing saved runtime flags.')
    if report.get('saved_hardware') is None:
        report['saved_hardware'] = hardware
    require(hardware == report['saved_hardware'], 'Runtime changed between saved policy/domain groups.')
    settings = binding['settings']
    effective = settings['effective_requested_generation']
    require(all(effective.get(k) == v for k, v in
                {'max_new_tokens': 512, 'temperature': .7, 'top_p': .9, 'do_sample': True}.items()),
            'Saved requested decoding differs.')
    require(effective['eos_token_id'] == settings['tokenizer_eos_token_id']
            and effective['pad_token_id'] == settings['tokenizer_pad_token_id'], 'Saved tokenizer IDs differ.')
    if reference_settings is not None:
        require(settings == reference_settings, 'Decoding settings changed between policies/domains.')
    units = folder / 'units'
    require(units.is_dir() and not units.is_symlink(), 'Missing/linked units folder; preserve and review.')
    paths = sorted(units.glob('*.json'))
    require(len(paths) <= len(rows) and all(p.name == f'{i:06d}.json' for i, p in enumerate(paths)),
            'Saved answer sequence has a gap, unexpected filename, or too many entries: ' + name)
    for parent, allowed in ((folder, {'BINDING.json', 'units'}),
                            (units, {p.name for p in paths})):
        require(all(p.name in allowed or p.name.startswith('.pending_') for p in parent.iterdir()),
                'Unknown result entry; preserve and review: ' + name)
        report['retained_temporary_files'].extend(pending_files(parent, root))
    binding_sha = core.checksum(binding)
    previous = binding_sha
    for i, path in enumerate(paths):
        record = read(path)
        require(set(record) == {'index', 'binding_sha256', 'previous_sha256', 'payload', 'rng_after',
                               'record_sha256'}, 'Unexpected checkpoint fields.')
        contents = {k: v for k, v in record.items() if k != 'record_sha256'}
        require(record['index'] == i and record['binding_sha256'] == binding_sha
                and record['previous_sha256'] == previous and record['record_sha256'] == core.checksum(contents),
                'Saved answer checksum/chain differs: ' + str(path))
        payload, row = record['payload'], rows[i]
        require(payload['row_index'] == i and payload['row_sha256'] == core.checksum(row)
                and payload['source_index'] == row['source_index'] and payload['policy'] == policy
                and payload['domain'] == domain and isinstance(payload['response'], str),
                'Saved response row identity differs.')
        require(payload['exact_reward'] == verifier.exact_reward(payload['response'], row['gold_final'])
                and payload['extracted_final'] == verifier.extract_designated_final(payload['response'])
                and payload['format_compliant'] == (payload['extracted_final'] is not None),
                'Saved exact-verifier fields differ.')
        ids, length = payload['generated_token_ids'], payload['response_tokens']
        require(type(length) is int and 0 <= length <= 512 and isinstance(ids, list) and len(ids) == length
                and all(type(x) is int and x >= 0 for x in ids), 'Saved response token fields differ.')
        eos = settings['tokenizer_eos_token_id']
        ended = eos is not None and eos in ids
        require((not ended or ids.index(eos) == length - 1)
                and payload['terminated_with_eos'] == ended
                and payload['truncated'] == (length >= 512 and not ended), 'Saved EOS/truncation fields differ.')
        prompt = prompt_checks[domain][i]
        require(payload['prompt_tokens'] == prompt['prompt_tokens']
                and payload['prompt_token_ids_sha256'] == prompt['token_ids_sha256']
                and prompt['row_sha256'] == core.checksum(row) and prompt['row_index'] == i
                and payload['prompt_tokens'] + 512 <= context_limit, 'Saved full prompt identity differs.')
        require(isinstance(payload['generation_seconds'], (int, float))
                and math.isfinite(payload['generation_seconds']) and payload['generation_seconds'] >= 0,
                'Invalid saved generation timing.')
        verify_rng(record['rng_after'])
        previous = core.checksum(record)
        report['verified_prefix_counts'][name] = i + 1
        if (i + 1) % 25 == 0 or i + 1 == len(paths):
            write_new(attempt / 'progress' / f'{policy}_{domain}_{i+1:06d}.json',
                      {'policy_domain': name, 'checked_answers': i + 1, 'last_record_sha256': previous})
            print(f'  Verified {i+1}/{len(paths)} saved answers', flush=True)
    report['verified_prefix_counts'][name] = len(paths)
    report['generation_bindings'][name] = {
        'binding_sha256': binding_sha, 'last_record_sha256': previous, 'answers': len(paths)}
    return len(paths), settings


def main():
    smi = shutil.which('nvidia-smi')
    gpu = subprocess.run([smi, '--query-gpu=name', '--format=csv,noheader'], capture_output=True,
                         text=True, timeout=15) if smi else None
    require(not ((gpu and gpu.returncode == 0 and gpu.stdout.strip())
                 or os.environ.get('COLAB_TPU_ADDR') or os.environ.get('TPU_NAME')),
            'Select Runtime > Change runtime type > Hardware accelerator: None, then rerun this cell.')
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    project = Path('/content/drive/MyDrive/ATML-Assignment-2')
    root = project / 'task5/experiment_v1'
    require(root.is_dir() and not root.is_symlink(), 'Existing Task 5 Drive folder unavailable.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid.uuid4().hex[:8]
    attempt = root / 'resume_audit_attempts' / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    report = {'status': 'CPU_RESUME_AUDIT_RUNNING', 'errors': [], 'audit_complete': False,
              'audit_source_sha256': globals().get('AUDIT_SOURCE_SHA256'),
              'verified_prefix_counts': {}, 'generation_bindings': {}, 'retained_temporary_files': [],
              'models_loaded': 0, 'answers_generated': 0, 'ai_judgments': 0, 'installs': 0,
              'downloads': 0, 'existing_evaluation_files_modified': False,
              'evaluation_release_sha256': RELEASE_SHA}
    write_new(attempt / 'START.json', report)
    try:
        print('1/3: checking saved code, approvals, fixed inputs and prior setup evidence.', flush=True)
        release = root / 'evaluation_releases' / RELEASE_SHA
        require(release.is_dir() and not release.is_symlink(), 'Missing/linked frozen evaluation release.')
        manifest = checked(release / 'MANIFEST.json', RELEASE_SHA)
        for name, expected in manifest.items():
            relative = PurePosixPath(name)
            require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source manifest entry.')
            checked_file = release / name
            require(checked_file.is_file() and not checked_file.is_symlink()
                    and file_sha(checked_file) == expected, 'Saved evaluation source differs: ' + name)
        require(file_sha(release / 'protocol.json') == PROTOCOL_SHA, 'Approved protocol differs.')
        core = types.ModuleType('verified_task5_core')
        exec(compile((release / 'task5_run/core.py').read_bytes(), 'verified_task5_core.py', 'exec'), core.__dict__)
        verifier = types.ModuleType('verified_task5_exact_verifier')
        exec(compile((release / 'task5_feedback/rlvr.py').read_bytes(), 'verified_task5_verifier.py', 'exec'), verifier.__dict__)
        approval = read(root / 'EXECUTION_APPROVAL.json')
        expected_approval = {'format': 'ATML_TASK5_PRETRAINED_EXECUTION_APPROVAL_V1', 'approved': True,
            'user_message': 'yes', 'generation': True, 'ai_judging': True, 'quantitative_aggregation': True,
            'protocol_sha256': PROTOCOL_SHA, 'evaluation_release_sha256': RELEASE_SHA,
            'planned_policy_answers': 1200, 'planned_ai_comparison_events': 880,
            'training': False, 'downloads': False, 'publishing': False, 'protocol_changes': False,
            'one_code_block_at_a_time': True, 'save_after_every_completed_unit': True}
        require(approval == expected_approval, 'Existing execution approval differs.')
        cpu_report = checked(root / CPU_REPORT, CPU_REPORT_SHA)
        gpu_report = checked(root / GPU_REPORT, GPU_REPORT_SHA)
        cpu = checked(root / 'CPU_READINESS_CONTRACT.json', cpu_report['cpu_readiness_sha256'])
        contract_sha = gpu_report['environment_contract_sha256']
        contract = checked(root / 'ISOLATED_ENVIRONMENT_CONTRACT.json', contract_sha)
        require(cpu_report['status'] == 'CPU_READY_FOR_APPROVED_A100_SETUP' and not cpu_report['errors']
                and cpu['complete'] is True and cpu['release_sha256'] == RELEASE_SHA
                and gpu_report['status'] == 'A100_ISOLATED_ENVIRONMENT_READY_PRETRAINED_EXECUTION_APPROVAL_PENDING'
                and not gpu_report['errors'] and gpu_report['isolated_packages_verified'] == 84
                and contract['complete'] is True and contract['synthetic_cuda_nf4_passed'] is True
                and contract['probe']['tests_run'] == 44 and contract['probe']['tests_passed'] is True
                and contract['cpu_readiness_sha256'] == file_sha(root / 'CPU_READINESS_CONTRACT.json')
                and contract['evaluation_release_sha256'] == RELEASE_SHA
                and contract['package_versions'] == read(release / 'package_versions.json'),
                'Prior CPU/A100 setup evidence differs.')
        b = read(release / 'bindings.json')
        for name, expected in b['asset_sha256'].items():
            path = project / name
            require(path.is_file() and not path.is_symlink() and file_sha(path) == expected,
                    'Fixed course input differs: ' + name)
        rows = {d: core.read_rows(project / b['datasets'][d]) for d in core.DOMAINS}
        core.generation_schedule(rows)
        snapshot = project / b['snapshots']['policy']['path']
        config = checked(snapshot / 'config.json', b['snapshots']['policy']['files']['config.json']['sha256'])
        prompt_checks = checked((root / GPU_REPORT).parent / 'POLICY_PROMPT_CHECKS.json',
                                contract['probe']['policy_prompt_checks_sha256'])
        expected_hardware = {'python': contract['python_version'], 'torch': contract['probe']['torch'],
            'cuda': contract['probe']['torch_cuda'], 'gpu': contract['probe']['gpu'],
            'driver': gpu_report['hardware']['driver'], 'packages_sha256': core.checksum(contract['package_versions']),
            'environment_contract_sha256': contract_sha}
        report['prior_environment_contract_sha256'] = contract_sha
        print('2/3: verifying saved answer chains, exact scores, full prompts and RNG state.', flush=True)
        counts, reference_settings = {}, None
        for policy in core.POLICIES:
            for domain in core.DOMAINS:
                name = policy + '/' + domain
                counts[name], reference_settings = audit_generation(
                    root / 'evaluation_v1/generations' / policy / domain, policy, domain, rows[domain],
                    core, verifier, expected_hardware, reference_settings,
                    b['asset_sha256'][b['datasets'][domain]], prompt_checks, root, attempt, report,
                    config['max_position_embeddings'])
        incomplete_seen = False
        for name, count in counts.items():
            require(not incomplete_seen or count == 0, 'Saved run does not follow the approved sequential schedule.')
            incomplete_seen |= count < core.COUNTS[name.split('/')[1]]
        require(counts['sft/gsm'] == 300 and counts['sft/transfer'] == 100
                and counts['rlvr/gsm'] == 300 and counts['rlvr/transfer'] == 100
                and counts['rlaif/gsm'] >= 200, 'Fewer committed answers than previously printed; preserve and review.')
        judgments = root / 'evaluation_v1/judgments'
        require(not judgments.exists(), 'A judgment store already exists; preserve it for separate review.')
        report.update({'audit_complete': True, 'counts': counts, 'completed_answers': sum(counts.values()),
                       'remaining_answers': 1200 - sum(counts.values()), 'ai_comparison_events_remaining': 880,
                       'saved_generation_settings': reference_settings,
                       'status': 'CPU_RESUME_AUDIT_PASSED_A100_REVALIDATION_PENDING'})
        report['generation_attempts'] = []
        for folder in sorted((root / 'generation_attempts').glob('*')):
            if folder.is_dir():
                item = {'path': str(folder.relative_to(root)), 'report_present': (folder / 'REPORT.json').is_file()}
                for name in ('START.json', 'REPORT.json', 'GENERATION.log'):
                    path = folder / name
                    if path.is_file():
                        require(not path.is_symlink(), 'Linked attempt evidence.')
                        item[name] = {'bytes': path.stat().st_size, 'sha256': file_sha(path)}
                report['generation_attempts'].append(item)
    except KeyboardInterrupt:
        report['audit_complete'] = False
        report['status'] = 'CPU_RESUME_AUDIT_INTERRUPTED_REVIEW_PENDING'
        report['errors'].append('Audit interrupted. Counts in verified_prefix_counts are partial only; no generation started.')
    except Exception as exc:
        report['audit_complete'] = False
        report['status'] = 'CPU_RESUME_AUDIT_STOPPED_REVIEW_PENDING'
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    finally:
        print('3/3: saving this audit attempt; no further scan is performed here.', flush=True)
        write_new(attempt / 'REPORT.json', report)
        print('Status:', report['status'])
        print('Audit complete:', report['audit_complete'])
        if report['audit_complete']:
            print('Verified saved answers:', json.dumps(report['counts'], indent=2))
            print('Completed / remaining:', report['completed_answers'], '/', report['remaining_answers'])
            print('AI comparison events remaining:', report['ai_comparison_events_remaining'])
        else:
            print('Partially verified prefixes:', json.dumps(report['verified_prefix_counts'], indent=2))
            print('Exact completed / remaining: UNCONFIRMED')
        print('Retained temporary files:', len(report['retained_temporary_files']))
        print('Errors:', report['errors'])
        print('Saved report:', attempt / 'REPORT.json')
        print('Report SHA-256:', file_sha(attempt / 'REPORT.json'))
        print('0 installs; 0 downloads; 0 models loaded; 0 answers generated; 0 AI judgments.')
        print('Send the printed output before continuing.')


if __name__ == '__main__':
    main()
