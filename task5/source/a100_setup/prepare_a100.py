"""Approved Task 5 isolated offline environment setup; no pretrained inference."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import venv
from guards import clean_environment, disk_requirement, install_command, requirement_lines, validate_inventory, require

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/content/drive/MyDrive/ATML-Assignment-2')
EXPERIMENT = PROJECT / 'task5/experiment_v1'
LOCAL = Path('/content/task5_isolated_runtime_v1')
EVALUATION_SHA = 'd6dd999e7fdf241c7136ee13f46babdf13338fa6dc12ff9046d2ea6a2e89fd87'
CPU_REPORT = 'combined_cpu_preflight_attempts/20261013T090737.971920Z_38745111/REPORT.json'
CPU_REPORT_SHA = '80b7e94d72a248c606f75d255e1315debdf9262c679455a5d69fc5def88388a1'


def child_environment(original, bootstrap_wheel=None):
    require(original.get('MPLBACKEND') in {None, '', 'module://matplotlib_inline.backend_inline'},
            'Unexpected plotting backend; preserve and review.')
    env = clean_environment(original, bootstrap_wheel)
    env.pop('MPLBACKEND', None)
    return env


def parse_hardware(text):
    import csv
    rows = list(csv.reader(text.strip().splitlines()))
    require(len(rows) == 1 and len(rows[0]) == 4 and 'A100' in rows[0][0].upper(),
            'Select one NVIDIA A100 runtime; no GPU fallback is approved.')
    name, memory, driver, device_uuid = [v.strip() for v in rows[0]]
    require(float(memory) > 0 and bool(driver), 'GPU inventory is incomplete.')
    return {'name': name, 'memory_mib': float(memory), 'driver': driver, 'uuid': device_uuid}


def validate_cpu(report, contract, contract_sha):
    require(report['status'] == 'CPU_READY_FOR_APPROVED_A100_SETUP' and report['tests_run'] == 44
            and report['tests_passed'] is True and not report['errors']
            and len(report['verified_snapshot_files']) == 16
            and len(report['wheel_validation']['verified_wheels']) == 90
            and report['base_package_inventory_unchanged'] is True
            and report['cpu_readiness_sha256'] == contract_sha
            and contract['complete'] is True and contract['release_sha256'] == EVALUATION_SHA,
            'The successful combined CPU checkpoint differs.')


def validate_approval(approval, proposal_sha, pins_sha, pins):
    require(approval['user_message'] == 'okay go ahead'
            and approval['approved_proposal_sha256'] == proposal_sha ==
                '20009615ee16b746ef0e4182004926b67ff7b39ce93600e5dabcf55400ccc19d'
            and approval['approved_candidate_versions_sha256'] == pins_sha ==
                '9f695dba56dfdbc4d3597586fe8ab6c90d38fa20a8bc3bcfccf50422bfe030ab'
            and approval['local_environment'] == str(LOCAL) and len(pins) == 84
            and approval['base_package_changes_approved'] is False
            and approval['downloads_approved'] is False
            and approval['pretrained_model_loading_approved'] is False,
            'The exact approved isolated setup scope differs.')


def main():
    # Verify the persisted evaluation source before importing its helpers.
    evaluation = EXPERIMENT / 'evaluation_releases' / EVALUATION_SHA
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    require(digest(evaluation / 'MANIFEST.json') == EVALUATION_SHA, 'Evaluation source manifest differs.')
    for name, expected in json.loads((evaluation / 'MANIFEST.json').read_text()).items():
        require(digest(evaluation / name) == expected, 'Evaluation source differs: ' + name)
    sys.path.insert(0, str(evaluation))
    from task5_run import core, preflight
    own_manifest = core.read_json(ROOT / 'MANIFEST.json')
    for name, expected in own_manifest.items():
        require(core.file_sha(ROOT / name) == expected, 'A100 setup source differs: ' + name)
    source_sha = core.file_sha(ROOT / 'MANIFEST.json')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid.uuid4().hex[:8]
    attempt = EXPERIMENT / 'a100_setup_attempts' / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'errors': [], 'commands': [], 'setup_release_sha256': source_sha,
              'evaluation_release_sha256': EVALUATION_SHA, 'base_package_changes_performed': False,
              'downloads': 0, 'pretrained_model_weights_loaded': 0, 'policy_answers_generated': 0,
              'ai_judgments': 0, 'installation_attempted': False, 'tokenizers_loaded': 0}
    before = None

    def command(argv, name, env=None):
        event = {'argv': [str(v) for v in argv], 'name': name}
        core.write_once(attempt / (name + '_COMMAND.json'), event)
        print('A100 setup:', name, flush=True)
        with (attempt / (name + '.log')).open('x') as log:
            process = subprocess.Popen(event['argv'], env=env or child_environment(os.environ),
                cwd='/content', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            event['returncode'] = process.wait()
        event['log_sha256'] = core.file_sha(attempt / (name + '.log'))
        core.write_once(attempt / (name + '_RESULT.json'), event)
        report['commands'].append(event)
        require(event['returncode'] == 0, name + ' failed; evidence preserved without fallback.')

    try:
        core.write_once(attempt / 'START.json', report)
        cpu_report = preflight.checked_json(EXPERIMENT / CPU_REPORT, CPU_REPORT_SHA)
        cpu_path = EXPERIMENT / 'CPU_READINESS_CONTRACT.json'
        cpu = preflight.checked_json(cpu_path, cpu_report['cpu_readiness_sha256'])
        validate_cpu(cpu_report, cpu, core.file_sha(cpu_path))
        pins = core.read_json(evaluation / 'package_versions.json')
        validate_approval(core.read_json(ROOT / 'APPROVAL.json'), core.file_sha(ROOT / 'APPROVED_PROPOSAL.json'),
                          core.file_sha(evaluation / 'package_versions.json'), pins)
        command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,uuid',
                 '--format=csv,noheader,nounits'], 'nvidia_inventory')
        report['hardware'] = parse_hardware((attempt / 'nvidia_inventory.log').read_text())
        command([sys.executable, '-I', '-B', ROOT / 'inspect_python.py', attempt / 'BASE_BEFORE.json'],
                'base_inventory_before')
        before = core.read_json(attempt / 'BASE_BEFORE.json')
        bindings = core.read_json(evaluation / 'bindings.json')
        print('Rechecking the cached wheels against this fresh A100 platform.', flush=True)
        wheel_check = preflight.verify_wheels(PROJECT, evaluation, bindings)
        core.write_once(attempt / 'FRESH_A100_WHEEL_CHECKS.json', wheel_check)
        wheel_contract_path = PROJECT / bindings['wheel_contract']['path']
        cache = preflight.checked_json(wheel_contract_path, bindings['wheel_contract']['sha256'])
        logical = cache['sizes']['isolated_84_wheels_zip_uncompressed_file_bytes']
        require(logical == 5522644346, 'Approved disk-sizing input differs.')
        core.write_once(attempt / 'DISK_CHECK.json', disk_requirement(logical, shutil.disk_usage('/content').free))

        import platform
        owner = {'format': 'ATML_TASK5_ISOLATED_OWNER_V1', 'directory': str(LOCAL),
                 'setup_release_sha256': source_sha, 'cpu_readiness_sha256': core.file_sha(cpu_path),
                 'approved_versions_sha256': core.file_sha(evaluation / 'package_versions.json'),
                 'base_python_executable': str(Path(sys.executable).resolve()),
                 'python_version': platform.python_version(), 'system_site_packages': False}
        owner_path = LOCAL / '.task5_isolation_owner.json'
        require(not LOCAL.is_symlink(), 'Linked environment directory.')
        if LOCAL.exists():
            require(owner_path.is_file() and core.read_json(owner_path) == owner,
                    'Unknown or different local environment; preserve and review.')
        else:
            LOCAL.mkdir()
            core.write_once(owner_path, owner)
        if not (LOCAL / 'pyvenv.cfg').exists():
            venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(LOCAL)
        python = LOCAL / 'bin/python'
        command([python, '-I', '-B', ROOT / 'inspect_python.py', attempt / 'ISOLATED_BEFORE.json'],
                'isolated_inventory_before')
        installed = validate_inventory(core.read_json(attempt / 'ISOLATED_BEFORE.json'), pins, LOCAL)
        wheels = wheel_contract_path.parent / 'wheels'
        requirements = attempt / 'OFFLINE_HASHED_REQUIREMENTS.txt'
        with requirements.open('x') as stream:
            stream.write(requirement_lines(cache['files'], pins, wheels))
        report['installation_attempted'] = True
        if 'pip' not in installed:
            pip_requirements = attempt / 'OFFLINE_PIP_BOOTSTRAP.txt'
            pip_requirements.write_text(requirement_lines(cache['files'], {'pip': pins['pip']}, wheels))
            pip_record = next(r for r in cache['files'].values() if r['package'] == 'pip' and r['version'] == pins['pip'])
            command(install_command(python, pip_requirements, attempt / 'PIP_INSTALL_REPORT.json', bootstrap=True),
                    'bootstrap_pinned_pip', child_environment(os.environ, wheels / pip_record['filename']))
        command(install_command(python, requirements, attempt / 'PACKAGE_INSTALL_REPORT.json'), 'install_exact_84')
        command([python, '-I', '-B', ROOT / 'inspect_python.py', attempt / 'ISOLATED_AFTER.json'],
                'isolated_inventory_after')
        isolated = core.read_json(attempt / 'ISOLATED_AFTER.json')
        validate_inventory(isolated, pins, LOCAL, complete=True)
        command([python, '-I', '-B', '-m', 'pip', '--disable-pip-version-check', 'check'], 'isolated_pip_check')
        command([python, '-I', '-B', ROOT / 'probe_a100.py', evaluation, PROJECT, attempt], 'isolated_gpu_probe')
        probe = core.read_json(attempt / 'GPU_PROBE.json')
        require(probe['fresh_imports_passed'] and probe['tiny_float16_cuda_passed'] and probe['tiny_nf4_cuda_passed']
                and probe['tests_passed'] and probe['tests_run'] == 44 and probe['policy_prompts_checked'] == 400
                and probe['diagnostic_judge_prompts_checked'] == 80 and probe['pretrained_model_weights_loaded'] == 0,
                'A100 synthetic/tokenizer validation is incomplete.')
        report['tokenizers_loaded'] = 2
        command([sys.executable, '-I', '-B', ROOT / 'inspect_python.py', attempt / 'BASE_AFTER.json'],
                'base_inventory_after')
        require(core.read_json(attempt / 'BASE_AFTER.json') == before, 'Base package inventory changed.')
        report['base_package_inventory_unchanged'] = True
        contract = {'format': 'ATML_TASK5_ISOLATED_ENVIRONMENT_V1', 'complete': True,
                    'synthetic_cuda_nf4_passed': True, 'package_versions': pins,
                    'cpu_readiness_sha256': core.file_sha(cpu_path), 'setup_release_sha256': source_sha,
                    'evaluation_release_sha256': EVALUATION_SHA, 'python_executable': str(python),
                    'python_version': isolated['python'], 'system_site_packages': False,
                    'user_site_enabled': False, 'isolated_dependency_check_passed': True,
                    'base_packages_unchanged': True, 'probe': probe,
                    'pretrained_model_checks_pending': True,
                    'actual_pretrained_execution_separate_approval_required': True}
        target = EXPERIMENT / 'ISOLATED_ENVIRONMENT_CONTRACT.json'
        if target.exists():
            require(core.read_json(target) == contract, 'Existing environment contract differs; preserve and review.')
        else:
            core.write_once(target, contract)
        report['environment_contract_sha256'] = core.file_sha(target)
        report['status'] = 'A100_ISOLATED_ENVIRONMENT_READY_PRETRAINED_EXECUTION_APPROVAL_PENDING'
        report['isolated_packages_verified'] = 84
        report['gpu_probe'] = probe
    except Exception as exc:
        report['status'] = 'A100_SETUP_STOPPED_EVIDENCE_PRESERVED'
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    finally:
        if before is not None and 'base_package_inventory_unchanged' not in report:
            try:
                command([sys.executable, '-I', '-B', ROOT / 'inspect_python.py', attempt / 'BASE_FAILURE_AFTER.json'],
                        'base_inventory_after_stop')
                report['base_package_inventory_unchanged'] = core.read_json(attempt / 'BASE_FAILURE_AFTER.json') == before
                if not report['base_package_inventory_unchanged']:
                    report['errors'].append('Base package inventory changed.')
            except Exception as exc:
                report['errors'].append('Post-stop inventory check: ' + str(exc))
        report['artifact_sha256'] = {str(p.relative_to(attempt)): core.file_sha(p)
                                    for p in attempt.rglob('*') if p.is_file()}
        core.write_once(attempt / 'REPORT.json', report)
        print('Status:', report['status'])
        print('Isolated packages:', report.get('isolated_packages_verified'), '/ 84')
        print('GPU checks:', json.dumps(report.get('gpu_probe', {}), indent=2))
        print('Base packages unchanged:', report.get('base_package_inventory_unchanged'))
        print('Errors:', report['errors'])
        print('Saved report:', attempt / 'REPORT.json')
        print('Report SHA-256:', core.file_sha(attempt / 'REPORT.json'))
        print('0 downloads; 0 pretrained model weight loads; 0 policy answers; 0 AI judgments.')
        print('Keep this A100 runtime connected and send the output before continuing.')
    return report


if __name__ == '__main__':
    raise SystemExit(0 if main()['status'] == 'A100_ISOLATED_ENVIRONMENT_READY_PRETRAINED_EXECUTION_APPROVAL_PENDING' else 1)
