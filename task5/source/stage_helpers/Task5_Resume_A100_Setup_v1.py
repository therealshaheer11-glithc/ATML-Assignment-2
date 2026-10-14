# Task 5: restore the approved A100 environment; generation follows in a later cell.
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import csv, hashlib, io, json, os, platform, shutil, signal, subprocess, sys, types, uuid

PROJECT = Path('/content/drive/MyDrive/ATML-Assignment-2')
EVALUATION_SHA = 'd6dd999e7fdf241c7136ee13f46babdf13338fa6dc12ff9046d2ea6a2e89fd87'
SETUP_SHA = 'cace5df894e95d9e7d4d7064da06f32edc183500b0df6fb64f7919b36085ff9d'
AUDIT_PATH = 'resume_audit_attempts/20261013T181642.449932Z_ac16e163/REPORT.json'
AUDIT_SHA = '3c4956879b74cc7b7dbdaaa2df4c564377bbd390758853c7aced813c9e1918e2'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    require(path.is_file() and not path.is_symlink(), 'Missing/linked file: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_release(folder, expected):
    require(folder.is_dir() and not folder.is_symlink(), 'Missing/linked saved source folder.')
    require(digest(folder / 'MANIFEST.json') == expected, 'Saved source manifest differs.')
    for name, sha in json.loads((folder / 'MANIFEST.json').read_text()).items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source manifest entry.')
        require(digest(folder / name) == sha, 'Saved source differs: ' + name)


def clean_child_environment(original):
    require(original.get('MPLBACKEND') in {None, '', 'module://matplotlib_inline.backend_inline'},
            'Unexpected plotting backend; preserve and review.')
    env = {k: v for k, v in original.items()
           if not k.startswith(('PYTHON', 'PIP_')) and k != 'VIRTUAL_ENV'}
    env.pop('MPLBACKEND', None)
    env.update({'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    return env


def verify_runtime(observed, saved):
    require(all(observed.get(k) == saved.get(k) for k in ('python', 'gpu', 'driver')),
            'The new Python/GPU/driver differs from the saved run. No installation started; send this report for review.')


def main():
    smi = shutil.which('nvidia-smi')
    require(smi is not None, 'Switch Colab to an A100 runtime before running this cell.')
    gpu = subprocess.run([smi, '--query-gpu=name,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True, timeout=15)
    rows = list(csv.reader(io.StringIO(gpu.stdout.strip())))
    require(gpu.returncode == 0 and len(rows) == 1 and len(rows[0]) == 2 and 'A100' in rows[0][0],
            'Select exactly one A100 runtime before running this cell.')
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    root = PROJECT / 'task5/experiment_v1'
    evaluation = root / 'evaluation_releases' / EVALUATION_SHA
    setup = root / 'a100_setup_releases' / SETUP_SHA
    verify_release(evaluation, EVALUATION_SHA)
    verify_release(setup, SETUP_SHA)
    require(digest(root / AUDIT_PATH) == AUDIT_SHA, 'Successful CPU resume audit differs.')
    audit = json.loads((root / AUDIT_PATH).read_text())
    require(audit['status'] == 'CPU_RESUME_AUDIT_PASSED_A100_REVALIDATION_PENDING'
            and audit['audit_complete'] is True and not audit['errors']
            and audit['evaluation_release_sha256'] == EVALUATION_SHA
            and audit['counts'] == {'sft/gsm': 300, 'sft/transfer': 100, 'rlvr/gsm': 300,
                                   'rlvr/transfer': 100, 'rlaif/gsm': 200, 'rlaif/transfer': 0},
            'CPU resume audit did not verify the expected saved run.')
    core = types.ModuleType('verified_task5_resume_persistence')
    exec(compile((evaluation / 'task5_run/core.py').read_bytes(), 'verified_task5_core.py', 'exec'), core.__dict__)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid.uuid4().hex[:8]
    attempt = root / 'a100_resume_attempts' / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    report = {'status': 'A100_RESUME_SETUP_RUNNING', 'errors': [], 'resume_cpu_audit_sha256': AUDIT_SHA,
              'resume_wrapper_source_sha256': globals().get('WRAPPER_SOURCE_SHA256'),
              'setup_release_sha256': SETUP_SHA, 'evaluation_release_sha256': EVALUATION_SHA,
              'previously_verified_answers': 1000, 'remaining_answers': 200,
              'models_loaded': 0, 'answers_generated': 0, 'ai_judgments': 0, 'downloads': 0,
              'setup_worker_started': False, 'protocol_changes': False}
    process = None
    core.write_once(attempt / 'START.json', report)
    try:
        observed = {'python': platform.python_version(), 'gpu': rows[0][0].strip(), 'driver': rows[0][1].strip()}
        report['observed_runtime'] = observed
        report['saved_runtime'] = {k: audit['saved_hardware'][k] for k in ('python', 'gpu', 'driver')}
        verify_runtime(observed, audit['saved_hardware'])
        contract_path = root / 'ISOLATED_ENVIRONMENT_CONTRACT.json'
        old_contract_sha = digest(contract_path)
        require(old_contract_sha == audit['prior_environment_contract_sha256'], 'Saved environment contract differs.')
        previous_attempts = set((root / 'a100_setup_attempts').iterdir())
        env = clean_child_environment(os.environ)
        argv = [sys.executable, '-s', '-B', str(setup / 'prepare_a100.py')]
        core.write_once(attempt / 'COMMAND.json', {'argv': argv})
        print('Verified saved run: 1,000 answers; 200 remaining; 880 AI comparisons pending.', flush=True)
        print('Restoring the approved isolated A100 environment from cached packages...', flush=True)
        with (attempt / 'SETUP.log').open('x') as log:
            process = subprocess.Popen(argv, cwd='/content', env=env, start_new_session=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            report['setup_worker_started'] = True
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            report['worker_returncode'] = process.wait()
        require(report['worker_returncode'] == 0, 'A100 setup stopped; preserve the output for review.')
        created = set((root / 'a100_setup_attempts').iterdir()) - previous_attempts
        require(len(created) == 1, 'Unexpected setup attempt coverage; preserve and review.')
        new_report_path = created.pop() / 'REPORT.json'
        new_report = json.loads(new_report_path.read_text())
        require(new_report['status'] == 'A100_ISOLATED_ENVIRONMENT_READY_PRETRAINED_EXECUTION_APPROVAL_PENDING'
                and not new_report['errors'] and new_report['isolated_packages_verified'] == 84
                and new_report['base_package_inventory_unchanged'] is True
                and new_report['gpu_probe']['tests_run'] == 44
                and new_report['gpu_probe']['tests_passed'] is True
                and new_report['environment_contract_sha256'] == old_contract_sha == digest(contract_path),
                'Fresh setup did not reproduce the approved saved environment.')
        report.update({'status': 'A100_RESUME_SETUP_VERIFIED_GENERATION_PENDING',
                       'fresh_setup_report': str(new_report_path), 'fresh_setup_report_sha256': digest(new_report_path),
                       'environment_contract_sha256': old_contract_sha, 'isolated_packages_verified': 84,
                       'base_packages_unchanged': True})
    except KeyboardInterrupt:
        report['status'] = 'A100_RESUME_SETUP_INTERRUPTED_REVIEW_PENDING'
        report['errors'].append('Setup interrupted; no generation started.')
    except Exception as exc:
        report['status'] = 'A100_RESUME_SETUP_STOPPED_REVIEW_PENDING'
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    finally:
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
            report['owned_setup_process_group_stopped'] = True
        core.write_once(attempt / 'REPORT.json', report)
        print('Resume status:', report['status'])
        print('Observed runtime:', report.get('observed_runtime'))
        print('Required saved runtime:', report.get('saved_runtime'))
        print('Errors:', report['errors'])
        print('Saved resume report:', attempt / 'REPORT.json')
        print('Resume report SHA-256:', digest(attempt / 'REPORT.json'))
        print('0 downloads; 0 pretrained model loads; 0 answers generated; 0 AI judgments.')
        print('Keep this A100 runtime connected and send the full printed output before continuing.')


if __name__ == '__main__':
    main()
