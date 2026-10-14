"""Resume the unchanged frozen runner. Only missing answers are generated."""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib, json, os, re, signal, subprocess, types, uuid

PROJECT = Path('/content/drive/MyDrive/ATML-Assignment-2')
PYTHON = Path('/content/task5_isolated_runtime_v1/bin/python')
RELEASE_SHA = 'd6dd999e7fdf241c7136ee13f46babdf13338fa6dc12ff9046d2ea6a2e89fd87'
PROTOCOL_SHA = '58632138b81533416b3bb46979e03ca508323dd56dbbba572e0249d1b1d8665a'
AUDIT_PATH = 'resume_audit_attempts/20261013T181642.449932Z_ac16e163/REPORT.json'
AUDIT_SHA = '3c4956879b74cc7b7dbdaaa2df4c564377bbd390758853c7aced813c9e1918e2'
RESUME_PATH = 'a100_resume_attempts/20261013T183737.071719Z_37a523ea/REPORT.json'
RESUME_SHA = 'f904a525c644e274fdb4e33c4f9c9943e2069950c5199cfe4b4e7f924223a3dd'
SETUP_SHA = 'a889212fd696664380bf93f3b93ef454c2fcb96f46748161e4a9a9f1208cb80f'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    require(path.is_file() and not path.is_symlink(), 'Missing/linked file: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path, expected):
    require(digest(path) == expected, 'Saved file checksum differs: ' + str(path))
    return json.loads(path.read_text())


def inventory(root, rows, b, audit, core, before):
    """Quick counts and anchored record tips; the frozen runner validates full chains."""
    counts = {}
    for policy in core.POLICIES:
        for domain in core.DOMAINS:
            name = policy + '/' + domain
            folder = root / 'evaluation_v1/generations' / policy / domain
            if not folder.exists():
                require(audit['counts'][name] == 0, 'Previously saved result folder is missing: ' + name)
                counts[name] = 0
                continue
            require(folder.is_dir() and not folder.is_symlink(), 'Linked/non-directory result folder.')
            binding_path = folder / 'BINDING.json'
            digest(binding_path)
            binding = core.read_json(binding_path)
            require(binding['phase'] == 'generation' and binding['policy'] == policy and binding['domain'] == domain
                    and binding['release_sha256'] == RELEASE_SHA and binding['seed'] == 6304
                    and binding['hardware'] == audit['saved_hardware']
                    and binding['settings'] == audit['saved_generation_settings']
                    and binding['dataset_sha256'] == b['asset_sha256'][b['datasets'][domain]]
                    and binding['row_sha256_in_order'] == [core.checksum(row) for row in rows[domain]],
                    'Saved generation binding differs: ' + name)
            units = folder / 'units'
            require(units.is_dir() and not units.is_symlink(), 'Missing/linked units folder.')
            paths = sorted(units.glob('*.json'))
            require(len(paths) <= len(rows[domain])
                    and all(p.name == f'{i:06d}.json' for i, p in enumerate(paths)),
                    'Saved result sequence has a gap or unexpected filename: ' + name)
            old_count = audit['counts'][name]
            require(len(paths) >= old_count, 'Previously saved answers are missing: ' + name)
            if name in audit['generation_bindings']:
                anchor = audit['generation_bindings'][name]
                require(core.checksum(binding) == anchor['binding_sha256'], 'Audited binding changed: ' + name)
                if old_count:
                    path = units / f'{old_count-1:06d}.json'
                    digest(path)
                    require(core.checksum(core.read_json(path)) == anchor['last_record_sha256'],
                            'Previously audited checkpoint tip changed: ' + name)
            if paths:
                path = paths[-1]
                digest(path)
                record = core.read_json(path)
                contents = {k: v for k, v in record.items() if k != 'record_sha256'}
                p = record['payload']
                require(record['record_sha256'] == core.checksum(contents)
                        and record['index'] == p['row_index'] == len(paths)-1
                        and record['binding_sha256'] == core.checksum(binding)
                        and p['policy'] == policy and p['domain'] == domain
                        and p['row_sha256'] == core.checksum(rows[domain][len(paths)-1])
                        and record['rng_after'] is not None, 'Saved final record/RNG differs: ' + name)
            counts[name] = len(paths)
    if before:
        require(counts == audit['counts'], 'Saved counts changed since the CPU audit; preserve and review.')
    return counts


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


def main():
    root = PROJECT / 'task5/experiment_v1'
    release = root / 'evaluation_releases' / RELEASE_SHA
    require(root.is_dir() and PYTHON.is_file(), 'Keep the prepared A100 runtime connected.')
    manifest = checked(release / 'MANIFEST.json', RELEASE_SHA)
    for name, expected in manifest.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source manifest entry.')
        require(digest(release / name) == expected, 'Frozen evaluation source differs: ' + name)
    require(digest(release / 'protocol.json') == PROTOCOL_SHA, 'Approved evaluation protocol differs.')
    core = types.ModuleType('verified_task5_generation_persistence')
    exec(compile((release / 'task5_run/core.py').read_bytes(), 'verified_task5_core.py', 'exec'), core.__dict__)
    audit = checked(root / AUDIT_PATH, AUDIT_SHA)
    resumed = checked(root / RESUME_PATH, RESUME_SHA)
    require(audit['audit_complete'] is True and not audit['errors'] and audit['completed_answers'] == 1000
            and audit['evaluation_release_sha256'] == RELEASE_SHA
            and resumed['status'] == 'A100_RESUME_SETUP_VERIFIED_GENERATION_PENDING' and not resumed['errors']
            and resumed['resume_cpu_audit_sha256'] == AUDIT_SHA
            and resumed['fresh_setup_report_sha256'] == SETUP_SHA,
            'Successful CPU/A100 resume evidence differs.')
    setup = checked(Path(resumed['fresh_setup_report']), SETUP_SHA)
    require(setup['isolated_packages_verified'] == 84 and not setup['errors']
            and setup['base_package_inventory_unchanged'] is True
            and setup['gpu_probe']['tests_passed'] is True and setup['gpu_probe']['tests_run'] == 44,
            'Fresh A100 setup checks are incomplete.')
    contract_sha = digest(root / 'ISOLATED_ENVIRONMENT_CONTRACT.json')
    require(contract_sha == resumed['environment_contract_sha256'] == audit['prior_environment_contract_sha256']
            == setup['environment_contract_sha256'], 'Saved environment contract differs.')
    approval = core.read_json(root / 'EXECUTION_APPROVAL.json')
    require(approval == {'format': 'ATML_TASK5_PRETRAINED_EXECUTION_APPROVAL_V1', 'approved': True,
        'user_message': 'yes', 'generation': True, 'ai_judging': True, 'quantitative_aggregation': True,
        'protocol_sha256': PROTOCOL_SHA, 'evaluation_release_sha256': RELEASE_SHA,
        'planned_policy_answers': 1200, 'planned_ai_comparison_events': 880,
        'training': False, 'downloads': False, 'publishing': False, 'protocol_changes': False,
        'one_code_block_at_a_time': True, 'save_after_every_completed_unit': True}, 'Execution approval differs.')
    b = core.read_json(release / 'bindings.json')
    rows = {}
    for domain, relative in b['datasets'].items():
        require(digest(PROJECT / relative) == b['asset_sha256'][relative], 'Fixed evaluation data differs.')
        rows[domain] = core.read_rows(PROJECT / relative)
    core.generation_schedule(rows)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid.uuid4().hex[:8]
    attempt = root / 'generation_attempts' / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    report = {'status': 'GENERATION_RESUME_RUNNING', 'errors': [], 'exact_final_count_checked': False,
              'wrapper_source_sha256': globals().get('WRAPPER_SOURCE_SHA256'),
              'evaluation_release_sha256': RELEASE_SHA, 'resume_cpu_audit_sha256': AUDIT_SHA,
              'a100_resume_report_sha256': RESUME_SHA, 'execution_approval_sha256': digest(root / 'EXECUTION_APPROVAL.json'),
              'last_reported_counts': dict(audit['counts']), 'ai_judgments': 0,
              'installs': 0, 'downloads': 0, 'training': False, 'protocol_changes': False,
              'count_check_method': 'Ordered committed filenames, frozen bindings and record tips; full chains validated by unchanged runner.'}
    core.write_once(attempt / 'START.json', report)
    process = None
    worker = r'''
import sys
from pathlib import Path
release, project, audit_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(release))
from task5_run import core
from task5_run.runtime import execution_gate
from task5_run.run import generate
core.require(execution_gate(project, release, 'generation') == core.read_json(audit_path)['saved_hardware'],
             'Current full runtime identity differs; no model loaded.')
print('Runtime matched. Rechecking saved policy groups; completed answers will be skipped.', flush=True)
generate(project, release)
'''
    try:
        report['counts_before'] = inventory(root, rows, b, audit, core, before=True)
        core.write_once(attempt / 'EXECUTION_APPROVAL.json', approval)
        print('Resuming generation: 1,000 saved; 200 answers remaining.', flush=True)
        print('Next new answer: RLAIF GSM 201/300; then 100 transfer answers.', flush=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'PIP_')) and k != 'VIRTUAL_ENV'}
        require(env.get('MPLBACKEND') in {None, '', 'module://matplotlib_inline.backend_inline'}, 'Unexpected plotting backend.')
        env.pop('MPLBACKEND', None)
        env.update({'PIP_CONFIG_FILE': os.devnull, 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        argv = [str(PYTHON), '-I', '-B', '-u', '-c', worker, str(release), str(PROJECT), str(root / AUDIT_PATH)]
        core.write_once(attempt / 'COMMAND.json', {'argv': argv, 'worker_sha256': hashlib.sha256(worker.encode()).hexdigest()})
        with (attempt / 'GENERATION.log').open('x') as log:
            process = subprocess.Popen(argv, cwd=release, env=env, start_new_session=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                event = re.fullmatch(r'(sft|rlvr|rlaif)/(gsm|transfer): (\d+)/(\d+) answers saved\s*', line)
                if event:
                    policy, domain, count, total = event[1], event[2], int(event[3]), int(event[4])
                    name = policy + '/' + domain
                    require(policy == 'rlaif' and total == core.COUNTS[domain]
                            and count == report['last_reported_counts'][name]+1 and count <= total,
                            'Unexpected generation progress; stopping for review.')
                    report['last_reported_counts'][name] = count
                    if count % 25 == 0 or count == total:
                        core.write_once(attempt / 'progress' / f'{policy}_{domain}_{count:06d}.json',
                                        {'reported_saved_counts': dict(report['last_reported_counts'])})
                if not event or int(event[3]) in (1, 201) or int(event[3]) % 25 == 0 or event[3] == event[4]:
                    print(line, end='', flush=True)
            report['worker_returncode'] = process.wait()
        require(report['worker_returncode'] == 0, 'Generation worker stopped; existing committed answers preserved.')
        report['generation_log_sha256'] = digest(attempt / 'GENERATION.log')
        report['counts_after'] = inventory(root, rows, b, audit, core, before=False)
        require(all(n == core.COUNTS[name.split('/')[1]] for name, n in report['counts_after'].items()),
                'Generation schedule remains incomplete.')
        report.update({'exact_final_count_checked': True, 'completed_answers': 1200, 'remaining_answers': 0,
                       'newly_saved_answers': 200, 'ai_comparison_events_remaining': 880,
                       'status': 'POLICY_GENERATION_COMPLETE_AI_JUDGING_PENDING'})
    except KeyboardInterrupt:
        report['status'] = 'POLICY_GENERATION_INTERRUPTED_EVIDENCE_PRESERVED'
        report['errors'].append('Interrupted. Last-reported counts are a lower bound; exact final count awaits checkpoint audit.')
    except Exception as exc:
        report['status'] = 'POLICY_GENERATION_STOPPED_EVIDENCE_PRESERVED'
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    finally:
        report['owned_worker_group_stopped'] = stop_owned(process)
        # No checkpoint rescan here: an interrupted closing audit cannot cause a second long audit.
        core.write_once(attempt / 'REPORT.json', report)
        print('Status:', report['status'])
        if report['exact_final_count_checked']:
            print('Saved answers:', json.dumps(report['counts_after'], indent=2))
            print('Completed / remaining:', report['completed_answers'], '/', report['remaining_answers'])
            print('New answers saved:', report['newly_saved_answers'])
        else:
            print('Last-reported saved counts:', json.dumps(report['last_reported_counts'], indent=2))
            print('Exact final completed / remaining: UNCONFIRMED')
        print('AI judgments in this cell: 0; pending AI comparison events: 880.')
        print('Errors:', report['errors'])
        print('Saved report:', attempt / 'REPORT.json')
        print('Report SHA-256:', digest(attempt / 'REPORT.json'))
        print('Keep the A100 connected and send the full output before continuing.')


if __name__ == '__main__':
    main()
