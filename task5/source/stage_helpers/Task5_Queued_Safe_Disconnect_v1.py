# Queue this cell immediately AFTER the running generation cell.
def task5_disconnect_after_saved_completion():
    from pathlib import Path
    import hashlib, json, types, uuid
    from google.colab import drive, runtime

    root = Path('/content/drive/MyDrive/ATML-Assignment-2/task5/experiment_v1')
    release_sha = 'd6dd999e7fdf241c7136ee13f46babdf13338fa6dc12ff9046d2ea6a2e89fd87'
    wrapper_sha = '58443d1ee5d98ed9bde26ea495d107c6a203a7550aaf067d8773c63526593533'
    audit_sha = '3c4956879b74cc7b7dbdaaa2df4c564377bbd390758853c7aced813c9e1918e2'
    expected = {p+'/'+d: n for p in ('sft', 'rlvr', 'rlaif')
                for d, n in (('gsm', 300), ('transfer', 100))}

    def require(ok, message):
        if not ok:
            raise RuntimeError('NOT disconnected: ' + message)

    def sha(path):
        require(path.is_file() and not path.is_symlink(), 'Missing/linked file: '+str(path))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    attempts = sorted(p for p in (root/'generation_attempts').iterdir() if p.is_dir())
    require(bool(attempts), 'No generation attempt found.')
    attempt = attempts[-1]  # Never substitute an older successful attempt.
    require(not attempt.is_symlink(), 'Linked generation attempt.')
    report_path = attempt/'REPORT.json'
    report_sha = sha(report_path)
    report = json.loads(report_path.read_text())
    require(report.get('status') == 'POLICY_GENERATION_COMPLETE_AI_JUDGING_PENDING'
            and report.get('wrapper_source_sha256') == wrapper_sha
            and report.get('evaluation_release_sha256') == release_sha
            and report.get('resume_cpu_audit_sha256') == audit_sha
            and report.get('worker_returncode') == 0 and not report.get('errors')
            and report.get('exact_final_count_checked') is True
            and report.get('completed_answers') == 1200 and report.get('remaining_answers') == 0
            and report.get('counts_after') == expected, 'Latest generation is not confirmed complete.')
    require(sha(attempt/'GENERATION.log') == report['generation_log_sha256'], 'Generation log differs.')
    release = root/'evaluation_releases'/release_sha
    require(sha(release/'MANIFEST.json') == release_sha, 'Frozen source manifest differs.')
    for name, value in json.loads((release/'MANIFEST.json').read_text()).items():
        require(sha(release/name) == value, 'Frozen source differs: '+name)
    core = types.ModuleType('verified_task5_disconnect_persistence')
    exec(compile((release/'task5_run/core.py').read_bytes(), 'verified_task5_core.py', 'exec'), core.__dict__)
    audit_path = root/'resume_audit_attempts/20261013T181642.449932Z_ac16e163/REPORT.json'
    require(sha(audit_path) == audit_sha, 'CPU checkpoint audit differs.')
    audit = core.read_json(audit_path)

    with core.process_lock(root/'evaluation_v1'):
        for name, total in expected.items():
            policy, domain = name.split('/')
            folder = root/'evaluation_v1/generations'/policy/domain
            binding = core.read_json(folder/'BINDING.json')
            require(binding['release_sha256'] == release_sha and binding['policy'] == policy
                    and binding['domain'] == domain and binding['phase'] == 'generation'
                    and binding['hardware'] == audit['saved_hardware']
                    and binding['settings'] == audit['saved_generation_settings'], 'Saved run binding differs: '+name)
            units = folder/'units'
            paths = sorted(units.glob('*.json'))
            require(len(paths) == total and all(p.name == f'{i:06d}.json' and p.is_file()
                    and not p.is_symlink() for i, p in enumerate(paths)), 'Saved answer files incomplete: '+name)
            last = core.read_json(paths[-1])
            contents = {k: v for k, v in last.items() if k != 'record_sha256'}
            require(last['record_sha256'] == core.checksum(contents) and last['index'] == total-1
                    and last['binding_sha256'] == core.checksum(binding) and last['rng_after'] is not None
                    and last['payload']['row_sha256'] == binding['row_sha256_in_order'][-1],
                    'Saved final answer/RNG differs: '+name)
            if audit['counts'][name]:
                old = core.read_json(units/f"{audit['counts'][name]-1:06d}.json")
                require(core.checksum(old) == audit['generation_bindings'][name]['last_record_sha256'],
                        'Previously audited checkpoint changed: '+name)
        receipt = attempt/('DISCONNECT_READY_'+uuid.uuid4().hex[:8]+'.json')
        core.write_once(receipt, {'status': 'READY_TO_FLUSH_AND_REQUEST_RUNTIME_RELEASE',
            'generation_report': str(report_path), 'generation_report_sha256': report_sha,
            'completed_answers': 1200, 'remaining_answers': 0, 'ai_comparisons_pending': 880,
            'checks': 'Completion report, all checkpoint filenames, frozen bindings and anchored record tips.',
            'disconnect_success_not_yet_confirmed': True})
        print('Verified saved completion: 1,200 / 1,200 answers.', flush=True)
        print('Generation report:', report_path, flush=True)
        print('Generation report SHA-256:', report_sha, flush=True)
        print('Disconnect receipt:', receipt, flush=True)
    print('Flushing pending Drive writes...', flush=True)
    drive.flush_and_unmount(timeout_ms=60000)
    print('Drive flush succeeded. Requesting runtime release now.', flush=True)
    runtime.unassign()

task5_disconnect_after_saved_completion()
