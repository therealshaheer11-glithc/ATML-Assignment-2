# Task 5 - Block 9: approved generation only.
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, re, subprocess, types, uuid

project = Path('/content/drive/MyDrive/ATML-Assignment-2')
root = project / 'task5/experiment_v1'
release_sha = 'd6dd999e7fdf241c7136ee13f46babdf13338fa6dc12ff9046d2ea6a2e89fd87'
release = root / 'evaluation_releases' / release_sha
python = Path('/content/task5_isolated_runtime_v1/bin/python')
setup_report = root / 'a100_setup_attempts/20261013T093754.042876Z_44eefae6/REPORT.json'
setup_report_sha = '440f9c23ddc59291a96730b99cc5d9c6c6df00ee8d90996f7cac40f911b5af06'

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

require(root.is_dir() and python.is_file(),
        'Keep the same prepared A100 runtime connected.')
require(digest(release / 'MANIFEST.json') == release_sha,
        'Evaluation source manifest differs.')
for name, expected in json.loads((release / 'MANIFEST.json').read_text()).items():
    path = release / name
    require(path.is_file() and not path.is_symlink() and digest(path) == expected,
            'Evaluation source differs: ' + name)

core = types.ModuleType('task5_block9_persistence')
exec(compile((release / 'task5_run/core.py').read_bytes(),
             'checked_task5_core.py', 'exec'), core.__dict__)

require(digest(setup_report) == setup_report_sha,
        'Successful A100 setup report differs.')
previous = core.read_json(setup_report)
require(previous['status'] ==
        'A100_ISOLATED_ENVIRONMENT_READY_PRETRAINED_EXECUTION_APPROVAL_PENDING'
        and previous['isolated_packages_verified'] == 84
        and not previous['errors']
        and previous['base_package_inventory_unchanged'] is True,
        'A100 setup did not finish successfully.')

contract_path = root / 'ISOLATED_ENVIRONMENT_CONTRACT.json'
require(digest(contract_path) == previous['environment_contract_sha256'],
        'A100 environment contract differs.')
contract = core.read_json(contract_path)
require(contract['complete'] is True
        and contract['synthetic_cuda_nf4_passed'] is True
        and contract['probe']['tests_run'] == 44
        and contract['probe']['tests_passed'] is True,
        'A100 checks are incomplete.')
require(digest(release / 'protocol.json') ==
        '58632138b81533416b3bb46979e03ca508323dd56dbbba572e0249d1b1d8665a',
        'Approved evaluation protocol differs.')

approval = {
    'format': 'ATML_TASK5_PRETRAINED_EXECUTION_APPROVAL_V1',
    'approved': True, 'user_message': 'yes',
    'generation': True, 'ai_judging': True,
    'quantitative_aggregation': True,
    'protocol_sha256': digest(release / 'protocol.json'),
    'evaluation_release_sha256': release_sha,
    'planned_policy_answers': 1200,
    'planned_ai_comparison_events': 880,
    'training': False, 'downloads': False, 'publishing': False,
    'protocol_changes': False,
    'one_code_block_at_a_time': True,
    'save_after_every_completed_unit': True,
}
approval_path = root / 'EXECUTION_APPROVAL.json'
if approval_path.exists():
    require(core.read_json(approval_path) == approval,
            'Existing execution approval differs; preserve and review.')
else:
    core.write_once(approval_path, approval)

bindings = core.read_json(release / 'bindings.json')

def saved_counts():
    counts = {}
    for policy in core.POLICIES:
        for domain in core.DOMAINS:
            name = policy + '/' + domain
            folder = root / 'evaluation_v1/generations' / policy / domain
            if not folder.exists():
                counts[name] = 0
                continue
            require((folder / 'BINDING.json').is_file(),
                    'Unbound result folder; preserve and review: ' + name)
            binding = core.read_json(folder / 'BINDING.json')
            require(binding['release_sha256'] == release_sha
                    and binding['policy'] == policy
                    and binding['domain'] == domain
                    and binding['phase'] == 'generation'
                    and binding['dataset_sha256'] ==
                        bindings['asset_sha256'][bindings['datasets'][domain]]
                    and len(binding['row_sha256_in_order']) == core.COUNTS[domain],
                    'Saved run identity differs: ' + name)
            records = core.Journal(folder, binding).records()
            require(len(records) <= core.COUNTS[domain],
                    'Too many saved answers: ' + name)
            for i, record in enumerate(records):
                payload = record['payload']
                require(payload['row_index'] == i
                        and payload['policy'] == policy
                        and payload['domain'] == domain
                        and record['rng_after'] is not None
                        and payload['row_sha256'] ==
                            binding['row_sha256_in_order'][i],
                        'Saved answer/RNG identity differs: ' + name)
            counts[name] = len(records)
    return counts

stamp = (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
         + '_' + uuid.uuid4().hex[:8])
attempt = root / 'generation_attempts' / stamp
attempt.mkdir(parents=True, exist_ok=False)
report = {
    'status': 'RUNNING', 'errors': [],
    'ai_judgments_in_this_block': 0,
    'downloads': 0, 'training': False, 'protocol_changes': False,
    'evaluation_release_sha256': release_sha,
    'setup_report_sha256': setup_report_sha,
    'execution_approval_sha256': digest(approval_path),
}
worker = r'''
import sys
from pathlib import Path
release, project = map(Path, sys.argv[1:])
sys.path.insert(0, str(release))
from task5_run.run import generate
generate(project, release)
'''
process = None

try:
    report['counts_before'] = saved_counts()
    core.write_once(attempt / 'START.json', report)
    core.write_once(attempt / 'EXECUTION_APPROVAL.json', approval)
    print('Starting frozen-policy generation: 1,200 answers total.', flush=True)
    print('Previously saved:', sum(report['counts_before'].values()),
          '/ 1200', flush=True)

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PYTHON', 'PIP_')) and k != 'VIRTUAL_ENV'}
    require(env.get('MPLBACKEND') in {
        None, '', 'module://matplotlib_inline.backend_inline'
    }, 'Unexpected child plotting backend; preserve and review.')
    env.pop('MPLBACKEND', None)
    env.update({
        'PIP_CONFIG_FILE': os.devnull,
        'PYTHONNOUSERSITE': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
    })
    argv = [str(python), '-I', '-B', '-u', '-c',
            worker, str(release), str(project)]
    core.write_once(attempt / 'COMMAND.json', {
        'argv': argv,
        'worker_sha256': hashlib.sha256(worker.encode()).hexdigest(),
    })

    with (attempt / 'GENERATION.log').open('x') as log:
        process = subprocess.Popen(
            argv, cwd=release, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            progress = re.fullmatch(
                r'(sft|rlvr|rlaif)/(gsm|transfer): (\d+)/(\d+) answers saved\s*',
                line
            )
            if (not progress or int(progress[3]) == 1
                    or int(progress[3]) % 25 == 0
                    or progress[3] == progress[4]):
                print(line, end='', flush=True)
        report['worker_returncode'] = process.wait()

    report['counts_after'] = saved_counts()
    total = sum(report['counts_after'].values())
    report['completed_answers'] = total
    report['newly_saved_answers'] = total - sum(report['counts_before'].values())
    report['remaining_answers'] = 1200 - total
    require(report['worker_returncode'] == 0,
            'Generation stopped; committed answers are preserved.')
    require(total == 1200, 'The generation schedule is incomplete.')
    report['status'] = 'POLICY_GENERATION_COMPLETE_AI_JUDGING_PENDING'

except KeyboardInterrupt:
    report['status'] = 'POLICY_GENERATION_INTERRUPTED_EVIDENCE_PRESERVED'
    report['errors'].append(
        'Generation interrupted; committed answers are preserved.'
    )
except Exception as exc:
    report['status'] = 'POLICY_GENERATION_STOPPED_EVIDENCE_PRESERVED'
    report['errors'].append(type(exc).__name__ + ': ' + str(exc))

finally:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
        report['worker_terminated_after_stop'] = True

    if 'counts_after' not in report:
        try:
            report['counts_after'] = saved_counts()
            report['completed_answers'] = sum(report['counts_after'].values())
            report['remaining_answers'] = 1200 - report['completed_answers']
        except Exception as exc:
            report['errors'].append('Saved-answer audit: ' + str(exc))

    report['artifact_sha256'] = {
        str(p.relative_to(attempt)): digest(p)
        for p in attempt.rglob('*') if p.is_file()
    }
    core.write_once(attempt / 'REPORT.json', report)
    print('Status:', report['status'])
    print('Saved answers by policy/domain:',
          json.dumps(report.get('counts_after', {}), indent=2))
    print('Completed / remaining answers:',
          report.get('completed_answers'), '/', report.get('remaining_answers'))
    print('AI judgments in this block: 0; remaining AI comparison events: 880.')
    print('Errors:', report['errors'])
    print('Saved report:', attempt / 'REPORT.json')
    print('Report SHA-256:', digest(attempt / 'REPORT.json'))
    print('Keep this A100 runtime connected and send the printed output before continuing.')