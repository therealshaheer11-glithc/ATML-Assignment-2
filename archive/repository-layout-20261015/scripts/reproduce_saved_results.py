"""Reproduce all five tasks' saved statistics on CPU, without model inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', action='store_true', help='Print commands without writing or executing.')
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error('Choose a new output directory; existing results are never overwritten.')
    if output.resolve().is_relative_to(ROOT) and not output.resolve().is_relative_to(ROOT / 'outputs'):
        parser.error('Inside the repository, write only under ignored outputs/.')
    py = [sys.executable, '-B']
    jobs = [
        ('integrity', py + ['scripts/verify_repository.py']),
        ('task1', py + ['-m', 'task1_dpo.analyze_results', '--output', str(output / 'task1')]),
        ('task2', py + ['task2/analysis/audit_export.py', '--original', 'task2/evidence/original', '--output', str(output / 'task2')]),
        ('task3', py + ['task3/analysis/present_saved_results.py', '--output', str(output / 'task3')]),
        ('task4', py + ['task4/reproduce.py', '--output', str(output / 'task4')]),
        ('task5', py + ['task5/verify_package.py', '--output', str(output / 'task5')]),
    ]
    if args.plan:
        print(json.dumps({'cwd': str(ROOT), 'jobs': jobs, 'model_calls': 0}, indent=2))
        return
    if not __debug__:
        parser.error('Run without -O; historical audits use assertions.')
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    receipt = {'status': 'RUNNING', 'model_calls': 0, 'jobs': [],
               'scope': 'Saved-result reproduction. Task 5 uses published score records; full exact rescoring needs its private bundle.'}
    try:
        for name, command in jobs:
            print('Checking ' + name, flush=True)
            with (output / (name + '.log')).open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt['jobs'].append({'name': name, 'command': command, 'returncode': result.returncode})
            if result.returncode:
                raise RuntimeError('Check failed: ' + name + '; inspect its saved log.')
        receipt['status'] = 'PASSED'
    except BaseException as error:
        receipt['status'] = 'FAILED'
        receipt['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        receipt['files_sha256'] = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(output.rglob('*')) if p.is_file()}
        (output / 'RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('ALL FIVE SAVED-RESULT CHECKS PASSED: ' + str(output))


if __name__ == '__main__':
    main()
