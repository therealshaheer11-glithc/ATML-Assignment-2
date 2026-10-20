"""Verify the organized publication and preserved historical evidence."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from wording_integrity import matches_recorded_file
ROOT = Path(__file__).resolve().parents[1]
INDEX = 'manifests/REPOSITORY_SHA256.json'
def read(name):
    return json.loads((ROOT/name).read_text())
def check(name, expected):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p)!=name or not p.parts:
        raise SystemExit('Unsafe manifest path: ' + name)
    path = ROOT/name
    if not matches_recorded_file(path, expected):
        raise SystemExit('Missing or changed file: ' + name)
def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group();g.add_argument('--stage',action='store_true');g.add_argument('--staged',action='store_true')
    args=p.parse_args()
    current=read(INDEX)
    for name,digest in current.items():check(name,digest)
    expected=set(current)|{INDEX}
    baseline=read('manifests/BASELINE_SHA256.json');mapping=read('manifests/BASELINE_PATH_MAP.json')
    if set(baseline)!=set(mapping) or len(set(mapping.values()))!=len(mapping):
        raise SystemExit('Invalid baseline relocation coverage.')
    for name,digest in baseline.items():check(mapping[name],digest)
    figures=read('manifests/TASK1_FIGURES_ORIGINAL_SHA256.json')
    for name,digest in figures.items():check(mapping[name],digest)
    print(f'CURRENT PUBLICATION VERIFIED: {len(expected)} files; all {len(baseline)} baseline entries checked against original seals or exact indexed wording revisions.',flush=True)
    for command in [('scripts/verify_task1_package.py',),('task2/verify_package.py',),('task3/verify_package.py',),('task4/verify_package.py',),('task5/verify_package.py',)]:
        subprocess.run([sys.executable,'-B',*command],cwd=ROOT,check=True)
    if args.stage:
        indexed=set(git('ls-files','-z').decode().split('\0'))-{''}
        if indexed-set(baseline)-expected:raise SystemExit('Unexpected tracked files.')
        removed=(set(baseline)-expected)&indexed
        git('add','-f','-A','--',*sorted(expected|removed))
    if args.stage or args.staged:
        indexed=set(git('ls-files','-z').decode().split('\0'))-{''}
        if indexed!=expected:raise SystemExit('Index inventory differs: '+repr(sorted(indexed^expected)))
        git('diff','--quiet','--exit-code')
        print(f'STAGED VERIFIED: {len(expected)} files.')
    print('REPOSITORY VERIFIED. No GPU used.')
if __name__=='__main__':main()
