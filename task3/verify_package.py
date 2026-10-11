"""Verify Task 3 publication and saved evidence using only the standard library."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE_SHA = 'f87021b8ee1057d2afbae29d12c5bd0e904d10b1e9888d9476f99f2959a05c07'
RELEASE_SHA = 'd3c55c4d0202d09d36c2a93dc7f3ccbc48a0c65690939ff7a429a585aeb9bbd5'


def require(ok, message):
    if not ok:
        raise SystemExit(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    require(__debug__, 'Run without -O: evidence audit assertions must be enabled.')
    manifest = json.loads((ROOT / 'PACKAGE_SHA256.json').read_text())
    for name, expected in manifest.items():
        p = ROOT / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe manifest path')
        require(not p.is_symlink() and p.is_file() and sha(p.read_bytes()) == expected,
                'Task 3 missing/changed file: ' + name)
    source = ROOT / 'source'
    require(sha((source / 'docs/SOURCE_SHA256.json').read_bytes()) == SOURCE_SHA, 'Executed source seal changed')
    seal = json.loads((source / 'docs/SOURCE_SHA256.json').read_text())
    expected_source = set(seal) | {'docs/SOURCE_SHA256.json'}
    actual_source = {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    require(actual_source == expected_source, 'Unexpected executed-source inventory')
    release = ROOT / 'provenance/releases/ATML_PA2_Task3_Colab.zip'
    require(sha(release.read_bytes()) == RELEASE_SHA, 'Original Colab release changed')
    with zipfile.ZipFile(release) as z:
        require(z.testzip() is None, 'Original release ZIP CRC failure')
        original = json.loads(z.read('PACKAGE_SHA256.json'))
        require(len(z.namelist()) == len(set(z.namelist())) and
                set(z.namelist()) == set(original) | {'PACKAGE_SHA256.json'}, 'Original release inventory differs')
        for name, expected in original.items():
            require(sha(z.read(name)) == expected, 'Original release payload differs: ' + name)
        for name in expected_source:
            require((source / name).read_bytes() == z.read('task3/source/' + name), 'Executed source differs from release')
        for p in (ROOT / 'notebooks').glob('*.ipynb'):
            require(p.read_bytes() == z.read('task3/notebooks/' + p.name), 'Historical notebook changed')
    with tempfile.TemporaryDirectory(prefix='task3-evidence-') as out:
        subprocess.run([sys.executable, str(ROOT / 'analysis/audit_saved_export.py'),
                        str(ROOT / 'results/actual_run_20261011'), '--output', out], check=True)
    print(f'TASK 3 VERIFIED: {len(manifest)} publication files; original source/release preserved. No GPU used.')


if __name__ == '__main__':
    main()
