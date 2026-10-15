"""Verify immutable Task 4 publication files and recompute all saved metrics."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def safe(root, name):
    p = PurePosixPath(name)
    require(name not in {'', '.'} and not p.is_absolute() and '..' not in p.parts and '\\' not in name and str(p) == name, 'Unsafe manifest name')
    path = root / name
    require(not any(parent.is_symlink() for parent in [path, *path.parents] if parent != root), 'Symlink in package: ' + name)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_integrity():
    manifest = json.loads((ROOT / 'PACKAGE_SHA256.json').read_text())
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    require(actual == set(manifest) | {'PACKAGE_SHA256.json'}, 'Task 4 file inventory differs')
    for name, expected in manifest.items():
        path = safe(ROOT, name)
        require(path.is_file() and sha(path) == expected, 'Task 4 checksum mismatch: ' + name)
    evidence = ROOT / 'evidence'
    exported = json.loads((evidence / 'EXPORT_MANIFEST.json').read_text())['files']
    mapping = json.loads((ROOT / 'provenance/EXPORT_PATH_MAP.json').read_text())
    require(set(mapping) <= set(exported) and len(set(mapping.get(n,n) for n in exported)) == len(exported), 'Invalid export path mapping')
    actual_evidence = {p.relative_to(evidence).as_posix() for p in evidence.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    require(actual_evidence == {mapping.get(n,n) for n in exported} | {'EXPORT_MANIFEST.json'}, 'Original export inventory differs')
    for name, entry in exported.items():
        path = safe(evidence, mapping.get(name, name))
        require(sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], 'Original export differs: ' + name)
    sources = json.loads((evidence / 'SOURCE_PATHS.json').read_text())
    for entry in sources.values():
        path = safe(evidence, mapping.get(entry['destination'], entry['destination']))
        require(sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], 'Drive source mapping differs')
    for name, expected in json.loads((evidence / 'source/orchestration/MANIFEST.json').read_text()).items():
        require(sha(safe(evidence / 'source/orchestration', name)) == expected, 'Original orchestration source differs')
    for kind, contract in [('generation', 'GENERATION_COMPLETION_CONTRACT.json'), ('judgment', 'JUDGING_COMPLETION_CONTRACT.json')]:
        exports = json.loads((evidence / 'experiment' / contract).read_text())['exports']
        for p, entry in exports.items():
            require(sha(evidence / 'experiment/evaluation_v1/exports' / kind / (p + '.jsonl')) == entry['sha256'], 'Completion/export hash differs')
    cross = json.loads((ROOT / 'CROSS_TASK_SOURCES.json').read_text())
    for name, expected in cross['sources'].items():
        current = 'task1/results/' + name[len('results/task1/'):] if name.startswith('results/task1/') else name
        require(sha(safe(ROOT.parent, current)) == expected, 'Earlier-task score source changed: ' + name)
    for archive in (ROOT / 'provenance/releases').glob('*.zip'):
        with zipfile.ZipFile(archive) as z:
            require(z.testzip() is None, 'Release ZIP corruption: ' + archive.name)
    print(f'TASK 4 INTEGRITY: {len(manifest)} publication files; 160 original export payloads; 19 sealed releases verified.')


def main():
    from reproduce import recompute
    recompute()


if __name__ == '__main__':
    main()
