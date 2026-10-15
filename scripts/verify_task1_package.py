"""Verify the original Task 1 manifest without altering its hashes."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MANIFEST_SHA256 = '1ea18dbe564c95ebb326b062305279c191344036730a896f78b87b5178dd63ca'
def current_path(name):
    if name == 'docs/SOURCE_ATTRIBUTION.md':
        name = 'archive/repository-maintenance-20261014-cleanup/docs/SOURCE_ATTRIBUTION.md'
    elif name == 'README.md':
        name = 'archive/task1-publication/README.md'
    elif name == 'scripts/verify_task1_package.py':
        name = 'archive/task1-publication/verify_task1_package.py'
    elif name.startswith(('task2_ppo/','task3_grpo/','task4_safety/')):
        name = 'archive/course-starter/' + name
    return json.loads((ROOT/'manifests/BASELINE_PATH_MAP.json').read_text())[name]
def main():
    raw = (ROOT/'manifests/TASK1_ORIGINAL_SHA256.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_MANIFEST_SHA256:
        raise SystemExit('Original Task 1 manifest changed.')
    manifest = json.loads(raw)
    for original, digest in manifest.items():
        path = ROOT/current_path(original)
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise SystemExit('Original Task 1 bytes differ: ' + original)
    print(f'PACKAGE_OK: {len(manifest)} original Task 1 files verified through the relocation map.')
if __name__ == '__main__':
    main()
