"""Check exact historical seals alongside explicitly recorded wording revisions."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = json.loads((ROOT / 'manifests/WORDING_REVISIONS.json').read_text())['files']


def matches_recorded_file(path, expected_sha256, expected_bytes=None):
    """Accept original bytes or the one exact, path-bound published revision."""
    path = Path(path)
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        return False
    try:
        name = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return False
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest == expected_sha256:
        return expected_bytes is None or len(data) == expected_bytes
    revision = REVISIONS.get(name)
    return bool(revision
                and revision['before_sha256'] == expected_sha256
                and revision['after_sha256'] == digest
                and revision['after_bytes'] == len(data)
                and (expected_bytes is None or revision['before_bytes'] == expected_bytes))
