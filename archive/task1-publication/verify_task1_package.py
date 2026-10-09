"""Verify every shipped publication file; no packages, network or GPU required."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'PACKAGE_SHA256.json').read_text())
    for name, expected in manifest.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing package file: {name}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Package file changed: {name}')
    print(f'PACKAGE_OK: {len(manifest)} shipped files match; additional local outputs are not checked.')


if __name__ == '__main__':
    main()
