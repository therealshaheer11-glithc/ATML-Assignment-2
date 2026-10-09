"""Verify the unchanged original Task 1 manifest at documented current paths."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MANIFEST_SHA256 = "1ea18dbe564c95ebb326b062305279c191344036730a896f78b87b5178dd63ca"

def current_path(name):
    if name == "README.md":
        return "archive/task1-publication/README.md"
    if name == "scripts/verify_task1_package.py":
        return "archive/task1-publication/verify_task1_package.py"
    if name.startswith("task2_ppo/"):
        return "archive/course-starter/" + name
    return name

def main():
    raw = (ROOT / "PACKAGE_SHA256.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_MANIFEST_SHA256:
        raise SystemExit("Original Task 1 manifest has changed.")
    manifest = json.loads(raw)
    for original, expected in manifest.items():
        path = ROOT / current_path(original)
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"Missing or unsafe original file: {original} -> {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Original file changed: {original} -> {path}")
    print(f"PACKAGE_OK: {len(manifest)} original Task 1 files match at documented current paths.")

if __name__ == "__main__":
    main()
