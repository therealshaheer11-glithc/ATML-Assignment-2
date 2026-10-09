"""Check the combined publication, optionally stage exactly its approved files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from verify_task1_package import current_path

ROOT = Path(__file__).resolve().parents[1]

def read_manifest(name):
    return json.loads((ROOT / name).read_text())

def verify_manifest(name):
    manifest = read_manifest(name)
    for rel, digest in manifest.items():
        path = ROOT / rel
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Checksum mismatch or missing file: {rel}")
    print(f"VERIFIED: {len(manifest)} files in {name}", flush=True)
    return set(manifest) | {name}

def git(*args, capture=False):
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          stdout=subprocess.PIPE if capture else None).stdout

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stage", action="store_true")
    group.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    expected = verify_manifest("REPOSITORY_MAINTENANCE_SHA256.json")
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_task1_package.py")], check=True)
    original = read_manifest("PACKAGE_SHA256.json")
    expected |= {current_path(n) for n in original} | {"PACKAGE_SHA256.json"}
    expected |= verify_manifest("TASK1_FIGURES_SHA256.json")
    subprocess.run([sys.executable, str(ROOT / "task2/verify_package.py")], check=True)
    expected |= {"task2/" + n for n in read_manifest("task2/PACKAGE_SHA256.json")}
    expected.add("task2/PACKAGE_SHA256.json")
    if args.stage or args.staged:
        tracked = set(git("ls-files", "-z", capture=True).decode().split("\0")) - {""}
        allowed = expected | set(original)
        if tracked - allowed:
            raise SystemExit("Unexpected tracked files; review before publishing: " + repr(sorted(tracked - allowed)))
        if args.stage:
            removed = set(original) - {current_path(n) for n in original}
            paths = expected | (removed & tracked)
            git("add", "-f", "-A", "--", *sorted(paths))
        indexed = set(git("ls-files", "-z", capture=True).decode().split("\0")) - {""}
        if indexed != expected:
            raise SystemExit("Git index inventory mismatch: " + repr(sorted(indexed ^ expected)))
        git("diff", "--quiet", "--exit-code")
        print(f"STAGED VERIFIED: all {len(expected)} expected files are indexed and match verified working files.")
    print("REPOSITORY VERIFIED. No GPU used.")

if __name__ == "__main__":
    main()
