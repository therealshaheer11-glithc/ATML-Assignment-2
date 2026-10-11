"""Reconstruct the isolated runtime from the sealed source, without touching prior tasks."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def restore(destination):
    source = Path(__file__).resolve().parent/'source'
    manifest_path = source/'docs/SOURCE_SHA256.json'
    manifest = json.loads(manifest_path.read_text())
    for name,h in manifest.items():
        p = source/name
        if p.is_symlink() or not p.is_file() or digest(p)!=h:
            raise RuntimeError('Packaged source differs: '+name)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    for name in [*manifest,'docs/SOURCE_SHA256.json']:
        old,new = source/name,destination/name
        if new.exists():
            if new.is_symlink() or digest(new)!=digest(old):
                raise RuntimeError('Existing runtime source differs; use a fresh runtime: '+str(new))
        else:
            new.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(old,new)
    print('Sealed Task 3 source verified and reconstructed at '+str(destination))
    return destination


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--destination',required=True)
    restore(ap.parse_args().destination)
