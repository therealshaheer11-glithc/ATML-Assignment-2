"""Extract preserved source releases into a new directory; never execute them."""
import argparse
from pathlib import Path, PurePosixPath
import zipfile
from verify_package import ROOT, require, verify_integrity


def restore(destination):
    verify_integrity()
    destination = Path(destination).absolute()
    require(not destination.exists() and not destination.is_symlink(), 'Choose a fresh destination')
    require(not destination.resolve().is_relative_to(ROOT), 'Restore outside frozen task4 package')
    payloads = []
    for archive in sorted((ROOT / 'provenance/releases').glob('*.zip')):
        with zipfile.ZipFile(archive) as z:
            seen = set()
            for entry in z.infolist():
                p = PurePosixPath(entry.filename)
                require(not p.is_absolute() and '..' not in p.parts and '\\' not in entry.filename,
                        'Unsafe release member')
                require((entry.external_attr >> 16) & 0o170000 != 0o120000, 'Linked release member')
                require(entry.filename not in seen, 'Duplicate release member')
                seen.add(entry.filename)
                if not entry.is_dir():
                    payloads.append((Path(archive.stem) / p, z.read(entry)))
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in payloads:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    print(f'RESTORED: {len(payloads)} original files from 19 releases into {destination}. Nothing executed.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--destination', required=True)
    restore(p.parse_args().destination)
