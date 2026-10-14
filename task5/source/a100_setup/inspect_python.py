"""Stdlib-only fresh interpreter inventory; works before pip is installed."""
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import site
import sys
import sysconfig


def main():
    prefix = Path(sys.prefix).resolve()
    versions, duplicates, outside = {}, [], []
    for item in metadata.distributions():
        if not item.metadata.get('Name'):
            continue
        name = re.sub(r'[-_.]+', '-', item.metadata['Name']).lower()
        if name in versions:
            duplicates.append(name)
        versions[name] = item.version
        location = Path(item.locate_file('')).resolve()
        if not location.is_relative_to(prefix):
            outside.append({'name': name, 'location': str(location)})
    config = prefix / 'pyvenv.cfg'
    configuration = {}
    if config.exists():
        for line in config.read_text().splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                configuration[key.strip().lower()] = value.strip()
    outside_paths = [value for value in sys.path if value and
                     ('site-packages' in Path(value).parts or 'dist-packages' in Path(value).parts) and
                     not Path(value).resolve().is_relative_to(prefix)]
    result = {'format': 'ATML_TASK4_PYTHON_ISOLATION_INVENTORY_V1',
              'python': platform.python_version(), 'executable': sys.executable,
              'prefix': str(prefix), 'base_prefix': str(Path(sys.base_prefix).resolve()),
              'user_site_enabled': site.ENABLE_USER_SITE,
              'system_site_packages': configuration.get('include-system-site-packages', '').lower() != 'false',
              'purelib': sysconfig.get_path('purelib'), 'platlib': sysconfig.get_path('platlib'),
              'package_versions': dict(sorted(versions.items())),
              'duplicate_distributions': duplicates, 'outside_prefix_distributions': outside,
              'outside_prefix_site_paths': outside_paths, 'sys_path': sys.path,
              'pyvenv_configuration': configuration}
    with Path(sys.argv[1]).open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print('Isolated interpreter inventory:', len(versions), 'distributions;', platform.python_version(), flush=True)


if __name__ == '__main__':
    main()
