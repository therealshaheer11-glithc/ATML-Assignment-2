"""Narrow isolated-environment identities and offline installation commands."""
import os
from pathlib import Path

from packaging.utils import canonicalize_name


def require(value, message):
    if not value:
        raise RuntimeError(message)


def clean_environment(original, bootstrap_wheel=None):
    # Preserve driver/OS settings; remove inherited Python/pip install targets.
    result = {key: value for key, value in original.items()
              if not key.startswith(('PYTHON', 'PIP_')) and key != 'VIRTUAL_ENV'}
    result.update({'PIP_CONFIG_FILE': os.devnull, 'PYTHONNOUSERSITE': '1',
                   'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1'})
    if bootstrap_wheel is not None:
        result['PYTHONPATH'] = str(bootstrap_wheel)
    return result


def requirement_lines(files, pins, wheel_directory):
    indexed = {}
    for record in files.values():
        key = (canonicalize_name(record['package']), record['version'])
        require(key not in indexed, 'Duplicate binary package/version')
        indexed[key] = record
    result = []
    for name, version in sorted(pins.items()):
        require((name, version) in indexed, 'Approved wheel missing: ' + name)
        record = indexed[(name, version)]
        filename = record['filename']
        require(Path(filename).name == filename and '\\' not in filename,
                'Unsafe wheel requirement filename')
        result.append(name + ' @ ' + (Path(wheel_directory) / filename).absolute().as_uri() +
                      ' --hash=sha256:' + record['sha256'])
    return '\n'.join(result) + '\n'


def install_command(python, requirements, report_path, bootstrap=False):
    flags = ['-P', '-s', '-B'] if bootstrap else ['-I', '-B']
    result = [str(python), *flags, '-m', 'pip', '--disable-pip-version-check',
              '--no-cache-dir', 'install', '--no-index', '--no-deps',
              '--only-binary=:all:', '--require-hashes', '--report', str(report_path)]
    if bootstrap:
        # The wheel on PYTHONPATH is the installer, not the installed package.
        result.append('--ignore-installed')
    return [*result, '-r', str(requirements)]


def validate_inventory(value, pins, environment, complete=False):
    environment = str(Path(environment).resolve())
    require(value['prefix'] == environment and value['prefix'] != value['base_prefix'] and
            value['user_site_enabled'] is False and value['system_site_packages'] is False,
            'Python environment is not fully isolated')
    require(not value['duplicate_distributions'] and not value['outside_prefix_distributions'] and
            not value['outside_prefix_site_paths'], 'Distribution/site path escaped the isolated environment')
    actual = value['package_versions']
    require(set(actual) <= set(pins), 'Unapproved package appeared in the isolated environment')
    for name, version in actual.items():
        require(version == pins[name], 'Isolated installed version differs: ' + name)
    require(not complete or actual == pins, 'Isolated package inventory is incomplete')
    return actual


def disk_requirement(logical_unpacked_bytes, free_bytes):
    require(type(logical_unpacked_bytes) is int and logical_unpacked_bytes > 0,
            'Missing measured wheel sizes')
    # Operational headroom for installed files, compilation and pip temporary
    # unpacking. It does not assert measured installation disk usage.
    threshold = 2 * logical_unpacked_bytes
    require(free_bytes >= threshold, 'Insufficient local runtime space for conservative installation headroom')
    return {'free_bytes_before': free_bytes, 'logical_unpacked_bytes': logical_unpacked_bytes,
            'required_free_bytes': threshold, 'rule': 'Twice measured logical unpacked bytes; installed disk usage remains unmeasured'}
