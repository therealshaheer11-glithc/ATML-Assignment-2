"""No real package installs: identity, isolation, offline and disk guards."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv
import zipfile

from guards import (clean_environment, disk_requirement, install_command,
                    requirement_lines, validate_inventory)

ROOT = Path(__file__).resolve().parent


class IsolationGuards(unittest.TestCase):
    def inventory(self):
        return {'prefix': '/task4_example', 'base_prefix': '/usr', 'user_site_enabled': False,
                'system_site_packages': False, 'duplicate_distributions': [],
                'outside_prefix_distributions': [], 'outside_prefix_site_paths': [],
                'package_versions': {'pip': '24.1.2'}}

    def test_partial_exact_inventory_allowed_before_install(self):
        self.assertEqual(validate_inventory(self.inventory(), {'pip': '24.1.2', 'demo': '1.0'}, '/task4_example'),
                         {'pip': '24.1.2'})

    def test_complete_inventory_required_after_install(self):
        with self.assertRaises(RuntimeError):
            validate_inventory(self.inventory(), {'pip': '24.1.2', 'demo': '1.0'}, '/task4_example', complete=True)

    def test_wrong_versions_and_extra_packages_rejected(self):
        for actual in [{'pip': '25'}, {'pip': '24.1.2', 'extra': '1'}]:
            value = self.inventory(); value['package_versions'] = actual
            with self.assertRaises(RuntimeError):
                validate_inventory(value, {'pip': '24.1.2'}, '/task4_example')

    def test_base_prefix_user_site_and_system_site_rejected(self):
        for changes in [{'base_prefix': '/task4_example'}, {'user_site_enabled': True},
                        {'system_site_packages': True}, {'prefix': '/other'}]:
            with self.assertRaises(RuntimeError):
                validate_inventory({**self.inventory(), **changes}, {'pip': '24.1.2'}, '/task4_example')

    def test_outside_distributions_paths_and_duplicates_rejected(self):
        for key in ['outside_prefix_distributions', 'outside_prefix_site_paths', 'duplicate_distributions']:
            value = self.inventory(); value[key] = ['unexpected']
            with self.assertRaises(RuntimeError):
                validate_inventory(value, {'pip': '24.1.2'}, '/task4_example')

    def test_inherited_install_targets_removed_driver_preserved(self):
        environment = clean_environment({'PYTHONPATH': '/global', 'PYTHONHOME': '/global',
            'PIP_TARGET': '/global', 'PIP_INDEX_URL': 'https://other', 'VIRTUAL_ENV': '/other',
            'LD_LIBRARY_PATH': '/driver', 'PATH': '/bin', 'HOME': '/home/example'})
        self.assertNotIn('PIP_TARGET', environment)
        self.assertNotIn('PYTHONPATH', environment)
        self.assertNotIn('PYTHONHOME', environment)
        self.assertNotIn('VIRTUAL_ENV', environment)
        self.assertEqual(environment['LD_LIBRARY_PATH'], '/driver')
        self.assertEqual(environment['HOME'], '/home/example')
        self.assertEqual(environment['PIP_CONFIG_FILE'], os.devnull)

    def test_bootstrap_only_adds_exact_pip_wheel_pythonpath(self):
        environment = clean_environment({'PYTHONPATH': '/other'}, '/approved/pip.whl')
        self.assertEqual(environment['PYTHONPATH'], '/approved/pip.whl')
        self.assertEqual(environment['PYTHONNOUSERSITE'], '1')

    def test_installer_commands_are_local_hashed_and_dependency_closed(self):
        command = install_command('/venv/bin/python', '/approved/req.txt', '/report.json')
        self.assertEqual(command[:4], ['/venv/bin/python', '-I', '-B', '-m'])
        for option in ['--no-index', '--no-deps', '--only-binary=:all:', '--require-hashes', '--no-cache-dir']:
            self.assertIn(option, command)
        self.assertNotIn('--ignore-installed', command)
        bootstrap = install_command('/venv/bin/python', '/pip.txt', '/report.json', bootstrap=True)
        self.assertIn('--ignore-installed', bootstrap)
        self.assertIn('-P', bootstrap)
        self.assertIn('-s', bootstrap)

    def test_hash_requirements_use_exact_local_artifacts(self):
        records = {'demo': {'package': 'demo', 'version': '1.0',
                           'filename': 'demo-1.0-py3-none-any.whl', 'sha256': 'a' * 64}}
        lines = requirement_lines(records, {'demo': '1.0'}, '/approved/cache')
        self.assertEqual(lines, 'demo @ file:///approved/cache/demo-1.0-py3-none-any.whl --hash=sha256:' + 'a' * 64 + '\n')
        with self.assertRaises(RuntimeError):
            requirement_lines(records, {'demo': '2.0'}, '/approved/cache')

    def test_wrong_wheel_path_and_duplicate_package_rejected(self):
        record = {'package': 'demo', 'version': '1.0', 'filename': '../demo.whl', 'sha256': 'a' * 64}
        with self.assertRaises(RuntimeError):
            requirement_lines({'one': record}, {'demo': '1.0'}, '/approved')
        with self.assertRaises(RuntimeError):
            requirement_lines({'one': record, 'two': copy.deepcopy(record)}, {'demo': '1.0'}, '/approved')

    def test_disk_headroom_uses_measured_sizes_and_stops_if_insufficient(self):
        self.assertEqual(disk_requirement(100, 200)['required_free_bytes'], 200)
        for logical, free in [(100, 199), (0, 1000)]:
            with self.assertRaises(RuntimeError):
                disk_requirement(logical, free)

    def test_real_empty_venv_isolation_probe_without_package_install(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'isolated'
            output = Path(directory) / 'inventory.json'
            venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(folder)
            subprocess.run([folder / 'bin/python', '-I', '-B', ROOT / 'inspect_python.py', output],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           env=clean_environment(os.environ))
            result = json.loads(output.read_text())
            self.assertEqual(validate_inventory(result, {}, folder, complete=True), {})

    def test_real_bootstrap_zip_import_without_package_install(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'isolated'
            wheel = Path(directory) / 'fixture_pip.whl'
            venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(folder)
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('pip/__init__.py', '')
                archive.writestr('pip/__main__.py', 'import json,site,sys\nprint(json.dumps({"file":__file__,"user_site":site.ENABLE_USER_SITE,"prefix":sys.prefix,"argv":sys.argv[1:]}))\n')
            result = subprocess.run(install_command(folder / 'bin/python', '/unused/requirements', '/unused/report', bootstrap=True),
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=clean_environment(os.environ, wheel))
            observed = json.loads(result.stdout)
            self.assertTrue(observed['file'].startswith(str(wheel)))
            self.assertFalse(observed['user_site'])
            self.assertEqual(Path(observed['prefix']).resolve(), folder.resolve())
            self.assertIn('--ignore-installed', observed['argv'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
