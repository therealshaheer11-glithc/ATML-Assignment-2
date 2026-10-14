import copy
import json
from pathlib import Path
import unittest
from prepare_a100 import child_environment, parse_hardware, validate_cpu, validate_approval, EVALUATION_SHA

ROOT = Path(__file__).resolve().parent


class Task5SetupChecks(unittest.TestCase):
    def test_one_a100_required(self):
        valid = parse_hardware('NVIDIA A100-SXM4-40GB, 40960, 580.10.1, GPU-example\n')
        self.assertEqual(valid['memory_mib'], 40960)
        for text in ['', 'NVIDIA T4, 15360, 580.10.1, GPU-example',
                     'NVIDIA A100, 40960, 580.10.1, one\nNVIDIA A100, 40960, 580.10.1, two']:
            with self.assertRaises(RuntimeError):
                parse_hardware(text)

    def test_only_child_notebook_backend_override_removed(self):
        original = {'MPLBACKEND': 'module://matplotlib_inline.backend_inline',
                    'PYTHONPATH': '/old', 'LD_LIBRARY_PATH': '/driver', 'PATH': '/bin'}
        saved = copy.deepcopy(original)
        child = child_environment(original)
        self.assertEqual(original, saved)
        self.assertNotIn('MPLBACKEND', child)
        self.assertNotIn('PYTHONPATH', child)
        self.assertEqual(child['LD_LIBRARY_PATH'], '/driver')

    def test_unknown_backend_stops(self):
        with self.assertRaises(RuntimeError):
            child_environment({'MPLBACKEND': 'unexpected_backend'})

    def test_offline_bootstrap_uses_only_supplied_wheel(self):
        env = child_environment({'PYTHONPATH': '/other'}, '/cache/pip.whl')
        self.assertEqual(env['PYTHONPATH'], '/cache/pip.whl')
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertEqual(env['TRANSFORMERS_OFFLINE'], '1')

    def test_exact_environment_approval(self):
        approval = json.loads((ROOT / 'APPROVAL.json').read_text())
        proposal_sha = '20009615ee16b746ef0e4182004926b67ff7b39ce93600e5dabcf55400ccc19d'
        pins_sha = '9f695dba56dfdbc4d3597586fe8ab6c90d38fa20a8bc3bcfccf50422bfe030ab'
        pins = {str(i): 'synthetic' for i in range(84)}
        validate_approval(approval, proposal_sha, pins_sha, pins)
        for field in ['base_package_changes_approved', 'downloads_approved', 'pretrained_model_loading_approved']:
            changed = copy.deepcopy(approval)
            changed[field] = True
            with self.assertRaises(RuntimeError):
                validate_approval(changed, proposal_sha, pins_sha, pins)

    def test_successful_combined_cpu_checkpoint_binding(self):
        report = {'status': 'CPU_READY_FOR_APPROVED_A100_SETUP', 'tests_run': 44,
                  'tests_passed': True, 'errors': [], 'verified_snapshot_files': dict.fromkeys(range(16)),
                  'wheel_validation': {'verified_wheels': dict.fromkeys(range(90))},
                  'base_package_inventory_unchanged': True, 'cpu_readiness_sha256': 'synthetic'}
        contract = {'complete': True, 'release_sha256': EVALUATION_SHA}
        validate_cpu(report, contract, 'synthetic')
        for field, value in [('tests_passed', False), ('tests_run', 43), ('errors', ['failed'])]:
            changed = copy.deepcopy(report)
            changed[field] = value
            with self.assertRaises(RuntimeError):
                validate_cpu(changed, contract, 'synthetic')

    def test_modified_cpu_contract_rejected(self):
        report = {'status': 'CPU_READY_FOR_APPROVED_A100_SETUP', 'tests_run': 44,
                  'tests_passed': True, 'errors': [], 'verified_snapshot_files': dict.fromkeys(range(16)),
                  'wheel_validation': {'verified_wheels': dict.fromkeys(range(90))},
                  'base_package_inventory_unchanged': True, 'cpu_readiness_sha256': 'synthetic'}
        with self.assertRaises(RuntimeError):
            validate_cpu(report, {'complete': True, 'release_sha256': 'wrong'}, 'synthetic')


if __name__ == '__main__':
    unittest.main()
