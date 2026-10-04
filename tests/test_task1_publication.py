"""CPU-only checks for post-experiment packaging; no model imports/downloads."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from task1_dpo import ablate_beta, prepare_assets
from task1_dpo.analyze_results import pair_summary, rows

ROOT = Path(__file__).resolve().parents[1]


class PublicationTests(unittest.TestCase):
    def test_wrong_accuracy_and_reference_sign_are_rejected(self):
        row = rows(ROOT / 'results/task1/evidence/evaluation/standard/policy_results/pairs_dpo_standard_eval.jsonl')[0]
        pair_summary([row], .1)
        wrong = dict(row, correct=not row['correct'])
        with self.assertRaises(ValueError):
            pair_summary([wrong], .1)
        wrong = dict(row, reference_chosen_logp=row['reference_rejected_logp'],
                     reference_rejected_logp=row['reference_chosen_logp'])
        with self.assertRaises(ValueError):
            pair_summary([wrong], .1)

    def test_beta_dispatch_is_fixed_and_does_not_tune(self):
        args = argparse.Namespace(config='configs/dpo.yaml', models_dir=Path('/tmp/models'),
                                  backup_root=Path('/tmp/backups'), run_root=Path('/tmp/runs'), phase='train')
        for phase in ('train', 'policy', 'reward'):
            args.phase = phase
            cmds = ablate_beta.commands(args)
            self.assertEqual(len(cmds), 3)
            for name, cmd in zip(('beta_003', 'beta_010', 'beta_030'), cmds):
                self.assertIn(name, cmd)
                self.assertIn(str(args.run_root / name / ('checkpoint' if phase == 'train' else phase)), cmd)
                self.assertNotIn('--beta', cmd)
                self.assertNotIn('--max-examples', cmd)
                if phase == 'policy': self.assertIn('--adapter', cmd)
                if phase == 'reward': self.assertIn('--policy-results', cmd)

    def test_bad_existing_asset_stops_without_overwrite_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'file'; target.write_bytes(b'keep')
            request = {'destination': str(target), 'sha256': '0'*64}
            with patch.object(prepare_assets, 'requests', return_value=[request]):
                with self.assertRaises(ValueError):
                    prepare_assets.prepare(Path(tmp), 'data')
            self.assertEqual(target.read_bytes(), b'keep')
            request = {'destination': str(Path(tmp)/'missing'), 'tracked_only': True, 'sha256': '0'*64}
            with patch.object(prepare_assets, 'requests', return_value=[request]):
                with self.assertRaises(FileNotFoundError):
                    prepare_assets.prepare(Path(tmp), 'data')

    def test_cpu_cli_plans_import_no_ml_packages_and_create_no_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = '''
import sys
from pathlib import Path
from task1_dpo.prepare_assets import requests
from task1_dpo import analyze_length, ablate_beta
jobs = requests(Path(sys.argv[1]), 'all')
assert len(jobs) == 20
assert sum(j.get('tracked_only', False) for j in jobs) == 1
assert all(len(j['revision']) == 40 and len(j['sha256']) == 64 for j in jobs)
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
assert 'huggingface_hub' not in sys.modules
assert list(Path(sys.argv[1]).iterdir()) == []
'''
            subprocess.run([sys.executable, '-c', code, tmp], cwd=ROOT, check=True)


if __name__ == '__main__':
    unittest.main()
