import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from task3_grpo.storage import atomic_json, commit_directory, publish_directory, restore_published, verify_directory, safe_extract, repair_missing_receipt
from task3_grpo.session_budget import SessionBudget, SessionBudgetReached


class StorageTests(unittest.TestCase):
    def test_commit_publish_restore_and_republish(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'tmp').mkdir(); atomic_json(p/'tmp/value.json',{'a':1})
            commit_directory(p/'tmp',p/'step_0001')
            receipt = publish_directory(p/'step_0001',p/'drive')
            restore_published(p/'drive/step_0001.zip',p/'restored')
            self.assertEqual(verify_directory(p/'restored'),verify_directory(p/'step_0001'))
            self.assertEqual(receipt,publish_directory(p/'step_0001',p/'drive'))
    def test_local_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'tmp').mkdir(); (p/'tmp/value').write_text('original')
            commit_directory(p/'tmp',p/'unit'); (p/'unit/value').write_text('changed')
            with self.assertRaises(RuntimeError): verify_directory(p/'unit')
    def test_persistent_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'tmp').mkdir(); (p/'tmp/value').write_text('a')
            commit_directory(p/'tmp',p/'unit'); publish_directory(p/'unit',p/'drive')
            with (p/'drive/unit.zip').open('ab') as f: f.write(b'changed')
            with self.assertRaises(RuntimeError): restore_published(p/'drive/unit.zip',p/'restored')
    def test_missing_receipt_can_be_repaired_from_local(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'tmp').mkdir(); (p/'tmp/value').write_text('a')
            commit_directory(p/'tmp',p/'unit'); receipt = publish_directory(p/'unit',p/'drive')
            (p/'drive/unit.sha256.json').unlink()
            self.assertEqual(receipt,publish_directory(p/'unit',p/'drive'))
    def test_uncommitted_partial_not_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); atomic_json(p/'partial/value.json',{'a':1})
            with self.assertRaises(FileNotFoundError): verify_directory(p/'partial')
    def test_orphan_receipt_repair_on_new_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'tmp').mkdir(); (p/'tmp/value').write_text('a')
            commit_directory(p/'tmp',p/'unit'); publish_directory(p/'unit',p/'drive')
            (p/'drive/unit.sha256.json').unlink()
            repair_missing_receipt(p/'drive/unit.zip')
            restore_published(p/'drive/unit.zip',p/'restored')
            self.assertEqual((p/'restored/value').read_text(),'a')
    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            with zipfile.ZipFile(p/'bad.zip','w') as z: z.writestr('../escape','x')
            with zipfile.ZipFile(p/'bad.zip') as z:
                with self.assertRaises(RuntimeError): safe_extract(z,p/'out')
    def test_symlink_archive_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); info = zipfile.ZipInfo('link'); info.external_attr = 0o120777<<16
            with zipfile.ZipFile(p/'bad.zip','w') as z: z.writestr(info,'target')
            with zipfile.ZipFile(p/'bad.zip') as z:
                with self.assertRaises(RuntimeError): safe_extract(z,p/'out')
    def test_existing_commit_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'a').mkdir(); (p/'b').mkdir()
            with self.assertRaises(FileExistsError): commit_directory(p/'a',p/'b')
    def test_expired_session_stops(self):
        with patch('task3_grpo.session_budget.time.time',return_value=1000.):
            budget = SessionBudget(1,session_start_epoch=0.)
        with self.assertRaises(SessionBudgetReached): budget.check()
    def test_invalid_budget_rejected(self):
        for m in (0,-1,float('nan')):
            with self.assertRaises(ValueError): SessionBudget(m)
