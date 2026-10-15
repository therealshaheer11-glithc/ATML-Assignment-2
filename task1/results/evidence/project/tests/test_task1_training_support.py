"""CPU tests of approved run selection, durable copying, and pinned sources."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from task1_dpo import support
from common.data import load_yaml


def approved_config():
    cfg = copy.deepcopy(load_yaml("configs/dpo.yaml"))
    cfg.update({"max_sequence_length": 4096, "max_prompt_length": 4096,
                "reward_max_length": 4096, "generation_batch_size": 1,
                "base_model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                "reward_tokenizer_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                "reward_model_revision": "f98759a4a1ccdc47a3136748b6bfea9a32ba8fcc"})
    return cfg


class TrainingSupportTests(unittest.TestCase):
    def test_all_five_conditions_and_original_short_order(self):
        cfg = approved_config()
        original = [{"prompt_id": f"original-{i}"} for i in range(1500)]
        with patch.object(support, "validated_rows", return_value=original):
            for name, (_, count, beta) in support.RUNS.items():
                rows, plan = support.run_plan(cfg, name)
                self.assertEqual(rows, original[:count])
                self.assertEqual((plan["pairs"], plan["beta"]), (count, beta))
        self.assertEqual(support.RUNS["standard"], ("dpo_standard_train", 1500, .10))
        self.assertEqual(support.RUNS["length_balanced"], ("dpo_length_train", 1500, .10))

    def test_unapproved_parameters_are_rejected(self):
        for key, value in (("seed", 1), ("epochs", 2), ("batch_size", 1),
                           ("learning_rate", 1e-4), ("max_sequence_length", 768)):
            cfg = approved_config()
            cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                support.validate_config(cfg)

    def test_changed_dataset_hash_is_rejected(self):
        with patch.object(support, "file_sha", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                support.validated_rows(approved_config(), "dpo_standard_train")

    def test_verified_copy_matches_and_preserves_existing_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "adapter.safetensors").write_bytes(b"fixture-bytes")
            support.save_json(source / "record.json", {"status": "fixture"})
            destination = root / "backup"
            receipt = support.verified_copy(source, destination)
            self.assertEqual(receipt["status"], "BACKUP_VERIFIED")
            self.assertEqual(receipt["files"], 2)
            before = (destination / "adapter.safetensors").read_bytes()
            with self.assertRaises(FileExistsError):
                support.verified_copy(source, destination)
            self.assertEqual((destination / "adapter.safetensors").read_bytes(), before)

    def test_bad_copy_cannot_report_backup_verified(self):
        real_copy = support.shutil.copytree
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source", root / "backup"
            source.mkdir()
            (source / "record.json").write_text("original")
            def corrupt_copy(a, b):
                real_copy(a, b)
                (Path(b) / "record.json").write_text("changed")
            with patch.object(support.shutil, "copytree", side_effect=corrupt_copy):
                with self.assertRaises(IOError):
                    support.verified_copy(source, destination)

    def test_nested_backup_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "record.json").write_text("fixture")
            with self.assertRaises(ValueError):
                support.verified_copy(source, source / "nested_backup")

    def test_pinned_cache_requests_cannot_download_during_training(self):
        source, kwargs = support.model_source(approved_config())
        self.assertEqual(source, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(kwargs["revision"], "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
        self.assertTrue(kwargs["local_files_only"])

    def test_prepared_snapshot_integrity(self):
        cfg = approved_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root / "policy"
            policy.mkdir()
            (policy / "config.json").write_text("fixture-config")
            manifest = {"policy": {"model_id": cfg["base_model"],
                "revision": cfg["base_model_revision"], "directory": "policy",
                "sha256": {"config.json": support.file_sha(policy / "config.json")}}}
            support.save_json(root / "model_manifest.json", manifest)
            source, kwargs = support.model_source(cfg, root)
            self.assertEqual(Path(source), policy.resolve())
            self.assertTrue(kwargs["local_files_only"])
            (policy / "config.json").write_text("changed")
            with self.assertRaises(ValueError):
                support.model_source(cfg, root)

    def test_base_digest_excludes_adapters_even_when_frozen_for_reload(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Parameter(torch.tensor([1., 2.]), requires_grad=False)
                self.lora_A = torch.nn.Parameter(torch.tensor([3., 4.]))
        model = Model()
        base_before = support.parameter_digest(model)
        adapter_before = support.parameter_digest(model, adapters=True)
        model.requires_grad_(False)
        model.lora_A.add_(1.)
        self.assertEqual(support.parameter_digest(model), base_before)
        self.assertNotEqual(support.parameter_digest(model, adapters=True), adapter_before)


if __name__ == "__main__":
    unittest.main()
