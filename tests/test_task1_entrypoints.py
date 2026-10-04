"""CPU checks that entry points respect plans and reject invalid execution."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from common.data import load_yaml
from task1_dpo import support, train, verify_checkpoint


class EntryPointTests(unittest.TestCase):
    def test_all_cli_plans_run_without_model_loading(self):
        cfg = copy.deepcopy(load_yaml("configs/dpo.yaml"))
        cfg.update({"max_sequence_length": 4096, "max_prompt_length": 4096,
                    "reward_max_length": 4096, "generation_batch_size": 1,
                    "base_model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                    "reward_tokenizer_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                    "reward_model_revision": "f98759a4a1ccdc47a3136748b6bfea9a32ba8fcc"})
        rows = [{"prompt_id": str(i)} for i in range(1500)]
        for name, (_, count, beta) in support.RUNS.items():
            with self.subTest(name=name), patch.object(train, "load_yaml", return_value=cfg), \
                 patch.object(support, "validated_rows", return_value=rows), \
                 patch.object(train, "load_pinned_policy", side_effect=AssertionError("Weights loaded during plan")), \
                 patch.object(train, "load_pinned_tokenizer", side_effect=AssertionError("Tokenizer loaded during plan")), \
                 patch.object(sys, "argv", ["train", "--run-name", name, "--plan"]):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    train.main()
                plan = json.loads(buffer.getvalue())
                self.assertEqual((plan["pairs"], plan["beta"]), (count, beta))
                self.assertEqual(sum(plan["update_pair_counts"]), count)
                self.assertEqual(plan["optimizer_updates"], 38 if count == 600 else 94)

    def test_unapproved_cli_overrides_are_rejected(self):
        cfg = {"paths": {"dpo_standard_train": "data/dpo_standard_train.jsonl"}}
        plan = {"path_key": "dpo_standard_train", "beta": .1, "pairs": 1500}
        for dataset, beta, count in (("data/other.jsonl", None, None),
                                     (None, .3, None), (None, None, 600)):
            with self.subTest(dataset=dataset, beta=beta, count=count), self.assertRaises(ValueError):
                train.validate_overrides(cfg, plan, dataset, beta, count)

    def test_cpu_training_stops_before_creating_output_or_loading_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not_created"
            with patch.object(train, "load_yaml", return_value={}), \
                 patch.object(train, "run_plan", return_value=([], {})), \
                 patch.object(train.torch.cuda, "is_available", return_value=False), \
                 patch.object(train, "load_pinned_policy", side_effect=AssertionError("Unexpected model load")):
                with self.assertRaisesRegex(RuntimeError, "--plan on CPU"):
                    train.run_training("fixture", "standard", output_path=str(output))
            self.assertFalse(output.exists())

    def test_failed_checkpoint_cannot_pass_reload_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "run_record.json").write_text(json.dumps({"status": "FAILED"}))
            with patch.object(sys, "argv", ["verify", "--checkpoint", str(folder)]), \
                 patch.object(verify_checkpoint, "load_pinned_policy", side_effect=AssertionError("Unexpected model load")):
                with self.assertRaisesRegex(ValueError, "completed training budget"):
                    verify_checkpoint.main()
            result = json.loads((folder / "reload_verification.json").read_text())
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["held_out_examples_used"])


if __name__ == "__main__":
    unittest.main()
