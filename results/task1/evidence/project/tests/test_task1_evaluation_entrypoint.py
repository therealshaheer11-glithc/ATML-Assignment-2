"""CPU entrypoint safeguards; all fixtures are synthetic and load no weights."""
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
from task1_dpo import evaluate
from task1_dpo.runtime import update_pair_counts
from task1_dpo.support import RUNS, file_sha


def approved_cfg():
    cfg = copy.deepcopy(load_yaml("configs/dpo.yaml"))
    cfg.update({"max_sequence_length": 4096, "max_prompt_length": 4096,
        "reward_max_length": 4096, "generation_batch_size": 1,
        "base_model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "reward_tokenizer_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "reward_model_revision": "f98759a4a1ccdc47a3136748b6bfea9a32ba8fcc"})
    return cfg


class EvaluationEntryPointTests(unittest.TestCase):
    def test_all_six_cli_plans_use_full_pools_without_loading_models(self):
        cfg = approved_cfg()
        def rows(_, key):
            return [{"prompt_id": str(i)} for i in range(evaluate.DATA[key][0])]
        for name in (*RUNS, "sft"):
            with self.subTest(name=name), patch.object(evaluate, "load_yaml", return_value=cfg), \
                 patch.object(evaluate, "validated_rows", side_effect=rows), \
                 patch.object(evaluate, "load_pinned_policy", side_effect=AssertionError("Weights during plan")), \
                 patch.object(evaluate, "load_pinned_reward", side_effect=AssertionError("Weights during plan")), \
                 patch.object(evaluate, "load_pinned_tokenizer", side_effect=AssertionError("Tokenizer during plan")), \
                 patch.object(sys, "argv", ["evaluate", "--name", name, "--plan"]):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    evaluate.main()
                plan = json.loads(buffer.getvalue())
                self.assertEqual(plan["datasets"]["dpo_standard_eval"]["rows"], 300)
                self.assertEqual(plan["datasets"]["dpo_standard_eval"]["ordered_prompt_ids"], [str(i) for i in range(300)])
                if name in {"standard", "length_balanced", "sft"}:
                    self.assertEqual(plan["datasets"]["word_limit_prompts"]["rows"], 10)
                if name in {"standard", "length_balanced"}:
                    self.assertEqual(plan["datasets"]["dpo_length_eval"]["rows"], 246)
                if name == "sft":
                    self.assertEqual(plan["pair_pools"], [])

    def test_cpu_execution_stops_before_output_or_weight_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not_created"
            for phase in ("policy", "reward"):
                with self.subTest(phase=phase), patch.object(evaluate, "load_yaml", return_value={}), \
                     patch.object(evaluate, "fixed_pools", return_value=({}, {})), \
                     patch.object(evaluate.torch.cuda, "is_available", return_value=False), \
                     patch.object(evaluate, "load_pinned_policy", side_effect=AssertionError("Weights on CPU")), \
                     patch.object(evaluate, "load_pinned_reward", side_effect=AssertionError("Weights on CPU")):
                    with self.assertRaisesRegex(RuntimeError, "--plan on CPU"):
                        evaluate.run_evaluation("fixture", "standard", phase, output_path=str(output))
            self.assertFalse(output.exists())

    def test_baseline_adapter_and_incomplete_or_wrong_checkpoint_are_rejected(self):
        cfg = approved_cfg()
        with self.assertRaisesRegex(ValueError, "baseline"):
            evaluate.checkpoint_for_condition(cfg, "sft", "any_adapter")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = [{"prompt_id": str(i)} for i in range(600)]
            plan = {"name": "beta_003", "pairs": 600, "beta": .03}
            counts = update_pair_counts(600, 2, 8)
            training = {"status": "COMPLETE_RELOAD_VERIFIED", "effective_config": cfg,
                "plan": plan, "selected_prompt_ids": [row["prompt_id"] for row in rows],
                "completed_pairs": 600, "completed_updates": 38, "expected_update_pair_counts": counts}
            reload_check = {"status": "PASS", "held_out_examples_used": False, "probe_prompt_ids": ["0", "1"]}
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                (folder / name).write_text("synthetic placeholder; never loaded")
            (folder / "reload_verification.json").write_text(json.dumps(reload_check))
            for change in ({"status": "FAILED"}, {"completed_pairs": 599},
                           {"plan": {**plan, "name": "beta_030", "beta": .3}}):
                (folder / "run_record.json").write_text(json.dumps({**training, **change}))
                with patch.object(evaluate, "run_plan", return_value=(rows, plan)):
                    with self.assertRaisesRegex(ValueError, "incomplete, mismatched"):
                        evaluate.checkpoint_for_condition(cfg, "beta_003", folder)
            (folder / "run_record.json").write_text(json.dumps(training))
            with patch.object(evaluate, "run_plan", return_value=(rows, plan)):
                actual, _, hashes = evaluate.checkpoint_for_condition(cfg, "beta_003", folder)
            self.assertEqual(actual, folder)
            self.assertEqual(len(hashes), 4)

    def test_changed_policy_artifact_and_reordered_reward_inputs_stop_before_weights(self):
        cfg = approved_cfg()
        plan = {"name": "sft", "beta": None, "pair_pools": [],
            "generation_pools": ["dpo_standard_eval"],
            "datasets": {"dpo_standard_eval": {"rows": 1}}}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            metrics = {"name": "sft", "beta": None, "pairs": {},
                       "generation": {"dpo_standard_eval": {"responses": 1}}}
            (folder / "metrics.json").write_text(json.dumps(metrics))
            generated_path = folder / "generation_dpo_standard_eval.jsonl"
            generated_path.write_text(json.dumps({"fixture": True}) + "\n")
            record = {"status": "POLICY_COMPLETE", "phase": "policy", "plan": plan,
                "effective_config": cfg, "artifact_sha256": {name: file_sha(folder / name)
                    for name in ("metrics.json", generated_path.name)}}
            (folder / "evaluation_record.json").write_text(json.dumps(record))
            evaluate.checked_policy_results(folder, cfg, plan)
            generated_path.write_text(json.dumps({"fixture": "changed"}) + "\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                evaluate.checked_policy_results(folder, cfg, plan)
        class Tokenizer:
            def decode(self, ids, **kwargs):
                return "answer"
        row = {"prompt_id": "p", "prompt": "question"}
        good = {"row_index": 0, "prompt_id": "p", "messages": [{"role": "user", "content": "question"}],
                "response_length": 2, "response_token_ids": [2, 1], "response": "answer"}
        with patch.object(evaluate, "reward_input", return_value=("template", [1, 2, 3])):
            audit = evaluate.reward_preflight(Tokenizer(), {"dpo_standard_eval": [good]},
                                             {"dpo_standard_eval": [row]}, cfg)
        self.assertEqual(audit["dpo_standard_eval"]["maximum_complete_input_tokens"], 3)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(evaluate, "verified_copy"), \
             patch.object(evaluate, "checked_policy_results", return_value=({}, {}, {"dpo_standard_eval": [{**good, "row_index": 1}]})), \
             patch.object(evaluate, "model_source", return_value=("synthetic tokenizer", {})), \
             patch.object(evaluate, "load_pinned_tokenizer", return_value=Tokenizer()), \
             patch.object(evaluate, "load_pinned_reward", side_effect=AssertionError("Weights before preflight")), \
             patch.object(evaluate, "reward_input", side_effect=AssertionError("Bad membership reached tokenization")):
            with self.assertRaisesRegex(ValueError, "row/order"):
                evaluate.reward_phase(cfg, plan, {"dpo_standard_eval": [row]},
                    Path(temporary), "synthetic models", "synthetic prior output", {})

    def test_phase_failure_is_backed_up_without_success_and_existing_results_are_preserved(self):
        # Mock only the GPU/model phase. Exercise the actual output/status/copy
        # orchestration with temporary files; these are never PA measurements.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, backup, models = root / "output", root / "backup", root / "models"
            backup.mkdir()
            models.mkdir()
            (models / "model_manifest.json").write_text("synthetic manifest")
            preparation = root / "preparation.json"
            preparation.write_text(json.dumps({"model_manifest_sha256": file_sha(models / "model_manifest.json")}))
            source = root / "fixture.py"
            source.write_text("# synthetic source provenance\n")
            cfg = approved_cfg()
            cfg["paths"]["dpo_standard_eval"] = str(source)
            plan = {"name": "sft", "beta": None, "datasets": {
                "dpo_standard_eval": {"rows": 1, "sha256": file_sha(source)}}}
            original_repo_path = evaluate.repo_path
            def resolve(path):
                return preparation if str(path) == "docs/task1_model_preparation.json" else original_repo_path(path)
            def fail_phase(*args):
                (output / "partial.jsonl").write_text('{"synthetic": true}\n')
                raise RuntimeError("synthetic phase failure")
            with patch.object(evaluate, "load_yaml", return_value=cfg), \
                 patch.object(evaluate, "fixed_pools", return_value=(plan, {})), \
                 patch.object(evaluate, "source_files", return_value=[source]), \
                 patch.object(evaluate, "repo_path", side_effect=resolve), \
                 patch.object(evaluate.metadata, "version", return_value="synthetic environment"), \
                 patch.object(evaluate, "policy_phase", side_effect=fail_phase), \
                 patch.object(evaluate.torch.cuda, "is_available", return_value=True), \
                 patch.object(evaluate.torch.cuda, "get_device_name", return_value="synthetic GPU"), \
                 patch.object(evaluate.torch.cuda, "reset_peak_memory_stats"), \
                 patch.object(evaluate.torch.cuda, "max_memory_allocated", return_value=0), \
                 patch.object(evaluate.torch.cuda, "max_memory_reserved", return_value=0), \
                 patch.object(evaluate.torch.cuda, "empty_cache"), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "synthetic phase failure"):
                    evaluate.run_evaluation("fixture", "sft", "policy", models_dir=models,
                        output_path=str(output), backup_root=backup)
                with self.assertRaises(FileExistsError):
                    evaluate.run_evaluation("fixture", "sft", "policy", models_dir=models,
                        output_path=str(output), backup_root=backup)
            saved = json.loads((output / "evaluation_record.json").read_text())
            self.assertEqual(saved["status"], "FAILED")
            copies = [path for path in backup.iterdir() if path.is_dir()]
            self.assertEqual(len(copies), 1)
            self.assertEqual(file_sha(copies[0] / "partial.jsonl"), file_sha(output / "partial.jsonl"))
            self.assertEqual(json.loads((copies[0] / "evaluation_record.json").read_text())["status"], "FAILED")
            receipt = json.loads(next(backup.glob("*_backup_receipt.json")).read_text())
            self.assertEqual(receipt["status"], "BACKUP_VERIFIED")


if __name__ == "__main__":
    unittest.main()
