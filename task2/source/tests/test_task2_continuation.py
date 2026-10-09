"""Meaningful CPU integration checks; tiny random models, never course weights."""
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2ForSequenceClassification

from common.data import load_yaml
from common.models import trainable_parameters, value_parameter_groups
from task2_ppo.checkpoint import restore, snapshot, trainable_state
from task2_ppo.critic_precision import promote_critic_head
from task2_ppo.storage import (atomic_json, commit_directory, publish_directory,
                               restore_published, safe_extract, verify_directory)
from task2_ppo.training_core import freeze_rollout, optimize_rollout, ordinary_rollout


def models():
    torch.manual_seed(6304)
    config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, num_labels=1, pad_token_id=0, attention_dropout=0.)
    config._attn_implementation = "eager"
    actor = get_peft_model(Qwen2ForCausalLM(config).half(), LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=["q_proj", "v_proj"]))
    critic = get_peft_model(Qwen2ForSequenceClassification(config).half(), LoraConfig(
        task_type="SEQ_CLS", r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=["q_proj", "v_proj"], modules_to_save=["score"]))
    # Represent a nonzero supplied midpoint policy, rather than zero fresh LoRA.
    with torch.no_grad():
        for name, p in actor.named_parameters():
            if "lora_B" in name:
                p.fill_(.02)
    actor.eval()
    critic.eval()
    promote_critic_head(critic)
    actor_opt = torch.optim.AdamW(trainable_parameters(actor), lr=3e-6)
    critic_opt = torch.optim.AdamW(value_parameter_groups(critic, 1e-4, 3e-4), weight_decay=0.)
    return actor, critic, actor_opt, critic_opt


def batch(eos=True):
    with torch.inference_mode():
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    return ordinary_rollout({"sequences": ids, "attention_mask": torch.ones_like(ids),
        "prompt_width": 3, "response_ids": ids[:, 3:], "response_mask": torch.ones(1, 3),
        "terminated_with_eos": [eos], "truncated": [not eos],
        "responses": ["tiny CPU response"], "response_lengths": [3]})


class ContinuationTests(unittest.TestCase):
    def test_full_runner_recovers_locally_committed_update_without_retraining(self):
        from task2_ppo.continuation import run_one
        class Tokenizer:
            def save_pretrained(self, directory):
                Path(directory).mkdir(parents=True)
                (Path(directory) / "tokenizer_config.json").write_text("{}")
        def loader(cfg, folders):
            a, c, ao, co = models()
            return a, c, Tokenizer(), ao, co, {"test": "tiny CPU models"}
        specification = {"name": "standard", "updates": 4, "clip_epsilon": .2, "kl_beta": .1}
        train = [{"prompt_id": "cpu-test", "source_index": 7, "messages": [{"role": "user", "content": "tiny test"}]}]
        schedule = {"training": [{"update_index": i, "row_index": 0, "prompt_id": "cpu-test",
                                  "source_index": 7, "rollout_seed": 6304 + i} for i in range(4)]}
        audit = {"approvals": [], "source": {}, "chunk2_files": {}, "assets": {"verified_files": {}},
                 "public_models": {}, "schedule_sha256": "tiny-test", "environment": {"device": "CPU test"}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patches = [mock.patch("task2_ppo.continuation.load_actor_critic", side_effect=loader),
                       mock.patch("task2_ppo.continuation.batch_generate", side_effect=lambda *a, **k: batch()),
                       mock.patch("task2_ppo.continuation.score_reward_pairs", return_value=torch.tensor([.7])),
                       mock.patch("torch.cuda.synchronize"), mock.patch("torch.cuda.reset_peak_memory_stats"),
                       mock.patch("torch.cuda.max_memory_allocated", return_value=0),
                       mock.patch("torch.cuda.max_memory_reserved", return_value=0)]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                arguments = (load_yaml("configs/ppo.yaml"), specification, train, schedule, {}, audit, None, {})
                run_one(*arguments, root / "whole", root / "whole_saved")
                actual_publish = publish_directory
                def interrupted_publish(folder, destination):
                    if Path(folder).name == "step_0002":
                        raise OSError("simulated interrupted Drive upload")
                    return actual_publish(folder, destination)
                with mock.patch("task2_ppo.continuation.publish_directory", side_effect=interrupted_publish):
                    with self.assertRaisesRegex(OSError, "interrupted"):
                        run_one(*arguments, root / "resumed", root / "resumed_saved")
                self.assertTrue((root / "resumed/standard/step_0002/checkpoint.pt").is_file())
                run_one(*arguments, root / "resumed", root / "resumed_saved", resume=True)
            whole = torch.load(root / "whole/standard/step_0004/checkpoint.pt", weights_only=False)
            resumed = torch.load(root / "resumed/standard/step_0004/checkpoint.pt", weights_only=False)
            for group in ("policy", "critic"):
                for name in whole[group]:
                    torch.testing.assert_close(whole[group][name], resumed[group][name], rtol=0, atol=0)
            summary = json.loads((root / "resumed/standard/final/summary.json").read_text())
            self.assertEqual(summary["completed_updates"], 4)
            self.assertEqual(summary["optimization_steps"], 8)
            self.assertEqual(summary["actual_response_tokens"], 12)
            self.assertEqual(len(summary["sessions"]), 2)
            self.assertTrue(summary["wall_clock_complete"])

    def test_two_epochs_preserve_rollout_targets_and_distinguish_reference(self):
        actor, critic, ao, co = models()
        cfg, b = load_yaml("configs/ppo.yaml"), batch()
        fixed, diag = freeze_rollout(actor, critic, b, torch.tensor([.7]), cfg)
        original = {k: v.clone() for k, v in fixed.items()}
        old_actor = trainable_state(actor)
        self.assertGreater(float((fixed["old_logprobs"] - fixed["ref_logprobs"]).abs().max()), 0.)
        self.assertTrue(all(not t.requires_grad and not t.is_inference() for t in fixed.values()))
        self.assertFalse(b["sequences"].is_inference())
        steps = optimize_rollout(actor, critic, ao, co, b, fixed, cfg)
        self.assertEqual(len(steps), 2)
        self.assertAlmostEqual(steps[0]["ratio_max_abs_change"], 0., places=6)
        for key in original:
            torch.testing.assert_close(fixed[key], original[key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(p, trainable_state(actor)[n]) for n, p in old_actor.items()))
        for step in steps:
            self.assertAlmostEqual(step["weighted_value_loss"], .5 * step["value_mse"], places=7)
        for opt in (ao, co):
            self.assertTrue(all(int(s["step"]) == 2 for s in opt.state.values()))

    def test_missing_eos_penalty_and_terminal_shaping_are_exact(self):
        actor, critic, _, _ = models()
        cfg, b = load_yaml("configs/ppo.yaml"), batch(eos=False)
        fixed, diag = freeze_rollout(actor, critic, b, torch.tensor([2.]), cfg)
        self.assertEqual(float(fixed["effective_rewards"]), 1.)
        expected = -.1 * (fixed["old_logprobs"] - fixed["ref_logprobs"])
        expected[:, -1] += 1.
        torch.testing.assert_close(fixed["shaped_rewards"], expected)
        torch.testing.assert_close(fixed["returns"], fixed["raw_advantages"] + fixed["old_values"])
        self.assertEqual(diag["valid_generated_tokens"], 3)

    def test_critic_nonfinite_stops_before_actor_optimizer_step(self):
        actor, critic, ao, co = models()
        cfg, b = load_yaml("configs/ppo.yaml"), batch()
        fixed, _ = freeze_rollout(actor, critic, b, torch.tensor([.7]), cfg)
        before = trainable_state(actor)
        with torch.no_grad():
            critic.get_base_model().score.modules_to_save["default"].weight.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            optimize_rollout(actor, critic, ao, co, b, fixed, cfg)
        self.assertEqual(len(ao.state), 0)
        self.assertEqual(len(co.state), 0)
        for name, value in trainable_state(actor).items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_update_boundary_resume_matches_uninterrupted_training(self):
        cfg, b, contract = load_yaml("configs/ppo.yaml"), batch(), {"test": "same approved configuration"}
        a, c, ao, co = models()
        fixed, _ = freeze_rollout(a, c, b, torch.tensor([.7]), cfg)
        optimize_rollout(a, c, ao, co, b, fixed, cfg)
        saved = snapshot(a, c, ao, co, 1, contract)
        fixed2, _ = freeze_rollout(a, c, b, torch.tensor([.4]), cfg)
        optimize_rollout(a, c, ao, co, b, fixed2, cfg)
        a2, c2, ao2, co2 = models()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.save(saved, path)
            count = restore(torch.load(path, weights_only=False), a2, c2, ao2, co2, contract)
        self.assertEqual(count, 1)
        fixed3, _ = freeze_rollout(a2, c2, b, torch.tensor([.4]), cfg)
        optimize_rollout(a2, c2, ao2, co2, b, fixed3, cfg)
        for m1, m2 in ((a, a2), (c, c2)):
            for n, t in trainable_state(m1).items():
                torch.testing.assert_close(t, trainable_state(m2)[n], rtol=0, atol=0)

    def test_resume_rejects_different_contract_and_half_head(self):
        a, c, ao, co = models()
        saved = snapshot(a, c, ao, co, 0, {"clip": .2})
        with self.assertRaisesRegex(RuntimeError, "contract"):
            restore(saved, a, c, ao, co, {"clip": .5})
        with torch.no_grad():
            head = c.get_base_model().score.modules_to_save["default"].weight
            head.data = head.data.half()
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            restore(saved, a, c, ao, co, {"clip": .2})

    def test_approved_eval_mode_is_enforced(self):
        a, c, _, _ = models()
        a.train()
        with self.assertRaisesRegex(RuntimeError, "eval"):
            freeze_rollout(a, c, batch(), torch.tensor([.7]), load_yaml("configs/ppo.yaml"))

    def test_committed_archive_round_trip_republish_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            draft = root / "draft"
            draft.mkdir()
            atomic_json(draft / "update.json", {"completed_updates": 1})
            torch.save({"head": torch.tensor([.123456789], dtype=torch.float32)}, draft / "checkpoint.pt")
            step = root / "step_0001"
            commit_directory(draft, step)
            publish_directory(step, root / "drive")
            restored = root / "restored"
            restore_published(root / "drive/step_0001.zip", restored)
            self.assertEqual(verify_directory(step), verify_directory(restored))
            # Same logical checkpoint, different restored timestamps: safe reuse.
            original = root / "step_0001"
            import shutil
            shutil.rmtree(original)
            restored.rename(original)
            publish_directory(original, root / "drive")
            (original / "update.json").write_text("corrupted")
            with self.assertRaises(RuntimeError):
                verify_directory(original)

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            draft, destination = root / "draft", root / "step"
            draft.mkdir()
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                commit_directory(draft, destination)

    def test_zip_traversal_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with zipfile.ZipFile(root / "bad.zip", "w") as z:
                z.writestr("../escape.txt", "bad")
            with zipfile.ZipFile(root / "bad.zip") as z:
                with self.assertRaises(RuntimeError):
                    safe_extract(z, root / "destination")
            self.assertFalse((root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
