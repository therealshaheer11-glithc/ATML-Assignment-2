"""Analytical PPO checks; fixtures are tiny and download no model weights."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from common.generation import _response_mask, response_token_logprobs
from task2_ppo.ppo import (
    compute_gae, normalize_advantages, ppo_policy_loss,
    shaped_rewards, value_mse_loss,
)


class PPOObjectiveTests(unittest.TestCase):
    def policy_case(self, ratio, advantage):
        new = torch.tensor([[math.log(ratio)]], dtype=torch.float64, requires_grad=True)
        old = torch.zeros_like(new)
        adv = torch.full_like(new, advantage)
        loss, observed_ratio, fraction = ppo_policy_loss(
            new, old, adv, torch.ones_like(new), eps=0.2
        )
        loss.backward()
        return loss.item(), new.grad.item(), observed_ratio.item(), fraction.item()

    def test_positive_advantage_above_upper_bound(self):
        loss, grad, ratio, fraction = self.policy_case(1.5, 1.0)
        self.assertAlmostEqual(loss, -1.2)
        self.assertAlmostEqual(grad, 0.0)
        self.assertAlmostEqual(ratio, 1.5)
        self.assertEqual(fraction, 1.0)

    def test_negative_advantage_below_lower_bound(self):
        loss, grad, _, fraction = self.policy_case(0.5, -1.0)
        self.assertAlmostEqual(loss, 0.8)
        self.assertAlmostEqual(grad, 0.0)
        self.assertEqual(fraction, 1.0)

    def test_positive_advantage_below_lower_bound_remains_active(self):
        loss, grad, _, fraction = self.policy_case(0.5, 1.0)
        self.assertAlmostEqual(loss, -0.5)
        self.assertAlmostEqual(grad, -0.5)
        # Outside the interval still counts in the manual's affected fraction.
        self.assertEqual(fraction, 1.0)

    def test_negative_advantage_above_upper_bound_remains_active(self):
        loss, grad, _, fraction = self.policy_case(1.5, -1.0)
        self.assertAlmostEqual(loss, 1.5)
        self.assertAlmostEqual(grad, 1.5)
        self.assertEqual(fraction, 1.0)

    def test_inside_interval_has_unclipped_policy_gradient(self):
        for ratio, advantage, expected_loss, expected_grad in (
            (1.1, 1.0, -1.1, -1.1), (0.9, -1.0, 0.9, 0.9)
        ):
            with self.subTest(ratio=ratio, advantage=advantage):
                loss, grad, _, fraction = self.policy_case(ratio, advantage)
                self.assertAlmostEqual(loss, expected_loss)
                self.assertAlmostEqual(grad, expected_grad)
                self.assertEqual(fraction, 0.0)

    def test_padding_does_not_change_loss_diagnostics_or_gradient(self):
        ratios = torch.tensor([[1.5, 1.0, 4.0, 0.1]], dtype=torch.float64)
        new = ratios.log().requires_grad_()
        mask = torch.tensor([[1., 1., 0., 0.]], dtype=torch.float64)
        advantages = torch.tensor([[1., -1., 1000., -1000.]], dtype=torch.float64)
        loss, detached_ratio, fraction = ppo_policy_loss(
            new, torch.zeros_like(new), advantages, mask, eps=0.2
        )
        # Valid-token objective = mean([1.2, -1.0]) = 0.1.
        self.assertAlmostEqual(loss.item(), -0.1)
        self.assertAlmostEqual(fraction.item(), 0.5)
        self.assertFalse(detached_ratio.requires_grad)
        self.assertFalse(fraction.requires_grad)
        loss.backward()
        torch.testing.assert_close(new.grad[0, 2:], torch.zeros(2, dtype=torch.float64))

    def test_identical_old_and_new_policy_has_ratio_one(self):
        old = torch.tensor([[-2.0, -3.0]], dtype=torch.float64)
        new = old.clone().requires_grad_()
        adv = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
        loss, ratios, fraction = ppo_policy_loss(new, old, adv, torch.ones_like(old))
        torch.testing.assert_close(ratios, torch.ones_like(old))
        self.assertEqual(fraction.item(), 0.0)
        self.assertAlmostEqual(loss.item(), -0.5)

    def test_gae_matches_hand_calculation(self):
        rewards = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
        values = torch.tensor([[0.5, 0.25, 0.75]], dtype=torch.float64)
        advantages, returns = compute_gae(rewards, values, torch.ones_like(values), 1., .95)
        torch.testing.assert_close(
            advantages, torch.tensor([[5.155625, 4.6375, 2.25]], dtype=torch.float64)
        )
        torch.testing.assert_close(
            returns, torch.tensor([[5.655625, 4.8875, 3.]], dtype=torch.float64)
        )

    def test_gae_lambda_one_equals_monte_carlo_return(self):
        rewards = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
        values = torch.tensor([[4., -3., 2.]], dtype=torch.float64)
        _, returns = compute_gae(rewards, values, torch.ones_like(values), 1., 1.)
        torch.testing.assert_close(returns, torch.tensor([[6., 5., 3.]], dtype=torch.float64))

    def test_gae_cannot_bootstrap_through_padding(self):
        rewards = torch.tensor([[1., 2., 999.]], dtype=torch.float64)
        values = torch.tensor([[0.5, 0.25, 999.]], dtype=torch.float64)
        mask = torch.tensor([[1., 1., 0.]], dtype=torch.float64)
        advantages, returns = compute_gae(rewards, values, mask, 1., .95)
        torch.testing.assert_close(
            advantages, torch.tensor([[2.4125, 1.75, 0.]], dtype=torch.float64)
        )
        torch.testing.assert_close(returns[:, :2], torch.tensor([[2.9125, 2.]], dtype=torch.float64))
        # The helper's padded return values are ignored by the critic loss mask.

    def test_kl_shaping_sign_and_terminal_reward_position(self):
        old = torch.tensor([[-1., -2., 100.], [-3., -4., -5.]], dtype=torch.float64)
        ref = old - torch.tensor([[.2, .4, 50.], [.1, .2, .3]], dtype=torch.float64)
        mask = torch.tensor([[1., 1., 0.], [1., 1., 1.]], dtype=torch.float64)
        rewards = shaped_rewards(torch.tensor([2., 3.]), old, ref, mask, .1)
        expected = torch.tensor([[-.02, 1.96, 0.], [-.01, -.02, 2.97]], dtype=torch.float64)
        torch.testing.assert_close(rewards, expected)

    def test_zero_kl_coefficient_leaves_only_terminal_reward(self):
        mask = torch.tensor([[1., 1., 0.]])
        rewards = shaped_rewards(torch.tensor([2.]), torch.ones_like(mask), torch.zeros_like(mask), mask, 0.)
        torch.testing.assert_close(rewards, torch.tensor([[0., 2., 0.]]))

    def test_masked_population_advantage_normalization(self):
        original = torch.tensor([[1., 3., 100.]])
        result = normalize_advantages(original, torch.tensor([[1., 1., 0.]]))
        torch.testing.assert_close(result, torch.tensor([[-1., 1., 0.]]))
        torch.testing.assert_close(original, torch.tensor([[1., 3., 100.]]))

    def test_value_loss_is_masked_mse_without_extra_half_factor(self):
        predicted = torch.tensor([[1., 3., 100.]])
        returns = torch.tensor([[2., 1., -100.]])
        loss = value_mse_loss(predicted, returns, torch.tensor([[1., 1., 0.]]))
        self.assertAlmostEqual(loss.item(), 2.5)

    def test_response_mask_includes_first_eos_but_excludes_later_padding(self):
        ids = torch.tensor([[7, 2, 0], [8, 9, 10]])
        torch.testing.assert_close(_response_mask(ids, 2), torch.tensor([[1., 1., 0.], [1., 1., 1.]]))

    def test_response_log_probabilities_have_correct_causal_shift(self):
        logits = torch.zeros((1, 5, 6), dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            for position, token in [(1, 1), (2, 2), (3, 3)]:
                logits[0, position, token] = 2.
        class TinyModel:
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=logits)
        logp, _ = response_token_logprobs(
            TinyModel(), torch.tensor([[4, 5, 1, 2, 3]]),
            torch.ones((1, 5), dtype=torch.long), 2, torch.tensor([[1, 2, 3]])
        )
        expected = 2. - math.log(math.exp(2.) + 5.)
        torch.testing.assert_close(logp, torch.full((1, 3), expected))
        (-logp.sum()).backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0.)
        self.assertEqual(logits.grad[0, 4].abs().sum().item(), 0.)

    def test_reference_disables_adapter_and_restores_mode_even_on_error(self):
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from common.models import reference_mode
        torch.manual_seed(6304)
        cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=1, num_attention_heads=2,
                          num_key_value_heads=2, pad_token_id=0, eos_token_id=2)
        base = Qwen2ForCausalLM(cfg).eval()
        independent_base = copy.deepcopy(base)
        model = get_peft_model(base, LoraConfig(
            r=2, lora_alpha=4, lora_dropout=.05,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
        ))
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.normal_(std=.2)
        ids = torch.tensor([[3, 4, 5]])
        model.eval()
        with torch.no_grad():
            expected = independent_base(ids).logits
            adapted = model(ids).logits
        self.assertFalse(torch.allclose(adapted, expected, atol=1e-7, rtol=1e-7))
        model.train()
        with reference_mode(model), torch.no_grad():
            self.assertFalse(model.training)
            torch.testing.assert_close(model(ids).logits, expected)
        self.assertTrue(model.training)
        with self.assertRaisesRegex(RuntimeError, "intentional fixture error"):
            with reference_mode(model):
                raise RuntimeError("intentional fixture error")
        self.assertTrue(model.training)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(ids).logits, adapted)


class SetupTests(unittest.TestCase):
    def test_released_config_passes_and_changed_budget_is_rejected(self):
        from common.data import load_yaml
        from task2_ppo.preflight import verify_config
        cfg = load_yaml("configs/ppo.yaml")
        verify_config(cfg)
        changed = copy.deepcopy(cfg)
        changed["updates"] = 21
        with self.assertRaisesRegex(RuntimeError, "updates"):
            verify_config(changed)

    def test_training_schedule_prefix_and_eval_seeds_are_fixed(self):
        from task2_ppo.preflight import make_schedule
        train = [{"prompt_id": f"train_{i}", "source_index": i} for i in range(40)]
        evaluation = [{"prompt_id": f"eval_{i}", "source_index": i} for i in range(3)]
        a = make_schedule(train, evaluation, 6304)
        b = make_schedule(train, evaluation, 6304)
        self.assertEqual(a, b)
        self.assertEqual(len(a["training"]), 20)
        self.assertEqual(len({r["row_index"] for r in a["training"]}), 20)
        self.assertEqual([r["rollout_seed"] for r in a["training"][:8]], list(range(6304, 6312)))
        self.assertEqual(a["evaluation"][0]["generation_seed"], 106304)
        self.assertEqual(a["evaluation"][0]["prompt_id"], "eval_0")

    def reward_fixture(self):
        return {"model_type": "qwen2", "architectures": ["Qwen2ForSequenceClassification"],
                "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 2,
                "id2label": {"0": "LABEL_0"}, "label2id": {"LABEL_0": 0},
                "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.}}

    def test_reward_translation_preserves_file_and_only_changes_theta(self):
        from task2_ppo.reward_compat import compatible_reward_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(self.reward_fixture()))
            original = path.read_bytes()
            effective, record = compatible_reward_config(path)
            self.assertEqual(effective.rope_theta, 1000000.)
            self.assertEqual(set(record["changed_fields"]), {"rope_theta"})
            self.assertEqual(path.read_bytes(), original)
            self.assertTrue(record["position_frequencies_match_checkpoint"])
            self.assertFalse(record["model_weights_loaded"])

    def test_reward_translation_rejects_unapproved_rope_type(self):
        from task2_ppo.reward_compat import compatible_reward_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            raw = self.reward_fixture()
            raw["rope_parameters"]["rope_type"] = "linear"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "approved"):
                compatible_reward_config(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
