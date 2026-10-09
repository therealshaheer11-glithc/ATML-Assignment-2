"""CPU diagnostics only; no public or course weights are loaded."""
import copy
import unittest

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForSequenceClassification

from task2_ppo.critic_precision import promote_critic_head, token_values_fp32_head


def tiny_critic():
    torch.manual_seed(6304)
    cfg = Qwen2Config(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, num_labels=1, pad_token_id=0,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    base = Qwen2ForSequenceClassification(cfg).half()
    model = get_peft_model(base, LoraConfig(
        task_type="SEQ_CLS", r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], modules_to_save=["score"],
    ))
    return model.eval()


def optimizer(model):
    head, lora = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (lora if "lora_" in name else head).append(p)
    return torch.optim.AdamW(
        [{"params": lora, "lr": 1e-4}, {"params": head, "lr": 3e-4}],
        weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8,
    )


IDS = torch.tensor([[1, 2, 3, 4, 5]])
MASK = torch.ones_like(IDS)


class CriticPrecisionTests(unittest.TestCase):
    def test_only_active_head_changes_dtype_and_no_parameter_changes_value(self):
        model = tiny_critic()
        before = {n: (id(p), p.detach().clone()) for n, p in model.named_parameters()}
        audit = promote_critic_head(model)
        changed = []
        for n, p in model.named_parameters():
            old_id, old = before[n]
            self.assertEqual(id(p), old_id)
            torch.testing.assert_close(p.float(), old.float(), rtol=0, atol=0)
            if old.dtype != p.dtype:
                changed.append(n)
                self.assertTrue(p.requires_grad)
                self.assertIn("score.modules_to_save.default", n)
            if not p.requires_grad:
                self.assertEqual(p.dtype, torch.float16)
            if p.requires_grad and "lora_" in n:
                self.assertEqual(p.dtype, torch.float32)
        self.assertEqual(len(changed), 1)
        self.assertEqual(audit["approval_id"], "L")

    def test_values_match_manual_head_and_causal_state_slice(self):
        model = tiny_critic()
        promote_critic_head(model)
        base = model.get_base_model()
        with torch.no_grad():
            hidden = base.model(input_ids=IDS, attention_mask=MASK,
                                use_cache=False).last_hidden_state
            expected = torch.nn.functional.linear(
                hidden.float(), base.score.modules_to_save["default"].weight)
            actual = token_values_fp32_head(model, IDS, MASK)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected.squeeze(-1), rtol=0, atol=0)
        # A two-token prompt means response actions use states 1, 2, 3.
        torch.testing.assert_close(actual[:, 1:4], expected[:, [1, 2, 3], 0], rtol=0, atol=0)

    def test_backward_reaches_head_and_lora_then_adamw_uses_float32_moments(self):
        model = tiny_critic()
        promote_critic_head(model)
        opt = optimizer(model)
        targets = torch.tensor([[0.2, -0.1, 0.4, 0.0, 0.3]])
        loss = 0.5 * (token_values_fp32_head(model, IDS, MASK) - targets).square().mean()
        loss.backward()
        lora_nonzero = False
        head_nonzero = False
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                if "lora_" in name:
                    lora_nonzero |= bool(p.grad.abs().sum() > 0)
                else:
                    head_nonzero |= bool(p.grad.abs().sum() > 0)
            else:
                self.assertIsNone(p.grad)
        self.assertTrue(lora_nonzero)
        self.assertTrue(head_nonzero)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        for p, state in opt.state.items():
            self.assertTrue(torch.isfinite(p).all())
            self.assertEqual(state["exp_avg"].dtype, torch.float32)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)

    def test_small_gradient_mechanism_and_float32_correction(self):
        results = {}
        for dtype in (torch.float16, torch.float32):
            p = torch.nn.Parameter(torch.tensor([0.1], dtype=dtype))
            opt = torch.optim.AdamW([p], lr=3e-4, weight_decay=0,
                                   betas=(0.9, 0.999), eps=1e-8)
            p.grad = torch.tensor([1e-5], dtype=dtype)
            opt.step()
            results[dtype] = bool(torch.isfinite(p).all())
        self.assertFalse(results[torch.float16])
        self.assertTrue(results[torch.float32])

    def test_checkpoint_restore_preserves_float32_head_and_optimizer_exactly(self):
        model = tiny_critic()
        promote_critic_head(model)
        opt = optimizer(model)
        token_values_fp32_head(model, IDS, MASK).square().mean().backward()
        opt.step()
        state = copy.deepcopy(model.state_dict())
        opt_state = copy.deepcopy(opt.state_dict())
        restored = tiny_critic()
        promote_critic_head(restored)  # Must happen before loading float32 head state.
        restored.load_state_dict(state)
        restored_opt = optimizer(restored)
        restored_opt.load_state_dict(opt_state)
        torch.testing.assert_close(token_values_fp32_head(model, IDS, MASK),
                                   token_values_fp32_head(restored, IDS, MASK), rtol=0, atol=0)
        for group_a, group_b in zip(opt.param_groups, restored_opt.param_groups):
            for a, b in zip(group_a["params"], group_b["params"]):
                for key in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(opt.state[a][key], restored_opt.state[b][key], rtol=0, atol=0)

    def test_unpromoted_scoring_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "decision L"):
            token_values_fp32_head(tiny_critic(), IDS, MASK)


if __name__ == "__main__":
    unittest.main(verbosity=2)
