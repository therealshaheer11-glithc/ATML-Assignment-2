"""CPU numerical fixtures, not official coursework experiments."""
import copy
from pathlib import Path
import tempfile
import unittest
import json
import subprocess
import sys

import torch
from peft import LoraConfig, get_peft_model
from torch.optim import SGD
from transformers import Qwen2Config, Qwen2ForCausalLM
from common.models import reference_mode
from task1_dpo.runtime import (
    PairMetrics, SampledKL, pair_scores, train_epoch, update_pair_counts,
)

torch.set_num_threads(1)


def make_model():
    torch.manual_seed(6304)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=29, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, attention_dropout=0.0,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    ))
    return get_peft_model(model, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    ))


def batches(indices):
    def side(rejected):
        ids = [
            [1, 3+i, 9+i, (14+i if rejected else 20+i), 2]
            for i in indices
        ]
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.ones((len(ids), 5), dtype=torch.long),
            "response_mask": torch.tensor(
                [[0., 0., 0., 1., 1.]] * len(ids)
            ),
        }
    return side(False), side(True), [f"fixture-{i}" for i in indices]


def run(
    model, loader, batch_size, accumulation_steps, *,
    scaler=None, max_grad_norm=100000.,
):
    optimizer = SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    if scaler is None:
        scaler = torch.amp.GradScaler("cpu", enabled=False)
    events = []
    result = train_epoch(
        model, loader, optimizer, scaler, beta=.1, total_pairs=5,
        batch_size=batch_size, accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm, emit=events.append,
    )
    return result, events


class RuntimeTests(unittest.TestCase):
    def test_released_budgets_keep_partial_windows(self):
        full = update_pair_counts(1500, 2, 8)
        short = update_pair_counts(600, 2, 8)
        self.assertEqual(
            (len(full), full[-1], sum(full)), (94, 12, 1500)
        )
        self.assertEqual(
            (len(short), short[-1], sum(short)), (38, 8, 600)
        )

    def test_accumulation_equals_full_window_including_tail(self):
        accumulated = make_model()
        direct = copy.deepcopy(accumulated)
        before_frozen = {
            n: p.detach().clone()
            for n, p in accumulated.named_parameters()
            if not p.requires_grad
        }
        before_trainable = {
            n: p.detach().clone()
            for n, p in accumulated.named_parameters()
            if p.requires_grad
        }
        run_a, _ = run(
            accumulated,
            [batches([0, 1]), batches([2, 3]), batches([4])], 2, 2
        )
        run_b, _ = run(
            direct, [batches([0, 1, 2, 3]), batches([4])], 4, 1
        )
        self.assertEqual(run_a["update_pair_counts"], [4, 1])
        self.assertEqual(
            run_a["ordered_prompt_ids"], run_b["ordered_prompt_ids"]
        )

        changed = 0
        for (name, a), (other, b) in zip(
            accumulated.named_parameters(), direct.named_parameters()
        ):
            self.assertEqual(name, other)
            if a.requires_grad:
                torch.testing.assert_close(a, b, atol=2e-8, rtol=1e-5)
                changed += int(
                    not torch.equal(a.detach(), before_trainable[name])
                )
            else:
                self.assertTrue(torch.equal(a, before_frozen[name]))
        self.assertGreater(changed, 0)
        self.assertTrue(accumulated.training)

    def test_scaled_gradients_are_unscaled_before_clipping(self):
        scaled = make_model()
        unscaled = copy.deepcopy(scaled)
        loader = [batches([0, 1]), batches([2, 3]), batches([4])]
        _, events = run(
            scaled, loader, 2, 2,
            scaler=torch.amp.GradScaler(
                "cpu", enabled=True, init_scale=65536.
            ),
            max_grad_norm=1e-5,
        )
        _, plain_events = run(
            unscaled, loader, 2, 2, max_grad_norm=1e-5
        )
        norms = [
            e["gradient_norm_before_clipping"]
            for e in events if e["event"] == "optimizer_update"
        ]
        plain_norms = [
            e["gradient_norm_before_clipping"]
            for e in plain_events if e["event"] == "optimizer_update"
        ]
        self.assertEqual(norms, plain_norms)
        self.assertGreater(norms[0], 1e-5)
        for a, b in zip(scaled.parameters(), unscaled.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_nonfinite_gradient_stops_before_optimizer_update(self):
        model = make_model()
        parameters = [p for p in model.parameters() if p.requires_grad]
        original = [p.detach().clone() for p in parameters]
        hook = parameters[0].register_hook(
            lambda gradient: gradient * float("nan")
        )
        with self.assertRaises(FloatingPointError):
            run(
                model,
                [batches([0, 1]), batches([2, 3]), batches([4])], 2, 2
            )
        hook.remove()
        for a, b in zip(parameters, original):
            self.assertTrue(torch.equal(a, b))

    def test_incomplete_epoch_never_returns_success(self):
        with self.assertRaises(RuntimeError):
            run(make_model(), [batches([0, 1])], 2, 2)

    def test_adapter_reload_and_reference_invariance(self):
        model = make_model()
        probe = batches([0, 1])[:2]
        with tempfile.TemporaryDirectory() as directory:
            base_dir = Path(directory) / "base"
            adapter_dir = Path(directory) / "adapter"
            copy.deepcopy(model).unload().save_pretrained(base_dir)

            with torch.no_grad(), reference_mode(model):
                original_reference = pair_scores(model, probe)

            run(
                model,
                [batches([0, 1]), batches([2, 3]), batches([4])], 2, 2
            )
            model.eval()
            with torch.no_grad():
                expected_policy = pair_scores(model, probe)
            model.save_pretrained(adapter_dir)

            child = r"""
import json, sys, torch
from peft import PeftModel
from transformers import Qwen2ForCausalLM
from common.models import reference_mode
from task1_dpo.runtime import pair_scores

torch.set_num_threads(1)
base = Qwen2ForCausalLM.from_pretrained(sys.argv[1])
model = PeftModel.from_pretrained(base, sys.argv[2], is_trainable=False)
model.eval()
probe = torch.load(sys.argv[3], weights_only=True)
with torch.no_grad():
    policy = pair_scores(model, probe)
with torch.no_grad(), reference_mode(model):
    reference = pair_scores(model, probe)
print(json.dumps({
    "policy": [t.tolist() for t in policy],
    "reference": [t.tolist() for t in reference],
}))
"""
            probe_path = Path(directory) / "probe.pt"
            torch.save(probe, probe_path)
            completed = subprocess.run(
                [
                    sys.executable, "-c", child,
                    str(base_dir), str(adapter_dir), str(probe_path),
                ],
                text=True, capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            actual_policy = [
                torch.tensor(t) for t in result["policy"]
            ]
            actual_reference = [
                torch.tensor(t) for t in result["reference"]
            ]
            for a, b in zip(expected_policy, actual_policy):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            for a, b in zip(original_reference, actual_reference):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_fresh_run_does_not_inherit_previous_adapters(self):
        trained = make_model()
        initial = {
            n: p.detach().clone()
            for n, p in trained.named_parameters()
            if p.requires_grad
        }
        run(
            trained,
            [batches([0, 1]), batches([2, 3]), batches([4])], 2, 2
        )
        fresh = make_model()
        self.assertTrue(any(
            not torch.equal(p, initial[n])
            for n, p in trained.named_parameters()
            if p.requires_grad
        ))
        for name, parameter in fresh.named_parameters():
            if parameter.requires_grad:
                self.assertTrue(torch.equal(parameter, initial[name]))

    def test_skipped_optimizer_update_stops(self):
        class SkippedScaler:
            current_scale = 65536.
            def scale(self, loss):
                return loss
            def unscale_(self, optimizer):
                pass
            def get_scale(self):
                return self.current_scale
            def step(self, optimizer):
                pass
            def update(self):
                self.current_scale *= .5

        with self.assertRaisesRegex(FloatingPointError, "skipped"):
            run(
                make_model(),
                [batches([0, 1]), batches([2, 3]), batches([4])],
                2, 2, scaler=SkippedScaler(),
            )

    def test_nonfinite_loss_stops(self):
        model = make_model()
        def corrupt_output(module, args, output):
            output.logits = output.logits * float("nan")
            return output
        hook = model.register_forward_hook(corrupt_output)
        with self.assertRaisesRegex(FloatingPointError, "loss"):
            run(
                model,
                [batches([0, 1]), batches([2, 3]), batches([4])], 2, 2
            )
        hook.remove()

    def test_accuracy_includes_ties_in_denominator(self):
        metric = PairMetrics(.1)
        metric.add(
            torch.tensor([1., 2., 0.]), torch.zeros(3),
            torch.tensor([0., 2., 1.]), torch.zeros(3),
        )
        result = metric.result()
        self.assertEqual(result["preference_accuracy"], 1/3)
        self.assertEqual(result["ties"], 1)

    def test_pair_loss_weights_unequal_batches(self):
        whole, split = PairMetrics(.3), PairMetrics(.3)
        values = (
            torch.tensor([1., 2., -3.]),
            torch.zeros(3), torch.zeros(3), torch.zeros(3),
        )
        whole.add(*values)
        split.add(*(v[:2] for v in values))
        split.add(*(v[2:] for v in values))
        self.assertAlmostEqual(
            whole.result()["mean_dpo_loss"],
            split.result()["mean_dpo_loss"], places=6,
        )

    def test_kl_weights_tokens_across_unequal_response_lengths(self):
        metric = SampledKL()
        metric.add(
            torch.tensor([[4., 900.]]),
            torch.zeros((1, 2)), torch.tensor([[1., 0.]]),
        )
        metric.add(
            torch.tensor([[1., 1., 1.]]),
            torch.zeros((1, 3)), torch.ones((1, 3)),
        )
        self.assertEqual(metric.result()["sampled_kl"], 7/4)
        self.assertEqual(metric.result()["valid_response_tokens"], 4)

    def test_negative_sampled_kl_remains_negative(self):
        metric = SampledKL()
        metric.add(
            torch.tensor([[-2.]]), torch.zeros((1, 1)), torch.ones((1, 1))
        )
        self.assertEqual(metric.result()["sampled_kl"], -2.)


if __name__ == "__main__":
    unittest.main()
