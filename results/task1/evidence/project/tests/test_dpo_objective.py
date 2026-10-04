"""Independent objective tests using known scalar answers."""
import math
import unittest

import torch
from task1_dpo.dpo import dpo_loss


def tensor(values, grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=grad)


class DPOObjectiveTests(unittest.TestCase):
    def test_identical_policy_reference(self):
        chosen, rejected = tensor([-2., -10.]), tensor([-4., -3.])
        for beta in (0.03, 0.10, 0.30):
            loss, info = dpo_loss(chosen, rejected, chosen, rejected, beta)
            self.assertAlmostEqual(loss.item(), math.log(2), places=12)
            self.assertEqual(info["adjusted_margin_mean"].item(), 0)
            self.assertEqual(info["preference_accuracy"].item(), 0)

    def test_independent_scalar_formula(self):
        pc, pr = tensor([-2., -7.]), tensor([-4., -3.])
        rc, rr = tensor([-3., -5.]), tensor([-4., -4.])
        beta = 0.3
        # Adjusted margins calculated by hand: 1 and -3.
        expected = sum(
            math.log1p(math.exp(-beta * m)) for m in (1., -3.)
        ) / 2
        loss, info = dpo_loss(pc, pr, rc, rr, beta)
        self.assertAlmostEqual(loss.item(), expected, places=12)
        self.assertEqual(info["preference_accuracy"].item(), 0.5)

    def test_reference_changes_accuracy(self):
        # Policy gap +1; reference gap +3; adjusted gap -2.
        _, info = dpo_loss(
            tensor([-2.]), tensor([-3.]),
            tensor([-1.]), tensor([-4.]), 0.1,
        )
        self.assertEqual(info["preference_accuracy"].item(), 0)

    def test_analytical_gradient_and_frozen_reference(self):
        pc, pr = tensor([-2.], True), tensor([-4.], True)
        rc, rr = tensor([-3.], True), tensor([-4.], True)
        beta = 0.1
        loss, _ = dpo_loss(pc, pr, rc, rr, beta)
        loss.backward()
        expected = -beta / (1 + math.exp(beta * 1.0))
        self.assertAlmostEqual(pc.grad.item(), expected, places=12)
        self.assertAlmostEqual(pr.grad.item(), -expected, places=12)
        self.assertIsNone(rc.grad)
        self.assertIsNone(rr.grad)

    def test_common_policy_offset_cancels(self):
        pc, pr, rc, rr = map(tensor, ([-2.], [-4.], [-3.], [-4.]))
        a, _ = dpo_loss(pc, pr, rc, rr, 0.1)
        b, _ = dpo_loss(pc + 5, pr + 5, rc, rr, 0.1)
        self.assertAlmostEqual(a.item(), b.item(), places=12)

    def test_swapping_responses_reverses_margin(self):
        pc, pr, rc, rr = map(tensor, ([-2.], [-4.], [-3.], [-4.]))
        good, _ = dpo_loss(pc, pr, rc, rr, 0.1)
        swapped, info = dpo_loss(pr, pc, rr, rc, 0.1)
        self.assertGreater(swapped.item(), good.item())
        self.assertEqual(info["adjusted_margin_mean"].item(), -1)

    def test_nonpositive_beta_rejected(self):
        x = tensor([0.])
        for beta in (0, -0.1):
            with self.assertRaises(ValueError):
                dpo_loss(x, x, x, x, beta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
