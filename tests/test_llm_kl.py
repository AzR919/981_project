"""The on-policy KL loss terms in llm/train.py must have the gradient of the KL they are named after."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.train import draw_batch, drift_stats, kl_terms, parse_strata, policy_loss  # noqa: E402


def expected_gradient(direction, z, ref):
    """Gradient of E_{y ~ pi}[loss(y)] with samples held fixed (only log pi(y) differentiated)."""
    logp = torch.log_softmax(z, 0)
    loss, _ = kl_terms(logp, torch.log(ref), direction)
    return torch.autograd.grad((logp.exp().detach() * loss).sum(), z)[0]


class KLGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.z = torch.randn(6, dtype=torch.float64, requires_grad=True)
        self.ref = torch.softmax(torch.randn(6, dtype=torch.float64), 0)

    def true_gradient(self, direction):
        logp = torch.log_softmax(self.z, 0)
        if direction == "reverse":
            kl = (logp.exp() * (logp - torch.log(self.ref))).sum()
        else:
            kl = (self.ref * (torch.log(self.ref) - logp)).sum()
        return torch.autograd.grad(kl, self.z)[0]

    def test_reverse(self):
        torch.testing.assert_close(expected_gradient("reverse", self.z, self.ref), self.true_gradient("reverse"))

    def test_forward(self):
        torch.testing.assert_close(expected_gradient("forward", self.z, self.ref), self.true_gradient("forward"))

    def test_logged_value_estimates_reverse_kl(self):
        logp = torch.log_softmax(self.z, 0).detach()
        _, value = kl_terms(logp, torch.log(self.ref), "reverse")
        torch.testing.assert_close((logp.exp() * value).sum(), (logp.exp() * (logp - torch.log(self.ref))).sum())


class KLBoundTests(unittest.TestCase):
    def test_extreme_log_ratios_give_finite_losses_and_gradients(self):
        # A sampled token the trainer scores as impossible (-inf), and ratios far beyond exp range.
        for direction in ("reverse", "forward"):
            logp = torch.tensor([-1.0, -400.0, float("-inf"), -2.0], requires_grad=True)
            ref = torch.tensor([-1.2, -0.5, -0.7, -900.0])
            loss, value = kl_terms(logp, ref, direction)
            self.assertTrue(torch.isfinite(loss).all() and torch.isfinite(value).all(), direction)
            grad = torch.autograd.grad(loss.sum(), logp)[0]
            self.assertTrue(torch.isfinite(grad).all(), direction)
            self.assertEqual(float(grad[1]), 0.0)          # beyond the bound: no KL gradient

    def test_bound_leaves_ordinary_tokens_unchanged(self):
        logp = torch.tensor([-1.0, -3.0, -0.2])
        ref = torch.tensor([-1.5, -2.0, -0.1])
        for direction in ("reverse", "forward"):
            delta = ref - logp
            want = 0.5 * delta**2 if direction == "reverse" else torch.exp(delta) - delta - 1
            torch.testing.assert_close(kl_terms(logp, ref, direction)[0], want)


class DriftTests(unittest.TestCase):
    def test_tis_with_identical_log_probs_is_the_plain_loss(self):
        adv = torch.tensor([1.0, -0.5, 2.0])
        logp = torch.tensor([-3.0, -1.0, -2.0], requires_grad=True)
        plain = policy_loss(adv, logp)
        torch.testing.assert_close(policy_loss(adv, logp, logp.detach().clone(), tis=10.0), plain)

    def test_tis_weights_are_truncated_and_carry_no_gradient(self):
        adv = torch.tensor([1.0, 1.0])
        logp = torch.tensor([-1.0, -1.0], requires_grad=True)
        sampler = torch.tensor([-1.0 - 0.6931471805599453, -6.0])   # ratios 2 and e^5
        loss = policy_loss(adv, logp, sampler, tis=3.0)
        torch.testing.assert_close(loss, torch.tensor(-(2.0 * -1.0 + 3.0 * -1.0)))
        torch.testing.assert_close(torch.autograd.grad(loss, logp)[0], torch.tensor([-2.0, -3.0]))

    def test_drift_stats(self):
        out = drift_stats([-10.0, -20.0], [-10.5, -19.0], [10, 30])
        self.assertAlmostEqual(out["drift_abs_per_token"], 1.5 / 40)
        self.assertAlmostEqual(out["drift_per_token"], -0.5 / 40)
        self.assertAlmostEqual(out["drift_max_seq"], 1.0)


class BatchTests(unittest.TestCase):
    def test_strata_fix_the_batch_composition(self):
        import numpy as np

        problems = [{"id": i, "procedure": "each" if i < 50 else "total"} for i in range(100)]
        strata = parse_strata("procedure:each=7,total=3")
        idx = draw_batch(np.random.default_rng(0), problems, 64, strata)
        self.assertEqual(len(set(idx)), 10)
        self.assertEqual(sum(problems[i]["procedure"] == "each" for i in idx), 7)
        self.assertEqual(len(draw_batch(np.random.default_rng(0), problems, 64)), 64)


if __name__ == "__main__":
    unittest.main()
