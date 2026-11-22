"""The on-policy KL loss terms in llm/train.py must have the gradient of the KL they are named after."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.train import kl_terms  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
