"""Validate exact sequence KL and reference isolation."""

import itertools
import unittest

import torch

from synthetic_rl.policy import Policy


class PolicyKLTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.policy = Policy(num_features=3, num_tokens=3, response_length=2).double()
        self.prompts = torch.randn(2, 3, dtype=torch.float64)
        self.reference = self.policy.frozen_reference()

    def change_policy(self):
        with torch.no_grad():
            self.policy.output.bias.copy_(
                torch.tensor([0.7, -0.4, 0.1, -0.2, 1.1, 0.3])
            )

    def test_identical_policies_have_zero_kl(self):
        for direction in ("reverse", "forward"):
            actual = self.policy.kl_divergence(
                self.prompts, self.reference, direction
            )
            torch.testing.assert_close(actual, torch.zeros(2, dtype=torch.float64))

    def test_both_directions_match_enumerated_sequence_kl(self):
        self.change_policy()
        sequences = torch.tensor(list(itertools.product(range(3), repeat=2)))
        responses = sequences.unsqueeze(0).expand(2, -1, -1)
        log_p = self.policy.log_prob(self.prompts, responses)
        log_q = self.reference.log_prob(self.prompts, responses)
        expected = {
            "reverse": (log_p.exp() * (log_p - log_q)).sum(dim=1),
            "forward": (log_q.exp() * (log_q - log_p)).sum(dim=1),
        }
        for direction in expected:
            actual = self.policy.kl_divergence(
                self.prompts, self.reference, direction
            )
            torch.testing.assert_close(actual, expected[direction])
            self.assertTrue((actual > 0).all())

    def test_reference_is_independent_and_receives_no_gradients(self):
        self.change_policy()
        self.assertFalse(self.reference.training)
        self.assertTrue(all(not p.requires_grad for p in self.reference.parameters()))
        torch.testing.assert_close(
            self.reference.output.bias,
            torch.zeros_like(self.reference.output.bias),
        )
        # Even an unfrozen reference must not receive gradients from KL.
        self.reference.requires_grad_(True)
        for direction in ("reverse", "forward"):
            self.policy.zero_grad(set_to_none=True)
            self.policy.kl_divergence(
                self.prompts, self.reference, direction
            ).mean().backward()
            gradient = self.policy.output.bias.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in self.reference.parameters()))

    def test_rejects_invalid_direction_and_task_dimensions(self):
        with self.assertRaises(ValueError):
            self.policy.kl_divergence(self.prompts, self.reference, "invalid")
        for settings in (
            {"num_features": 4},
            {"num_tokens": 4},
            {"response_length": 3},
        ):
            dimensions = dict(num_features=3, num_tokens=3, response_length=2)
            dimensions.update(settings)
            with self.assertRaises(ValueError):
                self.policy.kl_divergence(self.prompts, Policy(**dimensions))


if __name__ == "__main__":
    unittest.main()
