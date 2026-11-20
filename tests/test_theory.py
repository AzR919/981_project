"""Check the advantage gap and fixed points against enumeration and known values."""

import itertools
import math
import unittest

import torch

from synthetic_rl.theory import advantage_gap, fixed_points
from synthetic_rl.trainer import calculate_advantages

RULES = [("mean", None), ("rloo", None), ("grpo", None), ("passk", 2), ("maxrl", None), ("sign", None)]


def enumerated_gap(p, G, rule, passk):
    """E[A_1 | r_1 = 1] - E[A_1 | r_1 = 0] over all 2^G reward vectors."""
    gap = 0.0
    for bits in itertools.product((0, 1), repeat=G):
        rewards = torch.tensor([bits], dtype=torch.float64)
        A = calculate_advantages(rewards, kind=rule, passk=passk)[0, 0].item()
        prob = p ** sum(bits) * (1 - p) ** (G - sum(bits))
        gap += prob / p * A if bits[0] else -prob / (1 - p) * A
    return gap


class AdvantageGapTests(unittest.TestCase):
    def test_matches_enumeration(self):
        for (rule, k), G, p in itertools.product(RULES, (2, 4, 6), (0.03, 0.3, 0.8)):
            with self.subTest(rule=rule, G=G, p=p):
                self.assertAlmostEqual(advantage_gap(p, G, rule, passk=k), enumerated_gap(p, G, rule, k), places=9)

    def test_closed_forms(self):
        for G, p in itertools.product((2, 5, 16), (0.05, 0.5, 0.9)):
            self.assertAlmostEqual(advantage_gap(p, G, "mean"), (G - 1) / G, places=12)
            self.assertAlmostEqual(advantage_gap(p, G, "rloo"), 1.0, places=12)
            self.assertAlmostEqual(advantage_gap(p, G, "maxrl"), (1 - (1 - p) ** (G - 1)) / p, places=10)
        # GRPO at G = 2 is RLOO scaled by 1 / sqrt(1 + 4 epsilon).
        self.assertAlmostEqual(advantage_gap(0.3, 2, "grpo"), 1 / math.sqrt(1 + 4e-5), places=12)


class FixedPointTests(unittest.TestCase):
    def test_guide_values(self):
        # Values quoted in docs/project_guide.html, section 8, H4.
        (p_star, stable), = fixed_points(0.05, 2, 4, "grpo")
        self.assertAlmostEqual(p_star, 0.109, places=3)
        self.assertTrue(stable)
        self.assertAlmostEqual(fixed_points(0.05, 2, math.inf, "mean")[0][0], 0.0798, places=4)
        self.assertAlmostEqual(fixed_points(0.05, 2, math.inf, "mean", "forward")[0][0], 0.0916, places=4)

    def test_finite_group_always_has_an_interior_root(self):
        # W_G <= sqrt(G - 1) is bounded, so the KL(pi || pi_ref) equation has a root.
        self.assertEqual(len(fixed_points(0.05, 0.5, 8, "grpo")), 1)
        self.assertEqual(fixed_points(0.05, 0.5, math.inf, "grpo"), [])


if __name__ == "__main__":
    unittest.main()
