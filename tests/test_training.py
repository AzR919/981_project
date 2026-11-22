"""Check advantage identities and that tabular training never lowers p."""

import unittest

import numpy as np
import torch

from synthetic_rl.policy import Policy
from synthetic_rl.synthetic_task import SyntheticTask
from synthetic_rl.trainer import Trainer, calculate_advantages
from synthetic_rl.warmstart import set_tabular_success


def all_groups(G):
    rows = [[1.0] * c + [0.0] * (G - c) for c in range(G + 1)]
    return torch.tensor(rows, dtype=torch.float64)


class AdvantageTests(unittest.TestCase):
    def test_zero_sum_rules(self):
        for rule in ("mean", "rloo", "grpo", "maxrl"):
            A = calculate_advantages(all_groups(6), kind=rule)
            torch.testing.assert_close(A.sum(dim=1), torch.zeros(7, dtype=torch.float64))

    def test_rloo_is_rescaled_mean(self):
        G = 5
        rewards = all_groups(G)
        torch.testing.assert_close(
            calculate_advantages(rewards, "rloo"),
            G / (G - 1) * calculate_advantages(rewards, "mean"),
        )

    def test_grpo_equals_scaled_rloo_at_two(self):
        rewards = all_groups(2)
        torch.testing.assert_close(
            calculate_advantages(rewards, "grpo"),
            calculate_advantages(rewards, "rloo") / np.sqrt(1 + 4e-5),
        )


def tabular_trainer(rule, L, eta, reward_noise=0.0, seed=0):
    task = SyntheticTask(num_tokens=4, response_length=L, num_features=32, features="onehot", seed=seed)
    data = task.generate_dataset(32)
    policy = Policy(num_features=32, num_tokens=4, response_length=L, arch="linear").double()
    set_tabular_success(policy, data[1], np.linspace(0.02, 0.6, 32))
    trainer = Trainer(policy, torch.optim.SGD(policy.parameters(), lr=eta), group_size=4,
                      advantage=rule, seed=seed, reward_noise=reward_noise)
    return trainer, data


class TabularMonotoneTests(unittest.TestCase):
    def test_p_never_decreases(self):
        # Deterministic verifier, zero-sum rule with A >= 0 on correct samples,
        # plain SGD, beta = 0, no shared parameters: p can only rise.
        for rule in ("mean", "rloo", "grpo", "maxrl"):
            for L in (1, 2):
                with self.subTest(rule=rule, L=L):
                    trainer, data = tabular_trainer(rule, L, eta=50.0)
                    for _ in range(100):
                        trainer.train_step(*data)
                    self.assertEqual(trainer.decreases, 0)

    def test_counter_sees_decreases_with_a_noisy_verifier(self):
        trainer, data = tabular_trainer("grpo", 1, eta=50.0, reward_noise=0.2)
        for _ in range(100):
            trainer.train_step(*data)
        self.assertGreater(trainer.decreases, 0)


if __name__ == "__main__":
    unittest.main()
