"""Check that the tabular trainer matches the original bandit in main.py.

Both implementations start with uniform logits, use the same fixed group
of responses, and take one SGD step with KL disabled. We compare the
change in each logit for mean-only, RLOO, and GRPO.

This checks that the implementations agree before H1 measures whether
the average sampled update matches the advantage-gap prediction.

Basically, a sanity check that the new tabular trainer reproduces the original bandit's SDG update for the same responses. 
I ran it already and it worked. 
"""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from main import advantages as bandit_advantages
from synthetic_rl.policy import Policy
from synthetic_rl.trainer import Trainer


class TabularBanditTests(unittest.TestCase):
    def test_one_sgd_step_matches_bandit(self):
        num_answers = 4
        correct_answer = 0
        learning_rate = 0.1
        sampled_answers = np.array([0, 0, 1, 2])
        group_size = len(sampled_answers)
        rewards = (sampled_answers == correct_answer).astype(float)

        prompt = torch.ones(1, 1)
        correct = torch.tensor([[correct_answer]])
        responses = torch.as_tensor(sampled_answers).reshape(1, group_size, 1)

        for rule in ("mean", "rloo", "grpo"):
            with self.subTest(rule=rule):
                # Both policies start with uniform probabilities.
                probabilities = np.full(num_answers, 1.0 / num_answers)
                advantages = bandit_advantages(rewards, rule)

                # Reproduce the NumPy bandit's update from main.py.
                weighted_counts = np.bincount(
                    sampled_answers, weights=advantages, minlength=num_answers
                )
                bandit_gradient = (
                    weighted_counts - advantages.sum() * probabilities
                ) / group_size
                expected_update = learning_rate * bandit_gradient

                policy = Policy(
                    num_features=1,
                    num_tokens=num_answers,
                    response_length=1,
                    architecture="linear",
                )
                optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate)
                trainer = Trainer(
                    policy,
                    optimizer,
                    group_size=group_size,
                    advantage=rule,
                    beta=0.0,
                )

                logits_before = policy(prompt).detach()
                with patch.object(policy, "sample", return_value=responses):
                    trainer.train_step(prompt, correct)
                logits_after = policy(prompt).detach()

                actual_update = (logits_after - logits_before)[0, 0].numpy()
                np.testing.assert_allclose(
                    actual_update, expected_update, rtol=1e-6, atol=1e-7
                )


if __name__ == "__main__":
    unittest.main()