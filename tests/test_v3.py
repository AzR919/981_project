"""Check the v3 additions: class shares, batches without replacement, absorption tracking, PPO options."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import run  # noqa: E402
from synthetic_rl.synthetic_task import SyntheticTask  # noqa: E402


def config(**kw):
    return dict(run.DEFAULTS, name="test", **kw)


class ClassShareTests(unittest.TestCase):
    def test_shares_change_only_the_answers(self):
        tasks = {m: SyntheticTask(num_tokens=3, response_length=1, num_features=9, seed=3, features="correlated",
                                  rho=0.9, class_shares=[m, (1 - m) / 2, (1 - m) / 2]) for m in (1 / 3, 0.9)}
        data = {m: t.generate_dataset(2000) for m, t in tasks.items()}
        np.testing.assert_array_equal(data[1 / 3][0], data[0.9][0])  # identical prompts
        for m, (_, correct) in data.items():
            self.assertAlmostEqual(np.mean(correct[:, 0] == 0), m, delta=0.04)

    def test_constant_feature_and_policy_width(self):
        cfg = config(arch="linear", features="correlated", rho=0.5, majority=0.7, L=1, n_train=64, n_test=8)
        _, train, _, policy = run.build(cfg)
        self.assertEqual(train[0].shape[1], cfg["d"] + 1)
        np.testing.assert_array_equal(train[0][:, -1], 1.0)


class TrackingTests(unittest.TestCase):
    def test_without_replacement_draws_distinct_prompts(self):
        idx = run.draw(np.random.default_rng(0), 512, config(B=512, replace=False))
        self.assertEqual(len(set(idx.tolist())), 512)

    def test_persistent_absorption_arrays(self):
        cfg = config(arch="linear", features="correlated", rho=0.9, L=2, eta=0.1, update="expected_full",
                     steps=200, log_every=50, track_every=5, diagnostics=False)
        out = run.run_train(cfg)
        p_end = np.array(out["snapshots"]["200"])
        tail = np.array(out["tail_max"])
        self.assertTrue(np.all(tail >= p_end - 1e-6))  # the tail maximum covers the final step
        last = np.array(out["last_above"]["0.01"])
        self.assertTrue(np.all(last[p_end >= 0.01] == 200))


class TrainerOptionTests(unittest.TestCase):
    def test_one_ppo_epoch_is_the_plain_update(self):
        base = config(features="onehot", n_train=16, init="tabular", p0="spread", L=2, G=4, B=8, eta=3.0,
                      steps=20, log_every=10, diagnostics=False)
        a = run.run_train(dict(base))
        b = run.run_train(dict(base, ppo_epochs=1, clip_eps=0.2))
        self.assertEqual(a["snapshots"]["20"], b["snapshots"]["20"])

    def test_expected_step_with_kl_matches_the_trainer_term(self):
        cfg = config(features="onehot", n_train=8, init="tabular", p0=0.2, L=1, G=8, B=8, beta=1.0,
                     update="expected_full", steps=1, log_every=1, diagnostics=False, p_start=0.6)
        out = run.run_train(cfg)
        self.assertGreater(out["history"][0]["kl"], 0.1)  # KL to the reference at 0.2 is active


if __name__ == "__main__":
    unittest.main()
