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


class V3bTests(unittest.TestCase):
    def test_drop_above_is_zero_when_p_only_rises(self):
        cfg = config(features="onehot", n_train=16, init="tabular", p0="spread", L=1, G=8, B=16, eta=1.0,
                     update="expected_full", steps=50, log_every=10, track_every=1, diagnostics=False)
        out = run.run_train(cfg)
        self.assertEqual(max(out["drop_above"]), 0.0)  # tabular exact flow never decreases p

    def test_stop_at_half_ends_the_run(self):
        cfg = config(features="onehot", n_train=16, init="tabular", p0=0.3, L=1, G=8, B=16, eta=30.0,
                     update="expected_full", steps=1000, log_every=100, track_every=1, diagnostics=False,
                     stop_at_half=True)
        out = run.run_train(cfg)
        self.assertIsNotNone(out["step_to_half"])
        self.assertEqual(out["history"][-1]["step"], out["step_to_half"])
        self.assertGreaterEqual(out["history"][-1]["train"]["mean"], 0.5)

    def test_bootstrap_null_is_returned(self):
        cfg = config(mode="gradcheck", features="onehot", n_train=8, init="tabular", p0=0.3, L=1, G=4, R=200,
                     probe=8, chunks=1, boot=20)
        out = run.run_gradcheck(cfg)
        self.assertEqual(len(out["mean_z2_boot"]), 20)
        self.assertTrue(all(v > 0 for v in out["mean_z2_boot"]))


class AllWrongRuleTests(unittest.TestCase):
    def test_all_wrong_groups_get_a_negative_advantage(self):
        from synthetic_rl.trainer import calculate_advantages

        wrong = torch.zeros(1, 8, dtype=torch.float64)
        self.assertTrue(torch.all(calculate_advantages(wrong, "grpo") == 0))
        expected = {"heckel": -1.0, "virtual": -1 / np.sqrt(8), "pm_alpha": -0.1, "nsr": -1.0}
        for rule, value in expected.items():
            np.testing.assert_allclose(calculate_advantages(wrong, rule).numpy(), value, atol=1e-4)

    def test_mixed_groups_match_the_definitions(self):
        from synthetic_rl.trainer import calculate_advantages

        r = torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64)
        np.testing.assert_allclose(calculate_advantages(r, "heckel").numpy(), [[3, -1, -1, -1]])
        np.testing.assert_allclose(calculate_advantages(r, "nsr").numpy(), [[0.1, -1, -1, -1]])
        for rule in ("virtual", "pm_alpha"):
            np.testing.assert_allclose(calculate_advantages(r, rule).numpy(),
                                       calculate_advantages(r, "grpo").numpy())

    def test_gap_adds_the_uniform_group_term(self):
        from synthetic_rl.theory import advantage_gap

        p, G = np.array([0.01, 0.2]), 8
        # virtual: + (1-p)^(G-1)/sqrt(G) from all-wrong groups and + p^(G-1)/sqrt(G) from all-correct ones
        extra = ((1 - p) ** (G - 1) + p ** (G - 1)) / np.sqrt(G)
        np.testing.assert_allclose(advantage_gap(p, G, "virtual") - advantage_gap(p, G, "grpo"), extra, rtol=2e-3)
        np.testing.assert_allclose(advantage_gap(p, G, "nsr"), 1.1)


if __name__ == "__main__":
    unittest.main()
