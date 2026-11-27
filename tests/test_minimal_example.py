"""The minimal example: explicit and block features, KL reference resets, a reward-rule switch, G = inf."""

import math
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import run  # noqa: E402
from experiments.exact_flow import flow  # noqa: E402

COUNTS = {0.9: [9, 1], 0.7: [7, 3], 0.5: [1, 1]}


def two_kind(gamma, m=0.9, **kw):
    return dict(K=2, L=1, arch="linear", features="explicit", phi=[[1.0, 0.0], [gamma, 0.5]], labels=[0, 1],
                counts=COUNTS[m], diagnostics=False, **kw)


def config(**kw):
    return dict(run.DEFAULTS, name="test", **kw)


def block(**kw):
    return dict(dict(K=3, L=1, arch="linear", features="block", majority=0.9, block_dim=4, n_train=40, n_test=20,
                     diagnostics=False), **kw)


def trajectory(cfg, every=10):
    """Per-prompt success (float64) every `every` steps."""
    out = run.run_train(config(**cfg, snapshot_at=list(range(0, cfg["steps"] + 1, every))))
    return out, np.array([out["snapshots"][str(s)] for s in range(0, cfg["steps"] + 1, every)])


class BuildTests(unittest.TestCase):
    def test_explicit_features(self):
        cfg = config(**two_kind(0.7, m=0.7))
        task, train, test, policy = run.build(cfg)
        self.assertIsNone(task)
        self.assertIsNone(test)
        self.assertEqual(cfg["n_train"], 10)
        np.testing.assert_array_equal(train[0], [[1.0, 0.0]] * 7 + [[0.7, 0.5]] * 3)  # no constant feature
        np.testing.assert_array_equal(train[1][:, 0], [0] * 7 + [1] * 3)
        self.assertEqual(policy.num_features, 2)
        self.assertTrue(torch.all(policy.output.weight == 0))                       # uniform start
        np.testing.assert_allclose(run.success(policy, train), 0.5)
        mlp = config(**dict(two_kind(0.7, m=0.7), arch="mlp"))
        np.testing.assert_allclose(run.success(run.build(mlp)[3], train), 0.5)

    def test_explicit_needs_one_token_and_counts(self):
        for bad in (dict(L=2), dict(majority=0.9), dict(counts=[9, 0]), dict(labels=[0, 2]), dict(counts=None)):
            with self.assertRaises(ValueError):
                run.build(config(**dict(two_kind(1.0), **bad)))

    def test_block_features(self):
        for scale in (0.0, 0.7):
            cfg = config(**block(shared_scale=scale))
            task, train, test, _ = run.build(cfg)
            self.assertIsNone(task)
            x, c = train[0], train[1][:, 0]
            self.assertEqual(x.shape, (40, 3 * 4 + 1))
            self.assertEqual(test[0].shape, (20, 13))
            np.testing.assert_array_equal(np.bincount(c), [36, 2, 2])               # round(0.9 * 40), rest split
            self.assertTrue(np.all(np.diff(c) >= 0))                                 # majority first
            np.testing.assert_array_equal(x[:, -1], scale)
            gram = x @ x.T
            cross = c[:, None] != c[None, :]
            np.testing.assert_array_equal(gram[cross], scale**2)                     # exactly s^2 (0 at s = 0)
            self.assertTrue(np.all(gram[~cross] >= 0))
            for k in range(3):                                                       # features only in own block
                np.testing.assert_array_equal(np.delete(x[c == k, :12], np.s_[4 * k:4 * k + 4], axis=1), 0)
        self.assertFalse(np.array_equal(train[0][:20], test[0]))                     # fresh test draws

    def test_block_runs_with_classes_logged(self):
        for arch in ("linear", "mlp"):
            out = run.run_train(config(**block(arch=arch, update="expected_full", eta=0.5, steps=5, log_every=5)))
            self.assertEqual(len(out["history"][-1]["class_mean"]), 2)
            self.assertIn("test_mean", out["history"][-1])
            self.assertAlmostEqual(out["start"]["p_b"], 1 / 3)


def flow_log_success(cfg, taus):
    return flow(cfg, taus)["log_success"]


class NonNegativeInteractionTests(unittest.TestCase):
    """With non-negative feature inner products across classes (two kinds, gamma <= 0) or none at all (block
    features, shared_scale = 0), no prompt's success ever decreases."""

    taus = np.logspace(-4, 8, 121)

    def cases(self):
        return [two_kind(-0.5, rule="mean"), two_kind(0.0, rule="grpo"), block(rule="grpo")]

    def test_exact_flow_never_decreases(self):
        for cfg in self.cases():
            logp = flow_log_success(cfg, self.taus)
            self.assertGreater(np.diff(logp, axis=0).min(), -1e-8, cfg["features"])

    def test_small_step_updates_never_decrease(self):
        for cfg in self.cases():
            out = run.run_train(config(**cfg, update="expected_full", eta=0.05, steps=400, log_every=400))
            self.assertEqual(out["decreases_total"], 0, cfg["features"])


class NegativeInteractionTests(unittest.TestCase):
    def test_strong_overlap_keeps_falling(self):
        logp = flow_log_success(two_kind(1.2, rule="mean"), [1e8, 1e10])[:, 9]
        self.assertLess(logp[0], math.log(1e-6))
        self.assertLess(logp[1], logp[0] - 1.0)

    def test_weak_overlap_recovers(self):
        logp = flow_log_success(two_kind(0.5, rule="mean"), np.logspace(-2, 6, 81))[:, 9]
        self.assertLess(logp.min(), math.log(0.3))      # it dips first
        self.assertGreater(logp[-1], math.log(0.5))     # and is back above 0.5 by tau = 1e6

    def test_first_step_ratio(self):
        """Majority-only change of the minority's success over the minority-only change, to first order."""
        for gamma in (0.25, 0.5, 1.2, 2.0):
            for m in (0.7, 0.9):
                cfg = config(**two_kind(gamma, m=m))
                _, train, _, policy = run.build(cfg)
                prompts, correct = run.tensors(policy, train)
                in_a = torch.as_tensor(train[1][:, 0] == 0)
                g_a, g_b, h = run.class_gradients(policy, prompts, correct, in_a, cfg)
                ratio = -(run.flat(g_a) @ run.flat(h)) / (run.flat(g_b) @ run.flat(h))
                want = gamma * m / ((1 - m) * (gamma**2 + 0.25))
                self.assertAlmostEqual(float(ratio) / want, 1.0, delta=0.01)


class ResetTests(unittest.TestCase):
    def test_reset_every_step_is_the_unregularised_run(self):
        """The KL gradient vanishes at the reference, so resetting it before every step removes the KL term."""
        base = two_kind(1.2, update="expected_full", optimizer="sgd", eta=0.5, steps=300)
        _, plain = trajectory(base)
        for kl in ("reverse", "forward"):
            out, reset = trajectory(dict(base, beta=1.0, kl=kl, reset_every=1))
            self.assertEqual(out["resets"], 299)
            np.testing.assert_allclose(reset, plain, rtol=1e-9, atol=1e-14)
        _, fixed = trajectory(dict(base, beta=1.0, kl="reverse"))
        _, every_10 = trajectory(dict(base, beta=1.0, kl="reverse", reset_every=10))
        self.assertGreater(np.abs(fixed - plain).max(), 1e-3)                      # the KL term matters
        self.assertGreater(np.abs(every_10 - fixed).max(), 1e-3)                   # and resets change it

    def test_reset_every_step_sampled(self):
        base = two_kind(1.2, update="sampled", G=8, B=16, eta=0.5, steps=60, seed=3)
        _, plain = trajectory(base)
        for kl in ("reverse", "forward"):
            _, reset = trajectory(dict(base, beta=1.0, kl=kl, reset_every=1))
            np.testing.assert_allclose(reset, plain, rtol=1e-9, atol=1e-14)

    def test_reset_checks(self):
        base = two_kind(1.2, update="expected_full", eta=0.5, steps=10)
        for bad in (dict(reset_every=5), dict(reset_every=5, beta=0.1, anchor_at=0.5, anchor_beta=0.1),
                    dict(reset_every=5, beta=0.1, release_at=0.5), dict(reset_every=0, beta=0.1)):
            with self.assertRaises(ValueError):
                run.run_train(config(**base, **bad))


class SwitchTests(unittest.TestCase):
    def test_switch_changes_only_later_steps(self):
        for update in ("expected_full", "sampled"):
            base = two_kind(1.2, update=update, rule="grpo", G=8, B=16, eta=0.5, steps=100, seed=1)
            plain_out, plain = trajectory(base, every=1)
            switch_out, switched = trajectory(dict(base, switch_at=0.4, switch_rule="rloo_k", switch_passk=2), 1)
            self.assertEqual(switch_out["switch_step"], 40)
            np.testing.assert_array_equal(switched[:41], plain[:41], update)       # steps 0..40 identical
            self.assertTrue(np.all(np.abs(switched[42:] - plain[42:]).max(axis=1) > 0), update)
            # A switch at step 0 is a run with the second rule from the start.
            _, at_zero = trajectory(dict(base, switch_at=0.0, switch_rule="rloo_k", switch_passk=2), 1)
            _, second = trajectory(dict(base, rule="rloo_k", passk=2), 1)
            np.testing.assert_array_equal(at_zero, second, update)

    def test_switch_checks(self):
        base = two_kind(1.2, update="expected_full", eta=0.5, steps=10)
        for bad in (dict(switch_at=0.5, switch_rule="rloo_k"), dict(switch_at=0.5, switch_rule="nope"),
                    dict(switch_at=0.5, release_at=0.5), dict(switch_at=0.5, anchor_at=0.5, anchor_beta=0.1)):
            with self.assertRaises(ValueError):
                run.run_train(config(**base, **bad))


class FlowAgreementTests(unittest.TestCase):
    def test_flow_matches_small_step_sgd(self):
        eta, steps = 0.01, 2000
        for gamma, rule, G in ((1.2, "mean", 8), (0.5, "grpo", 8), (1.2, "grpo", "inf")):
            cfg = two_kind(gamma, rule=rule, G=G)
            sim = run.run_train(config(**cfg, update="expected_full", optimizer="sgd", eta=eta, steps=steps,
                                       log_every=steps, snapshot_at=[steps]))
            got = np.exp(flow(cfg, [eta * steps])["log_success"][-1])
            np.testing.assert_allclose(got, sim["snapshots"][str(steps)], rtol=5e-3, atol=1e-6)

    def test_stable_tail_agrees(self):
        cfg = two_kind(0.9, rule="mean")
        taus = np.logspace(-2, 9, 23)
        np.testing.assert_allclose(flow(cfg, taus)["log_success"], flow(cfg, taus, stable_tail=True)["log_success"],
                                   rtol=1e-5, atol=1e-9)

    def test_infinite_group_needs_expected_updates(self):
        with self.assertRaises(ValueError):
            run.run_train(config(**two_kind(1.2, rule="grpo", G="inf", update="sampled", steps=2)))


if __name__ == "__main__":
    unittest.main()
