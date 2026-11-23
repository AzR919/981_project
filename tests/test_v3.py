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


class PassAtKRuleTests(unittest.TestCase):
    def test_rloo_k_gap_is_the_pass_at_k_weight(self):
        from synthetic_rl.theory import advantage_gap

        p = np.array([0.01, 0.1, 0.5, 0.9])
        for K, G in ((2, 8), (4, 8), (8, 32)):
            np.testing.assert_allclose(advantage_gap(p, G, "rloo_k", passk=K), (1 - p) ** (K - 1), rtol=1e-9)

    def test_rloo_k_matches_monte_carlo(self):
        from synthetic_rl.theory import advantage_gap
        from synthetic_rl.trainer import calculate_advantages

        rng = np.random.default_rng(0)
        r = torch.tensor((rng.random((100000, 8)) < 0.2).astype(float))
        A = calculate_advantages(r, "rloo_k", passk=4).numpy()
        r = r.numpy()
        self.assertAlmostEqual(A[r == 1].mean() - A[r == 0].mean(),
                               advantage_gap(np.array([0.2]), 8, "rloo_k", passk=4)[0], delta=0.01)


class BranchTests(unittest.TestCase):
    """T11-T13: class-split updates from one state."""

    def two_contexts(self, angle_deg, n_a, n_b, optimizer="sgd", eta=1e-6):
        from synthetic_rl.policy import Policy

        angle = np.deg2rad(angle_deg)
        x = np.array([[1.0, 0.0]] * n_a + [[np.cos(angle), np.sin(angle)]] * n_b)
        prompts = torch.as_tensor(x)
        correct = torch.zeros(n_a + n_b, 1, dtype=torch.long)
        in_a = torch.as_tensor([True] * n_a + [False] * n_b)
        policy = Policy(num_features=2, num_tokens=2, response_length=1, arch="linear").double()
        cfg = config(G=8, eta=eta, optimizer=optimizer, B=n_a + n_b)
        out = run.measure_branches(policy, run.make_optimizer(cfg, policy), prompts, correct, in_a, cfg,
                                   np.random.default_rng(0))
        return out, policy

    def test_mixed_step_follows_the_analytic_control(self):
        # Two equally solvable contexts at p = 0.5: the mixed step moves B's logit in proportion to
        # (1 - f) + f cos(angle), the B-only step in proportion to (1 - f).
        for n_a, n_b, f in ((1, 1, 0.5), (2, 1, 2 / 3), (9, 1, 0.9)):
            out, _ = self.two_contexts(120, n_a, n_b)
            want = ((1 - f) + f * np.cos(np.deg2rad(120))) / (1 - f)
            self.assertAlmostEqual(out["d_b_mixed"] / out["d_b_b"], want, places=5)
        out, _ = self.two_contexts(90, 9, 1)
        self.assertAlmostEqual(out["d_b_a"] / out["d_b_b"], 0.0, places=6)   # A alone leaves B unchanged
        out, _ = self.two_contexts(60, 9, 1)
        self.assertGreater(out["d_b_a"], 0)                                  # A alone helps B

    def test_mixed_branch_is_the_real_step_and_the_parent_is_untouched(self):
        for optimizer, eta in (("sgd", 0.5), ("adam", 0.01)):
            out, policy = self.two_contexts(120, 9, 1, optimizer=optimizer, eta=eta)
            self.assertLess(out["mixed_check"], 1e-12)
            self.assertTrue(all(torch.count_nonzero(q) == 0 for q in policy.parameters()))

    def test_projection_removes_the_harmful_component(self):
        out, policy = self.two_contexts(120, 9, 1, eta=1e-6)
        self.assertLess(out["dot_a_h"], 0)
        self.assertAlmostEqual(out["d_b_a_proj"] / abs(out["d_b_a"]), 0.0, places=5)  # B no longer pushed down
        self.assertLess(out["d_b_a_short"], 0)                                        # a shorter A step still harms B

    def test_projected_update_and_class_means_run(self):
        base = config(arch="linear", features="correlated", rho=0.9, majority=0.9, L=1, n_train=64, n_test=8,
                      eta=0.1, steps=40, log_every=20, track_every=5, diagnostics=False, branch_at=[0, 0.5],
                      branch_samples=2)
        for update in ("expected_full", "projected", "shortened"):
            out = run.run_train(dict(base, update=update))
            self.assertEqual(sorted(out["branches"]), ["0", "20"])
            self.assertEqual(len(out["history"][-1]["class_mean"]), 2)
            self.assertIn("s_d_b_mixed", out["branches"]["0"])


class JunkAnswerTests(unittest.TestCase):
    """T22: tokens that are never a correct answer."""

    def test_only_the_used_classes_have_prompts(self):
        cfg = config(arch="linear", features="correlated", rho=0.9, majority=0.9, K=6, used_classes=2, L=1,
                     n_train=512, n_test=8, diagnostics=False)
        _, train, _, policy = run.build(cfg)
        counts = np.bincount(train[1][:, 0], minlength=6)
        self.assertEqual(counts[2:].sum(), 0)
        self.assertAlmostEqual(counts[0] / 512, 0.9, delta=0.03)
        self.assertEqual(policy.num_tokens, 6)

    def test_junk_weakens_the_majority_push_as_the_first_order_formula_says(self):
        # At a uniform start over K tokens, two classes: R = S_BA / ((K - 1) S_same), so R shrinks as 1 / (K - 1)
        # for the same kernel sums. Compare the measured R with the kernel formula for K = 2 and K = 6.
        for K in (2, 6):
            cfg = config(arch="linear", features="correlated", rho=0.9, majority=0.9, K=K, used_classes=2, L=1,
                         n_train=256, n_test=8, eta=1e-4, steps=1, log_every=1, snapshot_every=1, diagnostics=False,
                         update="expected_full", branch_at=[0])
            out = run.run_train(cfg)
            _, train, _, _ = run.build(cfg)
            x, cls = train[0], train[1][:, 0]
            k = x @ x.T
            s_ba, s_same = k[cls == 1][:, cls == 0].sum(), k[cls == 1][:, cls == 1].sum()
            b = out["branches"]["0"]
            self.assertAlmostEqual(-b["d_b_a"] / b["d_b_b"], s_ba / ((K - 1) * s_same), delta=1e-3 * s_ba / s_same)


class WarmStartTests(unittest.TestCase):
    """T23: class-balanced supervised warm start."""

    def test_both_classes_start_near_the_target_and_apart(self):
        for arch in ("linear", "mlp"):
            cfg = config(arch=arch, features="correlated", rho=0.9, majority=0.9, K=3, L=1, n_train=256, n_test=8,
                         init="sft_balanced", p0=0.7, steps=1, log_every=1, snapshot_every=1, diagnostics=False,
                         update="expected_full")
            out = run.run_train(cfg)
            start = out["start"]
            self.assertGreater(start["p_a"], 0.5)
            self.assertGreater(start["p_b"], 0.5)                    # the minority is not left behind
            self.assertLess(start["b_on_a_answer"], 1 / 3)           # less mass on the majority's answer than at a uniform start
            uniform = run.run_train(dict(cfg, init="uniform", p0=None))["start"]
            self.assertAlmostEqual(uniform["b_on_a_answer"], 1 / 3, places=6)

    def test_unweighted_warm_start_is_unchanged(self):
        cfg = config(arch="linear", features="correlated", rho=0.5, K=3, L=1, n_train=64, n_test=8, init="sft_up", p0=0.5,
                     steps=1, log_every=1, snapshot_every=1, diagnostics=False)
        a, b = run.build(dict(cfg))[3], run.build(dict(cfg))[3]
        for x, y in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(x, y))


class ReleaseTests(unittest.TestCase):
    """T18: after the release, class A's reward term is dropped."""

    base = dict(arch="linear", features="correlated", rho=0.9, majority=0.9, L=1, n_train=64, n_test=8, eta=0.1,
                steps=40, log_every=1, snapshot_every=40, diagnostics=False, update="expected_full")

    def test_run_is_unchanged_before_the_release_and_follows_class_b_after_it(self):
        plain = run.run_train(config(**self.base, branch_at=[0.5]))
        released = run.run_train(config(**self.base, branch_at=[0.5], release_at=0.5))
        self.assertEqual(released["release_step"], 20)
        self.assertEqual([h["class_mean"] for h in plain["history"][:21]],
                         [h["class_mean"] for h in released["history"][:21]])
        self.assertIn("20", released["snapshots"])
        # The first released step is the B-only branch measured from the same state.
        before, after = released["history"][20]["class_mean"], released["history"][21]["class_mean"]
        branch = released["branches"]["20"]
        self.assertAlmostEqual(after[1] - before[1], branch["d_b_b"], places=12)
        self.assertAlmostEqual(after[0] - before[0], branch["d_a_b"], places=12)
        self.assertNotEqual(plain["history"][21]["class_mean"], after)

    def test_kl_after_the_release_pulls_toward_the_start_policy(self):
        # With no reward left for anyone (all prompts in class A) and a forward KL anchor, the policy
        # relaxes toward the reference it started from.
        cfg = config(**dict(self.base, majority=1.0, steps=400, log_every=100), release_at=0.25, release_beta=1.0,
                     release_kl="forward")
        out = run.run_train(cfg)
        means = [h["train"]["mean"] for h in out["history"]]
        self.assertGreater(means[1], means[0] + 0.05)                 # rewarded phase: success rises
        self.assertLess(abs(means[-1] - means[0]), abs(means[1] - means[0]) / 2)   # released: back toward the start

    def test_release_needs_the_exact_full_update(self):
        with self.assertRaises(ValueError):
            run.run_train(config(**dict(self.base, update="sampled"), release_at=0.5))


if __name__ == "__main__":
    unittest.main()
