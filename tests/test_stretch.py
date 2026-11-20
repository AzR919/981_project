"""Check the stretch-batch additions to experiments/run.py."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import run  # noqa: E402
from synthetic_rl.diagnostics import success_jacobian  # noqa: E402
from synthetic_rl.theory import advantage_gap  # noqa: E402


def config(**kw):
    return dict(run.DEFAULTS, name="test", **kw)


class ExpectedStepTests(unittest.TestCase):
    def test_gradient_is_minus_mean_W_grad_p(self):
        cfg = config(arch="linear", L=2, G=4, rule="grpo", n_train=16, n_test=8)
        _, train, _, policy = run.build(cfg)
        # A non-uniform start so W varies across prompts.
        with torch.no_grad():
            for q in policy.parameters():
                q.normal_(0, 0.5)
        prompts, correct = run.tensors(policy, train)
        p, J = success_jacobian(policy, prompts, correct)
        W = torch.as_tensor(advantage_gap(p.numpy(), 4, "grpo"), dtype=p.dtype)
        predicted = -(W[:, None] * J).mean(dim=0)

        opt = torch.optim.SGD(policy.parameters(), lr=0.0)
        run.expected_step(policy, opt, prompts, correct, cfg)
        grad = torch.cat([q.grad.reshape(-1) for q in policy.parameters()])
        torch.testing.assert_close(grad, predicted)


class StartTests(unittest.TestCase):
    def test_p_start_keeps_the_reference(self):
        cfg = config(features="onehot", n_train=8, init="tabular", p0=0.2, p_start=0.7, L=1, G=8, B=8,
                     beta=1.0, steps=1, log_every=1, snapshot_every=1, diagnostics=False)
        out = run.run_train(cfg)
        np.testing.assert_allclose(out["p_init"], 0.7, rtol=1e-5)
        # The KL at step 0 is against the reference at 0.2, so it is positive.
        self.assertGreater(out["history"][0]["kl"], 0.1)

    def test_sft_mixed_spreads_p(self):
        cfg = config(arch="mlp", L=1, init="sft_mixed", sft_steps=300, n_train=256, n_test=8)
        _, train, _, policy = run.build(cfg)
        self.assertGreater(run.success(policy, train).std(), 0.1)


class GradcheckTests(unittest.TestCase):
    def test_chunks_keep_the_old_outputs(self):
        base = config(mode="gradcheck", arch="linear", L=1, G=4, R=200, probe=8)
        old = run.run_gradcheck(dict(base))
        new = run.run_gradcheck(dict(base, chunks=10))
        for k in ("rel_err", "mc_rel", "mean_z2", "cosine"):
            self.assertAlmostEqual(old[k], new[k], places=12)
        self.assertEqual(len(new["mean_z2_null"]), 50)
        # The projected bias is small relative to its own error bar.
        self.assertLess(abs(new["bias_par"]), 5 * new["bias_par_se"])


if __name__ == "__main__":
    unittest.main()
