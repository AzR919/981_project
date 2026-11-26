"""The exact gradient flow of the linear policy follows small-step SGD on the exact expected update."""

import os as _os, sys as _sys  # extra/ layout: repo root, extra/, and both experiments/ folders on the path
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [_REPO, _os.path.join(_REPO, "experiments"), _os.path.join(_REPO, "extra"),
                 _os.path.join(_REPO, "extra", "experiments")]
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import run  # noqa: E402
from experiments.exact_flow import flow  # noqa: E402


class ExactFlowTests(unittest.TestCase):
    def test_flow_matches_small_step_sgd(self):
        base = dict(arch="linear", features="correlated", rho=0.9, majority=0.9, L=1, K=3, n_train=64, n_test=8,
                    diagnostics=False, task_seed=0, seed=0)
        eta, steps = 0.01, 400
        sim = run.run_train(dict(run.DEFAULTS, name="t", **base, optimizer="sgd", eta=eta, update="expected_full",
                                 steps=steps, log_every=steps, snapshot_every=steps))
        out = flow(base, [eta * steps])
        got = np.exp(out["log_success"][-1])
        want = np.asarray(sim["snapshots"][str(steps)])
        np.testing.assert_allclose(got, want, rtol=5e-3, atol=1e-6)

    def test_rejects_other_policies(self):
        with self.assertRaises(ValueError):
            flow(dict(arch="mlp", features="correlated", rho=0.9, majority=0.9, L=1, K=3), [1.0])


if __name__ == "__main__":
    unittest.main()
