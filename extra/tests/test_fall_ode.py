"""The reduced class-level ODE of the fall (extra/experiments/fall_ode.py) against the exact flow (extra/experiments/exact_flow.py).

On a linear task whose features are replaced by their class means, the exact flow is a flow on U distinct points,
and the class model must reproduce it to solver accuracy. With K = 2 there is one minority class, so the
class-symmetric model must reproduce it too.
"""

import os as _os, sys as _sys  # extra/ layout: repo root, extra/, and both experiments/ folders on the path
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [_REPO, _os.path.join(_REPO, "experiments"), _os.path.join(_REPO, "extra"),
                 _os.path.join(_REPO, "extra", "experiments")]
import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import exact_flow, fall_ode  # noqa: E402


def class_mean_task(cfg):
    phi, cls, W0, K, U = fall_ode.features(cfg, cache=None)
    means, shares, k, pooled = fall_ode.blocks(phi, cls, U)
    return means[cls], cls, K, U, means, shares, k, pooled, W0


class FallOdeTests(unittest.TestCase):
    def run_case(self, K):
        cfg = dict(arch="linear", features="correlated", rho=0.9, majority=0.8, L=1, K=K, n_train=128, n_test=8,
                   task_seed=0, seed=0)
        phim, cls, K, U, means, shares, k, pooled, W0 = class_mean_task(cfg)
        full = dict(fall_ode.DEFAULTS, **cfg)
        w = fall_ode.weight_fn(cfg)
        T = 1e4
        taus_c, C, logp_c = fall_ode.solve_class(k, shares, means @ W0.T, w, T, num=80)
        taus_s, logA_s, logB_s, _ = fall_ode.solve_sym(pooled, shares[0], U, K, w, means @ W0.T, T, num=80)
        with mock.patch.object(exact_flow, "task_arrays", lambda c: (phim, cls, K, full)):
            ex = exact_flow.flow(cfg, taus_c)
        np.testing.assert_allclose(taus_c, ex["taus"], rtol=1e-10)
        per_class = np.stack([ex["log_success"][:, cls == a][:, 0] for a in range(U)], axis=1)
        return per_class, logp_c, logA_s, logB_s

    def test_class_model_is_the_exact_flow_on_class_means(self):
        per_class, logp_c, _, _ = self.run_case(K=3)
        self.assertLess(per_class[:, 1:].min(), np.log(1e-4))        # the minority falls deep in this case
        np.testing.assert_allclose(logp_c, per_class, atol=2e-3, rtol=1e-3)

    def test_symmetric_model_is_exact_with_one_minority_class(self):
        per_class, _, logA_s, logB_s = self.run_case(K=2)
        np.testing.assert_allclose(logA_s, per_class[:, 0], atol=2e-3, rtol=1e-3)
        np.testing.assert_allclose(logB_s, per_class[:, 1], atol=2e-3, rtol=1e-3)

    def test_first_step_sign_matches_the_odds_law(self):
        """Below m* = 1/(1 + c) the minority rises at the start, above it falls (uniform start)."""
        for m in (0.3, 0.6):
            cfg = dict(arch="linear", features="correlated", rho=0.9, majority=m, L=1, K=3, n_train=128, n_test=8,
                       task_seed=0, seed=0)
            phim, cls, K, U, means, shares, k, pooled, W0 = class_mean_task(cfg)
            M = U - 1
            c = M * pooled["k_AB"] / (M * pooled["k_same"] - (M - 1) * pooled["k_diff"])
            rhs = fall_ode.sym_rhs(shares[0], M, K - U, pooled["k_AA"], pooled["k_AB"], pooled["k_same"],
                                   pooled["k_diff"], fall_ode.weight_fn(cfg))
            d = rhs(np.zeros(5))
            dlogpB = -(d[2] + (M - 1) * d[3]) / K         # d log p_B at the uniform start
            self.assertEqual(dlogpB < 0, shares[0] > 1 / (1 + c))


if __name__ == "__main__":
    unittest.main()
