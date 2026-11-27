"""
Exact gradient flow of the linear policy under the exact expected GRPO update, to very long times.

For the linear policy (logits z_x = W phi_x, W of shape K x d, one answer token, no bias), the exact
expected update of experiments/run.py (`update="expected_full"`, plain SGD) is, in flow time tau = eta * t,

    dW/dtau = (1/n) sum_x W_G(p_x) p_x (e_{c_x} - pi_x) phi_x^T,

with W_G the group rule's weight. Small-eta SGD follows this flow. Here it is integrated with an adaptive
solver in log time (u = log tau), so flow times of 1e10 and beyond are reached cheaply. Probabilities are
computed in log space so that absorbed prompts (success near 1e-300) stay exact.

usage (one task, as a function): flow(cfg, taus) -> dict with per-prompt log-success at each requested tau.

cfg may set G = "inf" (the large-group limit of mean, rloo and grpo). The explicit and block feature tasks of
experiments/run.py work as well (K = 2 included).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import log_softmax

from experiments.run import DEFAULTS, build
from synthetic_rl.theory import advantage_gap


def task_arrays(cfg):
    """Features (n x d, as the linear policy sees them), correct answer per prompt, number of answers K."""
    full = dict(DEFAULTS, **cfg)
    if full["arch"] != "linear" or full["L"] != 1:
        raise ValueError("exact_flow covers the linear policy with one answer token")
    _, train, _, _ = build(full)
    return np.asarray(train[0], dtype=float), np.asarray(train[1][:, 0]), full["K"], full


def flow(cfg, taus, tau0=1e-6, rtol=1e-8, atol=1e-10, method="LSODA", stable_tail=False, start=None):
    """Integrate from a uniform start (W = 0) to each tau in `taus` (increasing). Returns per-prompt
    log-success (n_taus x n) and the class labels.

    stable_tail=True computes 1 - pi_c in e_c - pi as the sum of the other answers' probabilities instead of
    by subtraction. Once some prompt has 1 - p below about 1e-13 the subtraction makes the right-hand side
    noisy and LSODA's steps collapse (a run to tau = 1e14 with solved prompts can then take minutes or not
    finish); with stable_tail it takes well under a second. Values agree to solver tolerance where both
    finish. The default (False) keeps earlier results bit for bit.

    start: weights (K x d) to start from instead of W = 0, for example the "W" of an earlier flow (with another
    rule, to switch rules part-way). The taus are then flow times since that start."""
    phi, c, K, full = task_arrays(cfg)
    n, d = phi.shape
    G, rule, passk = full["G"], full["rule"], full["passk"]
    onehot = np.eye(K)[c]
    rows = np.arange(n)

    def rhs_tau(W):
        logpi = log_softmax(phi @ W.reshape(K, d).T, axis=1)
        pi = np.exp(logpi)
        p = pi[rows, c]
        w = advantage_gap(p, G, rule, passk=passk) * p
        resid = onehot - pi
        if stable_tail:
            resid[rows, c] = (pi * (1 - onehot)).sum(axis=1)
        return ((w[:, None] * resid).T @ phi / n).ravel()

    def rhs_u(u, W):                       # u = log tau, dW/du = tau dW/dtau
        return np.exp(u) * rhs_tau(W)

    if start is None:
        W0 = tau0 * rhs_tau(np.zeros(K * d))   # one tiny Euler step from W = 0, then log time
    else:
        start = np.asarray(start, dtype=float).ravel()
        W0 = start + tau0 * rhs_tau(start)
    us = np.log(np.asarray(taus, dtype=float))
    sol = solve_ivp(rhs_u, (np.log(tau0), us[-1]), W0, t_eval=us, method=method, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    logp = np.stack([log_softmax(phi @ Wt.reshape(K, d).T, axis=1)[np.arange(n), c] for Wt in sol.y.T])
    return {"taus": np.exp(sol.t), "log_success": logp, "cls": c, "W": sol.y.T[-1].reshape(K, d)}
