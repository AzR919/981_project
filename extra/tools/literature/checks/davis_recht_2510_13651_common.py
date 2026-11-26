"""Shared helpers for the Davis & Recht (arXiv 2510.13651) prior-art checks.

Notation
--------
Project:        group size G, success probability p, advantage A_i, W_G(p) = E[A|r=1] - E[A|r=0].
Davis & Recht:  group size M, weight Z_i = (1-R_i) a_{S_i} + R_i b_{S_i} with S_i = sum_{j != i} R_j   (Eq. 4.1, p.5)
                kappa(t) = sum_{s=0}^{M-1} (b_s - a_s) C(M-1,s) t^s (1-t)^{M-1-s}                      (p.5)
                h_M(t)   = (1/M) sum_s (b_s - a_s) I_t(s+1, M-s),   h_M' = kappa                        (Eq. 4.2, p.5)

Two independent routes to the same number are implemented here:

  W_enum   brute force over all 2^G reward vectors, applying the advantage rule to the whole
           vector as a trainer would (no use of a_s, b_s);
  kappa    Davis & Recht's formula, with a_s, b_s written down by hand for each rule.
"""
from math import comb

import numpy as np


# --------------------------------------------------------------------------------------
# Advantage rules, applied to a batch of reward vectors R of shape (n_vectors, G).
# --------------------------------------------------------------------------------------
def adv_mean_only(R):
    """Dr. GRPO / mean-only: A_i = r_i - rbar."""
    return R - R.mean(axis=1, keepdims=True)


def adv_rloo(R):
    """RLOO: A_i = r_i - mean of the other G-1 rewards."""
    G = R.shape[1]
    loo = (R.sum(axis=1, keepdims=True) - R) / (G - 1)
    return R - loo


def make_adv_grpo(ddof=0, eps=0.0, eps_inside=False):
    """GRPO: A_i = (r_i - rbar) / denom.

    ddof=0  population variance  (1/G) sum (r - rbar)^2  = rbar (1 - rbar) for binary rewards
    ddof=1  unbiased variance    (1/(G-1)) sum (r - rbar)^2
    eps_inside=False  denom = sqrt(var) + eps      (Davis & Recht Eq. 6.1, p.7)
    eps_inside=True   denom = sqrt(var + eps)      (Mroueh Eq. 4, p.5; PROJECT_CONTEXT.md)
    With eps = 0 an all-equal group gives 0/0, which is set to 0.
    """

    def adv(R):
        G = R.shape[1]
        mean = R.mean(axis=1, keepdims=True)
        var = ((R - mean) ** 2).sum(axis=1, keepdims=True) / (G - ddof)
        denom = np.sqrt(var + eps) if eps_inside else np.sqrt(var) + eps
        num = R - mean
        out = np.zeros_like(num)
        np.divide(num, denom, out=out, where=denom > 0)
        return out

    return adv


def all_reward_vectors(G):
    """All 2^G binary reward vectors, shape (2^G, G), as floats."""
    idx = np.arange(2 ** G)[:, None]
    return ((idx >> np.arange(G)[None, :]) & 1).astype(float)


def W_enum(adv, G, p_grid, i=0):
    """E[A_i | r_i = 1] - E[A_i | r_i = 0] by brute force over all 2^G reward vectors."""
    R = all_reward_vectors(G)
    A = adv(R)[:, i]
    ri = R[:, i]
    s_other = R.sum(axis=1) - ri
    out = []
    for p in p_grid:
        w_other = p ** s_other * (1 - p) ** (G - 1 - s_other)  # P(other G-1 rewards)
        out.append((A * w_other * (ri == 1)).sum() - (A * w_other * (ri == 0)).sum())
    return np.array(out)


# --------------------------------------------------------------------------------------
# Davis & Recht coefficients a_s, b_s (s = 0..G-1) for each rule, written by hand.
# --------------------------------------------------------------------------------------
def ab_mean_only(G):
    s = np.arange(G)
    a = -s / G                 # r_i = 0, group mean = s/G
    b = 1 - (s + 1) / G        # r_i = 1, group mean = (s+1)/G
    return a, b


def ab_rloo(G):
    s = np.arange(G)
    a = -s / (G - 1)           # r_i = 0, leave-one-out mean = s/(G-1)
    b = 1 - s / (G - 1)        # r_i = 1, leave-one-out mean = s/(G-1)
    return a, b


def ab_grpo(G, ddof=0, eps=0.0, eps_inside=False):
    """Davis & Recht p.7 (ddof=0, eps outside the root) and the other conventions."""
    s = np.arange(G)
    q0 = s / G                 # group mean when r_i = 0   (q_s in the paper)
    q1 = (s + 1) / G           # group mean when r_i = 1   (q_{s+1} in the paper)
    scale = G / (G - ddof)

    def denom(q):
        var = scale * q * (1 - q)
        return np.sqrt(var + eps) if eps_inside else np.sqrt(var) + eps

    a = np.zeros(G)
    b = np.zeros(G)
    d0, d1 = denom(q0), denom(q1)
    np.divide(-q0, d0, out=a, where=d0 > 0)
    np.divide(1 - q1, d1, out=b, where=d1 > 0)
    return a, b


def kappa(a, b, p_grid):
    """Davis & Recht's kappa(t) = sum_s (b_s - a_s) C(M-1,s) t^s (1-t)^(M-1-s)."""
    G = len(a)
    phi = b - a
    out = []
    for p in p_grid:
        out.append(sum(phi[s] * comb(G - 1, s) * p ** s * (1 - p) ** (G - 1 - s) for s in range(G)))
    return np.array(out)


def fits_form(adv, a, b, G, tol=1e-12):
    """Max over all reward vectors and all i of |A_i - ((1-R_i) a_{S_i} + R_i b_{S_i})|."""
    R = all_reward_vectors(G)
    A = adv(R)
    S = (R.sum(axis=1, keepdims=True) - R).astype(int)
    pred = (1 - R) * a[S] + R * b[S]
    return np.abs(A - pred).max()


P_GRID = np.array([0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
