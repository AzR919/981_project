"""Advantage rules for the checks on MaxRL (2602.02710) and reward budgeting (2611.05942).

Convention throughout (the project's): the single-prompt update is
    g_hat = (1/G) * sum_i A_i * grad log pi(y_i),
so every paper's estimator is first rewritten as a per-sample advantage A_i in this convention.

Two representations of each rule:
  * adv_from_rewards(rule, r): A computed from a raw 0/1 reward array of shape (..., G),
    written the way an implementer would write it. Used for brute-force enumeration.
  * adv_table(rule, G): (Ap, Am) with Ap[k] = advantage of a correct sample and Am[k] = advantage
    of an incorrect sample in a group with k correct. Used for enumeration over compositions.
The two are written independently and compared in check 1.

Rules
  mean        A = r - rbar                                   (Dr. GRPO / mean-only)
  rloo        A = r_i - mean of the other G-1 rewards
  grpo        A = (r - rbar) / sigma, sigma = population std (ddof 0), 0 on constant groups
  maxrl_alg1  MaxRL Algorithm 1 (p.6):  g = 1/(N rhat) sum (r_j - rhat) S_j, 0 if rhat = 0
              -> A = (r - rbar)/rbar, 0 if K = 0
  maxrl_appF  MaxRL App. F (p.39):      A = (r - rbar)/(rbar + 1e-6)
  maxrl_eq9   MaxRL Eq. (9) (p.5):      g = (1/K) sum r_i S_i, 0 if K = 0   -> A = G r / K
  maxrl_eq10  MaxRL Eq. (10) (p.6):     g = sum (r_i/K - 1/N) S_i, first term 0 if K = 0,
              baseline kept on all-incorrect groups        -> A = G r / K - 1  (A = -1 if K = 0)
  jia_lemma   Jia Lemma A.3 (p.16-17):  g = 1{1<=k<=G-1} sum (c_i/k - 1/G) S_i   (a SUM over i)
              -> A = G c/k - 1 on mixed groups, 0 otherwise
  jia_table   Jia Algorithm 1 / Table 1 (p.7-8): the advantage A = c/k - 1/G itself, fed to the
              project's (1/G)-averaged update. Equals jia_lemma / G.
"""
import numpy as np
from scipy.stats import binom

RULES = ["mean", "rloo", "grpo", "maxrl_alg1", "maxrl_appF", "maxrl_eq9", "maxrl_eq10",
         "jia_lemma", "jia_table"]


def adv_from_rewards(rule, r):
    r = np.asarray(r, dtype=float)
    G = r.shape[-1]
    K = r.sum(-1, keepdims=True)
    rbar = K / G
    if rule == "mean":
        return r - rbar
    if rule == "rloo":
        return r - (K - r) / (G - 1)
    if rule == "grpo":
        sd = np.sqrt(((r - rbar) ** 2).mean(-1, keepdims=True))
        return np.divide(r - rbar, sd, out=np.zeros_like(r), where=sd > 0)
    if rule == "maxrl_alg1":
        return np.divide(r - rbar, rbar, out=np.zeros_like(r), where=rbar > 0)
    if rule == "maxrl_appF":
        return (r - rbar) / (rbar + 1e-6)
    if rule == "maxrl_eq9":
        return np.divide(G * r, K, out=np.zeros_like(r), where=K > 0)
    if rule == "maxrl_eq10":
        return np.divide(G * r, K, out=np.zeros_like(r), where=K > 0) - 1.0
    if rule in ("jia_lemma", "jia_table"):
        # Algorithm 1 of Jia et al., literally: subtract min, normalise gaps, centre, skip if no gap.
        g = r - r.min(-1, keepdims=True)
        Z = g.sum(-1, keepdims=True)
        R = np.divide(g, Z, out=np.zeros_like(r), where=Z > 0)
        A = np.where(Z > 0, R - R.mean(-1, keepdims=True), 0.0)
        return G * A if rule == "jia_lemma" else A
    raise ValueError(rule)


def adv_table(rule, G):
    """Ap[k], Am[k] for k = 0..G. Ap[0] and Am[G] are never used (set to 0)."""
    k = np.arange(G + 1, dtype=float)
    Ap = np.zeros(G + 1)
    Am = np.zeros(G + 1)
    mixed = (k >= 1) & (k <= G - 1)
    pos = k >= 1
    if rule == "mean":
        Ap, Am = 1 - k / G, -k / G
    elif rule == "rloo":
        Ap, Am = 1 - (k - 1) / (G - 1), -k / (G - 1)
    elif rule == "grpo":
        sd = np.sqrt(k * (G - k)) / G
        Ap[mixed] = (1 - k[mixed] / G) / sd[mixed]
        Am[mixed] = (-k[mixed] / G) / sd[mixed]
    elif rule == "maxrl_alg1":
        Ap[pos] = G / k[pos] - 1
        Am[pos] = -1.0
    elif rule == "maxrl_appF":
        Ap = (1 - k / G) / (k / G + 1e-6)
        Am = (-k / G) / (k / G + 1e-6)
    elif rule == "maxrl_eq9":
        Ap[pos] = G / k[pos]
    elif rule == "maxrl_eq10":
        Ap[pos] = G / k[pos] - 1
        Am[:] = -1.0
    elif rule == "jia_lemma":
        Ap[mixed] = G / k[mixed] - 1
        Am[mixed] = -1.0
    elif rule == "jia_table":
        Ap[mixed] = 1 / k[mixed] - 1 / G
        Am[mixed] = -1.0 / G
    else:
        raise ValueError(rule)
    Ap = np.array(Ap, dtype=float)
    Am = np.array(Am, dtype=float)
    Ap[0] = 0.0
    Am[G] = 0.0
    return Ap, Am


def W_exact(rule, G, p):
    """W_G(p) = E[A_1 | r_1 = 1] - E[A_1 | r_1 = 0], by enumeration over the other G-1 samples."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    Ap, Am = adv_table(rule, G)
    j = np.arange(G)
    pmf = binom.pmf(j[None, :], G - 1, p[:, None])
    return (pmf * (Ap[1:] - Am[:-1])[None, :]).sum(1)


def W_closed_maxrl(G, p):
    """(1 - (1-p)^(G-1)) / p, stable for small p."""
    p = np.asarray(p, dtype=float)
    return -np.expm1((G - 1) * np.log1p(-p)) / p


def F_closed_maxrl(G, p):
    """sum_{m=1}^{G-1} (1 - (1-p)^m) / m  (Jia Eq. 4; MaxRL Eq. 6 at T = G-1 plus H_{G-1})."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    m = np.arange(1, G)
    return (-np.expm1(m[None, :] * np.log1p(-p[:, None])) / m[None, :]).sum(1)


def harmonic(n):
    return float((1.0 / np.arange(1, n + 1)).sum())
