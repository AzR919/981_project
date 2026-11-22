"""Shared helpers for the checks on Vojnovic & Yun, arXiv 2502.18548.

Conventions (the paper's, Eq. 2, p. 2):
  A_i = (r_i - mean) / std,  std = sqrt((1/G) sum (r_i - mean)^2)   (1/G, i.e. ddof = 0)
  A_i = 0 when all rewards in the group are equal; no epsilon inside the std.

Everything is parameterised by z = logit p so that p and q = 1 - p stay accurate near 0 and 1.
"""
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, gammaln

INF = float("inf")


def _log_binom_pmf(n, j, logp, logq):
    j = np.asarray(j, dtype=float)
    return gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1) + j * logp + (n - j) * logq


def log_p_q(z):
    """log p and log(1-p) for p = expit(z), accurate for large |z|."""
    return -np.logaddexp(0.0, -z), -np.logaddexp(0.0, z)


def gap_terms(G, eps=0.0, ddof=0):
    """Per-composition conditional advantages.

    j = number of correct responses among the OTHER G-1 group members.
    a[j] = advantage of a correct response, b[j] = minus the advantage of a wrong response,
    so that W_G(p) = sum_j Binom(G-1, p)(j) * (a[j] + b[j]).
    """
    j = np.arange(G)
    # the response we condition on is correct: k = j + 1 correct in the group
    k1 = j + 1
    var1 = k1 * (G - k1) / G**2 * (G / (G - ddof))
    num1 = 1.0 - k1 / G
    # the response we condition on is wrong: k = j correct in the group
    k0 = j
    var0 = k0 * (G - k0) / G**2 * (G / (G - ddof))
    num0 = k0 / G  # minus (0 - mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(var1 > 0, num1 / np.sqrt(var1 + eps), 0.0)
        b = np.where(var0 > 0, num0 / np.sqrt(var0 + eps), 0.0)
    return a, b


def W(z, G, eps=0.0, ddof=0):
    """Advantage gap W_G(p) = E[A | r=1] - E[A | r=0] at p = expit(z).

    G = INF gives the population limit 1/sqrt(p(1-p) + eps).
    """
    logp, logq = log_p_q(z)
    if G == INF:
        return 1.0 / math.sqrt(math.exp(logp + logq) + eps)
    a, b = gap_terms(G, eps, ddof)
    pmf = np.exp(_log_binom_pmf(G - 1, np.arange(G), logp, logq))
    return float(np.dot(pmf, a + b))


def cond_adv(z, G, eps=0.0, ddof=0):
    """(E[A | r=1], E[A | r=0]) at p = expit(z), finite G."""
    logp, logq = log_p_q(z)
    a, b = gap_terms(G, eps, ddof)
    pmf = np.exp(_log_binom_pmf(G - 1, np.arange(G), logp, logq))
    return float(np.dot(pmf, a)), -float(np.dot(pmf, b))


def drift(z, zref, beta, G, kl, eps=0.0):
    """Expected update of the logit z of a two-outcome softmax policy, per unit step size.

    kl = "theta||ref": penalty beta * KL(pi_theta || pi_ref)
         d/dz = p(1-p) * [ W - beta * (z - zref) ]
    kl = "ref||theta": penalty beta * KL(pi_ref || pi_theta)
         d/dz = p(1-p) * W - beta * (p - pref)
    Zero drift is the fixed-point equation; drift going from + to - is a stable root.
    """
    logp, logq = log_p_q(z)
    pq = math.exp(logp + logq)
    w = W(z, G, eps)
    if kl == "theta||ref":
        return pq * (w - beta * (z - zref))
    if kl == "ref||theta":
        # p - pref computed as q_ref - q when p is close to 1, to avoid cancellation
        p, q = expit(z), expit(-z)
        pref, qref = expit(zref), expit(-zref)
        diff = (p - pref) if p < 0.5 else (qref - q)
        return pq * w - beta * diff
    raise ValueError(kl)


def scaled_residual(z, zref, beta, G, kl, eps=0.0):
    """A positive multiple of drift() that does not underflow near p = 0 or 1."""
    if kl == "theta||ref":
        return W(z, G, eps) - beta * (z - zref)
    logp, logq = log_p_q(z)
    p, q = expit(z), expit(-z)
    pref, qref = expit(zref), expit(-zref)
    diff = (p - pref) if p < 0.5 else (qref - q)
    # divide the drift by p(1-p):  W - beta (p - pref) / (p (1-p))
    return W(z, G, eps) - beta * diff * math.exp(-(logp + logq))


def roots(zref, beta, G, kl, eps=0.0, zmin=-40.0, zmax=None, n=6000):
    """All interior roots of the fixed-point equation on a logit grid.

    Returns a list of (z, p, 1-p, 'stable' | 'unstable').
    """
    if zmax is None:
        if G == INF:
            zmax = 60.0
        else:
            # every root has z <= zref + sqrt(G-1)/beta under KL(theta||ref); pad generously
            zmax = max(60.0, zref + math.sqrt(G - 1) / beta + 10.0)
    zs = np.linspace(zmin, zmax, n)
    f = np.array([scaled_residual(z, zref, beta, G, kl, eps) for z in zs])
    out = []
    for i in range(n - 1):
        if not (np.isfinite(f[i]) and np.isfinite(f[i + 1])):
            continue
        if f[i] == 0.0 or f[i] * f[i + 1] < 0:
            r = brentq(scaled_residual, zs[i], zs[i + 1], args=(zref, beta, G, kl, eps), xtol=1e-13)
            out.append((r, float(expit(r)), float(expit(-r)), "stable" if f[i] > 0 else "unstable"))
    return out


def logit(p):
    return math.log(p) - math.log1p(-p)


# ---- closed forms printed in the paper (deterministic rewards: gamma_{a,b} = 1) ----

def eq14(pref, beta, gamma=1.0):
    """Eq. 14, p. 10: G = 2, GRPO penalty (acts as KL(pi_ref || pi_theta))."""
    c = beta / gamma
    return 0.5 * (1 - c + math.sqrt((1 - c) ** 2 + 4 * c * pref))


def eq15(pref, beta):
    """Eq. 15, p. 11: G -> infinity, GRPO penalty (acts as KL(pi_ref || pi_theta))."""
    return (2 * beta**2 * pref + 1 + math.sqrt(1 + 4 * beta**2 * pref * (1 - pref))) / (2 * (1 + beta**2))


def direct_kl_g2(pref, beta, gamma=1.0):
    """p. 13 / App. B.2 p. 19: G = 2, penalty KL(pi_theta || pi_ref)."""
    up = math.exp(gamma / (2 * beta)) * pref
    dn = math.exp(-gamma / (2 * beta)) * (1 - pref)
    return up / (up + dn)
