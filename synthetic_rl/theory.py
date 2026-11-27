"""
Exact predictions for binary rewards: advantage gap, surrogate, informative
fraction and KL fixed points.

For any rule whose advantage depends only on a sample's own reward and the
number s of correct samples among the other G - 1 (Davis & Recht 2025,
Eq. 4.1-4.2), the expected update is W_G(p) * grad p with

    W_G(p) = sum_s (b_s - a_s) * C(G-1, s) * p^s * (1 - p)^(G-1-s),

where b_s is the advantage of a correct sample and a_s that of an incorrect
one. b_s and a_s are read off calculate_advantages() itself, so the theory
always matches the trainer's implementation.

KL fixed points (one prompt, two outcomes, flow in the logit z of p):
    KL(pi || pi_ref), "reverse":  logit p - logit p_ref = W_G(p) / beta
    KL(pi_ref || pi), "forward":  (p - p_ref) / (p (1 - p)) = W_G(p) / beta
"""

import functools
import math

import numpy as np
import torch

from .trainer import calculate_advantages


def gap_coefficients(group_size, rule, passk=None, epsilon=1e-5):
    """Return (b, a), each of length G: advantages of a correct and of an
    incorrect sample when s = 0..G-1 of the other samples are correct."""
    G = group_size
    # Row c has c correct samples first, then G - c incorrect ones.
    rewards = torch.zeros(G + 1, G, dtype=torch.float64)
    for c in range(G + 1):
        rewards[c, :c] = 1.0
    A = calculate_advantages(rewards, kind=rule, epsilon=epsilon, passk=passk)
    A = A.numpy()

    s = np.arange(G)
    b = A[s + 1, 0]       # this sample correct, s others correct
    a = A[s, G - 1]       # this sample incorrect, s others correct
    return b, a


@functools.lru_cache(maxsize=None)
def _gap_terms(G, rule, passk, epsilon):
    """b - a and log C(G-1, s), cached: the expected updates call advantage_gap at every step."""
    b, a = gap_coefficients(G, rule, passk=passk, epsilon=epsilon)
    log_binom = np.array([math.lgamma(G) - math.lgamma(k + 1) - math.lgamma(G - k) for k in range(G)])
    gap = b - a
    gap.flags.writeable = log_binom.flags.writeable = False
    return gap, log_binom


def advantage_gap(p, group_size, rule, passk=None, epsilon=1e-5):
    """W_G(p) = E[A | r=1] - E[A | r=0] for each value in p.

    group_size may be math.inf for the large-group limit of the rules that
    have one (mean, rloo, grpo).
    """
    p = np.asarray(p, dtype=float)

    if group_size == "inf" or group_size == math.inf:   # "inf": a JSON config can carry it as a string
        if rule in ("mean", "rloo"):
            return np.ones_like(p)
        if rule == "grpo":
            return 1.0 / np.sqrt(p * (1 - p) + epsilon)
        raise ValueError(f"No large-group limit implemented for {rule}")

    G = int(group_size)
    gap, log_binom = _gap_terms(G, rule, passk, epsilon)
    s = np.arange(G)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(p)[..., None]
        log_q = np.log1p(-p)[..., None]
        log_w = (
            log_binom
            + np.where(s > 0, s * log_p, 0.0)
            + np.where(G - 1 - s > 0, (G - 1 - s) * log_q, 0.0)
        )
    return (np.exp(log_w) * gap).sum(axis=-1)


def surrogate(p_grid, group_size, rule, passk=None, epsilon=1e-5):
    """F_G(p) = integral of W_G from 0 to p, on an increasing grid in (0, 1)."""
    p_grid = np.asarray(p_grid, dtype=float)
    W = advantage_gap(p_grid, group_size, rule, passk=passk, epsilon=epsilon)
    F = np.concatenate([[0.0], np.cumsum(0.5 * (W[1:] + W[:-1]) * np.diff(p_grid))])
    return F + W[0] * p_grid[0]


def informative_fraction(p, group_size):
    """phi = probability that a group contains both outcomes."""
    p = np.asarray(p, dtype=float)
    return 1 - p**group_size - (1 - p) ** group_size


def logit(p):
    return np.log(p) - np.log1p(-p)


def fixed_points(p_ref, beta, group_size, rule, direction="reverse",
                 passk=None, epsilon=1e-5, z_range=(-40.0, 40.0), num=40001):
    """Return [(p, stable), ...] for the per-prompt KL fixed-point equation.

    direction "reverse" is a penalty beta * KL(pi || pi_ref);
    "forward" is beta * KL(pi_ref || pi).
    Roots are found on a grid in the logit, so points within 1e-17 of 1 are
    still resolved. A root is stable when the flow dz/dt changes sign from
    positive to negative.
    """
    z = np.linspace(*z_range, num)
    p = 1 / (1 + np.exp(-z))
    q = 1 / (1 + np.exp(z))  # 1 - p without cancellation near p = 1
    W = advantage_gap(p, group_size, rule, passk=passk, epsilon=epsilon)

    if direction == "reverse":
        pull = z - logit(p_ref)
    elif direction == "forward":
        pull = (p - p_ref) / (p * q)
    else:
        raise ValueError("direction must be 'reverse' or 'forward'")

    flow = W / beta - pull
    sign_change = np.where(np.sign(flow[:-1]) != np.sign(flow[1:]))[0]

    roots = []
    for i in sign_change:
        # Linear interpolation in z between the two grid points.
        t = flow[i] / (flow[i] - flow[i + 1])
        z_root = z[i] + t * (z[i + 1] - z[i])
        roots.append((float(1 / (1 + np.exp(-z_root))), bool(flow[i] > 0)))
    return roots
