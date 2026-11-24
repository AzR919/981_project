"""
Reduced class-level ODE for phase 1 (the fall) of the linear policy under the exact expected GRPO update.

Theory: analysis/theory/fall.md. Two reductions of the exact flow (experiments/exact_flow.py):

- "class" model: every prompt of class a is replaced by its class-mean feature phi_a, weighted by the class share
  f_a. With W = W0 + sum_a C_a phi_a^T the flow is exactly
      dC_a/dtau = f_a w(p_a) (e_a - pi_a),   pi_a = softmax(W0 phi_a + sum_b k_ab C_b),
  with k_ab = phi_a . phi_b the block kernel means (they equal the means of k(x, y) over the class blocks) and
  w(p) = W_G(p) p. State: U x K numbers (U classes with prompts, K answers).
- "sym" model: the same with the minority classes pooled (class-symmetric state) and the four block means k_AA,
  k_AB, k_same, k_diff. State: five log-odds (u_A, v_A, L_B, r_B, j_B); see fall.md for the equations.

Per-prompt "test particles": prompt y in the class model's field has logits W0 phi_y + sum_a (phi_a . phi_y) C_a(tau).

usage:
    from experiments.fall_ode import predict
    out = predict(cfg)          # cfg: a run config (dict); integrates to tau = eta * steps
"""

import hashlib
import json
import math
import os

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.special import log_softmax, logsumexp

from experiments.run import DEFAULTS, build
from synthetic_rl.theory import gap_coefficients

TASK_KEYS = ("K", "L", "d", "features", "rho", "answers", "n_train", "n_test", "arch", "width", "init", "p0",
             "spread", "mix", "sft_steps", "sft_lr", "majority", "used_classes", "task_seed", "seed")
ABSORB, TAIL = 1e-3, 0.2          # absorbed: p < 1e-3 over the last 20% of the horizon (as in the analyses)
CACHE = None                      # optional on-disk cache directory for features(); an in-memory memo is always used
_MEMO = {}


def features(cfg, cache=CACHE):
    """(phi, cls, W0, K, U) for a run config: last-layer features (linear: the inputs; MLP: last hidden layer plus a
    constant 1 for the output bias), correct class per prompt, initial output weights (K x d'), number of answers
    and number of classes that have prompts. Cached on disk by the task-defining keys."""
    full = dict(DEFAULTS, **cfg)
    if full["L"] != 1:
        raise ValueError("fall_ode covers one answer token")
    key = {k: full.get(k) for k in TASK_KEYS}
    memo = json.dumps(key, sort_keys=True)
    if memo in _MEMO:
        return _MEMO[memo]
    path = None
    if cache:
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16] + ".npz")
        if os.path.exists(path):
            z = np.load(path)
            return z["phi"], z["cls"], z["W0"], int(z["K"]), int(z["U"])
    _, train, _, policy = build(full)
    with torch.no_grad():
        phi = policy.features(torch.as_tensor(train[0], dtype=torch.float64)).numpy()
        W0 = policy.output.weight.detach().numpy().copy()
        if policy.output.bias is not None:
            phi = np.concatenate([phi, np.ones((len(phi), 1))], axis=1)
            W0 = np.concatenate([W0, policy.output.bias.detach().numpy()[:, None]], axis=1)
    cls = np.asarray(train[1][:, 0])
    K, U = full["K"], full["used_classes"] or full["K"]
    if path:
        np.savez(path, phi=phi, cls=cls, W0=W0, K=K, U=U)
    _MEMO.clear()
    _MEMO[memo] = (phi, cls, W0, K, U)
    return phi, cls, W0, K, U


def blocks(phi, cls, U):
    """Class-mean features (U x d), class shares, block kernel means k_ab = phi_a . phi_b, and the pooled
    class-symmetric means (k_AA, k_AB, k_same, k_diff)."""
    means = np.stack([phi[cls == a].mean(0) for a in range(U)])
    shares = np.array([(cls == a).mean() for a in range(U)])
    k = means @ means.T
    M = U - 1
    off = k[1:, 1:][~np.eye(M, dtype=bool)]
    pooled = dict(k_AA=k[0, 0], k_AB=k[0, 1:].mean(), k_same=np.diag(k)[1:].mean(),
                  k_diff=off.mean() if M > 1 else 0.0)
    return means, shares, k, pooled


def weight_fn(cfg):
    """w(p) = W_G(p) p, with W_G from synthetic_rl.theory (coefficients computed once, evaluated in log space)."""
    full = dict(DEFAULTS, **cfg)
    G, rule, passk = int(full["G"]), full["rule"], full["passk"]
    b, a = gap_coefficients(G, rule, passk=passk)
    s = np.arange(G)
    log_binom = np.array([math.lgamma(G) - math.lgamma(j + 1) - math.lgamma(G - j) for j in s])

    def w(p):
        p = np.atleast_1d(np.asarray(p, dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            lp, lq = np.log(p)[:, None], np.log1p(-p)[:, None]
            lw = log_binom + np.where(s > 0, s * lp, 0.0) + np.where(G - 1 - s > 0, (G - 1 - s) * lq, 0.0)
        return (np.exp(lw) * (b - a)).sum(axis=1) * p

    return w


def _log_grid(tau0, T, num):
    return np.exp(np.linspace(np.log(tau0), np.log(T), num))


def _integrate(rhs, y0, T, num, tau0=1e-6, rtol=1e-8, atol=1e-10):
    """Integrate dy/dtau = rhs(y) from tau = 0 to T in log time (one Euler step to tau0 first, as exact_flow)."""
    y1 = y0 + tau0 * rhs(y0)
    taus = _log_grid(tau0, T, num)
    sol = solve_ivp(lambda u, y: np.exp(u) * rhs(y), (np.log(tau0), np.log(T)), y1, t_eval=np.log(taus),
                    method="LSODA", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return np.exp(sol.t), sol.y.T


# ---------------------------------------------------------------- class-symmetric model (five log-odds)

def sym_probs(y, M, J):
    """Probabilities of the pooled state y = (u_A, v_A, L_B, r_B, j_B).
    A prompts: logits relative to A's answer (0, -u_A on each of M minority answers, -v_A on each of J junk).
    B prompts: logits relative to their own answer (0, L_B on A, r_B on each other minority answer, j_B on junk)."""
    uA, vA, LB, rB, jB = y
    termsA = [0.0, -uA + math.log(M)] + ([-vA + math.log(J)] if J else [])
    lzA = logsumexp(termsA)
    termsB = [0.0, LB] + ([rB + math.log(M - 1)] if M > 1 else []) + ([jB + math.log(J)] if J else [])
    lzB = logsumexp(termsB)
    sA, tA = math.exp(-uA - lzA), (math.exp(-vA - lzA) if J else 0.0)
    qB = math.exp(LB - lzB)
    oB = math.exp(rB - lzB) if M > 1 else 0.0
    iB = math.exp(jB - lzB) if J else 0.0
    return dict(log_pA=-lzA, log_pB=-lzB, sA=sA, tA=tA, qA_off=M * sA + J * tA,
                qB=qB, oB=oB, iB=iB, qB_off=qB + (M - 1) * oB + J * iB)


def sym_rhs(m, M, J, kAA, kAB, ks, kd, w):
    """Right-hand side of the class-symmetric ODE (fall.md, section 2)."""
    fB = (1 - m) / M

    def rhs(y):
        s = sym_probs(y, M, J)
        pA, pB = math.exp(s["log_pA"]), math.exp(s["log_pB"])
        gA, gB = m * w(pA)[0], fB * w(pB)[0]
        offA, offB = s["qA_off"], s["qB_off"]                    # 1 - p_A and 1 - p_B without cancellation
        qB, oB, iB, sA, tA = s["qB"], s["oB"], s["iB"], s["sA"], s["tA"]
        duA = gA * kAA * (offA + sA) - gB * kAB * (M * qB + offB - (M - 1) * oB)
        dvA = gA * kAA * (offA + tA) - gB * kAB * M * (qB - iB) if J else 0.0
        dLB = gA * kAB * (offA + sA) - gB * ks * (offB + qB) - gB * kd * (M - 1) * (qB - oB)
        drB = -gB * (ks - kd) * (offB + oB) if M > 1 else 0.0
        djB = gA * kAB * (sA - tA) - gB * ks * (offB + iB) + gB * kd * (M - 1) * (oB - iB) if J else 0.0
        return np.array([duA, dvA, dLB, drB, djB])

    return rhs


def sym_start(Z, U, K):
    """Pooled log-odds from the class-mean logits Z (U x K) at the start."""
    M, junk = U - 1, list(range(U, K))
    mins = list(range(1, U))
    uA = np.mean([Z[0, 0] - Z[0, b] for b in mins])
    vA = np.mean([Z[0, 0] - Z[0, j] for j in junk]) if junk else 0.0
    LB = np.mean([Z[b, 0] - Z[b, b] for b in mins])
    rB = np.mean([Z[b, c] - Z[b, b] for b in mins for c in mins if c != b]) if M > 1 else 0.0
    jB = np.mean([Z[b, j] - Z[b, b] for b in mins for j in junk]) if junk else 0.0
    return np.array([uA, vA, LB, rB, jB])


def solve_sym(pooled, m, U, K, w, Z0, T, num=400):
    M, J = U - 1, K - U
    rhs = sym_rhs(m, M, J, pooled["k_AA"], pooled["k_AB"], pooled["k_same"], pooled["k_diff"], w)
    taus, ys = _integrate(rhs, sym_start(Z0, U, K), T, num)
    pr = [sym_probs(y, M, J) for y in ys]
    return taus, np.array([p["log_pA"] for p in pr]), np.array([p["log_pB"] for p in pr]), \
        np.array([p["qB"] for p in pr])


# ---------------------------------------------------------------- class model (U x K) and test particles

def class_rhs(k, shares, Z0, w):
    U, K = Z0.shape
    E = np.eye(K)[:U]

    def rhs(c):
        C = c.reshape(U, K)
        logpi = log_softmax(Z0 + k @ C, axis=1)
        pi = np.exp(logpi)
        p = np.exp(logpi[np.arange(U), np.arange(U)])
        return ((shares * w(p))[:, None] * (E - pi)).ravel()

    return rhs


def solve_class(k, shares, Z0, w, T, num=400):
    U, K = Z0.shape
    taus, cs = _integrate(class_rhs(k, shares, Z0, w), np.zeros(U * K), T, num)
    C = cs.reshape(len(taus), U, K)
    logp = log_softmax(Z0[None] + np.einsum("ab,tbk->tak", k, C), axis=2)[:, np.arange(U), np.arange(U)]
    return taus, C, logp


def particles(phi, cls, W0, means, C):
    """log success of every prompt in the class model's field: (n_taus x n)."""
    base = phi @ W0.T                                   # n x K
    kx = phi @ means.T                                   # n x U
    out = np.empty((len(C), len(phi)))
    for t, Ct in enumerate(C):
        out[t] = log_softmax(base + kx @ Ct, axis=1)[np.arange(len(phi)), cls]
    return out


# ---------------------------------------------------------------- summaries

def summary(taus, logp, T):
    """Minimum (log10), time of the minimum, value at the end and the largest value over the last 20%."""
    i = int(np.argmin(logp))
    tail = taus >= (1 - TAIL) * T
    return dict(log10_min=float(logp[i] / np.log(10)), tau_min=float(taus[i]),
                log10_end=float(logp[-1] / np.log(10)), log10_tail_max=float(logp[tail].max() / np.log(10)),
                absorbed=bool(logp[tail].max() < np.log(ABSORB)))


def predict(cfg, T=None, num=400, with_particles=True, cache=CACHE):
    """Reduced-model predictions for one run config, integrated to tau = eta * steps (or T)."""
    full = dict(DEFAULTS, **cfg)
    T = float(T if T is not None else full["eta"] * full["steps"])
    phi, cls, W0, K, U = features(cfg, cache=cache)
    means, shares, k, pooled = blocks(phi, cls, U)
    w = weight_fn(cfg)
    Z0 = means @ W0.T
    out = dict(T=T, U=U, K=K, share_a=float(shares[0]), gamma_block=float(pooled["k_AB"] / pooled["k_AA"]),
               **{k_: float(v) for k_, v in pooled.items()})
    M = U - 1
    out["c_odds"] = float(M * pooled["k_AB"] / (M * pooled["k_same"] - (M - 1) * pooled["k_diff"]))

    taus, lA, lB, qB = solve_sym(pooled, shares[0], U, K, w, Z0, T, num)
    out.update({f"sym_{a}": b for a, b in summary(taus, lB, T).items()})
    out["sym_log10_pA_end"] = float(lA[-1] / np.log(10))
    out["sym_qB_max"] = float(qB.max())

    taus, C, logp = solve_class(k, shares, Z0, w, T, num)
    lB_geo = logp[:, 1:].mean(1)                         # geometric mean over the minority classes
    out.update({f"cls_{a}": b for a, b in summary(taus, lB_geo, T).items()})

    if with_particles:
        lp = particles(phi, cls, W0, means, C)
        minority = cls != 0
        mins = lp[:, minority].min(0) / np.log(10)
        tail = taus >= (1 - TAIL) * T
        tmax = lp[tail][:, minority].max(0) / np.log(10)
        out["tp_log10_min_median"] = float(np.median(mins))
        out["tp_absorbed_share"] = float((tmax < np.log10(ABSORB)).mean())
        out["tp_log10_min"] = mins                       # per minority prompt (in prompt order)
        out["tp_log10_tail_max"] = tmax
    return out
