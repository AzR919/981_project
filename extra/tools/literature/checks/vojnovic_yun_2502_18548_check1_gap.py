"""Check 1 for Vojnovic & Yun 2502.18548: the advantage gap W_G(p) and the paper's P_G.

(a) W_G(p) for GRPO by three independent routes:
      - sum over group compositions (binomial over the other G-1 members);
      - brute force over all 2^G reward vectors with numpy's std (ddof=0), literal Eq. 2 (G <= 16);
      - the identity W_G(p) p(1-p) = E[std of the group's rewards];
    plus Monte Carlo at G = 64.
(b) The paper's P_G(o | pi) (p. 6) evaluated literally for a policy with several outputs per
    reward class: is P_G constant on each class, is E_{o~pi} P_G = 0, and is
    P_G(correct) - P_G(wrong) = W_G(p)?
(c) Is the solution of Eq. 11 a zero of the exact expected GRPO update of a tabular softmax
    policy, with (i) GRPO's own penalty estimator at pi_old = pi and (ii) exact KL(pi_ref||pi)?
    Same for the p. 13 formula with exact KL(pi||pi_ref).
(d) Bound W_G <= sqrt(G-1), end values, W_3, and the G -> infinity limit.
(e) Effect of the std convention (1/(G-1)) and of an epsilon inside the square root.
"""
import itertools
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import binom

import vojnovic_yun_2502_18548_lib as L

GS = [2, 4, 8, 16, 64]
PS = [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]


def W_bruteforce(p, G):
    """Literal Eq. 2 over all reward vectors, conditioning on r_1."""
    e1 = e0 = 0.0
    for rest in itertools.product([0, 1], repeat=G - 1):
        pr = p ** sum(rest) * (1 - p) ** (G - 1 - sum(rest))
        for r1 in (0, 1):
            r = np.array((r1,) + rest, dtype=float)
            s = r.std()  # ddof = 0, the paper's convention
            a = 0.0 if s == 0 else (r[0] - r.mean()) / s
            if r1:
                e1 += pr * a
            else:
                e0 += pr * a
    return e1 - e0


def W_from_std(p, G):
    k = np.arange(G + 1)
    return float(np.dot(binom.pmf(k, G, p), np.sqrt(k * (G - k)) / G)) / (p * (1 - p))


def W_montecarlo(p, G, n=400_000, seed=0):
    rng = np.random.default_rng(seed)
    r = (rng.random((n, G)) < p).astype(float)
    s = r.std(axis=1)
    a = np.where(s > 0, (r[:, 0] - r.mean(axis=1)) / np.where(s > 0, s, 1), 0.0)
    m1 = a[r[:, 0] == 1].mean()
    m0 = a[r[:, 0] == 0].mean()
    se = math.sqrt(a[r[:, 0] == 1].var() / (r[:, 0] == 1).sum() + a[r[:, 0] == 0].var() / (r[:, 0] == 0).sum())
    return m1 - m0, se


def part_a():
    print("=== (a) W_G(p), GRPO, std with 1/G, A = 0 on uniform groups, no epsilon ===")
    print("G     " + "".join(f"p={p:<9}" for p in PS))
    worst_bf = worst_std = 0.0
    for G in GS:
        row = []
        for p in PS:
            w = L.W(L.logit(p), G)
            row.append(w)
            worst_std = max(worst_std, abs(w - W_from_std(p, G)))
            if G <= 16:
                worst_bf = max(worst_bf, abs(w - W_bruteforce(p, G)))
        print(f"{G:<5d} " + "".join(f"{w:<11.6f}" for w in row))
    print("inf   " + "".join(f"{1 / math.sqrt(p * (1 - p)):<11.6f}" for p in PS))
    print(f"max |composition sum - brute force over 2^G reward vectors|, G<=16: {worst_bf:.2e}")
    print(f"max |composition sum - E[std]/(p(1-p))|, all G: {worst_std:.2e}")
    for p in (0.05, 0.5):
        mc, se = W_montecarlo(p, 64)
        print(f"Monte Carlo G=64 p={p}: {mc:.4f} +- {se:.4f}   exact {L.W(L.logit(p), 64):.4f}")


def P_G_literal(pi, r, G):
    """Paper p. 6: P_G(o | pi) = E over o'_1..o'_{G-1} iid pi of (r(o) - mean)/std."""
    n = len(pi)
    P = np.zeros(n)
    for others in itertools.product(range(n), repeat=G - 1):
        pr = float(np.prod(pi[list(others)]))
        ro = r[list(others)]
        for o in range(n):
            grp = np.concatenate(([r[o]], ro))
            s = grp.std()
            if s > 0:
                P[o] += pr * (r[o] - grp.mean()) / s
    return P


def part_b():
    print("\n=== (b) the paper's P_G(o|pi) for a 5-output policy, 2 correct and 3 wrong outputs ===")
    r = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    for pi in (np.array([0.03, 0.02, 0.5, 0.3, 0.15]), np.array([0.4, 0.3, 0.1, 0.15, 0.05])):
        p = pi[r == 1].sum()
        for G in (2, 4, 8):
            P = P_G_literal(pi, r, G)
            spread = max(np.ptp(P[r == 1]), np.ptp(P[r == 0]))
            gap = P[0] - P[2]
            print(
                f"p={p:.2f} G={G}: P_G(correct)={P[0]:+.6f} P_G(wrong)={P[2]:+.6f} "
                f"within-class spread={spread:.1e} E_pi[P_G]={np.dot(pi, P):+.1e} "
                f"gap={gap:.6f} W_G(p)={L.W(L.logit(p), G):.6f} diff={gap - L.W(L.logit(p), G):+.1e}"
            )


def expected_grpo_grad(pi, r, G):
    """Exact E[(1/G) sum_i A_i grad_theta log pi(o_i)] for a tabular softmax, all o_i iid pi.

    By exchangeability this is E_{o~pi}[P_G(o|pi) (e_o - pi)].
    """
    P = P_G_literal(pi, r, G)
    return pi * P - pi * np.dot(pi, P)


def part_c():
    print("\n=== (c) is the Eq. 11 / p. 13 solution a zero of the exact expected update? ===")
    r = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    ref = np.array([0.03, 0.02, 0.5, 0.3, 0.15])
    pref = ref[r == 1].sum()
    for G, beta in ((4, 2.0), (8, 0.5), (4, 0.3)):
        # --- GRPO penalty, Eq. 11: class-level equation, then pi(o) = pi_ref(o) / (1 - P_G(o)/beta)
        f = lambda z: L.scaled_residual(z, L.logit(pref), beta, G, "ref||theta")
        zs = np.linspace(-20, 20, 4001)
        fs = [f(z) for z in zs]
        i = next(i for i in range(len(zs) - 1) if fs[i] * fs[i + 1] < 0)
        p = float(expit(brentq(f, zs[i], zs[i + 1], xtol=1e-14)))
        Pc, Pw = L.cond_adv(L.logit(p), G)
        pi = np.where(r == 1, ref / (1 - Pc / beta), ref / (1 - Pw / beta))
        P = P_G_literal(pi, r, G)
        res11 = (1 - (P - np.dot(pi, P)) / beta) * pi - ref  # Eq. 11 residual, literal
        g_reward = expected_grpo_grad(pi, r, G)
        # (i) GRPO's estimator D_i sampled from pi_old = pi: grad_theta E_{o~pi_old}[ref/pi - log(ref/pi) - 1]
        #     = E_{pi_old}[(1 - ref/pi) grad log pi] = sum_o (pi_o - ref_o)(e_o - pi) = pi - ref
        g_pen_estimator = pi - ref
        # (ii) exact KL(ref || pi): grad_theta = -sum_o ref_o (e_o - pi) = pi - ref   (identical)
        g_pen_exact = pi - ref
        print(
            f"GRPO penalty   G={G} beta={beta}: p*={p:.6f} sum(pi)={pi.sum():.12f} "
            f"max|Eq.11 residual|={np.abs(res11).max():.1e} "
            f"max|E[g] - beta*grad penalty|={np.abs(g_reward - beta * g_pen_estimator).max():.1e} "
            f"within-class ratios kept: {np.allclose(pi[:2] / ref[:2], pi[0] / ref[0]) and np.allclose(pi[2:] / ref[2:], pi[2] / ref[2])}"
        )
        assert np.allclose(g_pen_estimator, g_pen_exact)
        # --- KL(pi || ref), p. 13: pi(o) = ref(o) exp(P_G(o|pi)/beta) / Z
        rts = L.roots(L.logit(pref), beta, G, "theta||ref")
        p = rts[0][1]
        Pc, Pw = L.cond_adv(L.logit(p), G)
        pi = ref * np.exp(np.where(r == 1, Pc, Pw) / beta)
        Z = pi.sum()
        pi = pi / Z
        P = P_G_literal(pi, r, G)
        res13 = pi - ref * np.exp(P / beta) / np.sum(ref * np.exp(P / beta))
        g_reward = expected_grpo_grad(pi, r, G)
        logratio = np.log(pi / ref)
        g_pen = pi * (logratio - np.dot(pi, logratio))  # grad_theta KL(pi || ref) for a softmax
        print(
            f"KL(pi||ref)    G={G} beta={beta}: p*={p:.6f} class mass={pi[r == 1].sum():.6f} "
            f"max|p.13 residual|={np.abs(res13).max():.1e} "
            f"max|E[g] - beta*grad KL|={np.abs(g_reward - beta * g_pen).max():.1e}"
        )


def part_d():
    print("\n=== (d) bound, end values, W_3, large-G limit ===")
    worst = -1.0
    for G in range(2, 3001):
        a, b = L.gap_terms(G)
        worst = max(worst, float((a + b).max() - math.sqrt(G - 1)))
    print(f"max over G in 2..3000 of [max_j (a_j + b_j) - sqrt(G-1)] = {worst:.2e}  (<= 0 means the bound holds)")
    for G in (2, 3, 4, 8, 16, 64):
        a, b = L.gap_terms(G)
        zs = np.linspace(-12, 12, 2401)
        ws = np.array([L.W(z, G) for z in zs])
        print(
            f"G={G:<3d} sqrt(G-1)={math.sqrt(G - 1):.4f}  W_G(p->0)={L.W(-40, G):.4f}  W_G(p->1)={L.W(40, G):.4f}  "
            f"min_p W_G={ws.min():.4f} at p={expit(zs[ws.argmin()]):.3f}  max_p W_G={ws.max():.4f}  "
            f"min_j(a_j+b_j)={float((a + b).min()):.4f}"
        )
    print("G -> infinity: W_G(p) * sqrt(p(1-p)) at fixed p")
    for p in (0.05, 0.2, 0.5):
        print(f"  p={p}: " + "  ".join(f"G={G}: {L.W(L.logit(p), G) * math.sqrt(p * (1 - p)):.4f}" for G in (4, 16, 64, 256, 1024, 4096)))
    print("  non-uniform: sup_p W_G = sqrt(G-1) for every G, while sup over [d,1-d] of the limit is finite")


def part_e():
    print("\n=== (e) other conventions (not the paper's) ===")
    for G in (2, 4, 16):
        z = L.logit(0.2)
        print(
            f"G={G:<2d} p=0.2: 1/G std: {L.W(z, G):.5f}   1/(G-1) std: {L.W(z, G, ddof=1):.5f} "
            f"(ratio {L.W(z, G, ddof=1) / L.W(z, G):.5f}, sqrt((G-1)/G)={math.sqrt((G - 1) / G):.5f})   "
            f"eps=1e-4 inside sqrt: {L.W(z, G, eps=1e-4):.5f}   eps=1e-2: {L.W(z, G, eps=1e-2):.5f}"
        )


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
