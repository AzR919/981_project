"""Check 1 for MaxRL (2602.02710) + reward budgeting (2611.05942).

(a) The two papers' rules, compared reward vector by reward vector.
(b) W_G(p) by exact enumeration, two independent ways, against (1 - (1-p)^(G-1))/p.
(c) F_G(p) = int_0^p W_G by numerical integration, against the closed form, p, and log p.
(d) The same for GRPO against 2*arcsin(sqrt(p)).

Run:  python3 maxrl_rb_check1_W_F.py
"""
import itertools
import numpy as np
from scipy.integrate import quad
from maxrl_rb_rules import (RULES, adv_from_rewards, adv_table, W_exact, W_closed_maxrl,
                            F_closed_maxrl, harmonic)

GS = [2, 4, 8, 16, 64]
PS = np.array([0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
np.set_printoptions(linewidth=200)


def all_reward_vectors(G):
    return np.array(list(itertools.product([0, 1], repeat=G)), dtype=float)


def W_bruteforce(rule, G, p):
    """E[(1/G) sum_i A_i S_i] / grad p over all 2^G reward vectors, one-logit Bernoulli policy
    (S_i = r_i - p, grad p = p(1-p)). Uses adv_from_rewards, not the table."""
    R = all_reward_vectors(G)
    A = adv_from_rewards(rule, R)
    K = R.sum(1)
    out = []
    for q in np.atleast_1d(p):
        prob = q ** K * (1 - q) ** (G - K)
        ghat = (A * (R - q)).mean(1)
        out.append((prob * ghat).sum() / (q * (1 - q)))
    return np.array(out)


print("=" * 100)
print("(a) Are the rules the same function of the reward vector?  max |A_rule1 - A_rule2| over all 2^G vectors")
print("=" * 100)
for G in [2, 4, 8, 16]:
    R = all_reward_vectors(G)
    A = {rule: adv_from_rewards(rule, R) for rule in RULES}
    print(f"G={G:3d}  maxrl_alg1 vs jia_lemma: {np.abs(A['maxrl_alg1'] - A['jia_lemma']).max():.2e}"
          f"   maxrl_alg1 vs G*jia_table: {np.abs(A['maxrl_alg1'] - G * A['jia_table']).max():.2e}"
          f"   maxrl_alg1 vs maxrl_appF: {np.abs(A['maxrl_alg1'] - A['maxrl_appF']).max():.2e}"
          f"   maxrl_alg1 vs maxrl_eq10: {np.abs(A['maxrl_alg1'] - A['maxrl_eq10']).max():.2e}")
    if G == 2:
        print(f"        at G=2: maxrl_alg1 vs grpo: {np.abs(A['maxrl_alg1'] - A['grpo']).max():.2e}"
              f"   vs rloo: {np.abs(A['maxrl_alg1'] - A['rloo']).max():.2e}"
              f"   vs 2*mean: {np.abs(A['maxrl_alg1'] - 2 * A['mean']).max():.2e}")
    # table and from-rewards implementations agree?
    K = R.sum(1).astype(int)
    for rule in RULES:
        Ap, Am = adv_table(rule, G)
        T = np.where(R == 1, Ap[K][:, None], Am[K][:, None])
        assert np.abs(T - A[rule]).max() < 1e-12, (rule, G)
print("(table and from-rewards implementations agree for every rule and G <= 16: assert passed)")
print("Jia Table 1 reproduced (G=8, advantage of correct / incorrect sample, gap, 1000*gap):")
Ap, Am = adv_table("jia_table", 8)
Gp, Gm = adv_table("grpo", 8)
for k in range(1, 8):
    print(f"   k={k}:  RB {Ap[k]:+.2f} {Am[k]:+.2f} gap {Ap[k]-Am[k]:.4f} kappa {1000*(Ap[k]-Am[k]):6.0f}"
          f"   | GRPO {Gp[k]:+.2f} {Gm[k]:+.2f} gap {Gp[k]-Gm[k]:.4f} kappa {1000*(Gp[k]-Gm[k]):6.0f}")

print()
print("=" * 100)
print("(b) W_G(p): enumeration over compositions vs brute force over 2^G vectors vs closed form")
print("=" * 100)
closed = {
    "maxrl_alg1": lambda G, p: W_closed_maxrl(G, p),
    "maxrl_appF": lambda G, p: W_closed_maxrl(G, p),      # up to O(1e-6)
    "jia_lemma": lambda G, p: W_closed_maxrl(G, p),
    "jia_table": lambda G, p: W_closed_maxrl(G, p) / G,
    "maxrl_eq9": lambda G, p: W_closed_maxrl(G + 1, p),   # order N = G
    "maxrl_eq10": lambda G, p: W_closed_maxrl(G + 1, p),
    "mean": lambda G, p: np.full_like(p, (G - 1) / G),
    "rloo": lambda G, p: np.ones_like(p),
}
for rule in ["maxrl_alg1", "maxrl_appF", "jia_lemma", "jia_table", "maxrl_eq9", "maxrl_eq10", "mean", "rloo"]:
    for G in GS:
        w_comp = W_exact(rule, G, PS)
        w_cl = closed[rule](G, PS)
        err_cl = np.abs(w_comp / w_cl - 1).max()
        if G <= 16:
            w_bf = W_bruteforce(rule, G, PS)
            err_bf = np.abs(w_bf / w_comp - 1).max()
            bf = f"{err_bf:.1e}"
        else:
            bf = "   n/a "
        print(f"{rule:11s} G={G:3d}  max rel.err composition-vs-closed {err_cl:.1e}   brute-vs-composition {bf}")
print()
print("W_G(p) for the MaxRL / reward-budgeting rule, project convention (maxrl_alg1 = jia_lemma):")
print("    p      " + "".join(f"   G={G:<6d}" for G in GS) + "   1/p (ML)")
for q in PS:
    print(f"  {q:6.3f}  " + "".join(f" {W_exact('maxrl_alg1', G, q)[0]:10.4f}" for G in GS) + f" {1/q:10.2f}")
print("same rule with Jia's advantage scale fed to a (1/G)-mean update (jia_table) = the row above / G:")
for q in PS:
    print(f"  {q:6.3f}  " + "".join(f" {W_exact('jia_table', G, q)[0]:10.4f}" for G in GS))
print("W_G(p) for GRPO (eps = 0, population std):")
print("    p      " + "".join(f"   G={G:<6d}" for G in GS) + "   1/sqrt(p(1-p))")
for q in PS:
    print(f"  {q:6.3f}  " + "".join(f" {W_exact('grpo', G, q)[0]:10.4f}" for G in GS) + f" {1/np.sqrt(q*(1-q)):10.4f}")
print("limits p->0+ (evaluated at p=1e-9):   rule: " + ", ".join(
    f"G={G}: mean {W_exact('mean', G, 1e-9)[0]:.3f} grpo {W_exact('grpo', G, 1e-9)[0]:.3f} "
    f"[sqrt(G-1)={np.sqrt(G-1):.3f}] maxrl {W_exact('maxrl_alg1', G, 1e-9)[0]:.3f} [G-1={G-1}]" for G in GS))
print("limits p->1- (evaluated at p=1-1e-9): rule: " + ", ".join(
    f"G={G}: grpo {W_exact('grpo', G, 1-1e-9)[0]:.3f} maxrl {W_exact('maxrl_alg1', G, 1-1e-9)[0]:.3f}" for G in GS))

print()
print("=" * 100)
print("(c) F_G(p) = int_0^p W_G for the MaxRL / RB rule: quadrature of the enumerated W vs closed form")
print("    closed form  F_G(p) = sum_{m=1}^{G-1} (1-(1-p)^m)/m = H_{G-1} + J^{(G-1)}_MaxRL(p)")
print("=" * 100)
for G in GS:
    Fq = np.array([quad(lambda t: W_exact("maxrl_alg1", G, t)[0], 0, q, epsabs=1e-12, epsrel=1e-12, limit=200)[0] for q in PS])
    Fc = F_closed_maxrl(G, PS)
    print(f"G={G:3d}  max |quad - closed| = {np.abs(Fq - Fc).max():.2e}   F_G(1) = H_(G-1) = {harmonic(G-1):.4f}")
print()
print("F_G(p) - H_{G-1}  (so that the G->inf limit is log p), next to p - 1 (the G=2 value) and log p:")
print("    p      " + "".join(f"   G={G:<6d}" for G in GS) + "      log p")
for q in PS:
    print(f"  {q:6.3f}  " + "".join(f" {F_closed_maxrl(G, q)[0] - harmonic(G-1):10.4f}" for G in GS) + f" {np.log(q):10.4f}")
print("normalised F_G(p)/F_G(1)  (0 at p=0, 1 at p=1), next to p:")
for q in PS:
    print(f"  {q:6.3f}  " + "".join(f" {F_closed_maxrl(G, q)[0] / harmonic(G-1):10.4f}" for G in GS))
print("sup over p in [1e-4,1] of |F_G(p) - H_{G-1} - log p| and the p where F_G - H_{G-1} is within 0.01 of log p:")
grid = np.logspace(-4, 0, 4001)
for G in GS:
    d = np.abs(F_closed_maxrl(G, grid) - harmonic(G - 1) - np.log(grid))
    ok = grid[d < 0.01]
    print(f"  G={G:3d}  sup = {d.max():.3f} (at p={grid[d.argmax()]:.4f});  within 0.01 of log p for p >= {ok.min() if ok.size else float('nan'):.3f}")

print()
print("=" * 100)
print("(d) GRPO finite-G surrogate F_G(p) = int_0^p W_G^GRPO, against p (G=2) and 2*arcsin(sqrt(p)) (G->inf)")
print("=" * 100)
print("    p      " + "".join(f"   G={G:<6d}" for G in GS) + "  2asin(sqrt p)")
FG = {}
for G in GS:
    FG[G] = np.array([quad(lambda t: W_exact("grpo", G, t)[0], 0, q, epsabs=1e-11, epsrel=1e-11, limit=400)[0] for q in list(PS) + [1.0]])
for i, q in enumerate(PS):
    print(f"  {q:6.3f}  " + "".join(f" {FG[G][i]:10.4f}" for G in GS) + f" {2*np.arcsin(np.sqrt(q)):10.4f}")
print("  F_G(1)   " + "".join(f" {FG[G][-1]:10.4f}" for G in GS) + f" {np.pi:10.4f}")
print("normalised F_G(p)/F_G(1), GRPO, next to (2/pi)*arcsin(sqrt(p)):")
for i, q in enumerate(PS):
    print(f"  {q:6.3f}  " + "".join(f" {FG[G][i]/FG[G][-1]:10.4f}" for G in GS) + f" {2/np.pi*np.arcsin(np.sqrt(q)):10.4f}")
print()
print("Ratio W_G^MaxRL / W_G^GRPO (how much more MaxRL weights a prompt than GRPO at the same p, G):")
for q in PS:
    print(f"  {q:6.3f}  " + "".join(f" {W_exact('maxrl_alg1', G, q)[0] / W_exact('grpo', G, q)[0]:10.4f}" for G in GS))
print()
print("Identity used in section 3/5 of the review: for a rule centred within the group,")
print("  W_G(p) = E_k[ rbar(1-rbar) * Delta(k) ] / (p(1-p)),  Delta(k) = A+(k) - A-(k)  (Jia's per-group gap)")
from scipy.stats import binom
for rule in ["mean", "grpo", "maxrl_alg1", "jia_table"]:
    worst = 0.0
    for G in GS:
        Ap, Am = adv_table(rule, G)
        k = np.arange(G + 1)
        for q in PS:
            lhs = W_exact(rule, G, q)[0]
            rhs = (binom.pmf(k, G, q) * (k / G) * (1 - k / G) * (Ap - Am)).sum() / (q * (1 - q))
            worst = max(worst, abs(lhs / rhs - 1))
    print(f"  {rule:11s} max rel.err over G, p: {worst:.1e}")
