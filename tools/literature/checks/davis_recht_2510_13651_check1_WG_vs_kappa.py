"""Check 1 for the Davis & Recht (arXiv 2510.13651) prior-art audit.

(a) W_G(p) = E[A|r=1] - E[A|r=0] by brute force over all 2^G reward vectors, for mean-only,
    RLOO and GRPO, G in {2,4,8,16}, against Davis & Recht's kappa(p) (p.5) with matching
    a_s, b_s, and against the team's claimed values.
(b) Each rule has the form Z_i = (1-R_i) a_{S_i} + R_i b_{S_i} (Eq. 4.1) on every reward vector.
(c) Which standard-deviation convention makes "GRPO = 1 at G = 2" true.
(d) Per-sample equivalences in H1.

Run:  python3 davis_recht_2510_13651_check1_WG_vs_kappa.py
"""
import numpy as np

from davis_recht_2510_13651_common import (
    P_GRID, W_enum, ab_grpo, ab_mean_only, ab_rloo, adv_mean_only, adv_rloo,
    all_reward_vectors, fits_form, kappa, make_adv_grpo,
)

np.set_printoptions(precision=6, suppress=True, linewidth=160)
GS = [2, 4, 8, 16]

print("=" * 100)
print("(a)+(b)  brute-force W_G  vs  Davis & Recht kappa  vs  claimed value")
print("=" * 100)
print("p grid:", P_GRID)

for G in GS:
    print(f"\n--- G = {G}  ({2 ** G} reward vectors) ---")

    # mean-only
    a, b = ab_mean_only(G)
    W = W_enum(adv_mean_only, G, P_GRID)
    K = kappa(a, b, P_GRID)
    print(f"mean-only : max|W_enum - kappa| = {np.abs(W - K).max():.2e}   "
          f"max|W_enum - (G-1)/G| = {np.abs(W - (G - 1) / G).max():.2e}   "
          f"form-fit residual = {fits_form(adv_mean_only, a, b, G):.2e}   "
          f"b_s - a_s constant = {np.allclose(b - a, (G - 1) / G)}")

    # RLOO
    a, b = ab_rloo(G)
    W = W_enum(adv_rloo, G, P_GRID)
    K = kappa(a, b, P_GRID)
    print(f"RLOO      : max|W_enum - kappa| = {np.abs(W - K).max():.2e}   "
          f"max|W_enum - 1|       = {np.abs(W - 1).max():.2e}   "
          f"form-fit residual = {fits_form(adv_rloo, a, b, G):.2e}   "
          f"b_s - a_s constant = {np.allclose(b - a, 1.0)}")

    # GRPO, Davis & Recht convention: population variance, eps outside the root
    for eps in (0.0, 1e-8):
        adv = make_adv_grpo(ddof=0, eps=eps, eps_inside=False)
        a, b = ab_grpo(G, ddof=0, eps=eps, eps_inside=False)
        W = W_enum(adv, G, P_GRID)
        K = kappa(a, b, P_GRID)
        print(f"GRPO pop-std eps={eps:g} (outside root): max|W_enum - kappa| = {np.abs(W - K).max():.2e}   "
              f"form-fit residual = {fits_form(adv, a, b, G):.2e}")
        if eps == 0.0:
            print("   W_G(p)                 :", W)
            with np.errstate(divide="ignore"):
                lim = 1 / np.sqrt(P_GRID * (1 - P_GRID))
            print("   1/sqrt(p(1-p))         :", lim)
            print("   W_G(p)*sqrt(p(1-p))    :", W * np.sqrt(P_GRID * (1 - P_GRID)))
            print(f"   W_G(0) = {W[0]:.6f}   sqrt(G-1) = {np.sqrt(G - 1):.6f}")

# large G through the composition formula only (2^G vectors is not feasible)
print("\n--- GRPO (pop-std, eps=0) at large G via kappa: ratio W_G(p) * sqrt(p(1-p)) ---")
interior = np.array([0.01, 0.05, 0.1, 0.25, 0.5])
print("p =", interior)
for G in (16, 64, 256, 1024):
    a, b = ab_grpo(G)
    print(f"G = {G:5d}:", kappa(a, b, interior) * np.sqrt(interior * (1 - interior)))

print()
print("=" * 100)
print("(c)  GRPO at G = 2 under each convention (W is constant in p at G = 2; max deviation from")
print("     constancy over the grid is printed as a check)")
print("=" * 100)
print(f"{'variance':>10} {'eps placement':>16} {'eps':>8} {'W_2':>12} {'closed form':>14} {'spread over p':>14}")
for ddof, name in ((0, "population"), (1, "unbiased")):
    for inside in (False, True):
        for eps in (0.0, 1e-8, 1e-6, 1e-4, 1e-1):
            adv = make_adv_grpo(ddof=ddof, eps=eps, eps_inside=inside)
            W = W_enum(adv, 2, P_GRID)
            sd = 0.5 if ddof == 0 else np.sqrt(0.5)      # std of a mixed pair
            closed = 0.5 / (np.sqrt(sd ** 2 + eps) if inside else sd + eps)
            print(f"{name:>10} {('sqrt(var+eps)' if inside else 'sqrt(var)+eps'):>16} {eps:8.0e} "
                  f"{W[5]:12.8f} {closed:14.8f} {W.max() - W.min():14.2e}")

print("\nUnbiased-variance GRPO equals sqrt((G-1)/G) times population-variance GRPO at eps = 0:")
for G in GS:
    Wp = W_enum(make_adv_grpo(0), G, P_GRID)
    Wu = W_enum(make_adv_grpo(1), G, P_GRID)
    print(f"  G = {G:2d}: max|W_unbiased - sqrt((G-1)/G) W_pop| = {np.abs(Wu - np.sqrt((G - 1) / G) * Wp).max():.2e}"
          f"   factor = {np.sqrt((G - 1) / G):.6f}")

print()
print("=" * 100)
print("(d)  per-sample equivalences in H1 (max over all reward vectors and all samples)")
print("=" * 100)
for G in GS:
    R = all_reward_vectors(G)
    d = np.abs(adv_rloo(R) - G / (G - 1) * adv_mean_only(R)).max()
    print(f"G = {G:2d}: max|A_RLOO - G/(G-1) A_mean-only| = {d:.2e}")
R = all_reward_vectors(2)
for ddof, name in ((0, "population"), (1, "unbiased")):
    d = np.abs(make_adv_grpo(ddof)(R) - adv_rloo(R)).max()
    print(f"G =  2: max|A_GRPO({name} std, eps=0) - A_RLOO| = {d:.6f}")
for eps in (1e-8, 1e-6):
    d = np.abs(make_adv_grpo(0, eps, False)(R) * (1 + 2 * eps) - adv_rloo(R)).max()
    print(f"G =  2: max|(1+2 eps) A_GRPO(pop std, sqrt(var)+eps, eps={eps:g}) - A_RLOO| = {d:.2e}")
