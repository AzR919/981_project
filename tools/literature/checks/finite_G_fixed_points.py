"""Finite-G GRPO KL fixed points: existence, count and stability.

W_G(p) from Davis & Recht Eq. (4.2) with population std and eps = 0 (an all-equal group gets advantage 0).
Fixed point of KL(pi||pi_ref):  logit p - logit p_ref = W_G(p)/beta.
Fixed point of KL(pi_ref||pi):  (p - p_ref)/(p(1-p)) = W_G(p)/beta.
"""
import numpy as np
from math import comb, sqrt

def W(p, G):
    if G == np.inf:
        return 1 / np.sqrt(p * (1 - p))
    tot = 0.0
    for s in range(G):                       # s = correct among the other G-1 samples
        qa, qb = s / G, (s + 1) / G          # group success rate if this sample is wrong / right
        b = 0.0 if qb in (0, 1) else (1 - qb) / sqrt(qb * (1 - qb))
        a = 0.0 if qa in (0, 1) else -qa / sqrt(qa * (1 - qa))
        tot += (b - a) * comb(G - 1, s) * p**s * (1 - p)**(G - 1 - s)
    return tot

p = np.linspace(1e-6, 1 - 1e-6, 200001)
logit = lambda x: np.log(x / (1 - x))
print("max_p W_G vs sqrt(G-1):", [(G, round(max(W(p[::500], G)), 4), round(sqrt(G - 1), 4)) for G in (2, 4, 8, 16, 64)])
print("W_G at p->0 and p=1/2:", [(G, round(W(1e-9, G), 4), round(W(0.5, G), 4)) for G in (2, 4, 8, 16, 64)])

def roots(f):
    v = f(p); i = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    return [(round(p[k], 4), "stable" if v[k] > 0 else "unstable") for k in i]   # f = RHS - LHS; + to - crossing is stable

print("\nKL(pi||pi_ref): roots of W_G(p)/beta - (logit p - logit p_ref)")
for pref in (0.01, 0.05, 0.2):
    for beta in (0.25, 0.5, 1, 2, 4):
        row = []
        for G in (2, 4, 8, 16, 64, np.inf):
            Wv = np.array([W(x, G) for x in p[::20]]) if G != np.inf else W(p[::20], G)
            pp = p[::20]; v = Wv / beta - (logit(pp) - logit(pref))
            i = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
            row.append(f"G={G}:" + ",".join(f"{pp[k]:.3f}{'s' if v[k] > 0 else 'u'}" for k in i) if len(i) else f"G={G}:none(->1)")
        print(f"p_ref={pref} beta={beta}: " + "  ".join(row))
