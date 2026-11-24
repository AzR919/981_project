"""
Yang (arXiv 2609.22547), check 2: the no-change null for "locked" prompts, from the released per-prompt counts.

usage:
    python tools/literature/checks/yang_2609_22547_check2_exact_null.py <dir with base.jsonl and rl.jsonl>

Data: github.com/cafferychen777/passk-inference, examples/deepscaler32k (MIT): 1,060 prompts, c_i correct of
n = 128 for the base model and d_i of 128 for the RL model. A prompt is "locked" if c_i > 0 and d_i = 0.

Question: how many locked prompts would there be if the RL model were the base model (no change)?
Under no change, d_i ~ Binomial(n, p_i) with the same p_i that produced c_i, so

    P(d_i = 0 | c_i) = E[(1 - p)^n | c_i],

an average over the prompts' success rates given the base count. The distribution G of p is estimated from the
base counts alone in three ways (the low tail is only weakly identified, so the spread across them is the
honest uncertainty): the nonparametric maximum-likelihood mixture on a grid, a Beta fitted by maximum
likelihood, and the Beta(a, 1) priors assumed in check 1.

A test that needs no G is also reported: under no change the pair (c_i, d_i) is exchangeable, so locked and
"newly solved" (c_i = 0, d_i > 0) prompts are equally likely.

Also reported: the same counts thinned to 64, 32 and 16 samples per prompt (hypergeometric thinning), to show
how the locked count and its no-change expectation depend on the number of samples.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln
from scipy.stats import betabinom, binomtest


def load(folder):
    rows = {}
    for name in ("base", "rl"):
        for line in open(os.path.join(folder, f"{name}.jsonl")):
            r = json.loads(line)
            rows.setdefault(r["id"], {})[name] = (r["c"], r["n"])
    ids = sorted(rows)
    c = np.array([rows[i]["base"][0] for i in ids])
    d = np.array([rows[i]["rl"][0] for i in ids])
    n = {rows[i][k][1] for i in ids for k in ("base", "rl")}
    assert len(n) == 1
    return c, d, n.pop()


def log_binom(c, n, p):
    """log Binomial(c; n, p) for arrays c (rows) and p (columns), with p = 0 and p = 1 allowed."""
    c, p = np.asarray(c)[:, None], np.asarray(p)[None, :]
    coef = gammaln(n + 1) - gammaln(c + 1) - gammaln(n - c + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = coef + np.where(c > 0, c * np.log(p), 0.0) + np.where(n - c > 0, (n - c) * np.log1p(-p), 0.0)
    return np.where(np.isnan(out), -np.inf, out)


def npmle(c, n, iters=5000):
    grid = np.unique(np.concatenate([[0.0], np.logspace(-6, np.log10(0.5), 300), np.linspace(0.5, 1.0, 101)]))
    like = np.exp(log_binom(c, n, grid))
    w = np.full(len(grid), 1 / len(grid))
    for _ in range(iters):
        w = w * (like / (like @ w)[:, None]).mean(axis=0)
    return grid, w


def null_share_mixture(c_values, n, n_after, grid, w):
    """P(d = 0 | c) under the mixture (grid, w), for each base count in c_values."""
    like = np.exp(log_binom(c_values, n, grid)) * w
    with np.errstate(divide="ignore"):
        zero = np.where(grid < 1, np.exp(n_after * np.log1p(-np.minimum(grid, 1 - 1e-300))), 0.0)
    return (like * zero).sum(axis=1) / like.sum(axis=1)


def null_share_beta(c_values, n, n_after, a, b):
    """P(d = 0 | c) when p ~ Beta(a, b): B(a + c, b + n - c + n') / B(a + c, b + n - c)."""
    c = np.asarray(c_values)
    return np.exp(betaln(a + c, b + n - c + n_after) - betaln(a + c, b + n - c))


def fit_beta(c, n):
    nll = lambda t: -betabinom.logpmf(c, n, np.exp(t[0]), np.exp(t[1])).sum()  # noqa: E731
    best = minimize(nll, x0=[np.log(0.3), np.log(0.5)], method="Nelder-Mead")
    return float(np.exp(best.x[0])), float(np.exp(best.x[1]))


def report(c, d, n, title):
    locked = (c > 0) & (d == 0)
    rescued = (c == 0) & (d > 0)
    print(f"\n{title}: n = {n} samples per prompt per model, {len(c)} prompts")
    print(f"  base never solves (c = 0): {(c == 0).sum()};  RL never solves (d = 0): {(d == 0).sum()}")
    print(f"  observed locked (c > 0, d = 0): {locked.sum()}   observed newly solved (c = 0, d > 0): {rescued.sum()}")
    # Under no change the pair (c, d) is exchangeable, so a prompt with exactly one zero count is as likely
    # to be locked as newly solved. This needs no assumption about the distribution of p.
    test = binomtest(int(locked.sum()), int(locked.sum() + rescued.sum()), 0.5, alternative="greater")
    print(f"  prior-free test (locked against newly solved, one-sided binomial): p = {test.pvalue:.4f}")
    grid, w = npmle(c, n)
    a, b = fit_beta(c, n)
    priors = {"NPMLE": lambda cv: null_share_mixture(cv, n, n, grid, w),
              f"Beta MLE (a = {a:.3f}, b = {b:.3f})": lambda cv: null_share_beta(cv, n, n, a, b)}
    for a0 in (0.25, 0.5, 1.0):
        priors[f"Beta({a0}, 1), as in check 1"] = lambda cv, a0=a0: null_share_beta(cv, n, n, a0, 1.0)
    solved = c[c > 0]
    strata = [("c = 1", 1, 1), ("c = 2", 2, 2), ("c = 3-4", 3, 4), ("c = 5-8", 5, 8), ("c > 8", 9, n)]
    print("  expected locked prompts under no change, with a Poisson-binomial sd:")
    for name, share in priors.items():
        q = share(solved)
        print(f"    {name:32s} {q.sum():6.1f} +- {np.sqrt((q * (1 - q)).sum()):.1f}")
    print("  by base count (observed locked / prompts; expected under no change with the NPMLE and the Beta MLE):")
    for label, lo, hi in strata:
        m = (c >= lo) & (c <= hi)
        if not m.any():
            continue
        e1 = priors["NPMLE"](c[m]).sum()
        e2 = null_share_beta(c[m], n, n, a, b).sum()
        print(f"    {label:8s} {int(locked[m].sum()):3d} / {int(m.sum()):3d}    NPMLE {e1:5.1f}    Beta MLE {e2:5.1f}")
    mass = w[grid < 1 / n].sum()
    print(f"  NPMLE mass on p < 1/n: {mass:.3f} (of which at p = 0: {w[0]:.3f})")


def main():
    c, d, n = load(sys.argv[1])
    report(c, d, n, "Released counts (DeepScaleR-1.5B against its base, 32k-token limit)")
    rng = np.random.default_rng(0)
    for size in (64, 32, 16):
        tallies, nulls = [], []
        for _ in range(20):
            cs, ds = rng.hypergeometric(c, n - c, size), rng.hypergeometric(d, n - d, size)
            tallies.append(int(((cs > 0) & (ds == 0)).sum()))
            a, b = fit_beta(cs, size)
            nulls.append(float(null_share_beta(cs[cs > 0], size, size, a, b).sum()))
        print(f"\nThinned to {size} samples per prompt (20 draws): observed locked {np.mean(tallies):.1f} "
              f"(range {min(tallies)}-{max(tallies)}); expected under no change (Beta MLE) {np.mean(nulls):.1f}")


if __name__ == "__main__":
    main()
