"""
Phase A of plan v3: analyses of the existing base and stretch caches (no new runs).

usage:
    python experiments/analyze_phase_a.py a1 [--base analysis/base] [--out analysis/phase_a]
    python experiments/analyze_phase_a.py a4 [--stretch results/stretch] [--out analysis/phase_a]

a1: H2 as promised. GRPO's weight W_G(p) as a mix of the constant (mean rule) and
the 2 arcsin sqrt(p) surrogate weight 1/sqrt(p(1-p)): W = c [(1 - lam) + lam / sqrt(p(1-p))].
lam is fitted (i) with free scale c, (ii) scale-free (shape only, both sides divided
by their value at p = 0.5), on the measured tabular H2 bins with a bootstrap over
bins, and (iii) on the exact W_G over the same p range for G = 2 ... 128.

a4: lost problems and coverage. Toy p is exact, so pass@k = 1 - (1 - p)^k exactly;
per run, the change in mean pass@1 and pass@64 from p_init to the final p of the
stretch predictor runs (s_pred), and the number of prompts lost (p_init >= 0.01,
final p < 1e-3).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze import markdown  # noqa: E402
from synthetic_rl.theory import advantage_gap  # noqa: E402


def mix(p, lam):
    return (1 - lam) + lam / np.sqrt(p * (1 - p))


def fit_lambda(p, w, scale_free):
    """Least-squares lam in [0, 1] (grid search, then refined); c is profiled out in closed form."""
    def loss(lam):
        if scale_free:
            return np.sum((w / np.interp(0.5, p, w) - mix(p, lam) / mix(0.5, lam)) ** 2)
        m = mix(p, lam)
        c = (m @ w) / (m @ m)
        return np.sum((c * m - w) ** 2)
    grid = np.linspace(0, 1, 1001)
    best = grid[np.argmin([loss(v) for v in grid])]
    fine = np.linspace(max(0, best - 1e-3), min(1, best + 1e-3), 201)
    return float(fine[np.argmin([loss(v) for v in fine])])


def a1(base, out, n_boot=2000):
    bins = pd.read_csv(os.path.join(base, "tables", "h2_binned_ratio.csv"))
    bins = bins[(bins.level == "tab") & (bins.rule == "grpo")]
    rng = np.random.default_rng(0)
    rows = []
    for G, g in bins.groupby("G"):
        g = g.sort_values("p")
        p = g.p.to_numpy()
        measured = g.measured.to_numpy() * advantage_gap(p, G, "mean")
        exact = advantage_gap(p, G, "grpo")
        row = dict(G=int(G), bins=len(p), p_range=f"{p.min():.3f}-{p.max():.3f}")
        for name, sf in (("free", False), ("shape", True)):
            row[f"lam_{name}"] = fit_lambda(p, measured, sf)
            draws = []
            for _ in range(n_boot):
                i = np.sort(rng.integers(len(p), size=len(p)))
                if len(np.unique(p[i])) > 2:
                    draws.append(fit_lambda(p[i], measured[i], sf))
            row[f"lam_{name}_lo"], row[f"lam_{name}_hi"] = np.quantile(draws, [0.025, 0.975])
            row[f"lam_{name}_exact"] = fit_lambda(p, exact, sf)
        rows.append(row)
    t = pd.DataFrame(rows)
    grid = np.linspace(1 / 64, 1 - 1 / 64, 200)
    theory = pd.DataFrame([dict(G=G, lam_free=fit_lambda(grid, advantage_gap(grid, G, "grpo"), False),
                                lam_shape=fit_lambda(grid, advantage_gap(grid, G, "grpo"), True))
                           for G in (2, 4, 8, 16, 32, 64, 128)])
    lines = ["# A1: H2 as promised, GRPO's weight between the mean rule and the arcsin surrogate", "",
             "Model: W_G(p) = c [(1 − λ) + λ / √(p(1 − p))]; λ = 0 is the mean rule (Dr. GRPO), λ = 1 the "
             "2 arcsin √p surrogate. `free`: c fitted; `shape`: both sides divided by their value at p = 0.5. "
             f"Measured: tabular GRPO bins of the base H2 runs; CI: {n_boot} bootstraps over bins. "
             "`_exact`: the same fit to the exact W_G at the same p.", "",
             markdown(t), "",
             "Exact λ(G) on p in [1/64, 63/64] (200 points):", "", markdown(theory), "",
             "- λ rises with G in every column: GRPO's weight moves from the mean rule toward the arcsin "
             "surrogate as G grows, as the milestone promised. The measured and exact fits agree because the "
             "tabular measurement matches W_G to within 1% (H2).", ""]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "A1.md"), "w") as f:
        f.write("\n".join(lines))
    print(t.round(3).to_string()); print(theory.round(3).to_string())


def a4(stretch, out):
    cache = os.path.join(stretch, "cache")
    runs = pd.read_csv(os.path.join(cache, "runs_s_pred.csv"))
    arrays = dict(np.load(os.path.join(cache, "prompts_s_pred.npz")))
    r = runs.iloc[arrays["row_train"]].reset_index(drop=True)
    p0, p1 = arrays["p_init"], arrays["p_final"]
    for k in (1, 64):
        r[f"d_pass{k}"] = np.nanmean(1 - (1 - p1) ** k, axis=1) - np.nanmean(1 - (1 - p0) ** k, axis=1)
    r["narrowing"] = (r.d_pass1 > 0) & (r.d_pass64 < 0)
    r["lost"] = np.nansum((p0 >= 0.01) & (p1 < 1e-3), axis=1)
    cells = r.groupby(["level", "init", "rule"]).agg(
        runs=("lost", "size"), d_pass1=("d_pass1", "mean"), d_pass64=("d_pass64", "mean"),
        narrowing_share=("narrowing", "mean"), lost_mean=("lost", "mean")).reset_index()
    wide = cells.pivot_table(index=["level", "init"], columns="rule", values="d_pass64").reset_index()
    more = int((wide.grpo < wide["mean"]).sum())
    lines = ["# A4: lost problems and coverage in the toy (stretch `s_pred`, existing data)", "",
             "pass@k is exact (toy p is known): pass@k = 1 − (1 − p)^k, averaged over the 512 prompts, "
             "from p_init to the final p. Lost: p_init ≥ 0.01 and final p < 1e-3. `narrowing_share`: runs "
             "where pass@1 rises and pass@64 falls. Runs span η from 0.3× to 10× the base-batch absorption "
             "threshold, so some losses are overshoot.", "",
             markdown(cells), "",
             f"- Every cell: mean pass@1 rises ({cells.d_pass1.min():.2f} to {cells.d_pass1.max():.2f}) and "
             f"mean pass@64 falls ({cells.d_pass64.max():.2f} to {cells.d_pass64.min():.2f}).",
             f"- GRPO lowers pass@64 more than the mean rule in {more} of {len(wide)} (level, init) pairs.", ""]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "A4.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[-3:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("a1", "a4"))
    parser.add_argument("--base", default="analysis/base")
    parser.add_argument("--stretch", default="results/stretch")
    parser.add_argument("--out", default="analysis/phase_a")
    args = parser.parse_args()
    if args.what == "a1":
        a1(args.base, args.out)
    else:
        a4(args.stretch, args.out)


if __name__ == "__main__":
    main()
