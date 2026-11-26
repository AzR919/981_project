"""
Reproduce the main-session checks behind extra/analysis/review/SUMMARY.md and the
plan critique, from the result caches.

usage (from 981_project/, after `analyze.py collect` for base and stretch):
    python3 extra/analysis/review/checks/verify_summary.py

Each section prints the numbers quoted in SUMMARY.md.
"""

import collections
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))  # repo root (extra/analysis/review/checks/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import run  # noqa: E402
from synthetic_rl.theory import advantage_gap  # noqa: E402

BASE = os.path.join(ROOT, "results", "base", "cache")
STRETCH = os.path.join(ROOT, "results", "stretch", "cache")


def drift_runs():
    runs = pd.read_csv(os.path.join(STRETCH, "runs_s_drift.csv"))
    arrays = np.load(os.path.join(STRETCH, "prompts_s_drift.npz"))
    return runs.iloc[arrays["row_train"]].reset_index(drop=True), arrays


def minority_collapse():
    """Item 1: absorbed prompts on c09/c05 are exactly minority-answer prompts."""
    print("== 1. absorbed prompts against the minority-answer set (expected_full, mid-grid eta)")
    runs, arrays = drift_runs()
    for level, rho in (("c09", 0.9), ("c05", 0.5)):
        for seed in range(5):
            cfg = dict(run.DEFAULTS, arch="linear", features="correlated", rho=rho, L=2, seed=seed, name="x")
            _, train, _, _ = run.build(cfg)
            answers = [tuple(a) for a in train[1]]
            majority = collections.Counter(answers).most_common(1)[0][0]
            minority = np.array([a != majority for a in answers])
            sel = np.where(((runs.level == level) & (runs.seed == seed) & (runs["update"] == "expected_full")
                            & (runs.block == "main")).to_numpy())[0]
            j = sel[len(sel) // 2]
            absorbed = arrays["p_final"][j][:512] < 0.01
            print(f"{level} task {seed} eta={runs.eta[j]:.3g}: minority share {minority.mean():.3f}, "
                  f"absorbed {absorbed.sum()}, absorbed with majority answer {(absorbed & ~minority).sum()}, "
                  f"share of minority absorbed {(absorbed & minority).sum() / minority.sum():.2f}")


def flow_time():
    """Item 2: c09 absorption happens at a fixed flow time eta * t."""
    print("== 2. c09 median absorption step and eta * t50 (expected_full, >= 5% absorbed)")
    runs, _ = drift_runs()
    m = runs[(runs.block == "main") & (runs.level == "c09") & (runs["update"] == "expected_full") & (runs.absorbed >= 0.05)]
    g = m.groupby("eta").t50.median().reset_index()
    g["eta_x_t50"] = g.eta * g.t50
    print(g.round(4).to_string(index=False))


def kick_size():
    """Item 3: threshold ratio expected_full / expected (B = 32) against n / B = 16."""
    print("== 3. eta* (median absorbed >= 0.1) by update mode, B = 32")
    runs, _ = drift_runs()
    med = runs[runs.block == "main"].groupby(["level", "update", "eta"]).absorbed.median().reset_index()
    for level in ("m64", "lin", "c05"):
        star = {}
        for update, g in med[med.level == level].groupby("update"):
            hit = g[g.absorbed >= 0.1].eta
            star[update] = hit.min() if len(hit) else np.nan
        print(level, star, f"ratio full/expected = {star['expected_full'] / star['expected']:.1f}")
    c = runs[(runs.block == "main") & (runs.level == "c09") & (runs["update"] == "sampled") & np.isclose(runs.eta, 0.01)]
    print("c09 sampled eta=0.01 absorbed by seed:", c.sort_values("seed")[["seed", "absorbed"]].values.tolist())


def h5_ties():
    """Item 4: tabular best-G ties in the base H5 table."""
    print("== 4. H5 tabular: G values tied at the minimum samples-to-target")
    best = pd.read_csv(os.path.join(ROOT, "analysis", "base", "tables", "h5_best_eta.csv"))
    tab = best[(best.level == "tab") & ~best.censored]
    for (p0, rule), g in tab.groupby(["p0", "rule"]):
        low = g.samples.min()
        print(f"{rule} p0={p0}: min {low:.0f} samples at G = {sorted(g[g.samples == low].G.tolist())}")


def arcsin_share():
    """Item 5: GRPO weight as a mix of constant and 1/sqrt(p(1-p)), free scale."""
    from scipy.optimize import least_squares
    print("== 5. H2 arcsin share lambda(G), tabular GRPO bins")
    bins = pd.read_csv(os.path.join(ROOT, "analysis", "base", "tables", "h2_binned_ratio.csv"))
    bins = bins[(bins.level == "tab") & (bins.rule == "grpo")]
    for G, g in bins.groupby("G"):
        p = g.p.to_numpy()
        measured = g.measured.to_numpy() * advantage_gap(p, G, "mean")
        exact = advantage_gap(p, G, "grpo")
        fit = least_squares(lambda w: w[0] * ((1 - w[1]) + w[1] / np.sqrt(p * (1 - p))) - measured,
                            [1, 0.5], bounds=([0, 0], [np.inf, 1])).x
        print(f"G={G}: lambda={fit[1]:.3f}, median |measured/exact - 1| = {100 * np.median(np.abs(measured / exact - 1)):.3f}%")


def critique_numbers():
    """Plan-critique checks: the 0/128 false-loss rate and with-replacement coverage."""
    print("== 6. plan critique")
    for p in (0.1, 0.05, 0.02):
        print(f"P(0 of 128 correct | p={p}) = {(1 - p) ** 128:.4g}")
    print(f"distinct prompts when drawing 512 of 512 with replacement: {1 - (1 - 1 / 512) ** 512:.3f}")


if __name__ == "__main__":
    for check in (minority_collapse, flow_time, kick_size, h5_ties, arcsin_share, critique_numbers):
        check()
        print()
