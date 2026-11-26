"""
Analysis of v3j (T22: junk answers), implementing PREREG_v3d_addendum4.md.

usage:
    python extra/experiments/analyze_v3j.py [--raw ~/rlvr-data/v3j] [--out extra/analysis/v3j]

Two answer classes have prompts (class 0, the majority with share m, and class 1); the other K - 2 tokens are
never correct. R = -(change of the minority's mean success under a majority-only step) / (change under a
minority-only step) at step 0; m* is where the median change under the actual step crosses zero.
The prediction is computed from each task's last-layer features with nothing fitted (check_odds_law.predict).
"""

import argparse
import glob
import gzip
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # repo root
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]  # extra/

from experiments.analyze import markdown  # noqa: E402
from experiments.analyze_v3e import crossing  # noqa: E402
from experiments.check_odds_law import predict  # noqa: E402
from experiments.run import DEFAULTS, build  # noqa: E402


def read_one(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    cfg = d["config"]
    small = {k: v for k, v in cfg.items() if k in DEFAULTS or k == "task_seed"}
    _, train, _, _ = build(dict(DEFAULTS, **small))
    cls = train[1][:, 0]
    p0, tail = np.asarray(d["p_init"]), np.asarray(d["tail_max"])
    b = d.get("branches", {}).get("0", {})
    row = dict(K=cfg["K"], arch=cfg["arch"], update=cfg["update"], m=cfg["majority"], task_seed=cfg["task_seed"],
               junk_prompts=int((cls > 1).sum()),
               absorbed_3=float(((tail < 1e-3) & (p0 >= 1e-2))[cls == 1].mean()),
               absorbed_2=float((tail < 1e-2)[cls == 1].mean()),
               majority_final=d["history"][-1]["class_mean"][0], minority_final=d["history"][-1]["class_mean"][1],
               d_b_a=b.get("d_b_a", np.nan), d_b_b=b.get("d_b_b", np.nan), d_b_mixed=b.get("d_b_mixed", np.nan))
    if cfg["update"] == "expected_full":
        pred = predict(small)
        row.update(pred_d_b_a=pred["pred_d_b_a"], pred_d_b_b=pred["pred_d_b_b"], pred_d_b_mixed=pred["pred_d_b_mixed"])
    return row


def threshold(m, y):
    """Zero crossing of y over m, or a note when the sign never changes on the grid."""
    c = crossing(m, y)
    if c == c:
        return c, f"{c:.3f}"
    return np.nan, "above 0.98" if (y > 0).all() else "below 0.5" if (y <= 0).all() else "no clean crossing"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=os.path.expanduser("~/rlvr-data/v3j"))
    parser.add_argument("--out", default="extra/analysis/v3j")
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.raw, "*", "results", "*", "*.json.gz")))
    with Pool(8) as pool:
        runs = pd.DataFrame(pool.map(read_one, paths, chunksize=8))
    os.makedirs(args.out, exist_ok=True)
    runs.to_csv(os.path.join(args.out, "runs.csv"), index=False)
    exact = runs[runs["update"] == "expected_full"].copy()
    exact["R"], exact["pred_R"] = -exact.d_b_a / exact.d_b_b, -exact.pred_d_b_a / exact.pred_d_b_b

    rows, within, order_ok = [], [], []
    for (arch, K), g in exact.groupby(["arch", "K"]):
        med = g.groupby("m").agg(d=("d_b_mixed", "median"), pred=("pred_d_b_mixed", "median"), R=("R", "median"),
                                 pred_R=("pred_R", "median"), a3=("absorbed_3", "mean"), a2=("absorbed_2", "mean"),
                                 maj=("majority_final", "median"), mino=("minority_final", "median")).sort_index()
        m = med.index.to_numpy()
        got, got_text = threshold(m, med.d.to_numpy())
        want, want_text = threshold(m, med.pred.to_numpy())
        ok = abs(got - want) <= 0.02 if got == got and want == want else got_text == want_text
        within.append(ok)
        first = med[med.a3 >= 0.5]
        start = float(first.index.min()) if len(first) else np.nan
        lowest_any = med[med.a3 > 0]
        low = float(lowest_any.index.min()) if len(lowest_any) else np.nan
        order_ok.append(bool(low != low or got != got and got_text == "below 0.5" or (got == got and low >= got - 1e-9)))
        rows.append({"arch": arch, "K": K, "(K − 1)/K": (K - 1) / K, "m* measured": got_text, "m* predicted": want_text,
                     "within 0.02": ok, "R at m = 0.9": float(med.R.loc[0.9]), "R predicted": float(med.pred_R.loc[0.9]),
                     "absorbed ≥ half from m": start, "any absorbed from m": low})
    cells = pd.DataFrame(rows)
    cells.to_csv(os.path.join(args.out, "cells.csv"), index=False)
    ratio = {a: float(g[g.K == 20]["R at m = 0.9"].iloc[0] / g[g.K == 2]["R at m = 0.9"].iloc[0]) for a, g in cells.groupby("arch")}
    k2 = exact[(exact.K == 2) & (exact.m >= 0.8)].groupby(["arch", "m"]).absorbed_3.mean()
    k20 = exact[(exact.K == 20) & (exact.m <= 0.9)].groupby(["arch", "m"]).apply(lambda g: (g.absorbed_3 == 0).sum() >= 0.9 * len(g))
    grid = lambda update, col: markdown(runs[runs["update"] == update].pivot_table(  # noqa: E731
        index=["arch", "K"], columns="m", values=col, aggfunc="mean").reset_index())
    lines = ["# Findings: v3j, T22 junk answers (PREREG_v3d addendum 4, exploratory)", "",
             f"Generated by `extra/experiments/analyze_v3j.py` from {len(runs)} runs (two answer classes with prompts, K − 2 junk "
             f"tokens; ρ = 0.9; uniform start; SGD, η = 0.03; 10,000 steps). Prompts whose correct answer is a junk token: "
             f"{int(runs.junk_prompts.sum())}.", "", markdown(cells), "",
             f"1. **The threshold moves as the formula says** (measured m\\* within 0.02 of the kernel prediction, or the same "
             f"side of the grid): {int(np.sum(within))} of {len(within)} cells.",
             "2. **The push shrinks as 1/(K − 1)** (R at m = 0.9, K = 20 over K = 2, within 0.03–0.08): "
             + "; ".join(f"{a}: {v:.3f}" for a, v in ratio.items()) + ".",
             f"3. **Absorption follows.** K = 2, m ≥ 0.8: mean absorbed share ≥ 0.5 in {int((k2 >= 0.5).sum())} of {len(k2)} "
             f"(architecture, m) cells. K = 20, m ≤ 0.9: no minority prompt absorbed in ≥ 9 of 10 tasks in "
             f"{int(k20.sum())} of {len(k20)} cells.",
             f"4. **Absorption only at or above m\\*:** {int(np.sum(order_ok))} of {len(order_ok)} cells.", "",
             "## Exact update: minority prompts absorbed at 1e-3 (mean share)", "", grid("expected_full", "absorbed_3"), "",
             "## Exact update: minority prompts below 1e-2 over the tail (mean share)", "", grid("expected_full", "absorbed_2"), "",
             "## Exact update: majority at the end (mean)", "", grid("expected_full", "majority_final"), "",
             "## Exact update: minority at the end (mean)", "", grid("expected_full", "minority_final"), "",
             "## Sampled update: minority prompts absorbed at 1e-3 (mean share)", "", grid("sampled", "absorbed_3"), "",
             "## Sampled update: majority at the end (mean)", "", grid("sampled", "majority_final"), "",
             "## Sampled update: minority at the end (mean)", "", grid("sampled", "minority_final"), ""]
    with open(os.path.join(args.out, "FINDINGS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
