"""
Analysis of v3h (T20: a correct minority class in Li, Pandey and Chen's enumerable environment),
implementing PREREG_T20.md.

usage:
    python experiments/analyze_v3h.py [--raw ~/rlvr-data/v3h] [--out analysis/v3h]

Absorbed: a prompt whose success stays below 1e-3 over the last 20% of steps (and started at or above 1e-2).
R = -(change of the minority's mean success under a majority-only step) / (change under a minority-only step).
"""

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze import markdown  # noqa: E402

TAU = 1e-3


def read_one(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    cfg = d["config"]
    cls, p0 = np.asarray(d["cls"]), np.asarray(d["p_init"])
    tail, low = np.asarray(d["tail_max"]), np.asarray(d["min_p"])
    absorbed = (tail < TAU) & (p0 >= 10 * TAU)
    b = d.get("branches", {})
    means = d["history"][-1]["class_mean"]
    dip = min(h["class_mean"][1] for h in d["history"])
    return dict(K=cfg["K"], n_major=cfg["n_major"], partition=cfg["partition"], policy=cfg["enum_policy"],
                algo=cfg["algo"], optimizer=cfg["optimizer"], eta=cfg["eta"], seed=cfg["seed"], m=d["share_a"],
                p_init=float(p0.mean()), absorbed_b=float(absorbed[cls != 0].mean()),
                absorbed_a=float(absorbed[cls == 0].mean()), majority_final=means[0], minority_final=means[1],
                minority_lowest_mean=dip, largest_drop=float((p0 - low).max()),
                d_b_a=b.get("d_b_a", np.nan), d_b_b=b.get("d_b_b", np.nan), d_b_mixed=b.get("d_b_mixed", np.nan))


def n_of(mask):
    return f"{int(mask.sum())}/{len(mask)}"


def cross_one(m, r):
    """m at which R crosses 1, interpolating log R linearly in logit m; nan if it does not."""
    z, y = np.log(m / (1 - m)), np.log(r)
    for i in range(len(z) - 1):
        if y[i] < 0 <= y[i + 1]:
            zc = z[i] + (z[i + 1] - z[i]) * (0 - y[i]) / (y[i + 1] - y[i])
            return float(1 / (1 + np.exp(-zc)))
    return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=os.path.expanduser("~/rlvr-data/v3h"))
    parser.add_argument("--out", default="analysis/v3h")
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.raw, "*", "results", "*", "*.json.gz")))
    runs = pd.DataFrame([read_one(p) for p in paths])
    os.makedirs(args.out, exist_ok=True)
    runs.to_csv(os.path.join(args.out, "runs.csv"), index=False)
    runs["task"] = runs.apply(lambda r: f"K={r.K}, {r.n_major}/64" + (" (parity)" if r.partition == "parity" else ""), axis=1)
    runs["stepper"] = runs.optimizer + " " + runs.eta.map("{:g}".format)

    cells = runs.groupby(["policy", "algo", "partition", "K", "n_major", "stepper"]).apply(lambda g: pd.Series({
        "seeds": len(g), "minority absorbed (mean)": float(g.absorbed_b.mean()),
        "seeds with ≥ half absorbed": n_of(g.absorbed_b >= 0.5), "seeds with none absorbed": n_of(g.absorbed_b == 0),
        "majority final": float(g.majority_final.median()), "minority final": float(g.minority_final.median()),
        "minority lowest mean": float(g.minority_lowest_mean.median()),
        "largest drop below start": float(g.largest_drop.max())})).reset_index()
    cells.to_csv(os.path.join(args.out, "cells.csv"), index=False)

    exact = runs[(runs.policy == "shared") & (runs.algo == "pg") & (runs.partition == "random") & (runs.optimizer == "sgd")]
    high = exact[(exact.n_major >= 56) & exact.eta.isin([1, 3])].groupby(["K", "n_major", "eta"]).apply(
        lambda g: (g.absorbed_b >= 0.5).sum() >= 0.8 * len(g))
    balanced = exact[((exact.K == 3) & (exact.n_major == 22)) | ((exact.K == 2) & (exact.n_major == 32))]
    bal = balanced.groupby(["K", "eta"]).apply(lambda g: (g.absorbed_b == 0).sum() >= 0.9 * len(g))
    unshared = runs[runs.policy.isin(["tabular", "copies"])]
    drops = unshared.groupby("policy").largest_drop.agg(["size", "max"])

    first = runs[(runs.policy == "shared") & (runs.partition == "random")].drop_duplicates(["K", "n_major", "seed"]).copy()
    first["R"] = -first.d_b_a / first.d_b_b
    per_m = first.groupby(["K", "n_major"]).agg(m=("m", "first"), R=("R", "median"), seeds=("R", "size"),
                                                 mixed_down=("d_b_mixed", lambda v: int((v < 0).sum()))).reset_index()
    law = []
    for K, g in per_m.groupby("K"):
        g = g.sort_values("m")
        pos = g[g.R > 0]
        slope = float(np.polyfit(np.log(pos.m / (1 - pos.m)), np.log(pos.R), 1)[0]) if len(pos) > 1 else np.nan
        law.append(dict(K=K, slope=slope, m_star=cross_one(pos.m.to_numpy(), pos.R.to_numpy()), one_over_K=1 / K,
                        R_at_smallest_m=float(g.R.iloc[0])))
    law = pd.DataFrame(law)
    par = runs[(runs.policy == "shared") & (runs.partition == "parity")].drop_duplicates("seed")
    par_r = float((-par.d_b_a / par.d_b_b).median()) if len(par) else np.nan

    show = ["K", "n_major", "stepper", "minority absorbed (mean)", "seeds with ≥ half absorbed", "seeds with none absorbed",
            "majority final", "minority final", "minority lowest mean"]
    pick = lambda pol, algo, part: cells[(cells.policy == pol) & (cells.algo == algo) & (cells.partition == part)][show]  # noqa: E731
    lines = ["# Findings: v3h, T20 a correct minority class in Li, Pandey and Chen's environment (PREREG_T20, exploratory)", "",
             f"Generated by `experiments/analyze_v3h.py` from {len(runs)} runs (64 prompts, 2-layer transformer, 100 responses, "
             f"exact success; 2,000 steps). Mean start success: {runs.p_init.mean():.3f}.", "",
             f"1. **Collapse at high share** (shared, exact update, SGD at lr 1 and 3, n_major ≥ 56: at least half of the "
             f"minority prompts absorbed in ≥ 8 of 10 seeds): {int(high.sum())} of {len(high)} cells. **None at balance** "
             f"(no minority prompt absorbed in ≥ 9 of 10 seeds, every SGD lr): {int(bal.sum())} of {len(bal)} cells.",
             "2. **No sharing, no decline** (largest fall of any prompt below its start): "
             + "; ".join(f"{k}: {v['max']:.2e} over {int(v['size'])} runs" for k, v in drops.iterrows()) + ".",
             "3. **Odds law** (slope of log R on logit m within 0.8–1.2): "
             + "; ".join(f"K = {r.K}: {r.slope:.2f}" for r in law.itertuples()) + ".",
             "4. **Threshold within 0.1 of 1/K:** "
             + "; ".join(f"K = {r.K}: R = 1 at m = {r.m_star:.3f} (1/K = {r.one_over_K:.3f})" for r in law.itertuples()) + ".",
             f"5. Parity task (48/64): median R at step 0 = {par_r:.2f}.", "",
             "## Step 0: the push ratio by share (shared transformer, random partition)", "", markdown(per_m), "",
             "## Shared transformer, exact update (random partition)", "", markdown(pick("shared", "pg", "random")), "",
             "## Shared transformer, sampled GRPO (random partition)", "", markdown(pick("shared", "grpo", "random")), "",
             "## Parity task", "", markdown(cells[cells.partition == "parity"][["policy", "algo"] + show]), "",
             "## Without sharing", "", markdown(cells[cells.policy.isin(["tabular", "copies"]) & (cells.partition == "random")][
                 ["policy"] + show + ["largest drop below start"]]), ""]
    with open(os.path.join(args.out, "FINDINGS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
