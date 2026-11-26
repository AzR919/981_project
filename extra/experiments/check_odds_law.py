"""
Check the odds law (extra/analysis/theory/odds_law.md) against the v3e runs (T15).

usage:
    python extra/experiments/check_odds_law.py [--raw ~/rlvr-data/v3e] [--out extra/analysis/theory]

For every run, the first-order change of the mean class-B success under the class-A step, the class-B step
and the mixed step is computed from the kernel k(x, y) = phi(x) . phi(y) of the last-layer features at the
uniform start, with no fitted quantity, and compared with the change the simulator measured at step 0.
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
import torch

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # repo root
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]  # extra/

from experiments.analyze import markdown  # noqa: E402
from experiments.analyze_v3e import crossing  # noqa: E402
from experiments.run import DEFAULTS, build  # noqa: E402
from synthetic_rl.theory import advantage_gap  # noqa: E402


def kernel_sums(phi, cls):
    """(S_BA, S_same, S_diff): sums of k over pairs (y in B, x in A), (y, x in B, same class), (y, x in B, other class)."""
    k = phi @ phi.T
    in_b = cls != 0
    kb = k[in_b]
    same = cls[in_b][:, None] == cls[None, :]
    return float(kb[:, ~in_b].sum()), float(kb[same].sum()), float(kb[:, in_b][~same[:, in_b]].sum())


def predict(cfg):
    """First-order step-0 changes of the mean class-B success, and the kernel means, for one configuration."""
    full = dict(DEFAULTS, **cfg)
    _, train, _, policy = build(full)
    x = torch.as_tensor(train[0], dtype=torch.float64)
    with torch.no_grad():
        phi = policy.features(x).numpy()
    if policy.output.bias is not None:
        phi = np.concatenate([phi, np.ones((len(phi), 1))], axis=1)
    cls = train[1][:, 0]
    K, n, n_b = full["K"], len(cls), int((cls != 0).sum())
    s_ba, s_same, s_diff = kernel_sums(phi, cls)
    w = float(advantage_gap(np.array([1 / K]), full["G"], full["rule"], passk=full["passk"])[0])
    scale = full["eta"] * w / (n * n_b * K ** 2)
    d_b_a = -scale * s_ba / K
    d_b_b = scale * ((1 - 1 / K) * s_same - s_diff / K)
    n_a, per = n - n_b, n_b / (K - 1)
    k_ba = s_ba / (n_b * n_a) if n_a else np.nan
    k_same = s_same / (n_b * per)
    k_diff = s_diff / (n_b * (n_b - per)) if K > 2 else 0.0
    c = (K - 1) * k_ba / ((K - 1) * k_same - (K - 2) * k_diff)
    return dict(pred_d_b_a=d_b_a, pred_d_b_b=d_b_b, pred_d_b_mixed=d_b_a + d_b_b, share_a=n_a / n, c=c,
                k_ba=k_ba, k_same=k_same, k_diff=k_diff)


def read_one(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    cfg, b = d["config"], d["branches"]["0"]
    row = dict(K=cfg["K"], arch=cfg["arch"], rho=cfg["rho"], m=cfg["majority"], task_seed=cfg["task_seed"],
               d_b_a=b["d_b_a"], d_b_b=b["d_b_b"], d_b_mixed=b["d_b_mixed"])
    row.update(predict({k: v for k, v in cfg.items() if k in DEFAULTS or k == "task_seed"}))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=os.path.expanduser("~/rlvr-data/v3e"))
    parser.add_argument("--out", default="extra/analysis/theory")
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.raw, "*", "results", "*", "*.json.gz")))
    with Pool(8) as pool:
        runs = pd.DataFrame(pool.map(read_one, paths, chunksize=8))
    os.makedirs(args.out, exist_ok=True)
    runs["R"], runs["pred_R"] = -runs.d_b_a / runs.d_b_b, -runs.pred_d_b_a / runs.pred_d_b_b
    for name in ("d_b_a", "d_b_b", "d_b_mixed"):
        runs[f"err_{name}"] = (runs[f"pred_{name}"] - runs[name]).abs() / runs[name].abs().clip(lower=1e-300)
    runs["err_R"] = (runs.pred_R - runs.R).abs() / runs.R.abs()
    runs["odds_form"] = runs.c * runs.share_a / (1 - runs.share_a)       # c m / (1 - m) with the realised share
    runs["err_odds_form"] = (runs.odds_form - runs.pred_R).abs() / runs.pred_R.abs()
    runs.to_csv(os.path.join(args.out, "odds_law_runs.csv"), index=False)

    rows = []
    for (K, arch, rho), g in runs.groupby(["K", "arch", "rho"]):
        med = g.groupby("m")[["d_b_mixed", "pred_d_b_mixed"]].median().sort_index()
        m = med.index.to_numpy()
        at = g[np.isclose(g.m, round(1 / K, 4))]
        rows.append({"K": K, "arch": arch, "rho": rho, "runs": len(g),
                     "median rel. error of R": float(g.err_R.median()), "max rel. error of R": float(g.err_R.max()),
                     "max rel. error, A-only change": float(g.err_d_b_a.max()),
                     "max rel. error, B-only change": float(g.err_d_b_b.max()),
                     "c at 1/K (median)": float(at.c.median()), "1/(1+c)": float(1 / (1 + at.c.median())),
                     "m* predicted": crossing(m, med.pred_d_b_mixed.to_numpy()),
                     "m* measured": crossing(m, med.d_b_mixed.to_numpy())})
    cells = pd.DataFrame(rows)
    cells.to_csv(os.path.join(args.out, "odds_law_cells.csv"), index=False)
    gap = (cells["m* predicted"] - cells["m* measured"]).abs()
    lines = ["# Check of the odds law against the v3e runs", "",
             f"Generated by `extra/experiments/check_odds_law.py` from {len(runs)} runs (step 0, η = 0.01). Nothing is fitted: "
             "the prediction uses only the task's last-layer features at the uniform start.", "", markdown(cells), "",
             f"- Relative error of the predicted R over all runs: median {runs.err_R.median():.2e}, "
             f"95th percentile {runs.err_R.quantile(0.95):.2e}, largest {runs.err_R.max():.2e}.",
             f"- Predicted against measured m\\*: largest gap {gap.max():.4f} over {len(cells)} cells.",
             f"- The closed form c·m/(1 − m), with c from the three kernel means, against the exact kernel sums: "
             f"largest relative gap {runs.err_odds_form.max():.2e} (it is an identity when the minority classes have "
             "equal sizes).", ""]
    with open(os.path.join(args.out, "odds_law_check.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
