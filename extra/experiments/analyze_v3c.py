"""
Analysis of v3c (pass@k training rules, T10), implementing extra/analysis/review/PREREG_v3c.md.

usage:
    python extra/experiments/analyze_v3c.py [--raw results/v3c] [--out extra/analysis/v3c]

Per run, from the final snapshot (exact p): mean pass@1, pass@8, pass@64, persistent
absorption at tau = 1e-3, and for the minority task the share of minority prompts
absorbed. Each rule is compared with grpo at the same (level, G, eta), paired over seeds.
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
from scipy import stats

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # repo root
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]  # extra/

from experiments.analyze import markdown  # noqa: E402

TAU = 1e-3


def read_one(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    c = d["config"]
    snaps = d["snapshots"]
    p = np.asarray(snaps[str(max(int(s) for s in snaps))])
    p0, tail = np.asarray(d["p_init"]), np.asarray(d["tail_max"])
    eligible = p0 >= 10 * TAU
    absorbed = (tail < TAU) & eligible
    row = dict(level=c["level"], rule=c["rule"] if c.get("passk") is None else f"{c['rule']}{c['passk']}",
               G=c["G"], eta=c["eta"], unit=c.get("task_seed", c["seed"]) if c["level"] != "tab" else c["seed"],
               pass1=p.mean(), pass8=(1 - (1 - p) ** 8).mean(), pass64=(1 - (1 - p) ** 64).mean(),
               absorbed=absorbed.sum() / max(eligible.sum(), 1))
    if c["level"] == "minority":
        from experiments.analyze_v3 import t1_labels
        cls, _ = t1_labels(c)
        row["minority_absorbed"] = absorbed[cls != 0].mean()
    return row


def compare(runs, value, alternative):
    """Per (level, G, eta, rule): mean difference rule - grpo over paired seeds and a one-sided Wilcoxon."""
    w = runs.pivot_table(index=["level", "G", "eta", "unit"], columns="rule", values=value).reset_index()
    rows = []
    for (level, G, eta), g in w.groupby(["level", "G", "eta"]):
        for rule in [c for c in w.columns if c not in ("level", "G", "eta", "unit", "grpo")]:
            d = (g[rule] - g.grpo).dropna()
            p = stats.wilcoxon(d, alternative=alternative).pvalue if (d != 0).any() else 1.0
            rows.append(dict(level=level, G=G, eta=eta, rule=rule, diff=d.mean(), p=p))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="results/v3c")
    parser.add_argument("--out", default="extra/analysis/v3c")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.raw, "*", "results", "*", "*.json.gz")))
    with Pool(args.jobs) as pool:
        runs = pd.DataFrame(pool.map(read_one, paths, chunksize=16))
    os.makedirs(args.out, exist_ok=True)
    runs.to_csv(os.path.join(args.out, "runs.csv"), index=False)
    lines = ["# v3c: pass@k training rules (exploratory, PREREG_v3c.md)", "",
             f"{len(runs)} runs. Each cell: mean over seeds of (rule − grpo) at the same (level, G, η); "
             "one-sided Wilcoxon over paired seeds. Exact pass@k from the final p.", ""]
    mino = compare(runs[runs.level == "minority"], "minority_absorbed", "less")
    agree = mino.groupby("rule").apply(lambda g: f"{int((g.p < 0.05).sum())}/{len(g)}").rename("cells_lower").reset_index()
    mean = mino.groupby("rule")["diff"].mean().rename("mean_diff").reset_index()
    lines += ["## 1. Minority task: share of minority prompts absorbed (prediction: pass@k rules lower)", "",
              markdown(agree.merge(mean)), ""]
    shared = runs[runs.level.isin(["lin", "m64"])]
    p64 = compare(shared, "pass64", "greater").groupby("rule").apply(
        lambda g: pd.Series(dict(cells_higher=f"{int((g.p < 0.05).sum())}/{len(g)}", mean_diff=g["diff"].mean())))
    p1 = compare(shared, "pass1", "less").groupby("rule").apply(
        lambda g: pd.Series(dict(cells_lower=f"{int((g.p < 0.05).sum())}/{len(g)}", mean_diff=g["diff"].mean())))
    lines += ["## 2. Shared levels (lin, m64): pass@64 higher and pass@1 lower than grpo?", "",
              "pass@64:", "", markdown(p64.reset_index()), "", "pass@1:", "", markdown(p1.reset_index()), ""]
    tab = runs[runs.level == "tab"].groupby("rule").absorbed.max().rename("max_absorbed").reset_index()
    lines += ["## 3. Tabular: absorption by rule (prediction: none)", "", markdown(tab), ""]
    cell = runs.groupby(["level", "rule"])[["pass1", "pass64", "absorbed"]].mean().reset_index()
    lines += ["## 4. Means by level and rule (over G, η, seeds)", "", markdown(cell), ""]
    with open(os.path.join(args.out, "FINDINGS.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
