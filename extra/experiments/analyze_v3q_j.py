"""
Analysis of v3q batch J (T29: the T20 transformer environment under SGD at small step sizes), implementing
prediction J of PREREG_v3d_addendum10.md. Reads only `results/v3_t29` (`v3_a4` belongs to battery A).

usage:
    python experiments/analyze_v3q_j.py [--raw ~/rlvr-data/v3q] [--out analysis/v3q]

Definitions as in T20 (`experiments/analyze_v3h.py`, `analysis/v3h/SUMMARY.md`): a prompt is absorbed if its
success stays below 1e-3 over the last 20% of steps and started at or above 1e-2. Class-level rule: a class is
absorbed if at least half of its prompts are. "Minority absorbed": at least half of the non-zero classes' prompts
are absorbed while class 0 is not; "class 0 absorbed": at least half of class 0's prompts are. High share:
n_major >= 56 of 64 (share >= 0.875); balanced: n_major = 22 (K = 3).
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
    cls, p0, tail = np.asarray(d["cls"]), np.asarray(d["p_init"]), np.asarray(d["tail_max"])
    absorbed = (tail < TAU) & (p0 >= 10 * TAU)
    means = d["history"][-1]["class_mean"]
    row = dict(K=cfg["K"], n_major=cfg["n_major"], share=d["share_a"], algo=cfg["algo"], eta=cfg["eta"],
               steps=cfg["steps"], seed=cfg["seed"], p_init=float(p0.mean()),
               absorbed_a=float(absorbed[cls == 0].mean()), absorbed_b=float(absorbed[cls != 0].mean()),
               majority_final=means[0], minority_final=means[1],
               minority_lowest_mean=min(h["class_mean"][1] for h in d["history"]))
    for c in range(cfg["K"]):
        row[f"absorbed_class{c}"] = float(absorbed[cls == c].mean())
    row["class0_absorbed"] = bool(row["absorbed_a"] >= 0.5)
    row["minority_absorbed"] = bool(row["absorbed_b"] >= 0.5 and not row["class0_absorbed"])
    row["any_class_absorbed"] = bool(max(row[f"absorbed_class{c}"] for c in range(cfg["K"])) >= 0.5)
    return row


def n_of(mask):
    return f"{int(mask.sum())}/{len(mask)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=os.path.expanduser("~/rlvr-data/v3q"))
    parser.add_argument("--out", default="analysis/v3q")
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.raw, "*", "results", "v3_t29", "*.json.gz")))
    runs = pd.DataFrame([read_one(p) for p in paths]).sort_values(["n_major", "algo", "eta", "seed"])
    os.makedirs(args.out, exist_ok=True)
    runs.to_csv(os.path.join(args.out, "runs_J.csv"), index=False)

    high = runs[runs.n_major >= 56]
    bal = runs[runs.n_major == 22]
    a_ok = int(high.class0_absorbed.sum()) == 0
    b_share, c_share = high.minority_absorbed.mean(), bal.minority_absorbed.mean()
    b_ok, c_ok = b_share >= 0.8, c_share <= 0.2
    holds = a_ok and b_ok and c_ok

    def arm(g):
        return pd.Series({"runs": len(g), "class 0 absorbed": n_of(g.class0_absorbed),
                          "minority absorbed (class rule)": n_of(g.minority_absorbed),
                          "any class absorbed": n_of(g.any_class_absorbed),
                          "minority prompts absorbed (mean)": float(g.absorbed_b.mean()),
                          "class 0 prompts absorbed (mean)": float(g.absorbed_a.mean()),
                          "majority final": float(g.majority_final.median()),
                          "minority final": float(g.minority_final.median()),
                          "minority lowest mean": float(g.minority_lowest_mean.median())})

    by_arm = runs.groupby(["n_major", "algo", "eta"]).apply(arm, include_groups=False).reset_index()
    by_arm.insert(1, "share", by_arm.n_major / 64)
    group = lambda df: df.groupby(["algo", "eta"]).apply(  # noqa: E731
        lambda g: pd.Series({"high: class 0 absorbed": n_of(g[g.n_major >= 56].class0_absorbed),
                             "high: minority absorbed": n_of(g[g.n_major >= 56].minority_absorbed),
                             "balanced: minority absorbed": n_of(g[g.n_major == 22].minority_absorbed)}),
        include_groups=False).reset_index()
    per_rule = group(runs)
    balanced_classes = bal.groupby(["algo", "eta"])[["absorbed_class0", "absorbed_class1", "absorbed_class2"]].mean().reset_index()
    verdict = pd.DataFrame([
        dict(part="(a) class 0 never absorbed at share ≥ 0.875", measured=f"{int(high.class0_absorbed.sum())} of {len(high)} runs",
             verdict="holds" if a_ok else "fails"),
        dict(part="(b) minority absorbed in ≥ 80% of high-share runs", measured=f"{n_of(high.minority_absorbed)} = {b_share:.3f}",
             verdict="holds" if b_ok else "fails"),
        dict(part="(c) minority absorbed in ≤ 20% of balanced runs (n_major = 22)", measured=f"{n_of(bal.minority_absorbed)} = {c_share:.3f}",
             verdict="holds" if c_ok else "fails")])
    lines = ["# Findings: v3q batch J, T29 the transformer environment under SGD at small step sizes "
             "(PREREG_v3d addendum 10, exploratory)", "",
             f"Generated by `experiments/analyze_v3q_j.py` from {len(runs)} runs (shared 2-layer transformer, 64 prompts, "
             "K = 3, n_major ∈ {22, 32, 40, 48, 56, 60}, SGD η = 0.03 × 20,000 and η = 0.1 × 6,000 steps (flow time 600, "
             f"as SGD 0.3 × 2,000), exact (pg) and sampled GRPO, 10 seeds). Mean start success: {runs.p_init.mean():.3f}.", "",
             f"1. **With η ≤ 0.1, class 0 is never absorbed at share ≥ 0.875, and the minority is absorbed in ≥ 80% of "
             f"high-share runs and ≤ 20% of balanced runs** (pooled over both η and both algorithms): "
             f"**{'HOLDS' if holds else 'FAILS'}.**", "", markdown(verdict), "",
             "### The three parts by algorithm and step size", "", markdown(per_rule), "",
             "### Balanced task (n_major = 22): share of each class's prompts absorbed (mean over seeds)", "",
             markdown(balanced_classes), "",
             "### Every share, algorithm and step size", "", markdown(by_arm), ""]
    with open(os.path.join(args.out, "FINDINGS_J.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
