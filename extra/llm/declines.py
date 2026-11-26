"""
Which problems got worse? Compare two evaluations (evaluate.py outputs).

usage:
    python llm/declines.py before.json.gz after.json.gz [--fdr 0.1] [--out table.csv]

Per problem, a one-sided Fisher exact test of p_after < p_before on the
(correct, n) counts; Benjamini-Hochberg across problems at the given FDR
(plan v3, C0). A *lost* problem is a significant decline that ends at
p_after <= 0.05. Also prints the transition matrix of p-hat bins and the
format-failure (unparsed answer) rates. Comparing two evaluations of the same
model gives the empirical false-decline rate.
"""

import argparse
import gzip
import json

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

BINS = [0, 0.05, 0.2, 0.5, 0.8, 1.0001]
LABELS = ["<0.05", "0.05-0.2", "0.2-0.5", "0.5-0.8", ">=0.8"]


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)["counts"]


def bh(pvalues, q):
    """Benjamini-Hochberg: boolean mask of rejections at FDR q."""
    p = np.asarray(pvalues)
    order = np.argsort(p)
    ranked = p[order] <= q * np.arange(1, len(p) + 1) / len(p)
    keep = np.zeros(len(p), dtype=bool)
    if ranked.any():
        keep[order[: np.nonzero(ranked)[0].max() + 1]] = True
    return keep


def compare(before, after, fdr=0.1):
    ids = sorted(set(before) & set(after))
    rows = []
    for i in ids:
        k1, f1, n1 = before[i]
        k2, f2, n2 = after[i]
        p = fisher_exact([[k2, n2 - k2], [k1, n1 - k1]], alternative="less").pvalue
        rows.append({"id": i, "k_before": k1, "n_before": n1, "k_after": k2, "n_after": n2,
                     "p_before": k1 / n1, "p_after": k2 / n2, "parsed_before": f1 / n1, "parsed_after": f2 / n2,
                     "pvalue": p})
    df = pd.DataFrame(rows)
    df["decline"] = bh(df.pvalue, fdr)
    df["lost"] = df.decline & (df.p_after <= 0.05)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--fdr", type=float, default=0.1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    df = compare(load(args.before), load(args.after), args.fdr)
    print(f"{len(df)} problems; mean p {df.p_before.mean():.3f} -> {df.p_after.mean():.3f}; "
          f"significant declines {df.decline.sum()} (FDR {args.fdr}); lost {df.lost.sum()}; "
          f"format failures {1 - df.parsed_before.mean():.3f} -> {1 - df.parsed_after.mean():.3f}")
    table = pd.crosstab(pd.cut(df.p_before, BINS, labels=LABELS, right=False),
                        pd.cut(df.p_after, BINS, labels=LABELS, right=False))
    print("transition matrix (rows: before, columns: after)")
    print(table.to_string())
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
