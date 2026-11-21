"""
C0 pilot analysis for one model: false-decline rate, problem pool, learning-rate pilots.

usage:
    python llm/analyze_c0.py --name qwen --base s0.json.gz s1.json.gz --test test.json.gz \\
        --pilot 1e-6=run_dir 3e-6=run_dir ... --out analysis/llm/c0_qwen.md --pool llm/pools/qwen.json

Each pilot run_dir holds run/log.jsonl (train.py) and after.json.gz
(evaluate.py on the training problems). The pool is every train problem whose
base p-hat, pooled over both base evaluations, lies in [0.1, 0.9].
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.declines import BINS, LABELS, compare, load  # noqa: E402

POOL = (0.1, 0.9)


def smallest_detectable(n, alpha):
    """Smallest p_before whose drop to 0/n gives a one-sided Fisher p below alpha."""
    for k in range(1, n + 1):
        if fisher_exact([[0, n], [k, n - k]], alternative="less").pvalue < alpha:
            return k / n
    return np.nan


def markdown(df):
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, rule] + rows)


def transitions(df):
    t = pd.crosstab(pd.cut(df.p_before, BINS, labels=LABELS, right=False),
                    pd.cut(df.p_after, BINS, labels=LABELS, right=False))
    t.index.name, t.columns.name = None, None
    return markdown(t.reset_index().rename(columns={"index": "before \\ after"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--base", nargs=2, required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--pilot", nargs="*", default=[], help="lr=run_dir")
    parser.add_argument("--fdr", type=float, default=0.1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool", required=True)
    args = parser.parse_args()

    s0, s1 = load(args.base[0]), load(args.base[1])
    test = load(args.test)
    lines = [f"# C0 pilot: {args.name}", ""]

    # False-decline rate: the same model evaluated twice with different seeds.
    null = compare(s0, s1, args.fdr)
    back = compare(s1, s0, args.fdr)
    lines += ["## Base model and the false-decline rate", "",
              f"- Train problems evaluated: {len(null)} × 32 samples, twice (seeds 0 and 1).",
              f"- Mean p̂: {null.p_before.mean():.3f} (seed 0), {null.p_after.mean():.3f} (seed 1); "
              f"test set ({len(test)} problems): {np.mean([c[0] / c[2] for c in test.values()]):.3f}.",
              f"- Format failures (no parsable answer): {1 - null.parsed_before.mean():.4f}.",
              f"- Same model against itself at FDR {args.fdr}: {int(null.decline.sum())} declines "
              f"({int(back.decline.sum())} in the reverse direction), {int(null.lost.sum())} lost. "
              f"Raw one-sided p < 0.05 in {(null.pvalue < 0.05).mean():.2%} of problems (Fisher is conservative).",
              ""]

    rows = [dict(N=n, **{"after = 0, p < 0.05": smallest_detectable(n, 0.05),
                         "after = 0, p < 1e-4": smallest_detectable(n, 1e-4)}) for n in (32, 64, 256)]
    lines += ["### Power of the decline test", "",
              "The smallest p_before that the test can flag when the problem drops all the way to 0 "
              "(N samples before and after). With about 1,000 problems, BH at 10% needs p of about 1e-4 for "
              "the first few rejections.", "", markdown(pd.DataFrame(rows)), ""]

    pooled = (null.k_before + null.k_after) / (null.n_before + null.n_after)
    in_pool = (pooled >= POOL[0]) & (pooled <= POOL[1])
    pool_ids = sorted(null.id[in_pool], key=lambda i: int(i.split("-")[1]))
    os.makedirs(os.path.dirname(args.pool) or ".", exist_ok=True)
    with open(args.pool, "w") as f:
        json.dump(pool_ids, f)
    edges = [0, 1e-9, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1 - 1e-9, 1.01]
    names = ["0", "(0, 0.05)", "[0.05, 0.1)", "[0.1, 0.2)", "[0.2, 0.5)", "[0.5, 0.8)", "[0.8, 0.9)",
             "[0.9, 0.95)", "[0.95, 1)", "1"]
    hist = pd.DataFrame({"p̂ (64 samples)": names, "problems": np.histogram(pooled, edges)[0]})
    lines += ["## Problem pool", "",
              f"Base p̂ pooled over both evaluations (64 samples). {len(pool_ids)} of {len(null)} problems "
              f"lie in [{POOL[0]}, {POOL[1]}]; their ids are in `{args.pool}`.", "", markdown(hist), ""]

    first = {k: v for k, v in s0.items() if int(k.split("-")[1]) < 1000}
    pool_set = set(pool_ids)
    summary, details = [], []
    for item in args.pilot:
        lr, path = item.split("=", 1)
        log = pd.DataFrame([json.loads(line) for line in open(os.path.join(path, "run", "log.jsonl"))])
        after = load(os.path.join(path, "after.json.gz"))
        df = compare(first, after, args.fdr)
        gains = compare(after, first, args.fdr)
        pooled_rows = df.id.isin(pool_set)
        blocks = log.reward.groupby(log.index // 25).mean().round(2).tolist()
        summary.append(dict(lr=lr, steps=len(log), reward_first10=log.reward[:10].mean(),
                            reward_last10=log.reward[-10:].mean(), length_first10=log.length[:10].mean(),
                            length_last10=log.length[-10:].mean(), p_before=df.p_before.mean(),
                            p_after=df.p_after.mean(), declines=int(df.decline.sum()),
                            declines_in_pool=int((df.decline & pooled_rows).sum()), lost=int(df.lost.sum()),
                            gains=int(gains.decline.sum()), format_fail_after=1 - df.parsed_after.mean(),
                            sec_per_step=log.t_step.mean()))
        details += [f"### lr {lr}", "", f"Training reward by 25-step block: {blocks}.", "",
                    "p̂ transitions on the 1,000 training problems (rows: base seed 0; columns: after):", "",
                    transitions(df), ""]
    if not summary:
        lines += ["No learning-rate pilots given.", ""]
    else:
        lines += ["## Learning-rate pilots (150 steps on train problems 0–999, G = 8, 64 prompts per step)", "",
                  "Declines, losses and gains are per-problem Fisher tests at FDR 10% against the seed-0 base "
                  "evaluation (32 samples each). Lost: a significant decline that ends at p̂ ≤ 0.05.", "",
                  markdown(pd.DataFrame(summary)), ""] + details
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} and {args.pool} ({len(pool_ids)} problems)")


if __name__ == "__main__":
    main()
