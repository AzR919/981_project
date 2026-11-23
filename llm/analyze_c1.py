"""
C1 / C5 analysis for one model family, as pre-registered in analysis/review/PREREG_C1.md.

usage:
    python llm/analyze_c1.py --name qwen --base0 b0.json.gz --base1 b1.json.gz \\
        --train-ids llm/pools/qwen_c1_train.json --heldout-ids llm/pools/qwen_c1_heldout.json \\
        --arm "lr 1e-6=dir0,dir1,dir2" --arm "lr 3e-6=..." --primary "lr 1e-6" \\
        --out analysis/llm/c1_qwen.md --csv analysis/llm/c1_qwen_runs.csv

Each run dir holds after_n256.json.gz, optionally step250.json.gz, and run/log.jsonl.
Lost = BH-significant decline (one-sided Fisher, q = 0.1, within the training
problems) that ends at p-hat <= 0.05. P4 per run: binomial test of the loss
count against the base-vs-base null rate (L0 + 1) / (n + 1); per family the
second-smallest seed p-value (2 of 3 seeds).
"""

import argparse
import gzip
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom, hypergeom, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.analyze_c0 import markdown, transitions  # noqa: E402
from llm.declines import compare, load  # noqa: E402
from llm.passk import pass_at_k  # noqa: E402

FDR = 0.1


def subset(counts, ids):
    return {i: counts[i] for i in ids if i in counts}


def check_counts(name, counts, ids, n=256):
    """Every expected problem present with exactly n samples (no partial or truncated evaluations)."""
    missing = [i for i in ids if i not in counts]
    wrong_n = [i for i in ids if i in counts and counts[i][2] != n]
    if missing or wrong_n:
        raise SystemExit(f"{name}: {len(missing)} problems missing, {len(wrong_n)} with n != {n}; refusing to score")


def check_run(path, steps):
    """The run finished its pre-registered number of steps; returns its seed."""
    config = json.load(open(os.path.join(path, "run", "config.json")))
    last = [json.loads(line) for line in open(os.path.join(path, "run", "log.jsonl"))][-1]["step"]
    if config["steps"] != steps or last != steps:
        raise SystemExit(f"{path}: configured {config['steps']} steps, logged {last}; expected {steps}")
    return config["seed"]


def census(before, after, ids):
    """Gains and losses on one problem set: BH-tested (gained, declined, lost), Yang's zero-count
    definitions (newly solved: 0 -> >= 1; locked: >= 1 -> 0), and strict-lost (0 of n after, base >= 0.1)."""
    down = compare(subset(before, ids), subset(after, ids), FDR)
    up = compare(subset(after, ids), subset(before, ids), FDR)
    return dict(problems=len(down), gained=int(up.decline.sum()), declined=int(down.decline.sum()),
                lost=int(down.lost.sum()), strict_lost=int(((down.k_after == 0) & (down.p_before >= 0.1)).sum()),
                newly_solved=int(((down.k_before == 0) & (down.k_after > 0)).sum()),
                locked=int(((down.k_before > 0) & (down.k_after == 0)).sum()),
                mean_change=float((down.p_after - down.p_before).mean()))


def problem_bootstrap(df, draws=2000, seed=0):
    """95% CIs from resampling problems: mean change in p-hat, and the lost count (decline flags fixed)."""
    rng = np.random.default_rng(seed)
    change, lost = (df.p_after - df.p_before).to_numpy(), df.lost.to_numpy()
    idx = rng.integers(len(df), size=(draws, len(df)))
    return (np.quantile(change[idx].mean(axis=1), [0.025, 0.975]), np.quantile(lost[idx].sum(axis=1), [0.025, 0.975]))


def passk_gap_band(df, ks=(1, 2, 4, 8, 16, 32, 64, 128, 256), draws=4000, seed=0):
    """Paired pass@k gap (after - before) with a simultaneous 95% band over k: prompt-level multiplier
    bootstrap, one N(0, 1) multiplier per prompt shared across k (Yang 2609.22547, Prop. 1)."""
    gaps = np.stack([pass_at_k(df.k_after, df.n_after, k) - pass_at_k(df.k_before, df.n_before, k) for k in ks], axis=1)
    m = len(gaps)
    mean, sd = gaps.mean(axis=0), gaps.std(axis=0, ddof=1)
    se = np.where(sd > 0, sd, np.inf) / np.sqrt(m)
    xi = np.random.default_rng(seed).standard_normal((draws, m))
    t_star = np.abs(xi @ (gaps - mean) / m / se).max(axis=1)
    c = float(np.quantile(t_star, 0.95))
    return pd.DataFrame({"k": ks, "gap": mean, "lo": mean - c * se, "hi": mean + c * se})


def modal_wrong(hist, k_correct, gold=None):
    """(answer, count) of the most frequent wrong answer. Without the reference answer, the entry whose
    count equals the correct count is taken to be the correct one."""
    items = [(a, c) for a, c in hist.items() if a != "unparsed"]
    if gold is not None:
        items = [(a, c) for a, c in items if a != gold]
    elif k_correct > 0:
        match = [a for a, c in items if c == k_correct]
        items = [(a, c) for a, c in items if not match or a != match[0]]
    return max(items, key=lambda ac: ac[1]) if items else (None, 0)


def partial_conjunction(pvalues, u=2):
    """p-value for 'at least u of the n hypotheses are false', valid under arbitrary dependence:
    (n - u + 1) * p_(u) (Benjamini & Heller 2008, Bonferroni form). For u = 2, n = 3 this is 2 * p_(2):
    under the null at most one hypothesis is false, so p_(2) <= t needs a true null with p <= t, and
    E[#true nulls with p <= t] <= 3t while two such are needed when all three are null (Markov: 3t / 2),
    or one of two when one is false (union bound: 2t)."""
    p = np.sort(np.asarray(pvalues, dtype=float))
    return float(min(1.0, (len(p) - u + 1) * p[u - 1]))


def run_row(label, path, base0, train, held, r0, n_train):
    after = load(os.path.join(path, "after_n256.json.gz"))
    check_counts(path, after, train + held)
    tr = compare(subset(base0, train), subset(after, train), FDR)
    ho = compare(subset(base0, held), subset(after, held), FDR)
    gains = compare(subset(after, train), subset(base0, train), FDR)
    lost = int(tr.lost.sum())
    log = pd.DataFrame([json.loads(line) for line in open(os.path.join(path, "run", "log.jsonl"))])
    row = dict(arm=label, run=os.path.basename(path.rstrip("/")), steps=len(log), lost=lost,
               p_binom=float(binom.sf(lost - 1, n_train, r0)), declines=int(tr.decline.sum()),
               gains=int(gains.decline.sum()), lost_heldout=int(ho.lost.sum()),
               declines_heldout=int(ho.decline.sum()), p_before=tr.p_before.mean(), p_after=tr.p_after.mean(),
               p_after_heldout=ho.p_after.mean(),
               pass64_before=pass_at_k(tr.k_before, tr.n_before, 64).mean(),
               pass64_after=pass_at_k(tr.k_after, tr.n_after, 64).mean(),
               format_fail_after=1 - tr.parsed_after.mean(), length_last50=log.length[-50:].mean(),
               reward_last50=log.reward[-50:].mean())
    step = os.path.join(path, "step250.json.gz")
    if os.path.exists(step):
        mid = subset(load(step), train)
        p_mid = np.array([c[0] / c[2] for c in mid.values()])
        row.update(p_step250=p_mid.mean(), below005_step250=int((p_mid <= 0.05).sum()))
    lo_hi_change, lo_hi_lost = problem_bootstrap(tr)
    row.update(change_ci=f"{lo_hi_change[0]:+.3f} to {lo_hi_change[1]:+.3f}",
               lost_ci=f"{lo_hi_lost[0]:.0f} to {lo_hi_lost[1]:.0f}")
    return row, tr


def answers_of(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    return d.get("answers", {}), d.get("gold", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--base0", required=True)
    parser.add_argument("--base1", required=True)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--heldout-ids", required=True)
    parser.add_argument("--arm", action="append", required=True, help="label=dir1,dir2,...")
    parser.add_argument("--primary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    base0, base1 = load(args.base0), load(args.base1)
    train, held = json.load(open(args.train_ids)), json.load(open(args.heldout_ids))
    check_counts(args.base0, base0, train + held)
    check_counts(args.base1, base1, train + held)
    null = compare(subset(base0, train), subset(base1, train), FDR)
    L0, n = int(null.lost.sum()), len(null)
    r0 = (L0 + 1) / (n + 1)

    rows, tables, turnover = [], {}, []
    for name, ids in (("train", train), ("held-out", held)):
        turnover.append(dict(arm="null: base seed 0 → seed 1", run="", set=name, **census(base0, base1, ids)))
    base_answers, gold = answers_of(args.base0)
    modal_rows, bands, sensitivity = [], [], []
    for item in args.arm:
        label, dirs = item.split("=", 1)
        paths = [d for d in dirs.split(",") if d]
        seeds = [check_run(p, args.steps) for p in paths]
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"arm {label}: repeated seeds {seeds}")
        for path in paths:
            row, tr = run_row(label, path, base0, train, held, r0, n)
            rows.append(row)
            tables[(label, row["run"])] = tr
            after = load(os.path.join(path, "after_n256.json.gz"))
            for name, ids in (("train", train), ("held-out", held)):
                turnover.append(dict(arm=label, run=row["run"], set=name, **census(base0, after, ids)))
            # Sensitivity to the choice of "before" draw (regression to the mean): base seed 1 instead of 0.
            alt = compare(subset(base1, train), subset(after, train), FDR)
            sensitivity.append(dict(arm=label, run=row["run"], lost_before_seed0=row["lost"],
                                    lost_before_seed1=int(alt.lost.sum()),
                                    lost_in_both=int((tr.set_index("id").lost & alt.set_index("id").lost).sum())))
            band = passk_gap_band(tr)
            band.insert(0, "run", row["run"])
            band.insert(0, "arm", label)
            bands.append(band)
            after_answers, _ = answers_of(os.path.join(path, "after_n256.json.gz"))
            if after_answers:
                for group, sel in (("lost", tr[tr.lost]), ("not lost", tr[~tr.lost])):
                    shares, same = [], []
                    for r in sel.itertuples():
                        answer, count = modal_wrong(after_answers[r.id], r.k_after, gold.get(r.id))
                        shares.append(count / r.n_after)
                        if base_answers and answer is not None and r.id in base_answers:
                            same.append(answer == modal_wrong(base_answers[r.id], r.k_before, gold.get(r.id))[0])
                    if shares:
                        modal_rows.append(dict(arm=label, run=row["run"], problems=group, n=len(shares),
                                               modal_wrong_share=float(np.mean(shares)),
                                               same_as_base_modal_wrong=float(np.mean(same)) if same else np.nan))
    runs = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    runs.to_csv(args.csv, index=False)

    lines = [f"# C1: {args.name}", "",
             "Generated by `llm/analyze_c1.py`; tests as pre-registered in `analysis/review/PREREG_C1.md`.", "",
             "## Null", "",
             f"Base against base (seed 0 against seed 1, N = 256) on the {n} training problems: "
             f"{int(null.decline.sum())} BH-significant declines, L₀ = {L0} lost. Null rate r₀ = (L₀ + 1)/{n + 1} "
             f"= {r0:.2e}.", "",
             "## Runs", "", markdown(runs.drop(columns=["run"])), "",
             "`change_ci` and `lost_ci`: 95% intervals from resampling problems (2,000 draws).", "",
             "## Turnover (gains and losses together, with the null)", "",
             "gained / declined / lost: BH-tested at 10%. newly solved: 0 correct before, ≥ 1 after; locked: "
             "≥ 1 before, 0 after (Yang 2609.22547). strict lost: 0 of 256 after with base p̂ ≥ 0.1. With "
             "256 samples, a fall to ≤ 0.05 is detectable only from base p̂ ≳ 0.15.", "",
             markdown(pd.DataFrame(turnover)), "",
             "## Sensitivity to the 'before' draw", "", markdown(pd.DataFrame(sensitivity)), ""]

    primary = runs[runs.arm == args.primary].sort_values("p_binom")
    p_family = float(primary.p_binom.iloc[1]) if len(primary) == 3 else np.nan
    p_pc = partial_conjunction(primary.p_binom) if len(primary) == 3 else np.nan
    lines += ["## P4 for this family", "",
              f"Primary arm `{args.primary}`: seed p-values {', '.join(f'{p:.3g}' for p in primary.p_binom)}; "
              f"p_family (second smallest, as first registered) = {p_family:.3g}; partial-conjunction p for "
              f"'at least 2 of 3 seeds' (2 × second smallest; amendment 1, the rule used) = {p_pc:.3g}. "
              + (f"**{'PASS' if p_pc < 0.05 else 'FAIL'}** for this family (P4 needs both families)."
                 if len(primary) == 3 else f"**UNTESTABLE**: {len(primary)} of 3 seeds available."), ""]

    # Secondary: base p-hat of lost problems, cross-seed overlap.
    lines += ["## Paired pass@k gap (after − before) with a simultaneous 95% band", "",
              markdown(pd.concat(bands)), ""]
    if modal_rows:
        lines += ["## Most frequent wrong answer", "",
                  "Share of the 256 completions on the single most frequent wrong answer, and how often that "
                  "answer is also the base model's most frequent wrong answer.", "",
                  markdown(pd.DataFrame(modal_rows)), ""]
    lines += ["## Secondary (exploratory)", ""]
    for label, g in runs.groupby("arm"):
        lost_sets = {r: set(tables[(label, r)].id[tables[(label, r)].lost]) for r in g.run}
        base_p = tables[(label, g.run.iloc[0])].set_index("id").p_before
        lost_p = [base_p[i] for s in lost_sets.values() for i in s]
        if lost_p:
            mw = mannwhitneyu(lost_p, base_p.values, alternative="less")
            lines.append(f"- `{label}`: median base p̂ of lost problems {np.median(lost_p):.3f} against "
                         f"{np.median(base_p):.3f} for all (one-sided Mann–Whitney p = {mw.pvalue:.3g}).")
        for a, b in itertools.combinations(lost_sets, 2):
            k, na, nb = len(lost_sets[a] & lost_sets[b]), len(lost_sets[a]), len(lost_sets[b])
            if na and nb:
                lines.append(f"  - overlap {a} ∩ {b}: {k} of {na} and {nb} (hypergeometric p = "
                             f"{hypergeom.sf(k - 1, n, na, nb):.3g}; {na * nb / n:.1f} expected by chance)")
    lines.append("")
    for (label, r), tr in tables.items():
        lines += [f"### {label}, {r}: p̂ transitions on training problems (base → after, N = 256)", "",
                  transitions(tr), ""]
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}; P4 {args.name}: p_family = {p_family:.3g}, partial conjunction = {p_pc:.3g}")


if __name__ == "__main__":
    main()
