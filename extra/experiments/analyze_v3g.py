"""
Analysis of v3g (T17 KL anchor, T18 release test, T19 group size), implementing PREREG_v3d_addendum3.md.

usage:
    python experiments/analyze_v3g.py [--raw ~/rlvr-data/v3g] [--out analysis/v3g]

Class A is answer class 0 (the majority); class B is the rest. Matched point: the first logged step at which
the class-A mean reaches 0.80. Absorbed: tail max of p below 1e-3 among prompts that start at or above 1e-2.
"""

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze import markdown  # noqa: E402
from experiments.analyze_v3 import t1_labels  # noqa: E402
from experiments.analyze_v3e import crossing  # noqa: E402

TAU, TARGET, RETURN = 1e-3, 0.80, 0.2
LABELS = {}


def labels(cfg):
    key = (cfg["K"], cfg["rho"], round(cfg["majority"], 6), cfg["task_seed"])
    if key not in LABELS:
        LABELS[key] = t1_labels(cfg)[0]
    return LABELS[key]


def read(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def common(d):
    cfg = d["config"]
    cls = labels(cfg)
    absorbed = (np.asarray(d["tail_max"]) < TAU) & (np.asarray(d["p_init"]) >= 10 * TAU)
    steps = np.array([h["step"] for h in d["history"]])
    means = np.array([h["class_mean"] for h in d["history"]])
    return cfg, cls, absorbed, steps, means


def t17_row(d):
    cfg, cls, absorbed, steps, means = common(d)
    hit = np.nonzero(means[:, 0] >= TARGET)[0]
    return dict(arch=cfg["arch"], update=cfg["update"], eta=cfg["eta"], kl=cfg["kl"] if cfg["beta"] else "none",
                beta=cfg["beta"], task_seed=cfg["task_seed"], reached=bool(len(hit)),
                matched_step=int(steps[hit[0]]) if len(hit) else -1,
                minority_at_match=float(means[hit[0], 1]) if len(hit) else np.nan,
                majority_final=float(means[-1, 0]), minority_final=float(means[-1, 1]),
                minority_absorbed=float(absorbed[cls != 0].mean()))


def t18_row(d):
    cfg, cls, absorbed, steps, means = common(d)
    rel = d.get("release_step")
    row = dict(arch=cfg["arch"], task_seed=cfg["task_seed"], release=rel if rel is not None else -1,
               arm="control" if rel is None else ("no KL" if not cfg["release_beta"] else
                                                  f"{cfg['release_kl']} {cfg['release_beta']}"),
               minority_final=float(means[-1, 1]), majority_final=float(means[-1, 0]),
               returned=bool(means[-1, 1] >= RETURN))
    if rel is not None:
        p = np.asarray(d["snapshots"][str(rel)])
        row["p_at_release"] = float(p[cls != 0].mean())
        after = np.nonzero((steps > rel) & (means[:, 1] >= RETURN))[0]
        row["steps_to_return"] = int(steps[after[0]] - rel) if len(after) else -1
    return row


def t19_row(d):
    cfg, cls, absorbed, steps, means = common(d)
    b = d.get("branches", {}).get("0", {})
    return dict(arch=cfg["arch"], rho=cfg["rho"], rule=cfg["rule"], update=cfg["update"], G=cfg["G"], m=cfg["majority"],
                task_seed=cfg["task_seed"], d_b_mixed=b.get("d_b_mixed", np.nan),
                absorbed_b=float(absorbed[cls != 0].mean()), minority_final=float(means[-1, 1]))


def frac(mask):
    return f"{int(mask.sum())}/{len(mask)}"


def t17(runs, out):
    cells = runs.groupby(["arch", "update", "eta", "kl", "beta"]).apply(lambda g: pd.Series({
        "reached 0.80": frac(g.reached), "matched step (median)": float(g.matched_step[g.reached].median()) if g.reached.any() else np.nan,
        "minority at match (median)": float(g.minority_at_match.median()),
        "majority final": float(g.majority_final.median()), "minority final": float(g.minority_final.median()),
        "minority absorbed": float(g.minority_absorbed.mean()),
        "tasks with none absorbed": frac(g.minority_absorbed == 0)})).reset_index()
    cells.to_csv(os.path.join(out, "t17_cells.csv"), index=False)
    n9 = lambda g, mask: mask.sum() >= 0.9 * len(g)  # noqa: E731
    base = runs[runs.beta == 0]
    p1 = p1n = p5 = p5n = 0
    for (arch, update, eta), g in runs.groupby(["arch", "update", "eta"]):
        zero = float(base[(base.arch == arch) & (base["update"] == update) & (base.eta == eta)].minority_absorbed.mean())
        series = {}
        for kl in ("reverse", "forward"):
            s = g[g.kl == kl].groupby("beta").minority_absorbed.mean().sort_index()
            series[kl] = s
            seq = np.array([zero] + s.tolist())
            p1 += int(np.all(np.diff(seq) <= 0.02))
            p1n += 1
        gap = series["forward"] - series["reverse"]
        p5 += int((gap <= 0.02).sum())
        p5n += len(gap)
    kl_runs = runs[runs.beta > 0]
    low = kl_runs[kl_runs.beta <= 0.3].groupby(["arch", "update", "eta", "kl", "beta"]).apply(lambda g: n9(g, g.reached))
    top = kl_runs[kl_runs.beta == 3].groupby(["arch", "update", "eta", "kl"]).apply(lambda g: not g.reached.any())
    fwd = kl_runs[(kl_runs.kl == "forward") & kl_runs.beta.isin([0.03, 0.1, 0.3])]
    p3 = fwd.groupby(["arch", "update", "eta", "beta"]).apply(
        lambda g: n9(g, g.minority_absorbed == 0) and n9(g, g.reached) and n9(g, g.minority_at_match < 0.2))
    p3a = fwd.groupby(["arch", "update", "eta", "beta"]).apply(lambda g: n9(g, g.minority_absorbed == 0) and n9(g, g.reached))
    rev = kl_runs[(kl_runs.kl == "reverse") & (kl_runs.beta <= 0.3)]
    p4 = rev.groupby(["arch", "update", "eta", "beta"]).apply(lambda g: n9(g, g.minority_absorbed > 0.5))
    best = cells[(cells.beta > 0)].copy()
    best["ok"] = best["reached 0.80"].str.split("/").str[0].astype(int) >= 9
    lines = ["## T17: KL anchor on the minority task", "",
             f"{len(runs)} runs. A cell is one architecture × update × η (8 cells).", "",
             f"1. **Monotone protection** (absorbed share does not rise with β, slack 0.02): {p1} of {p1n} cell × direction series.",
             f"2. **Majority cost:** reaches 0.80 in ≥ 9 of 10 tasks at β ≤ 0.3: {int(low.sum())} of {len(low)} cells; "
             f"never reaches it at β = 3: {int(top.sum())} of {len(top)}.",
             f"3. **Forward KL at β ∈ {{0.03, 0.1, 0.3}}** (none absorbed, majority reaches 0.80, and minority below 0.2 at "
             f"the matched point): {int(p3.sum())} of {len(p3)} cells. The first two parts alone: {int(p3a.sum())} of {len(p3a)}.",
             f"4. **Reverse KL at β ≤ 0.3 does not prevent absorption** (absorbed share above 0.5 in ≥ 9 of 10 tasks): "
             f"{int(p4.sum())} of {len(p4)} cells.",
             f"5. **Forward protects at least as well as reverse at equal β** (slack 0.02): {p5} of {p5n} comparisons.", "",
             markdown(cells), ""]
    return lines


def t18(runs, out):
    arms = runs.groupby(["arch", "release", "arm"]).apply(lambda g: pd.Series({
        "returned": frac(g.returned), "minority final (median)": float(g.minority_final.median()),
        "majority final (median)": float(g.majority_final.median()),
        "p at release (median)": float(g.p_at_release.median()) if "p_at_release" in g and g.release.iloc[0] >= 0 else np.nan,
        "steps to return (median)": float(g.steps_to_return[g.steps_to_return >= 0].median())
        if (g.release.iloc[0] >= 0 and (g.steps_to_return >= 0).any()) else np.nan})).reset_index()
    arms.to_csv(os.path.join(out, "t18_arms.csv"), index=False)
    share = lambda g: g.returned.mean()  # noqa: E731
    none9 = lambda g: (~g.returned).sum() >= 0.9 * len(g)  # noqa: E731
    all9 = lambda g: g.returned.sum() >= 0.9 * len(g)  # noqa: E731
    p1 = runs[runs.arm == "control"].groupby("arch").apply(none9)
    nokl = runs[runs.arm == "no KL"]
    rate = nokl.groupby(["arch", "release"]).apply(share).unstack("release")
    p2a = int((rate.diff(axis=1).iloc[:, 1:] <= 1e-9).all(axis=1).sum())
    p2b = nokl[nokl.release == 5000].groupby("arch").apply(none9)
    back = nokl[nokl.steps_to_return >= 0]
    rho = spearmanr(1 / back.p_at_release, back.steps_to_return).statistic if len(back) > 3 else np.nan
    rev = runs[runs.arm.str.startswith("reverse") & runs.release.isin([1000, 5000])]
    p3 = rev.groupby(["arch", "release", "arm"]).apply(none9)
    p4 = runs[runs.arm == "forward 0.1"].groupby(["arch", "release"]).apply(all9)
    lines = ["## T18: release test", "",
             f"{len(runs)} runs. Phase 1 is plain GRPO at m = 0.9; at the release step the majority is no longer rewarded. "
             "\"Returned\": the minority class mean is at least 0.2 at step 20,000.", "",
             f"1. **Control (no release): no return** in ≥ 9 of 10 tasks: {int(p1.sum())} of {len(p1)} architectures.",
             f"2. **No KL after the release:** return rate does not rise with the release step: {p2a} of {len(rate)} "
             f"architectures; no return at release step 5,000: {int(p2b.sum())} of {len(p2b)}; Spearman of steps to return "
             f"against 1/p at the release: {rho:.2f} ({len(back)} returning runs).",
             f"3. **Reverse KL: no return at release steps 1,000 and 5,000:** {int(p3.sum())} of {len(p3)} "
             "(architecture × release × β).",
             f"4. **Forward KL at β = 0.1 returns at every release step:** {int(p4.sum())} of {len(p4)} (architecture × release).",
             "5. Forward KL at β = 0.01: reported in the table.", "", markdown(arms), ""]
    return lines


def t19(runs, out):
    rows = []
    for (arch, rho, rule, update, G), g in runs.groupby(["arch", "rho", "rule", "update", "G"]):
        per = g.groupby("m").agg(d=("d_b_mixed", "median"), mean_abs=("absorbed_b", "mean"),
                                 med_abs=("absorbed_b", "median")).sort_index()
        reg, old = per[per.mean_abs >= 0.05], per[per.med_abs > 0.5]
        rows.append(dict(arch=arch, rho=rho, rule=rule, update=update, G=G,
                         m_star=crossing(per.index.to_numpy(), per.d.to_numpy()) if update == "expected_full" else np.nan,
                         absorption_from=float(reg.index.min()) if len(reg) else np.nan,
                         absorption_from_median=float(old.index.min()) if len(old) else np.nan))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "t19_thresholds.csv"), index=False)
    top = runs.m.max() + 1                                 # stands for "no absorption on the grid"
    exact = t[(t["update"] == "expected_full")]
    grpo = exact[exact.rule == "grpo"]
    span = grpo.groupby(["arch", "rho"]).m_star.agg(lambda v: v.max() - v.min())
    wide = grpo.pivot_table(index=["arch", "rho"], columns="G", values="absorption_from", dropna=False).fillna(top)
    mono = (wide.diff(axis=1).iloc[:, 1:] >= -1e-9).all(axis=1)
    larger = wide[64] > wide[2]
    shift = (wide[64] - wide[2]).where((wide[64] < top) & (wide[2] < top))
    mean = exact[exact.rule == "mean"].pivot_table(index=["arch", "rho"], columns="G", values="absorption_from",
                                                   dropna=False).fillna(top)
    same = mean[64] == mean[2]
    show = t.pivot_table(index=["arch", "rho", "rule", "update"], columns="G", values="absorption_from", dropna=False)
    show_med = t.pivot_table(index=["arch", "rho", "rule", "update"], columns="G", values="absorption_from_median", dropna=False)
    stars = grpo.pivot_table(index=["arch", "rho"], columns="G", values="m_star")
    lines = ["## T19: group size and the collapse threshold", "",
             f"{len(runs)} runs (K = 3, η = 0.01, 10,000 steps). `absorption_from`: the smallest grid m whose mean minority "
             "absorption over the 10 tasks is at least 0.05 (as registered); blank: none on the grid (largest m = "
             f"{runs.m.max():.3f}).", "",
             f"1. **m\\* does not depend on G** (range across G below 0.005): {int((span < 0.005).sum())} of {len(span)} cells "
             f"(largest range {span.max():.5f}).",
             f"2. **The absorption threshold rises with G under grpo:** non-decreasing in G in {int(mono.sum())} of {len(mono)} "
             f"cells; larger at G = 64 than at G = 2 in {int(larger.sum())} of {len(larger)}.",
             f"3. **Not under the mean rule** (same grid value at G = 2 and G = 64): {int(same.sum())} of {len(same)} cells.",
             f"4. **Size** (G = 64 threshold at most 0.2 above the G = 2 threshold, where both exist): largest shift "
             f"{shift.max():.3f}.", "",
             "Step-0 threshold m\\* by G (exact update, grpo):", "", markdown(stars.reset_index()), "",
             "`absorption_from` by G (mean ≥ 0.05, as registered):", "", markdown(show.reset_index()), "",
             "For comparison with v3e, the same with its definition (median minority absorption above 0.5):", "",
             markdown(show_med.reset_index()), ""]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=os.path.expanduser("~/rlvr-data/v3g"))
    parser.add_argument("--out", default="analysis/v3g")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lines = ["# Findings: v3g (PREREG_v3d addendum 3, exploratory)", "",
             "Generated by `experiments/analyze_v3g.py`. Class-share task with K = 3, uniform start, SGD; the KL "
             "reference is the start policy.", ""]
    for name, row, part in (("v3_t17", t17_row, t17), ("v3_t18", t18_row, t18), ("v3_t19", t19_row, t19)):
        paths = sorted(glob.glob(os.path.join(args.raw, "*", "results", name, "*.json.gz")))
        runs = pd.DataFrame([row(read(p)) for p in paths])
        runs.to_csv(os.path.join(args.out, f"{name[3:]}_runs.csv"), index=False)
        lines += part(runs, args.out)
        print(name, len(runs), flush=True)
    with open(os.path.join(args.out, "FINDINGS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
