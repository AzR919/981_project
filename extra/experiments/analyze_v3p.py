"""
Analysis of v3p (G = T27: forward KL toward a collapsed reference; H = T28: Adam's epsilon), implementing
PREREG_v3d_addendum10.md.

usage:
    python experiments/analyze_v3p.py [--raw ~/rlvr-data/v3p] [--out analysis/v3p] [--jobs 3]

Class A is answer class 0 (the majority); class B is the rest ("the minority"). "Minority" values are the class-B
mean success from `history[].class_mean`. Absorbed (as in T17): tail max of p over the last 20% of steps below
1e-3, among prompts that start at or above 1e-2. Recovered (H): the minority mean fell below 1e-3 at some logged
step and is above 0.1 at the end. Late exponent g (M2 of T12): growth of the minority's wrong-side margin
relative to the majority's over the last decade of training. gamma (H, reported): gamma_y = u . phi_y over the
minority prompts, where u is the majority's max-margin direction with unit margin on its closest prompts, in the
l2 norm (as in T24) and in the l-infinity norm (Zhang, Zou and Cao 2024), both on the input features.
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
from scipy.optimize import linprog
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze import markdown  # noqa: E402
from experiments.check_late_time import l2_max_margin  # noqa: E402
from experiments.run import DEFAULTS, build  # noqa: E402

TAU, FELL, BACK = 1e-3, 1e-3, 0.1
EPS = [1e-8, 1e-5, 1e-3, 1e-2, 1e-1, 1.0]
LABELS = {}


def task_key(cfg):
    return (cfg["rho"], round(cfg["majority"], 6), cfg["task_seed"])


def task_info(cfg):
    """Class labels and the l2 / l-infinity margins of the minority along the majority's max-margin direction."""
    small = {k: v for k, v in cfg.items() if k in DEFAULTS or k == "task_seed"}
    _, train, _, _ = build(dict(DEFAULTS, **small))
    phi, cls = np.asarray(train[0], dtype=float), train[1][:, 0]
    a = cls == 0
    g2 = phi[~a] @ l2_max_margin(phi[a])
    # l-infinity max margin: min t s.t. phi_a u >= 1, |u_i| <= t.  Variables (u, t).
    n, d = phi[a].shape
    c = np.r_[np.zeros(d), 1.0]
    eye = np.eye(d)
    a_ub = np.vstack([np.c_[-phi[a], np.zeros(n)], np.c_[eye, -np.ones(d)], np.c_[-eye, -np.ones(d)]])
    b_ub = np.r_[-np.ones(n), np.zeros(2 * d)]
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=[(None, None)] * d + [(0, None)], method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    ginf = phi[~a] @ res.x[:d]
    return dict(cls=cls, gamma_min_l2=float(g2.min()), gamma_median_l2=float(np.median(g2)),
                gamma_min_linf=float(ginf.min()), gamma_median_linf=float(np.median(ginf)))


def init_worker(labels):
    LABELS.update(labels)


def read_one(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    cfg = d["config"]
    info = LABELS[task_key(cfg)]
    cls = info["cls"]
    p0, tail = np.asarray(d["p_init"]), np.asarray(d["tail_max"])
    absorbed = (tail < TAU) & (p0 >= 10 * TAU)
    steps = np.array([h["step"] for h in d["history"]])
    means = np.array([h["class_mean"] for h in d["history"]])
    row = dict(batch=cfg["hypothesis"], name=cfg["name"], arch=cfg["arch"], rho=cfg["rho"], m=cfg["majority"],
               task_seed=cfg["task_seed"], optimizer=cfg["optimizer"], eta=cfg["eta"], adam_eps=cfg["adam_eps"],
               minority_absorbed=float(absorbed[cls != 0].mean()), majority_absorbed=float(absorbed[cls == 0].mean()),
               b_min=float(means[:, 1].min()), b_final=float(means[-1, 1]), a_final=float(means[-1, 0]))
    row["recovered"] = bool(row["b_min"] < FELL and row["b_final"] > BACK)
    tenth = int(np.argmin(np.abs(steps - steps[-1] / 10)))
    safe = lambda v: -np.log(np.clip(v, 1e-300, 1.0))  # noqa: E731
    den = safe(1 - means[-1, 0]) - safe(1 - means[tenth, 0])
    row["late_exponent"] = float((safe(means[-1, 1]) - safe(means[tenth, 1])) / den) if den > 0 else np.nan
    if cfg["hypothesis"] == "v3_t27":
        # First tracked step (every 5 steps) at which each minority prompt is below 1e-2; -1 if never.
        fall = np.asarray(d["absorbed_at"])[cls != 0]
        row["fall_1e2_median"] = float(np.median(fall[fall >= 0])) if (fall >= 0).any() else np.nan
        row["fall_1e2_max"] = float(fall[fall >= 0].max()) if (fall >= 0).any() else np.nan
        if cfg.get("anchor_at") is not None:
            row.update(arm=f"anchor {cfg['anchor_at']:g}", beta=cfg["anchor_beta"], anchor_at=cfg["anchor_at"])
            s = int(d["anchor_step"])
            hit = np.nonzero(steps == s)[0]
            row["anchor_step"] = s
            row["b_at_anchor"] = float(means[hit[0], 1]) if len(hit) else np.nan
            row["a_at_anchor"] = float(means[hit[0], 0]) if len(hit) else np.nan
        elif cfg["beta"]:
            row.update(arm="start anchor", beta=cfg["beta"], anchor_at=0.0)
        else:
            row.update(arm="no anchor", beta=0.0, anchor_at=np.nan)
    else:
        row["arm"] = "sgd" if cfg["optimizer"] == "sgd" else f"adam {cfg['adam_eps']:g}"
        for key in ("gamma_min_l2", "gamma_median_l2", "gamma_min_linf", "gamma_median_linf"):
            row[key] = info[key]
    return row


def frac(mask):
    return f"{int(mask.sum())}/{len(mask)}"


def g_section(g, out):
    arms = ["no anchor", "start anchor", "anchor 0.005", "anchor 0.05", "anchor 0.25"]
    g["arm"] = pd.Categorical(g.arm, arms, ordered=True)
    cells = g.groupby(["arch", "arm", "beta"], observed=True).apply(lambda x: pd.Series({
        "runs": len(x), "minority at anchor (median)": float(x.b_at_anchor.median()) if "b_at_anchor" in x else np.nan,
        "majority at anchor (median)": float(x.a_at_anchor.median()) if "a_at_anchor" in x else np.nan,
        "minority final (median)": float(x.b_final.median()), "majority final (median)": float(x.a_final.median()),
        "minority absorbed (mean share)": float(x.minority_absorbed.mean()),
        "runs with none absorbed": frac(x.minority_absorbed == 0)}), include_groups=False).reset_index()
    cells.insert(0, "batch", "G")

    late = g[g.arm.astype(str).str.startswith("anchor")].copy()
    elig = late[late.b_at_anchor < 1e-2].copy()
    elig["not_restored"] = elig.b_final <= 10 * elig.b_at_anchor
    p1 = elig.not_restored.mean() if len(elig) else np.nan
    per1 = elig.groupby(["arch", "anchor_at", "beta"]).apply(lambda x: frac(x.not_restored), include_groups=False)
    start = g[(g.arm == "start anchor") & (g.beta >= 0.1)]
    none = start.minority_absorbed == 0
    p2 = none.mean()
    per2 = start.groupby(["arch", "beta"]).apply(lambda x: frac(x.minority_absorbed == 0), include_groups=False)
    cmp3 = g[g.arm.isin(["start anchor", "anchor 0.005"])].groupby(["arch", "beta", "arm"], observed=True).apply(
        lambda x: pd.Series({"none absorbed": frac(x.minority_absorbed == 0),
                             "absorbed share": float(x.minority_absorbed.mean()),
                             "minority final (median)": float(x.b_final.median()),
                             "majority final (median)": float(x.a_final.median())}), include_groups=False).reset_index()
    below_at = late.groupby(["arch", "anchor_at"]).apply(lambda x: frac(x.b_at_anchor < 1e-2), include_groups=False)
    ctrl = g[g.arm == "no anchor"].groupby("arch").agg(med=("fall_1e2_median", "median"), top=("fall_1e2_max", "max"))
    lines = ["## G (T27): forward KL toward a collapsed reference", "",
             f"{len(g)} runs (linear and MLP, ρ = 0.9, m = 0.9, SGD η = 0.1, exact update, 20,000 steps, 10 tasks).", "",
             "Late-anchor runs whose minority mean is below 1e-2 at the anchor step: "
             + "; ".join(f"{a} at {t:g}: {v}" for (a, t), v in below_at.items()) + ".", "",
             "When the fall happens (no-anchor control; first tracked step, every 5 steps, at which a minority prompt is "
             "below 1e-2): " + "; ".join(f"{a}: median over runs of the per-run median step {r.med:.0f}, latest prompt "
                                          f"at step {r.top:.0f}" for a, r in ctrl.iterrows())
             + ". The earliest anchor (0.5% = step 100) therefore comes after the fall, not before it.", "",
             f"1. **Anchors taken after the fall do not restore the minority** (minority at the end ≤ 10 × its value at "
             f"the anchor step, among late-anchor runs with the minority below 1e-2 at the anchor; pass if ≥ 80%): "
             f"{int(elig.not_restored.sum())} of {len(elig)} runs = {p1:.1%}. "
             f"**{'HOLDS' if p1 >= 0.8 else 'FAILS'}.** By cell (arch, anchor fraction, β): "
             + "; ".join(f"{a} {t:g} β={b:g}: {v}" for (a, t, b), v in per1.items()) + ".",
             f"2. **The start anchor prevents absorption at β ≥ 0.1** (no minority prompt absorbed; pass if ≥ 90% of "
             f"runs): {int(none.sum())} of {len(none)} runs = {p2:.1%}. **{'HOLDS' if p2 >= 0.9 else 'FAILS'}.** By cell: "
             + "; ".join(f"{a} β={b:g}: {v}" for (a, b), v in per2.items()) + ".",
             "3. **Early anchor (0.5%) against the start anchor** (reported):", "", markdown(cmp3), "",
             "### All arms", "", markdown(cells.drop(columns="batch")), ""]
    verdicts = [dict(prediction="G1", rule="final ≤ 10 × at-anchor in ≥ 80% of late-anchor runs below 1e-2 at the anchor",
                     measured=f"{int(elig.not_restored.sum())}/{len(elig)} = {p1:.3f}", verdict="holds" if p1 >= 0.8 else "fails"),
                dict(prediction="G2", rule="start anchor, β ≥ 0.1: none absorbed in ≥ 90% of runs",
                     measured=f"{int(none.sum())}/{len(none)} = {p2:.3f}", verdict="holds" if p2 >= 0.9 else "fails")]
    return lines, cells, verdicts


def h_section(h, out):
    arms = [f"adam {e:g}" for e in EPS] + ["sgd"]
    keys = ["arch", "rho", "m"]
    share = h.pivot_table(index=keys, columns="arm", values="recovered", aggfunc="mean")[arms]
    fell = h.assign(fell=h.b_min < FELL).pivot_table(index=keys, columns="arm", values="fell", aggfunc="sum")[arms]
    cond = h[h.b_min < FELL].pivot_table(index=keys, columns="arm", values="recovered", aggfunc="mean").reindex(
        index=share.index, columns=arms)
    first, last, sgd = share["adam 1e-08"], share["adam 1"], share["sgd"]
    decreases = last < first
    close = (last - sgd).abs() <= 0.1 + 1e-12
    ok = decreases & close
    mono = (share[arms[:-1]].diff(axis=1).iloc[:, 1:] <= 0.1 + 1e-12).all(axis=1)
    ok_lenient = (last <= first) & close
    verdict = pd.DataFrame({"recovered ε=1e-8": first, "recovered ε=1": last, "recovered SGD": sgd,
                            "decreases": decreases, "ε=1 within 0.1 of SGD": close, "pass": ok,
                            "non-increasing along ε (slack 0.1)": mono}).reset_index()
    exponent = h.pivot_table(index=keys, columns="arm", values="late_exponent", aggfunc="median")[arms].reset_index()
    a_final = h.pivot_table(index=keys, columns="arm", values="a_final", aggfunc="median")[arms].reset_index()
    b_final = h.pivot_table(index=keys, columns="arm", values="b_final", aggfunc="median")[arms].reset_index()
    absorbed = h.pivot_table(index=keys, columns="arm", values="minority_absorbed", aggfunc="mean")[arms].reset_index()
    tasks = h.drop_duplicates(["rho", "m", "task_seed"])
    gam = tasks.groupby(["rho", "m"]).agg(
        gamma_min_l2=("gamma_min_l2", "median"), gamma_min_linf=("gamma_min_linf", "median"),
        gamma_median_l2=("gamma_median_l2", "median"), gamma_median_linf=("gamma_median_linf", "median"),
        rank_corr_l2_linf=("gamma_min_l2", lambda v: np.nan)).reset_index()
    for i, r in gam.iterrows():
        t = tasks[(tasks.rho == r.rho) & (tasks.m == r.m)]
        gam.loc[i, "rank_corr_l2_linf"] = float(spearmanr(t.gamma_min_l2, t.gamma_min_linf).statistic)
    corr = []
    for (arch, rho, m, arm), x in h.groupby(keys + ["arm"]):
        x = x.dropna(subset=["late_exponent"])
        row = dict(arch=arch, rho=rho, m=m, arm=arm)
        for norm in ("l2", "linf"):
            v = x[f"gamma_min_{norm}"]
            row[f"spearman(g, γ_min {norm})"] = float(spearmanr(v, x.late_exponent).statistic) if len(x) > 2 and np.ptp(v) > 0 and np.ptp(x.late_exponent) > 0 else np.nan
        corr.append(row)
    corr = pd.DataFrame(corr)
    corr_med = corr.groupby(["arch", "arm"])[["spearman(g, γ_min l2)", "spearman(g, γ_min linf)"]].median().reset_index()
    corr_med["arm"] = pd.Categorical(corr_med.arm, arms, ordered=True)
    corr_med = corr_med.sort_values(["arch", "arm"])
    n_ok = int(ok.sum())
    cells = h.groupby(keys + ["arm"]).apply(lambda x: pd.Series({
        "runs": len(x), "eta": float(x.eta.iloc[0]), "fell below 1e-3": int((x.b_min < FELL).sum()),
        "recovered": int(x.recovered.sum()), "recovered share": float(x.recovered.mean()),
        "minority absorbed (mean share)": float(x.minority_absorbed.mean()),
        "minority final (median)": float(x.b_final.median()), "majority final (median)": float(x.a_final.median()),
        "late exponent (median)": float(x.late_exponent.median()),
        "gamma_min_l2 (median)": float(x.gamma_min_l2.median()),
        "gamma_min_linf (median)": float(x.gamma_min_linf.median())}), include_groups=False).reset_index()
    cells.insert(0, "batch", "H")
    lines = ["## H (T28): Adam's epsilon", "",
             f"{len(h)} runs (linear and MLP, ρ ∈ {{0.5, 0.9}}, m ∈ {{0.7, 0.9}}, exact update, 10,000 steps, 10 tasks; "
             "Adam lr = max(1e-3, 0.1 ε); control SGD η = 0.1).", "",
             f"1. **Recovery share falls from ε = 1e-8 to ε = 1 and ends within 0.1 of SGD** (recovered: minority mean "
             f"below 1e-3 at some logged step and above 0.1 at the end; share over all 10 runs of the cell; pass if ≥ 6 "
             f"of 8 cells): {n_ok} of 8 cells. **{'HOLDS' if n_ok >= 6 else 'FAILS'}.** "
             f"The 'within 0.1 of SGD' part alone: {int(close.sum())} of 8; the 'decreases' part alone: "
             f"{int(decreases.sum())} of 8 (with ≤ instead of <: {int(ok_lenient.sum())} of 8 cells pass both).", "",
             markdown(verdict), "",
             "### Share of runs recovered (of 10)", "", markdown(share.reset_index()), "",
             "### Runs whose minority fell below 1e-3 (of 10)", "", markdown(fell.reset_index()), "",
             "### Recovered among runs that fell", "", markdown(cond.reset_index()), "",
             "### Minority prompts absorbed (mean share)", "", markdown(absorbed), "",
             "### Minority at the end (median)", "", markdown(b_final), "",
             "### Majority at the end (median)", "", markdown(a_final), "",
             "2. **Late exponent and γ_min under each ε** (reported).", "",
             "### Late exponent g (median over tasks)", "", markdown(exponent), "",
             "### γ_min of the task (median over the 10 tasks; the same for every ε and architecture)", "",
             "Unit margin on the majority's closest prompts; ℓ2 as in T24, ℓ∞ from the LP min ‖u‖∞ s.t. φ_A u ≥ 1. "
             "rank_corr_l2_linf: Spearman of the two γ_min across the 10 tasks.", "",
             markdown(gam), "",
             "### Within-cell Spearman of g with each γ_min across tasks (median over the 4 (ρ, m) cells)", "",
             markdown(corr_med), ""]
    verdicts = [dict(prediction="H1", rule="share(ε=1) < share(ε=1e-8) and share(ε=1) within 0.1 of share(SGD) in ≥ 6 of 8 cells",
                     measured=f"{n_ok}/8 cells", verdict="holds" if n_ok >= 6 else "fails")]
    corr.to_csv(os.path.join(out, "h_gamma_corr.csv"), index=False)
    return lines, cells, verdicts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=os.path.expanduser("~/rlvr-data/v3p"))
    parser.add_argument("--out", default="analysis/v3p")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.raw, "*", "results", "v3_t2[78]", "*.json.gz")))
    first = {}
    for p in paths:
        with gzip.open(p, "rt") as f:
            cfg = json.load(f)["config"]
        first.setdefault(task_key(cfg), cfg)
    print(f"building {len(first)} tasks, reading {len(paths)} files")
    with Pool(args.jobs) as pool:
        labels = dict(zip(first, pool.map(task_info, first.values())))
    with Pool(args.jobs, initializer=init_worker, initargs=(labels,)) as pool:
        runs = pd.DataFrame(pool.map(read_one, paths, chunksize=8))
    os.makedirs(args.out, exist_ok=True)
    runs.to_csv(os.path.join(args.out, "runs.csv"), index=False)
    g_lines, g_cells, g_v = g_section(runs[runs.batch == "v3_t27"].copy(), args.out)
    h_lines, h_cells, h_v = h_section(runs[runs.batch == "v3_t28"].copy(), args.out)
    pd.concat([g_cells, h_cells], ignore_index=True).to_csv(os.path.join(args.out, "cells.csv"), index=False)
    verdicts = pd.DataFrame(g_v + h_v)
    lines = ["# Findings: v3p, G (T27) collapsed-reference KL and H (T28) Adam's epsilon (PREREG_v3d addendum 10, exploratory)",
             "", f"Generated by `experiments/analyze_v3p.py` from {len(runs)} runs "
             f"({int((runs.batch == 'v3_t27').sum())} G, {int((runs.batch == 'v3_t28').sum())} H).", "",
             markdown(verdicts), ""] + g_lines + h_lines
    with open(os.path.join(args.out, "FINDINGS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
