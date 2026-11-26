"""
Analysis of v3d (T11 branch test, T12 plateau or permanent, T13 projection control), implementing
extra/analysis/review/PREREG_v3d.md.

usage:
    python extra/experiments/analyze_v3d.py collect [--raw ~/rlvr-data/v3d]
    python extra/experiments/analyze_v3d.py all [--raw ~/rlvr-data/v3d] [--out extra/analysis/v3d]

Class A is answer class 0 (share m), class B the other classes. collect writes <raw>/cache/runs.csv
(one row per run) and branches.csv (one row per run and branch point).
"""

import argparse
import glob
import gzip
import json
import os
import sys
from multiprocessing import Pool

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # repo root
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]  # extra/

from experiments.analyze import markdown  # noqa: E402
from experiments.analyze_v3 import t1_labels, task_key  # noqa: E402

TAU = 1e-3
KEYS = ("name", "hypothesis", "arch", "rho", "majority", "optimizer", "eta", "update", "steps", "task_seed")
LABELS = {}


def init_worker(labels):
    LABELS.update(labels)


def read_one(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    cfg = d["config"]
    row = {k: cfg[k] for k in KEYS}
    cls = LABELS[task_key(cfg)][0]
    p0, tail = np.asarray(d["p_init"]), np.asarray(d["tail_max"])
    absorbed = (tail < TAU) & (p0 >= 10 * TAU)
    steps = np.array([h["step"] for h in d["history"]])
    means = np.array([h["class_mean"] for h in d["history"]])       # columns: class A, class B
    low = int(means[:, 1].argmin())
    row.update(minority_absorbed=float(absorbed[cls != 0].mean()), majority_absorbed=float(absorbed[cls == 0].mean()),
               b_min=float(means[low, 1]), b_min_step=int(steps[low]), b_final=float(means[-1, 1]),
               a_final=float(means[-1, 0]), b_start=float(means[0, 1]), a_start=float(means[0, 0]))
    row["recovered"] = bool(row["b_min"] < TAU and row["b_final"] > 0.01)
    # M2: growth of the minority's wrong-side margin relative to the majority's margin over the last decade.
    tenth = int(np.argmin(np.abs(steps - steps[-1] / 10)))
    safe = lambda v: -np.log(np.clip(v, 1e-300, 1.0))  # noqa: E731
    den = safe(1 - means[-1, 0]) - safe(1 - means[tenth, 0])
    row["margin_growth"] = float((safe(means[-1, 1]) - safe(means[tenth, 1])) / den) if den > 0 else np.nan
    # M4: absorption by tercile of score margin (terciles over all prompts of the task), per class.
    margin = LABELS[task_key(cfg)][1]
    tercile = np.digitize(margin, np.quantile(margin, [1 / 3, 2 / 3]))
    for t in range(3):
        for name, members in (("b", cls != 0), ("a", cls == 0)):
            sel = members & (tercile == t)
            row[f"abs_{name}_t{t}"] = float(absorbed[sel].mean()) if sel.any() else np.nan
    hit = np.nonzero(means[:, 0] >= means[0, 0] + 0.10)[0]
    row["b_at_a_plus_010"] = float(means[hit[0], 1]) if len(hit) else np.nan
    row["step_a_plus_010"] = int(steps[hit[0]]) if len(hit) else -1
    branches = [dict({k: cfg[k] for k in KEYS}, at=int(s), frac=int(s) / cfg["steps"], **b)
                for s, b in d.get("branches", {}).items()]
    return row, branches


def collect(raw, jobs):
    paths = sorted(glob.glob(os.path.join(raw, "*", "results", "*", "*.json.gz")))
    first = {}
    for p in paths:
        with gzip.open(p, "rt") as f:
            cfg = json.load(f)["config"]
        first.setdefault(task_key(cfg), cfg)
    print(f"building {len(first)} tasks, reading {len(paths)} files")
    with Pool(jobs) as pool:
        labels = dict(zip(first, pool.map(t1_labels, first.values())))
    with Pool(jobs, initializer=init_worker, initargs=(labels,)) as pool:
        results = pool.map(read_one, paths, chunksize=16)
    cache = os.path.join(raw, "cache")
    os.makedirs(cache, exist_ok=True)
    pd.DataFrame([r for r, _ in results]).to_csv(os.path.join(cache, "runs.csv"), index=False)
    pd.DataFrame([b for _, bs in results for b in bs]).to_csv(os.path.join(cache, "branches.csv"), index=False)
    print("done")


def agree(series, n=10, need=9):
    return f"{int(series.sum())}/{len(series)}" + ("" if series.sum() >= need * len(series) / n else " ✗")


def t11(br, lines, out):
    br = br.copy()
    br["m"] = br.majority.round(3)
    exact = br[(br.optimizer == "sgd") & (br["update"] == "expected_full")]
    lines += ["## T11: one-step branches from a common state (SGD, η = 0.01, exact expected update)", "",
              f"Mixed-branch check: largest |mixed − expected_step| = {br.mixed_check.max():.1e} over all "
              f"{len(br)} branch points.", "",
              "Δ_B(X): change in the mean success of class B (the minority when m > 1/3) after one step along X. "
              "Counts are tasks out of 10; ✗ marks a cell below 9 of 10.", ""]
    rows = []
    for (arch, rho, m, frac), g in exact.groupby(["arch", "rho", "m", "frac"]):
        rows.append(dict(arch=arch, rho=rho, m=m, at=f"{frac:g}T", p_B=g.p_b.median(),
                         **{"Δ_B(A) < 0": agree(g.d_b_a < 0), "Δ_B(B) > 0": agree(g.d_b_b > 0),
                            "Δ_B(mixed) < 0": agree(g.d_b_mixed < 0), "cos(g_A, ∇p_B) < 0": agree(g.cos_a_h < 0)},
                         med_dB_A=g.d_b_a.median(), med_dB_B=g.d_b_b.median(), med_dB_mixed=g.d_b_mixed.median(),
                         med_ratio=(-g.d_b_a / g.d_b_b).median()))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "tables", "t11_cells.csv"), index=False)
    early = t[t["at"].isin(["0T", "0.1T"])]
    lines += ["### Predictions 1 and 2: steps 0 and 0.1T", "", markdown(early), ""]
    imb = exact[(exact.m > 0.34) & (exact.frac <= 0.1)]
    cells = imb.groupby(["arch", "rho", "m", "frac"]).apply(
        lambda g: (g.d_b_a < 0).sum() >= 9 and (g.d_b_b > 0).sum() >= 9 and (g.d_b_mixed < 0).sum() >= 9)
    cells0 = cells[cells.index.get_level_values("frac") == 0]
    bal = exact[(exact.m < 0.34) & (exact.frac == 0)].groupby(["arch", "rho"]).apply(lambda g: (g.d_b_mixed > 0).sum() >= 9)
    lines += [f"- **Prediction 1 (active reversal before collapse):** holds in {int(cells.sum())} of {len(cells)} "
              f"imbalanced cells over both early points ({int(cells0.sum())} of {len(cells0)} at step 0).",
              f"- **Prediction 2 (balanced control, Δ_B(mixed) > 0 at step 0):** holds in {int(bal.sum())} of {len(bal)} cells.",
              ""]
    # Prediction 3: the minority's own signal after collapse.
    base = exact[exact.frac == 0].set_index(["arch", "rho", "m", "task_seed"]).d_b_b
    late = exact[(exact.frac >= 0.5) & (exact.m > 0.34) & (exact.p_b < TAU)].copy()
    late["rel"] = late.d_b_b.abs().to_numpy() / base.reindex(late.set_index(["arch", "rho", "m", "task_seed"]).index).abs().to_numpy()
    lines += [f"- **Prediction 3 (starvation after collapse):** at 0.5T and 0.9T, {len(late)} branch points have "
              f"minority mean below 1e-3; |Δ_B(B)| is below 1% of its step-0 value in {int((late.rel < 0.01).sum())} "
              f"of them (median ratio {late.rel.median():.1e})." if len(late) else
              "- **Prediction 3:** no branch point at 0.5T or 0.9T has minority mean below 1e-3.", ""]
    # Prediction 4: dose.
    r0 = exact[exact.frac == 0].assign(ratio=lambda g: -g.d_b_a / g.d_b_b)
    wide = r0.pivot_table(index=["arch", "rho", "task_seed"], columns="m", values="ratio").dropna()
    dose = []
    for (arch, rho), g in wide.groupby(level=["arch", "rho"]):
        p = stats.wilcoxon(g[0.9], g[0.7], alternative="greater").pvalue
        dose.append(dict(arch=arch, rho=rho, ratio_m07=g[0.7].median(), ratio_m09=g[0.9].median(),
                         tasks_larger=f"{int((g[0.9] > g[0.7]).sum())}/{len(g)}", p=p))
    lines += ["### Prediction 4: the majority's push relative to the minority's own gain, −Δ_B(A)/Δ_B(B), at step 0",
              "", "Above 1 means the mixed step lowers the minority.", "", markdown(pd.DataFrame(dose)), ""]
    # Prediction 5: sampled branches.
    smp = br[(br["update"] == "sampled") & (br.frac == 0)]
    ok = [(np.sign(smp["s_d_b_" + k]) == np.sign(smp["d_b_" + k])).mean() for k in ("a", "b", "mixed")]
    lines += [f"- **Prediction 5 (sampled GRPO, 16 batches per point, step 0):** the mean sampled Δ_B has the sign of the "
              f"exact one in {ok[0]:.0%} (A-only), {ok[1]:.0%} (B-only) and {ok[2]:.0%} (mixed) of {len(smp)} runs.", ""]
    adam = br[(br.optimizer == "adam") & (br.frac <= 0.1) & (br.m > 0.34)]
    a = adam.groupby(["eta", "frac"]).apply(lambda g: pd.Series({
        "Δ_B(A) < 0": f"{(g.d_b_a < 0).mean():.0%}", "Δ_B(B) > 0": f"{(g.d_b_b > 0).mean():.0%}",
        "Δ_B(mixed) < 0": f"{(g.d_b_mixed < 0).mean():.0%}", "runs": len(g)})).reset_index()
    lines += ["### Adam (descriptive; actual displacements, imbalanced tasks)", "", markdown(a), ""]

    # Figure: paired task values at step 0.
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, ((arch, rho), g) in zip(axes, exact[exact.frac == 0].groupby(["arch", "rho"])):
        for j, m in enumerate(sorted(g.m.unique())):
            gm = g[g.m == m]
            for i, (k, color) in enumerate((("a", "C3"), ("b", "C0"), ("mixed", "k"))):
                x = j * 4 + i
                ax.scatter(np.full(len(gm), x) + np.linspace(-0.25, 0.25, len(gm)), gm["d_b_" + k], s=12, color=color,
                           label={"a": "majority-only step", "b": "minority-only step", "mixed": "actual (mixed) step"}[k] if j == 0 else None)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks([1, 5, 9]); ax.set_xticklabels([f"m = {m:.2f}" for m in sorted(g.m.unique())])
        ax.set_title(f"{arch}, ρ = {rho}")
    axes[0].set_ylabel("change in class-B mean success (one step)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "t11_branches.png"), dpi=130)
    plt.close(fig)
    lines += ["![One-step change in class-B success by branch, step 0, one dot per task](figures/t11_branches.png)", ""]


def t12(runs, lines, out):
    r = runs[runs.hypothesis == "v3_t12"].copy()
    r["m"] = r.majority.round(3)
    r["ever_below"] = r.b_min < TAU
    cells = r.groupby(["arch", "rho", "m", "optimizer", "eta", "update"]).agg(
        runs=("recovered", "size"), ever_below=("ever_below", "sum"), recovered=("recovered", "sum"),
        b_min=("b_min", "median"), b_final=("b_final", "median"), minority_absorbed=("minority_absorbed", "median")).reset_index()
    cells.to_csv(os.path.join(out, "tables", "t12_cells.csv"), index=False)
    lines += ["## T12: plateau or permanent (10× the T1 horizon)", "",
              "`ever_below`: runs whose minority class mean fell below 1e-3; `recovered`: of those, runs ending "
              "above 0.01. `minority_absorbed`: share of minority prompts below 1e-3 over the last 20% of steps.", ""]
    sgd = r[r.optimizer == "sgd"]
    n_below, n_rec = int(sgd.ever_below.sum()), int(sgd.recovered.sum())
    lines += [f"- **Prediction 1 (no recovery, SGD):** {n_below} of {len(sgd)} SGD runs fell below 1e-3; "
              f"{n_rec} of them recovered ({n_rec / max(n_below, 1):.1%}; predicted ≤ 5%).", ""]
    by = sgd.groupby(["arch", "rho", "m", "eta"]).agg(runs=("recovered", "size"), ever_below=("ever_below", "sum"),
                                                     recovered=("recovered", "sum"), b_min=("b_min", "median"),
                                                     b_final=("b_final", "median"),
                                                     minority_absorbed=("minority_absorbed", "median")).reset_index()
    lines += [markdown(by), ""]
    lin = r[(r.arch == "linear") & (r.optimizer == "adam") & (r.eta == 1e-4) & (r.rho == 0.9)]
    lin_cells = lin.groupby(["m", "update"]).apply(lambda g: pd.Series({
        "tasks with minority absorption > 0.5": f"{int((g.minority_absorbed > 0.5).sum())}/{len(g)}",
        "median minority absorption": g.minority_absorbed.median(), "median b_final": g.b_final.median()})).reset_index()
    lines += ["### Prediction 2: linear policy, Adam at η = 1e-4, 50,000 steps, ρ = 0.9", "",
              markdown(lin_cells), ""]
    mp = r[(r.m == 0.9) & (((r.optimizer == "sgd") & (r.eta == 0.01)) | ((r.optimizer == "adam") & (r.eta == 1e-4)))]
    mp_cells = mp.groupby(["arch", "rho", "optimizer", "update"]).apply(lambda g: pd.Series({
        "reached +0.10": f"{int(g.b_at_a_plus_010.notna().sum())}/{len(g)}",
        "minority below start there": f"{int((g.b_at_a_plus_010 < g.b_start).sum())}/{int(g.b_at_a_plus_010.notna().sum())}",
        "median minority mean there": g.b_at_a_plus_010.median()})).reset_index()
    lines += ["### Prediction 3: minority mean when the majority has gained +0.10 (m = 0.9)", "", markdown(mp_cells), ""]
    adam = r[r.optimizer == "adam"].groupby(["arch", "rho", "m", "eta"]).agg(
        runs=("recovered", "size"), ever_below=("ever_below", "sum"), recovered=("recovered", "sum"),
        b_final=("b_final", "median"), minority_absorbed=("minority_absorbed", "median")).reset_index()
    lines += ["### Adam cells (descriptive)", "", markdown(adam), ""]


def t13(runs, lines, out):
    r = runs[runs.hypothesis == "v3_t13"].copy()
    r["m"] = r.majority.round(3)
    wide = r.pivot_table(index=["arch", "rho", "m", "eta", "task_seed"], columns="update",
                         values=["minority_absorbed", "a_final", "b_final"])
    rows = []
    for key, g in wide.groupby(level=["arch", "rho", "m", "eta"]):
        ab, af = g["minority_absorbed"], g["a_final"]
        d = ab["projected"] - ab["shortened"]
        p = stats.wilcoxon(ab["projected"], ab["shortened"], alternative="less").pvalue if (d != 0).any() else 1.0
        rows.append(dict(zip(("arch", "rho", "m", "eta"), key), full=ab["expected_full"].mean(),
                         shortened=ab["shortened"].mean(), projected=ab["projected"].mean(), p_proj_lt_short=p,
                         a_final_full=af["expected_full"].mean(), a_final_proj=af["projected"].mean(),
                         b_final_proj=g["b_final"]["projected"].mean()))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "tables", "t13_cells.csv"), index=False)
    lines += ["## T13: projection control (SGD, exact expected update)", "",
              "Mean share of minority prompts absorbed, over 10 tasks. `projected`: the majority part without its "
              "component against the minority's success gradient; `shortened`: the majority part scaled to the same norm.",
              "", markdown(t), "",
              f"- **Prediction 1 (projected below shortened, p < 0.05):** {int((t.p_proj_lt_short < 0.05).sum())} of {len(t)} cells.",
              f"- **Prediction 2 (shortened within 0.05 of the unmodified update):** "
              f"{int(((t.shortened - t.full).abs() <= 0.05).sum())} of {len(t)} cells.",
              f"- **Prediction 3 (majority final mean within 0.05 under projection):** "
              f"{int(((t.a_final_proj - t.a_final_full).abs() <= 0.05).sum())} of {len(t)} cells.", ""]


def m2_m4(runs, lines):
    r = runs[runs.hypothesis == "v3_t12"].copy()
    r["m"] = r.majority.round(3)
    g = r[r["update"] == "expected_full"].groupby(["arch", "rho", "m", "optimizer", "eta"]).agg(
        runs=("margin_growth", "size"), median_g=("margin_growth", "median"),
        sinking=("margin_growth", lambda v: int((v >= 1).sum())), slower=("margin_growth", lambda v: int(((v > 0) & (v < 1)).sum())),
        recovering=("margin_growth", lambda v: int((v <= 0).sum())), b_final=("b_final", "median")).reset_index()
    lines += ["## M2 (addendum 1): is the minority still sinking at 10× the horizon?", "",
              "g = Δ[−log p̄_B] / Δ[−log(1 − p̄_A)] over the last decade of training (exact expected update). "
              "`sinking`: runs with g ≥ 1; `slower`: 0 < g < 1; `recovering`: g ≤ 0.", "", markdown(g), ""]
    t = runs[(runs.hypothesis == "v3_t13") & (runs["update"] == "expected_full")].copy()
    t["m"] = t.majority.round(3)
    cols = [f"abs_{c}_t{k}" for c in ("b", "a") for k in range(3)]
    m4 = t.groupby(["arch", "rho", "m"])[cols].mean().reset_index()
    m4.columns = ["arch", "rho", "m", "minority, low margin", "minority, mid", "minority, high margin",
                  "majority, low margin", "majority, mid", "majority, high margin"]
    spread = (m4.iloc[:, 3:6].max(axis=1) - m4.iloc[:, 3:6].min(axis=1))
    whole = m4.iloc[:, 3:6].min(axis=1) > 0.5
    lines += ["## M4 (addendum 1): imbalance or margin?", "",
              "Share of prompts absorbed, by class and by tercile of score margin (terciles over all prompts of the "
              "task); T13 runs with the unmodified update (the T1 configuration), mean over tasks and both η.", "",
              markdown(m4), "",
              f"- Minority terciles are within 0.15 of each other in {int((spread[whole] <= 0.15).sum())} of {int(whole.sum())} cells "
              f"where every tercile is mostly absorbed ({int((spread <= 0.15).sum())} of {len(m4)} cells overall).",
              f"- Majority prompts in the lowest-margin tercile: largest absorbed share {m4['majority, low margin'].max():.3f} "
              "(prediction: below 0.05).", ""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("collect", "all"))
    parser.add_argument("--raw", default=os.path.expanduser("~/rlvr-data/v3d"))
    parser.add_argument("--out", default="extra/analysis/v3d")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.what == "collect":
        collect(args.raw, args.jobs)
        return
    for sub in ("tables", "figures"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    runs = pd.read_csv(os.path.join(args.raw, "cache", "runs.csv"))
    br = pd.read_csv(os.path.join(args.raw, "cache", "branches.csv"))
    lines = ["# Findings: v3d (pre-registered in PREREG_v3d.md, exploratory)", "",
             f"Generated by `extra/experiments/analyze_v3d.py` from {len(runs)} runs. Class A is answer class 0 (share m); "
             "class B is the rest (the minority when m > 1/3).", ""]
    t11(br, lines, args.out)
    t12(runs, lines, args.out)
    t13(runs, lines, args.out)
    m2_m4(runs, lines)
    with open(os.path.join(args.out, "FINDINGS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}/FINDINGS.md")


if __name__ == "__main__":
    main()
