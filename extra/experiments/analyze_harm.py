"""
Why losing the minority is harmful: pass@k, crossover k*, coverage and worst-class pass@1 on the saved confirmatory
runs of battery A (v3o: A1, A2, A3). Descriptive, not a registered test; no new training.

usage:
    python extra/experiments/analyze_harm.py collect [--raw ~/rlvr-data/v3o] [--jobs 3]
    python extra/experiments/analyze_harm.py analyze [--raw ~/rlvr-data/v3o] [--out extra/analysis/v3o]

collect reads every v3_a1, v3_a2 and v3_a3 result once and writes <raw>/cache/harm_runs.csv: per run, the measures
below for the start policy (p_init) and the final policy (the snapshot at the last step), overall, for the majority
(class 0) and for the minority (classes 1..K-1 pooled). The class labels come from <raw>/cache/labels.npz, written by
`analyze_v3o.py collect`.

- pass@k = mean over prompts of 1 - (1 - p)^k, k in KS (Yue et al., NeurIPS 2025), from the exact per-prompt p.
- k* = the smallest k in KS at which the start policy's pass@k is at least the final policy's (none if never).
- coverage C = mean over prompts of 1/p (Huang et al. 2026, Definition 1.1), reported as log10 C. Snapshots are
  float32 with denormals flushed, so a p below 1.18e-38 is stored as 0. `logC` clips p at that float32 floor;
  `logC30` (the reported one) clips p at 1e-30 and is unaffected by the storage floor. Both are lower bounds when
  prompts sit at the clip. `floor_*` is the share of prompts stored as 0, `below30_*` the share below 1e-30.
- worst-class pass@1 = the smallest class mean of p (Sagawa et al., ICLR 2020, worst-group accuracy).

analyze aggregates per A1 cell (arch/rho/optimizer/update) and majority share m over the cell's A1 eta set
(`analyze_v3o.a1`) and per A2 and A3 arm over that grid's cells: each task is the mean over its runs in the group,
then the median and the range over the 20 tasks. k* is computed on each task's mean pass@k curves and summarized
by its lower median over tasks.
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

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # repo root
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]  # extra/

from experiments import analyze_v3o  # noqa: E402
from experiments.analyze import markdown  # noqa: E402
from experiments.analyze_v3o import LABELS, label_key  # noqa: E402

KS = (1, 2, 4, 8, 16, 32, 64)
F32_FLOOR = float(np.finfo(np.float32).tiny)  # 1.18e-38: the smallest stored non-zero p
CLIP = 1e-30
GRIDS = ("v3_a1", "v3_a2", "v3_a3")
EXPECTED = {"v3_a1": 9280, "v3_a2": 640, "v3_a3": 360}
GROUPS = ("all", "maj", "min")


# ---------------------------------------------------------------- measures

def pass_at_k(p, k):
    """Mean over prompts of 1 - (1 - p)^k."""
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        return float(np.mean(-np.expm1(k * np.log1p(-p))))


def crossover(start, final, ks=KS):
    """Smallest k at which start[i] >= final[i]; None if the final policy wins at every k."""
    for k, s, f in zip(ks, start, final):
        if s >= f:
            return k
    return None


def log10_coverage(p, floor):
    """log10 of the mean over prompts of 1/p, with p clipped below at floor."""
    return float(np.log10(np.mean(1.0 / np.maximum(np.asarray(p, float), floor))))


def measures(p, cls, when):
    row = {}
    masks = {"all": np.ones(len(p), bool), "maj": cls == 0, "min": cls != 0}
    for g, mask in masks.items():
        q = p[mask]
        for k in KS:
            row[f"pass{k}_{when}_{g}"] = pass_at_k(q, k)
        row[f"logC_{when}_{g}"] = log10_coverage(q, F32_FLOOR)
        row[f"logC30_{when}_{g}"] = log10_coverage(q, CLIP)
        row[f"floor_{when}_{g}"] = float(np.mean(q == 0))
        row[f"below30_{when}_{g}"] = float(np.mean(q < CLIP))
    row[f"log_median_inv_{when}_min"] = float(np.log10(np.median(1.0 / np.maximum(p[cls != 0], F32_FLOOR))))
    means = [float(p[cls == c].mean()) for c in np.unique(cls)]
    row[f"worst_{when}"] = min(means)
    for c, v in enumerate(means):
        row[f"class{c}_{when}"] = v
    return row


def arm(cfg):
    if cfg["hypothesis"] == "v3_a2":
        return "none" if not cfg["beta"] else f"{cfg['kl']} {cfg['beta']:g}"
    if cfg["hypothesis"] == "v3_a3":
        return cfg["rule"] if cfg.get("passk") is None else f"{cfg['rule']}{cfg['passk']}"
    return ""


def read_one(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    cfg = d["config"]
    cls = LABELS[label_key(cfg)]
    last = max(d["snapshots"], key=int)
    row = {k: cfg.get(k) for k in ("hypothesis", "arch", "rho", "majority", "optimizer", "update", "eta", "kl", "beta",
                                   "rule", "passk", "G", "init", "steps", "task_seed")}
    row.update(arm=arm(cfg), final_step=int(last), n_train=len(cls))
    row.update(measures(np.asarray(d["p_init"], float), cls, "start"))
    row.update(measures(np.asarray(d["snapshots"][last], float), cls, "final"))
    for g in GROUPS:
        k = crossover([row[f"pass{k}_start_{g}"] for k in KS], [row[f"pass{k}_final_{g}"] for k in KS])
        row[f"kstar_{g}"] = np.nan if k is None else k
    return row


def init_worker(labels):
    LABELS.update(labels)


def collect(raw, jobs):
    cache = os.path.join(raw, "cache")
    z = np.load(os.path.join(cache, "labels.npz"), allow_pickle=True)
    labels = dict(zip(map(tuple, z["keys"].tolist()), z["cls"]))
    paths = sorted(p for h in GRIDS for p in glob.glob(os.path.join(raw, "*", "results", h, "*.json.gz")))
    print(f"reading {len(paths)} runs", flush=True)
    with Pool(jobs, initializer=init_worker, initargs=(labels,)) as pool:
        rows = pool.map(read_one, paths, chunksize=32)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cache, "harm_runs.csv"), index=False)
    print(df.hypothesis.value_counts().to_string(), flush=True)


# ---------------------------------------------------------------- aggregation

def value_cols(df):
    return [c for c in df.columns if c.startswith(("pass", "logC", "floor_", "below30_", "log_median", "worst_",
                                                    "class"))]


def summarize(sel, **key):
    """Per task: mean over the group's runs; then median, min and max over tasks. k* from each task's mean curves."""
    cols = value_cols(sel)
    task = sel.groupby("task_seed")[cols].mean()
    row = dict(key, runs=len(sel), tasks=len(task))
    for g in GROUPS:
        ks = [crossover([t[f"pass{k}_start_{g}"] for k in KS], [t[f"pass{k}_final_{g}"] for k in KS])
              for _, t in task.iterrows()]
        found = [k for k in ks if k is not None]
        row[f"kstar_{g}_tasks"] = f"{len(found)}/{len(ks)}"
        # lower median over tasks (so it is a k in KS), with "none" as +inf
        med = sorted(np.inf if k is None else k for k in ks)[(len(ks) - 1) // 2]
        row[f"kstar_{g}"] = med if np.isfinite(med) else np.nan
        row[f"kstar_{g}_min"] = min(found) if found else np.nan
        row[f"kstar_{g}_max"] = max(found) if found else np.nan
    for c in cols:
        row[c] = task[c].median()
        row[f"{c}_lo"], row[f"{c}_hi"] = task[c].min(), task[c].max()
    return row


def a1_etas(raw):
    runs = pd.read_csv(os.path.join(raw, "cache", "v3_a1.csv"))
    _, cells, _ = analyze_v3o.a1(runs)
    return {c: [float(x) for x in str(e).split()] for c, e in zip(cells.cell, cells.etas)}


def aggregate(runs, etas):
    rows = []
    a1 = runs[(runs.hypothesis == "v3_a1") & (runs["update"] != "supervised")].copy()
    a1["m"] = a1.majority.round(3)
    for (arch, rho, opt, update), g in a1.groupby(["arch", "rho", "optimizer", "update"]):
        cell = f"{arch}/{rho}/{opt}/{update}"
        sel = g[np.isclose(g.eta.values[:, None], etas[cell]).any(axis=1)]
        for m, gm in sel.groupby("m"):
            rows.append(summarize(gm, grid="v3_a1", cell=cell, m=m, arm="grpo",
                                  etas=" ".join(f"{e:g}" for e in etas[cell])))
    for h in ("v3_a2", "v3_a3"):
        g = runs[runs.hypothesis == h]
        for a, ga in g.groupby("arm"):
            first = ga.iloc[0]
            cell = f"{'/'.join(sorted(ga.arch.unique()))}/{first.rho}/{first.optimizer}/{'+'.join(sorted(ga['update'].unique()))}"
            rows.append(summarize(ga, grid=h, cell=cell, m=round(first.majority, 3), arm=a,
                                  etas=" ".join(f"{e:g}" for e in sorted(ga.eta.unique()))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report

def rng(row, c, f="{:.3g}"):
    return f"{f.format(row[c])} ({f.format(row[c + '_lo'])}–{f.format(row[c + '_hi'])})"


def kstar(row):
    return "none" if np.isnan(row["kstar_all"]) else f"{row['kstar_all']:g} ({row['kstar_all_tasks']})"


def headline(cells):
    t = cells[(cells.grid == "v3_a1") & cells.m.isin([0.333, 0.9])]
    return pd.DataFrame([{
        "cell": r.cell, "m": f"{r.m:g}",
        "pass@1 start": f"{r.pass1_start_all:.3f}", "pass@1 final": rng(r, "pass1_final_all", "{:.3f}"),
        "worst-class pass@1 final": rng(r, "worst_final", "{:.2g}"),
        "pass@8 start → final": f"{r.pass8_start_all:.3f} → {r.pass8_final_all:.3f}",
        "pass@64 start → final": f"{r.pass64_start_all:.3f} → {r.pass64_final_all:.3f}",
        "k* (tasks)": kstar(r),
        "log10 C start → final": f"{r.logC30_start_all:.2f} → {r.logC30_final_all:.1f}",
        "minority at floor": f"{r.floor_final_min:.2f}"} for _, r in t.iterrows()])


def secondary(cells):
    t = cells[cells.grid != "v3_a1"]
    return pd.DataFrame([{
        "grid": r.grid, "cell": r.cell, "m": f"{r.m:g}", "η": r.etas, "arm": r.arm, "runs": r.runs,
        "pass@1 final": rng(r, "pass1_final_all", "{:.3f}"),
        "minority pass@1 final": rng(r, "pass1_final_min", "{:.2g}"),
        "minority pass@64 final": f"{r.pass64_final_min:.3g}",
        "worst-class pass@1 final": f"{r.worst_final:.2g}",
        "pass@8 start → final": f"{r.pass8_start_all:.3f} → {r.pass8_final_all:.3f}",
        "pass@64 start → final": f"{r.pass64_start_all:.3f} → {r.pass64_final_all:.3f}",
        "k* (tasks)": kstar(r),
        "log10 C start → final": f"{r.logC30_start_all:.2f} → {r.logC30_final_all:.2f}",
        "minority at floor": f"{r.floor_final_min:.2f}"} for _, r in t.iterrows()])


def analyze(raw, out):
    runs = pd.read_csv(os.path.join(raw, "cache", "harm_runs.csv"))
    counts = runs.hypothesis.value_counts().to_dict()
    assert counts == EXPECTED, counts
    uniform = runs[runs.init == "uniform"]
    dev = max(abs(uniform[f"pass{k}_start_all"] - (1 - (2 / 3) ** k)).max() for k in KS)
    print(f"runs: {counts}; uniform start: {len(uniform)}/{len(runs)}; "
          f"max |start pass@k - (1 - (2/3)^k)| = {dev:.2g}")
    assert dev < 1e-6
    final = runs[runs["update"] != "supervised"]
    print(f"final snapshots: p stored as 0 in {(final.floor_final_all > 0).sum()} of {len(final)} runs; "
          f"share of all prompts at 0: {final.floor_final_all.mean():.3f}")
    cells = aggregate(runs, a1_etas(raw))
    os.makedirs(out, exist_ok=True)
    cells.to_csv(os.path.join(out, "harm_cells.csv"), index=False)
    print(markdown(headline(cells)))
    print()
    print(markdown(secondary(cells)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("collect", "analyze", "all"))
    parser.add_argument("--raw", default=os.path.expanduser("~/rlvr-data/v3o"))
    parser.add_argument("--out", default="extra/analysis/v3o")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.what in ("collect", "all"):
        collect(args.raw, args.jobs)
    if args.what in ("analyze", "all"):
        analyze(args.raw, args.out)


if __name__ == "__main__":
    main()
