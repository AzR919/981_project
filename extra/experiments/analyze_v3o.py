"""
Confirmatory replication battery A1-A5 (batches v3o and v3q), implementing PREREG_v3d_addendum9.md.

usage:
    python experiments/analyze_v3o.py collect [--raw ~/rlvr-data/v3o] [--raw-q ~/rlvr-data/v3q] [--jobs 3]
    python experiments/analyze_v3o.py analyze [--raw ~/rlvr-data/v3o] [--out analysis/v3o]

collect reads every result once and writes one CSV per test to <raw>/cache. The class labels of each toy task
(rho, m, task seed) are rebuilt once with run.build (via analyze_v3.t1_labels) and shared by A1, A2, A3 and A5;
the architecture does not change the task. The per-run measures reuse the original analysis code:
A2 analyze_v3g.t17_row, A3 analyze_v3f.read_one, A4 analyze_v3h.read_one, A5 analyze_v3k.read_one (the last
two with the cached labels in place of a rebuild per run); A1 repeats the T1 branch of analyze_v3.read_one.
collect also reads the original T17 runs (~/rlvr-data/v3g) for A2's 1e-2 secondary on the original tasks.

Five one-sided sign tests (ties dropped), one Holm family at alpha = 0.05. Absorbed (toy): tail max below 1e-3
over the last 20% of steps, among prompts starting at or above 1e-2.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import analyze_v3, analyze_v3f, analyze_v3g, analyze_v3h, analyze_v3k  # noqa: E402
from experiments.analyze import markdown  # noqa: E402

ALPHA = 0.05
TAU = 1e-3
PERSIST = analyze_v3.PERSIST  # "persist_0.001"
TOY = ("v3_a1", "v3_a2", "v3_a3", "v3_a5")
TESTS = ("A1", "A2", "A3", "A4", "A5")
LABELS = {}


# ---------------------------------------------------------------- labels

def label_key(cfg):
    return (int(cfg["K"]), float(cfg["rho"]), round(float(cfg["majority"]), 6), int(cfg["task_seed"]))


def build_labels(cfg):
    return analyze_v3.t1_labels(cfg)[0]


def cached_t1_labels(cfg):
    """Drop-in for analyze_v3.t1_labels inside the original readers (the margin is not used there)."""
    return LABELS[label_key(cfg)], None


def cached_build(cfg):
    """Drop-in for run.build inside analyze_v3k.read_one, which only uses train[1][:, 0]."""
    return None, (None, LABELS[label_key(cfg)][:, None]), None, None


def init_worker(labels):
    LABELS.update(labels)
    analyze_v3g.LABELS.update(labels)          # t17_row looks labels up in its own cache
    analyze_v3f.t1_labels = cached_t1_labels  # read_one would rebuild the task for every run
    analyze_v3k.build = cached_build


# ---------------------------------------------------------------- per-run rows

def a1_row(d):
    """The T1 branch of analyze_v3.read_one (it checks the hypothesis name, so it cannot be called on v3_a1)."""
    cfg = d["config"]
    row = {k: cfg.get(k) for k in analyze_v3.CONFIG_KEYS}
    p_init, tail_max = np.asarray(d["p_init"]), np.asarray(d["tail_max"])
    for tau in analyze_v3.TAUS:
        row[f"persist_{tau:g}"] = analyze_v3.persistent(p_init, tail_max, tau)
    cls = LABELS[label_key(cfg)]
    absorbed = (tail_max < TAU) & (p_init >= 10 * TAU)
    row["n_absorbed"] = int(absorbed.sum())
    row["n_absorbed_minority"] = int((absorbed & (cls != 0)).sum())
    row["minority_share"] = float((cls != 0).mean())
    row["n_train"] = len(p_init)
    return row


def a2_row(d):
    row = analyze_v3g.t17_row(d)
    cls = LABELS[label_key(d["config"])]
    tail, p0 = np.asarray(d["tail_max"]), np.asarray(d["p_init"])
    row["minority_absorbed_1e2"] = float(((tail < 1e-2) & (p0 >= 1e-1))[cls != 0].mean())
    return row


def read_toy(path):
    h = path.split(os.sep)[-2]
    if h == "v3_a3":
        return h, analyze_v3f.read_one(path)
    if h == "v3_a5":
        return h, analyze_v3k.read_one(path)
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    return h, (a1_row(d) if h in ("v3_a1",) else a2_row(d))


def collect(raw, raw_q, raw_g, jobs):
    cache = os.path.join(raw, "cache")
    os.makedirs(cache, exist_ok=True)
    paths = sorted(p for h in TOY for p in glob.glob(os.path.join(raw, "*", "results", h, "*.json.gz")))
    orig = sorted(glob.glob(os.path.join(raw_g, "*", "results", "v3_t17", "*.json.gz")))
    first = {}
    for p in paths + orig:
        if "/v3_a1/" in p and len(first) >= 160:
            continue
        with gzip.open(p, "rt") as f:
            cfg = json.load(f)["config"]
        first.setdefault(label_key(cfg), cfg)
    label_file = os.path.join(cache, "labels.npz")
    labels = {}
    if os.path.exists(label_file):
        z = np.load(label_file, allow_pickle=True)
        labels = dict(zip(map(tuple, z["keys"].tolist()), z["cls"]))
    todo = {k: c for k, c in first.items() if k not in labels}
    print(f"{len(first)} tasks, building {len(todo)}", flush=True)
    if todo:
        with Pool(jobs) as pool:
            labels.update(zip(todo, pool.map(build_labels, todo.values())))
        keys = list(labels)
        np.savez(label_file, keys=np.array(keys, dtype=object), cls=np.stack([labels[k] for k in keys]))
    print(f"reading {len(paths)} toy runs and {len(orig)} original T17 runs", flush=True)
    with Pool(jobs, initializer=init_worker, initargs=(labels,)) as pool:
        results = pool.map(read_toy, paths, chunksize=32)
        orig_rows = pool.map(read_toy, orig, chunksize=32)
    for h in TOY:
        df = pd.DataFrame([r for t, r in results if t == h])
        df.to_csv(os.path.join(cache, f"{h}.csv"), index=False)
        print(f"{h}: {len(df)} runs", flush=True)
    pd.DataFrame([r for _, r in orig_rows]).to_csv(os.path.join(cache, "orig_v3_t17.csv"), index=False)
    q = sorted(glob.glob(os.path.join(raw_q, "*", "results", "v3_a4", "*.json.gz")))
    df = pd.DataFrame([analyze_v3h.read_one(p) for p in q])
    df.to_csv(os.path.join(cache, "v3_a4.csv"), index=False)
    print(f"v3_a4: {len(df)} runs", flush=True)


# ---------------------------------------------------------------- statistics

def sign_test(diff):
    """One-sided exact sign test that the differences are > 0; exact zeros are dropped."""
    diff = np.asarray(diff, float)
    up, down = int((diff > 0).sum()), int((diff < 0).sum())
    n = up + down
    p = float(stats.binomtest(up, n, 0.5, alternative="greater").pvalue) if n else 1.0
    return dict(up=up, down=down, ties=int(len(diff) - n), n=n, p=p)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[j]))
        adj[j] = running
    return adj


def frac(mask):
    return f"{int(np.sum(mask))}/{len(mask)}"


def fmt(x):
    return f"{x:.3g}"


# ---------------------------------------------------------------- A1

def original_p1():
    """The original task-level table (analysis/v3a/P1_TASK_LEVEL.md), for the effect sizes."""
    path = os.path.join("analysis", "v3a", "P1_TASK_LEVEL.md")
    rows = [line.strip().strip("|").split("|") for line in open(path)
            if line.startswith("| ") and "/" in line.split("|")[1]]
    t = pd.DataFrame([[c.strip() for c in r] for r in rows],
                     columns=["cell", "tasks_up", "p_task", "m0.333", "m0.5", "m0.7", "m0.9"])
    return t.set_index("cell")


def original_etas():
    """The no-overshoot eta sets of the original T1 analysis (analysis/v3a/tables/v3_t1_cells.csv)."""
    o = pd.read_csv(os.path.join("analysis", "v3a", "tables", "v3_t1_cells.csv"))
    cell = o.arch + "/" + o.rho.astype(str) + "/" + o.optimizer + "/" + o["update"]
    return {c: sorted(float(x) for x in str(e).replace(",", " ").split()) for c, e in zip(cell, o.etas)}


def a1(runs, fixed=None):
    rl = runs[runs["update"] != "supervised"].copy()
    rl["m"] = rl.majority.round(3)
    rl["minority_rate"] = rl.n_absorbed_minority / (rl.minority_share * rl.n_train)
    orig = original_p1()
    rows, per_task = [], []
    for (arch, rho, opt, update), g in rl.groupby(["arch", "rho", "optimizer", "update"]):
        cell = f"{arch}/{rho}/{opt}/{update}"
        bal = g[g.m == 0.333].groupby("eta")[PERSIST].median()
        etas = sorted(bal.index[bal <= 0.02]) if fixed is None else fixed[cell]
        row = dict(cell=cell, n_eta=len(etas), etas=" ".join(f"{e:g}" for e in etas))
        if not etas:
            rows.append(dict(row, tasks="0", up=0, down=0, ties=0, p=1.0))
            continue
        sel = g[g.eta.isin(etas)]
        t = sel.groupby(["task_seed", "m"])[[PERSIST, "minority_rate"]].mean().unstack("m")
        rate = t["minority_rate"]
        diff = rate[0.9] - rate[0.333]
        s = sign_test(diff)
        sp = sign_test(t[(PERSIST, 0.9)] - t[(PERSIST, 0.333)])
        try:
            wp = float(stats.wilcoxon(t[(PERSIST, 0.9)], t[(PERSIST, 0.333)], alternative="greater").pvalue)
        except ValueError:
            wp = 1.0
        o = orig.loc[cell] if cell in orig.index else None
        rows.append(dict(row, tasks=len(diff), up=s["up"], down=s["down"], ties=s["ties"], p=s["p"],
                         **{f"rate_m{m:g}": rate[m].mean() for m in (0.333, 0.5, 0.7, 0.9)},
                         orig_rate_m0333=o["m0.333"] if o is not None else "", orig_rate_m09=o["m0.9"] if o is not None else "",
                         persist_up=f"{sp['up']}/{sp['n']}", persist_sign_p=sp["p"], persist_wilcoxon_p=wp))
        per_task.append(pd.DataFrame(dict(cell=cell, task_seed=diff.index, rate_m0333=rate[0.333].values,
                                          rate_m09=rate[0.9].values, diff=diff.values)))
    cells = pd.DataFrame(rows)
    worst = cells.loc[cells.p.idxmax()]
    if cells.p.nunique() == 1:
        stat = (f"cells with p < 0.05: {int((cells.p < ALPHA).sum())}/16; every cell {worst.up} up, {worst.down} down, "
                f"{worst.ties} ties")
    else:
        stat = (f"cells with p < 0.05: {int((cells.p < ALPHA).sum())}/16; worst cell {worst.cell}: "
                f"{worst.up} up, {worst.down} down, {worst.ties} ties")
    return dict(test="A1", p=float(cells.p.max()), stat=stat, n="16 cells × 20 tasks"), cells, pd.concat(per_task)


# ---------------------------------------------------------------- A2

FWD = (0.03, 0.1, 0.3)


def a2_core(runs):
    rows = []
    for (arch, update, eta), g in runs.groupby(["arch", "update", "eta"]):
        base = g[g.beta == 0].set_index("task_seed").minority_absorbed
        fwd = g[(g.kl == "forward") & g.beta.isin(FWD)].groupby("task_seed").minority_absorbed.max()
        d = (base - fwd).dropna()
        rows.append(pd.DataFrame(dict(arch=arch, update=update, eta=eta, task_seed=d.index, base=base[d.index].values,
                                      fwd_max=fwd[d.index].values, diff=d.values)))
    per = pd.concat(rows)
    task = per.groupby("task_seed")["diff"].median()
    return per, task


def a2_secondary(runs):
    fwd = runs[(runs.kl == "forward") & runs.beta.isin(FWD)]
    out = []
    for beta, g in list(fwd.groupby("beta")) + [("all", fwd)]:
        out.append(dict(beta=beta, runs=len(g), none_1e3=frac(g.minority_absorbed == 0),
                        none_1e2=frac(g.minority_absorbed_1e2 == 0),
                        none_both=frac((g.minority_absorbed == 0) & (g.minority_absorbed_1e2 == 0)),
                        mean_absorbed_1e3=g.minority_absorbed.mean(), mean_absorbed_1e2=g.minority_absorbed_1e2.mean(),
                        reached=frac(g.reached), minority_at_match=g.minority_at_match.median()))
    base = runs[runs.beta == 0]
    out.append(dict(beta=0.0, runs=len(base), none_1e3=frac(base.minority_absorbed == 0),
                    none_1e2=frac(base.minority_absorbed_1e2 == 0),
                    none_both=frac((base.minority_absorbed == 0) & (base.minority_absorbed_1e2 == 0)),
                    mean_absorbed_1e3=base.minority_absorbed.mean(), mean_absorbed_1e2=base.minority_absorbed_1e2.mean(),
                    reached=frac(base.reached), minority_at_match=base.minority_at_match.median()))
    n9 = lambda g, mask: mask.sum() >= 0.9 * len(g)  # noqa: E731
    p3 = fwd.groupby(["arch", "update", "eta", "beta"]).apply(
        lambda g: n9(g, g.minority_absorbed == 0) and n9(g, g.reached) and n9(g, g.minority_at_match < 0.2),
        include_groups=False)
    return pd.DataFrame(out), p3


def a2(runs, orig):
    per, task = a2_core(runs)
    s = sign_test(task)
    sec, p3 = a2_secondary(runs)
    orig_t17 = orig[orig.beta.eq(0) | (orig.kl.eq("forward") & orig.beta.isin(FWD))]
    _, otask = a2_core(orig_t17)
    osec, op3 = a2_secondary(orig_t17)
    stat = (f"{s['up']} up, {s['down']} down, {s['ties']} ties; median task difference {task.median():.3f} "
            f"(original 10 tasks: {sign_test(otask)['up']} up, median {otask.median():.3f})")
    extra = dict(sec=sec, osec=osec, p3=f"{int(p3.sum())}/{len(p3)}", op3=f"{int(op3.sum())}/{len(op3)}", task=task)
    return dict(test="A2", p=s["p"], stat=stat, n=f"{s['n']} tasks"), per, extra


# ---------------------------------------------------------------- A3

def a3_core(runs):
    r = runs.copy()
    r["value"] = r.minority_at_match
    pk = r.rule.isin(["passk4", "rloo_k4"])
    r.loc[pk & ~r.reached, "value"] = 0.0     # pass@4 runs that never reach 0.80 count against it
    w = r.pivot_table(index=["task_seed", "G", "eta"], columns="rule", values="value", aggfunc="first", dropna=False)
    w["pass4"] = w[["passk4", "rloo_k4"]].max(axis=1)
    w["cell_diff"] = w.pass4 - w.grpo
    task = w.groupby("task_seed").agg(pass4=("pass4", "median"), grpo=("grpo", "median"),
                                      median_cell_diff=("cell_diff", "median"))
    task["diff"] = task.pass4 - task.grpo
    return w.reset_index(), task.reset_index()


def a3(runs, orig):
    w, task = a3_core(runs)
    s = sign_test(task["diff"])
    sens = sign_test(task.median_cell_diff)
    _, otask = a3_core(orig[orig.rule.isin(["grpo", "passk4", "rloo_k4"])])
    os_ = sign_test(otask["diff"])
    cells = []
    for src, df in (("fresh", runs), ("original", orig[orig.rule.isin(["grpo", "passk4", "rloo_k4"])])):
        for (rule, G, eta), g in df.groupby(["rule", "G", "eta"]):
            rc = g[g.reached]
            cells.append(dict(data=src, rule=rule, G=G, eta=eta, reached=frac(g.reached),
                              minority_at_match=g.minority_at_match.median(),
                              above_02=f"{int((rc.minority_at_match > 0.2).sum())}/{len(rc)}",
                              below_001=f"{int((rc.minority_at_match < 0.01).sum())}/{len(rc)}",
                              minority_absorbed=g.minority_absorbed.mean(), minority_final=g.minority_final.median()))
    cells = pd.DataFrame(cells)
    grpo_unreached = int((~runs[runs.rule == "grpo"].reached).sum())
    stat = (f"{s['up']} up, {s['down']} down, {s['ties']} ties; median task difference {task['diff'].median():.3f} "
            f"(original 10 tasks: {os_['up']} up, median {otask['diff'].median():.3f})")
    extra = dict(task=task, otask=otask, sens=sens, grpo_unreached=grpo_unreached,
                 pass_unreached=int((~runs[runs.rule != "grpo"].reached).sum()))
    return dict(test="A3", p=s["p"], stat=stat, n=f"{s['n']} tasks"), cells, extra


# ---------------------------------------------------------------- A4

def a4_core(runs):
    r = runs[(runs.policy == "shared") & (runs.optimizer == "adam") & np.isclose(runs.eta, 1e-3)
             & (runs.partition == "random")].copy()
    r["share"] = r.n_major / 64
    r["absorbed"] = (r.absorbed_b >= 0.5) & (r.absorbed_a < 0.5)
    r["group"] = np.where(r.share >= 0.875, "high",
                          np.where(((r.K == 3) & (r.n_major == 22)) | ((r.K == 2) & (r.n_major == 32)), "balanced", ""))
    r = r[r.group != ""]
    seeds = r.pivot_table(index="seed", columns="group", values="absorbed", aggfunc="mean")
    seeds["diff"] = seeds.high - seeds.balanced
    n = r.groupby(["seed", "group"]).size().unstack()
    seeds["n_high"], seeds["n_balanced"] = n.high, n.balanced
    return r, seeds.reset_index()


def a4(runs, orig):
    r, seeds = a4_core(runs)
    s = sign_test(seeds["diff"])
    orr, oseeds = a4_core(orig)
    os_ = sign_test(oseeds["diff"])
    _, pg = a4_core(runs[runs.algo == "pg"])
    sens_pg = sign_test(pg["diff"])
    cells = []
    for src, df in (("fresh", r), ("original", orr)):
        for (group, algo), g in df.groupby(["group", "algo"]):
            cells.append(dict(data=src, group=group, algo=algo, runs=len(g), minority_absorbed_class=frac(g.absorbed),
                              class0_lost_half=frac(g.absorbed_a >= 0.5), per_prompt_minority=g.absorbed_b.mean(),
                              per_prompt_class0=g.absorbed_a.mean(), minority_final=g.minority_final.median(),
                              majority_final=g.majority_final.median()))
    cells = pd.DataFrame(cells)
    stat = (f"{s['up']} up, {s['down']} down, {s['ties']} ties; mean seed difference {seeds['diff'].mean():.3f} "
            f"(original 10 seeds: {os_['up']} up, {os_['down']} down, mean {oseeds['diff'].mean():.3f})")
    return dict(test="A4", p=s["p"], stat=stat, n=f"{s['n']} seeds"), seeds, dict(cells=cells, oseeds=oseeds,
                                                                                 sens_pg=sens_pg, pg=pg)


# ---------------------------------------------------------------- A5

def a5_core(runs):
    r = runs[np.isclose(runs.m, 0.7) & runs.start.astype(str).isin(["uniform", "0.5"])]
    w = r.pivot_table(index=["task_seed", "arch", "update"], columns="start", values="absorbed_3", aggfunc="first")
    w["diff"] = w["uniform"] - w["0.5"]
    task = w.groupby("task_seed")["diff"].median()
    return w.reset_index(), task


def a5(runs, orig):
    w, task = a5_core(runs)
    s = sign_test(task)
    _, otask = a5_core(orig)
    os_ = sign_test(otask)
    cells = []
    for src, df in (("fresh", runs), ("original", orig[np.isclose(orig.m, 0.7) & orig.start.astype(str).isin(["uniform", "0.5"])])):
        for (arch, update, start), g in df.groupby(["arch", "update", df.start.astype(str)]):
            cells.append(dict(data=src, arch=arch, update=update, start=start, tasks=len(g),
                              absorbed_1e3=g.absorbed_3.mean(), absorbed_1e2=g.absorbed_2.mean(),
                              none_absorbed=frac(g.absorbed_3 == 0), minority_final=g.minority_final.median(),
                              majority_final=g.majority_final.median(), minority_lowest=g.minority_lowest.median()))
    cells = pd.DataFrame(cells)
    stat = (f"{s['up']} up, {s['down']} down, {s['ties']} ties; median task difference {task.median():.3f} "
            f"(original 10 tasks: {os_['up']} up, median {otask.median():.3f})")
    return dict(test="A5", p=s["p"], stat=stat, n=f"{s['n']} tasks"), w, dict(cells=cells, task=task)


# ---------------------------------------------------------------- report

def analyze(raw, out):
    cache = os.path.join(raw, "cache")
    load = lambda h: pd.read_csv(os.path.join(cache, f"{h}.csv"))  # noqa: E731
    runs = {h: load(h) for h in TOY + ("v3_a4",)}
    os.makedirs(out, exist_ok=True)
    keep = ["arch", "rho", "majority", "optimizer", "update", "eta", "task_seed", "kl", "beta", "rule", "G", "start",
            "K", "n_major", "algo", "seed", "persist_0.001", "n_absorbed", "n_absorbed_minority", "minority_share",
            "reached", "matched_step", "minority_at_match", "minority_absorbed", "minority_absorbed_1e2",
            "absorbed_3", "absorbed_2", "absorbed_a", "absorbed_b", "majority_final", "minority_final"]
    pd.concat([df.assign(test=h) for h, df in runs.items()], ignore_index=True).pipe(
        lambda d: d[["test"] + [c for c in keep if c in d.columns]]).to_csv(os.path.join(out, "runs.csv"), index=False)

    r1, c1, t1 = a1(runs["v3_a1"])
    r1o, c1o, _ = a1(runs["v3_a1"], fixed=original_etas())
    r2, p2, x2 = a2(runs["v3_a2"], load("orig_v3_t17"))
    r3, c3, x3 = a3(runs["v3_a3"], pd.read_csv(os.path.join("analysis", "v3f", "runs.csv")))
    r4, s4, x4 = a4(runs["v3_a4"], pd.read_csv(os.path.join("analysis", "v3h", "runs.csv")))
    r5, w5, x5 = a5(runs["v3_a5"], pd.read_csv(os.path.join("analysis", "v3k", "runs.csv")))
    res = pd.DataFrame([r1, r2, r3, r4, r5])
    res["holm_p"] = holm(res.p)
    res["verdict"] = np.where(res.holm_p < ALPHA, "replicated (confirmatory)", "did not replicate on fresh tasks")

    c1.to_csv(os.path.join(out, "a1_cells.csv"), index=False)
    t1.to_csv(os.path.join(out, "a1_tasks.csv"), index=False)
    p2.to_csv(os.path.join(out, "a2_cell_tasks.csv"), index=False)
    pd.concat([x2["sec"].assign(data="fresh"), x2["osec"].assign(data="original")]).to_csv(
        os.path.join(out, "a2_secondary.csv"), index=False)
    c3.to_csv(os.path.join(out, "a3_cells.csv"), index=False)
    x3["task"].to_csv(os.path.join(out, "a3_tasks.csv"), index=False)
    s4.to_csv(os.path.join(out, "a4_seeds.csv"), index=False)
    x4["cells"].to_csv(os.path.join(out, "a4_cells.csv"), index=False)
    w5.to_csv(os.path.join(out, "a5_cell_tasks.csv"), index=False)
    x5["cells"].to_csv(os.path.join(out, "a5_cells.csv"), index=False)
    res.to_csv(os.path.join(out, "results.csv"), index=False)

    a1_show = c1[["cell", "n_eta", "etas", "tasks", "up", "down", "ties", "p", "rate_m0.333", "rate_m0.5", "rate_m0.7",
                  "rate_m0.9", "orig_rate_m0333", "orig_rate_m09", "persist_up", "persist_wilcoxon_p"]]
    counts = {h: len(df) for h, df in runs.items()}
    lines = ["# Findings: v3o and v3q, confirmatory replication battery A1–A5 (PREREG_v3d addendum 9)", "",
             f"Generated by `experiments/analyze_v3o.py` from {sum(counts.values()):,} runs "
             f"({', '.join(f'{h} {n:,}' for h, n in counts.items())}); fresh task seeds 10–29 (toy) and partition "
             "seeds 10–29 (transformer). One-sided exact sign tests, ties dropped; one Holm family at α = 0.05.", "",
             "## Results", "",
             markdown(res[["test", "stat", "n", "p", "holm_p", "verdict"]].rename(
                 columns={"stat": "statistic", "holm_p": "Holm-adjusted p"})), "",
             "## A1: absorption per minority prompt rises with the majority share (16 cells)", "",
             "Per cell, the no-overshoot η set is the η at which the balanced arm (m = 1/3) has median persistent "
             "absorption ≤ 0.02 over the 20 tasks; each task is averaged over that set. `rate_m*`: absorbed minority "
             "prompts ÷ minority prompts, mean over tasks; `orig_rate_*`: the same on the original 10 tasks "
             "(P1_TASK_LEVEL.md). `persist_up` and `persist_wilcoxon_p`: the original P1 statistic (absorbed fraction of "
             "all prompts, exact Wilcoxon) on the fresh tasks, for reference only.", "", markdown(a1_show), "",
             f"The η sets are recomputed on the fresh tasks; they equal the original sets except in "
             f"{int((c1.etas != c1o.etas).sum())} cells, which gain one larger η. Sensitivity with the original sets: "
             f"{r1o['stat']}, largest p = {r1o['p']:.3g}; balanced-arm rate at most {c1o['rate_m0.333'].max():.3g}.", "",
             "## A2: forward KL prevents absorption", "",
             f"Per task: median over the 8 (architecture, update, η) cells of [minority share absorbed at β = 0] − "
             f"[largest share under forward KL β ∈ {{0.03, 0.1, 0.3}}]. Task medians: "
             f"{', '.join(f'{v:.3f}' for v in x2['task'])}.", "",
             "Secondary (registered): forward-KL runs with no minority prompt absorbed, at 1e-3 and at 1e-2 (1e-2: tail "
             "max below 1e-2 among prompts starting at or above 1e-1). `minority_at_match`: median minority class mean "
             "when the majority first reaches 0.80.", "", "Fresh tasks (10–29):", "", markdown(x2["sec"]), "",
             "Original tasks (0–9, recomputed from the v3g raw results):", "", markdown(x2["osec"]), "",
             f"Original T17 prediction 3 (none absorbed, majority reaches 0.80, minority below 0.2 at the matched point, "
             f"each in ≥ 90% of a cell's runs): fresh {x2['p3']} cells, original {x2['op3']}.", "",
             "## A3: pass@4 protects the minority at matched progress", "",
             f"Per task: median over the 6 (G, η) cells of max(pass@4, unbiased pass@4) minority mean at the matched point "
             f"(0 if the majority never reaches 0.80) minus the median over cells under GRPO. pass@4 runs that never reach "
             f"0.80: {x3['pass_unreached']}; GRPO runs that never reach it: {x3['grpo_unreached']}. Sensitivity (median "
             f"over cells of the per-cell difference): {x3['sens']['up']} up, {x3['sens']['down']} down, p = "
             f"{x3['sens']['p']:.3g}.", "", markdown(x3["task"]), "", markdown(c3), "",
             "## A4: under Adam the minority is absorbed at a high share and not when balanced", "",
             "Adam lr 1e-3, shared transformer, random partition, both algorithms (exact `pg` and sampled `grpo`). High "
             "share: n_major/64 ≥ 0.875 (n_major 56, 60; K = 2 and 3; 8 runs per seed). Balanced: n_major 22 at K = 3 "
             "and 32 at K = 2 (4 runs per seed). Minority absorbed: at least half of the non-zero classes' prompts "
             "absorbed while class 0 is not. Sensitivity, exact update (`pg`) only: "
             f"{x4['sens_pg']['up']} up, {x4['sens_pg']['down']} down, {x4['sens_pg']['ties']} ties, p = "
             f"{x4['sens_pg']['p']:.3g} (mean seed difference {x4['pg']['diff'].mean():.3f}).", "", markdown(s4), "", markdown(x4["cells"]), "",
             "## A5: a warm start at p0 = 0.5 protects the minority at m = 0.7", "",
             f"Per task: median over the 4 (architecture, update) cells of [minority share absorbed, uniform start] − "
             f"[at p0 = 0.5]. Task medians: {', '.join(f'{v:.3f}' for v in x5['task'])}.", "", markdown(x5["cells"]), ""]
    with open(os.path.join(out, "FINDINGS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(res[["test", "stat", "p", "holm_p", "verdict"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("collect", "analyze", "all"))
    parser.add_argument("--raw", default=os.path.expanduser("~/rlvr-data/v3o"))
    parser.add_argument("--raw-q", default=os.path.expanduser("~/rlvr-data/v3q"))
    parser.add_argument("--raw-g", default=os.path.expanduser("~/rlvr-data/v3g"))
    parser.add_argument("--out", default="analysis/v3o")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.what in ("collect", "all"):
        collect(args.raw, args.raw_q, args.raw_g, args.jobs)
    if args.what in ("analyze", "all"):
        analyze(args.raw, args.out)


if __name__ == "__main__":
    main()
