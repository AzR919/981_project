"""
Analysis of v3 batch 1 (T1, T2, T4, T6, T8), implementing analysis/review/PREREG_v3a.md.

usage:
    python experiments/analyze_v3.py collect [--raw results/v3a] [--jobs 8]
    python experiments/analyze_v3.py all [--raw results/v3a] [--out analysis/v3a]

collect reads every result once and writes <raw>/cache/v3_<t>.csv with one row
per run: the config, the T0 absorption fractions at each tau, the largest drop
below p_init, and (T1) how many absorbed prompts belong to a minority class.
The class labels and margins come from rebuilding each T1 task with run.build.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze import Report, assemble, markdown  # noqa: E402
from synthetic_rl.theory import fixed_points  # noqa: E402

TESTS = ("v3_t1", "v3_t2", "v3_t4", "v3_t6", "v3_t8")
TAUS = (1e-3, 1e-2)
PRIMARY_TAU = 1e-3
MATERIAL = 1e-3  # T6: a material drop is max_drop > 1e-3
N_BOOT = 10_000
CONFIG_KEYS = (
    "name", "level", "features", "rho", "arch", "init", "p0", "L", "G", "B", "rule", "eta", "beta", "steps",
    "seed", "task_seed", "update", "optimizer", "momentum", "reward_noise", "ablation", "majority", "replace",
    "ppo_epochs", "clip_eps", "grad_clip", "p_start", "offset",
)


# ---------------------------------------------------------------- T0 metrics

def persistent(p_init, tail_max, tau):
    """Persistent-absorption fraction among eligible prompts (p_init >= 10 tau)."""
    eligible = p_init >= 10 * tau
    if not eligible.any():
        return np.nan
    return float((tail_max[eligible] < tau).mean())


def best_by_tiebreak(medians, resolution):
    """Smallest key whose median is within `resolution` of the minimum (keys sorted ascending)."""
    finite = {k: v for k, v in medians.items() if np.isfinite(v)}
    if not finite:
        return np.nan
    low = min(finite.values())
    return min(k for k, v in finite.items() if v <= low + resolution)


# ---------------------------------------------------------------- collect

def t1_labels(cfg):
    """Correct class (0 is the majority) and score margin of each T1 training prompt."""
    from experiments.run import DEFAULTS, build

    task, train, _, _ = build(dict(DEFAULTS, **cfg))
    scores = np.einsum("nd,lkd->nlk", train[0], task.weights) + task.offset
    top2 = np.sort(scores[:, 0, :], axis=-1)[:, -2:]
    return train[1][:, 0], top2[:, 1] - top2[:, 0]


def task_key(cfg):
    return (cfg["rho"], round(cfg["majority"], 6), cfg["task_seed"])


LABELS = {}


def init_worker(labels):
    LABELS.update(labels)


def read_one(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    cfg = d["config"]
    row = {k: cfg.get(k) for k in CONFIG_KEYS}
    if isinstance(row["p0"], str):
        row["p0"] = np.nan  # "spread"
    p_init = np.asarray(d["p_init"])
    tail_max = np.asarray(d["tail_max"])
    min_p = np.asarray(d["min_p"])
    snaps = d["snapshots"]
    p_final = np.asarray(snaps[str(max(int(s) for s in snaps))])
    for tau in TAUS:
        eligible = p_init >= 10 * tau
        row[f"persist_{tau:g}"] = persistent(p_init, tail_max, tau)
        row[f"end_{tau:g}"] = float((p_final[eligible] < tau).mean()) if eligible.any() else np.nan
        row[f"ever_{tau:g}"] = float((min_p[eligible] < tau).mean()) if eligible.any() else np.nan
    row["max_drop"] = float(np.max(p_init - min_p))
    row["final_mean"] = float(p_final.mean())
    row["step_to_half"] = np.nan if d["step_to_half"] is None else float(d["step_to_half"])
    if cfg["hypothesis"] == "v3_t8":
        row["p_final"] = float(np.median(p_final))
    if cfg["hypothesis"] == "v3_t1":
        cls, _ = LABELS[task_key(cfg)]
        absorbed = (tail_max < PRIMARY_TAU) & (p_init >= 10 * PRIMARY_TAU)
        row["n_absorbed"] = int(absorbed.sum())
        row["n_absorbed_minority"] = int((absorbed & (cls != 0)).sum())
        row["minority_share"] = float((cls != 0).mean())
    return cfg["hypothesis"], row


def collect(raw, jobs):
    paths = sorted(glob.glob(os.path.join(raw, "*", "results", "*", "*.json.gz")))
    t1 = sorted(p for p in paths if "/v3_t1/" in p)
    # One label set per T1 task (rho, m, task seed); arch does not change the task.
    first = {}
    for p in t1:
        with gzip.open(p, "rt") as f:
            cfg = json.load(f)["config"]
        first.setdefault(task_key(cfg), cfg)
        if len(first) == 80:
            break
    print(f"building {len(first)} T1 tasks")
    with Pool(jobs) as pool:
        labels = dict(zip(first, pool.map(t1_labels, first.values())))
    cache = os.path.join(raw, "cache")
    os.makedirs(cache, exist_ok=True)
    rows = [dict(rho=k[0], majority=k[1], task_seed=k[2], cls=c, margin=m)
            for k, (cls, mar) in labels.items() for c, m in zip(cls, mar)]
    pd.DataFrame(rows).to_csv(os.path.join(cache, "v3_t1_prompts.csv"), index=False)
    print(f"reading {len(paths)} files with {jobs} processes")
    with Pool(jobs, initializer=init_worker, initargs=(labels,)) as pool:
        results = pool.map(read_one, paths, chunksize=64)
    for h in TESTS:
        df = pd.DataFrame([r for t, r in results if t == h]).sort_values("name")
        df.to_csv(os.path.join(cache, f"{h}.csv"), index=False)
        print(f"{h}: {len(df)} runs")


def load(raw, h):
    return pd.read_csv(os.path.join(raw, "cache", f"{h}.csv"))


# ---------------------------------------------------------------- stats helpers

def boot_tasks(df, value, stat, n=N_BOOT, seed=0):
    """Percentile CI of stat(df) when task seeds are resampled with replacement."""
    rng = np.random.default_rng(seed)
    tasks = df["task_seed"].unique()
    groups = {t: g for t, g in df.groupby("task_seed")}
    draws = []
    for _ in range(n):
        pick = rng.choice(tasks, size=len(tasks))
        draws.append(stat(pd.concat([groups[t] for t in pick])[value]))
    return np.quantile(draws, [0.025, 0.975])


def wilcoxon_greater(x, y):
    diff = np.asarray(x) - np.asarray(y)
    if np.all(diff == 0):
        return 1.0
    return float(stats.wilcoxon(x, y, alternative="greater").pvalue)


def spearman(x, y, alternative):
    if np.ptp(y) == 0:
        return np.nan, 1.0
    r = stats.spearmanr(x, y, alternative=alternative)
    return float(r.statistic), float(r.pvalue)


# ---------------------------------------------------------------- P1 (T1)

PERSIST = f"persist_{PRIMARY_TAU:g}"


def analyze_t1(runs, rep, raw):
    rep.say("## P1 (T1): absorption rises with majority share at fixed geometry")
    rep.say()
    rl = runs[runs["update"] != "supervised"].copy()
    rl["m"] = rl.majority.round(3)
    cells, worst = [], {"sgd": [], "adam": []}
    for (arch, rho, opt, update), g in rl.groupby(["arch", "rho", "optimizer", "update"]):
        bal = g[g.m == 0.333].groupby("eta")[PERSIST].median()
        etas = sorted(bal.index[bal <= 0.02])
        row = dict(arch=arch, rho=rho, optimizer=opt, update=update, n_eta=len(etas),
                   etas=",".join(f"{e:g}" for e in etas))
        if etas:
            sel = g[g.eta.isin(etas)]
            hi = sel[sel.m == 0.9].set_index(["task_seed", "eta"])[PERSIST]
            lo = sel[sel.m == 0.333].set_index(["task_seed", "eta"])[PERSIST]
            hi, lo = hi.align(lo, join="inner")
            p_a = wilcoxon_greater(hi.values, lo.values)
            rho_s, p_b = spearman(sel.m, sel[PERSIST], "greater")
            row.update(abs_bal=lo.mean(), abs_m07=sel[sel.m == 0.7][PERSIST].mean(), abs_m09=hi.mean(),
                       wilcoxon_p=p_a, spearman=rho_s, spearman_p=p_b, passed=(p_a < 0.05) and (p_b < 0.05))
            worst[opt].append(p_a)
        else:
            row.update(passed=np.nan)
        cells.append(row)
    cells = pd.DataFrame(cells)
    rep.say("No-overshoot η set per cell: the η at which the balanced arm (m = 1/3) has median persistent "
            "absorption ≤ 0.02 over the 10 task seeds. Absorption is the persistent fraction at τ = 1e-3. "
            "Means are over (task, η) in that set.")
    rep.say()
    rep.table(cells, "cells", show=True)
    included = cells.dropna(subset=["wilcoxon_p"])
    sgd = included[included.optimizer == "sgd"]
    adam = included[included.optimizer == "adam"]
    holds_sgd = len(sgd) > 0 and bool(sgd.passed.all())
    holds_adam = len(adam) > 0 and bool(adam.passed.all())
    p_holm = max(included.wilcoxon_p) if len(included) else 1.0
    excluded = cells[cells.n_eta == 0]
    if len(excluded):
        names = excluded.arch + "/" + excluded.rho.astype(str) + "/" + excluded.optimizer + "/" + excluded["update"]
        rep.say(f"Excluded (empty no-overshoot set): {', '.join(names)}.")
        rep.say()
    rep.say(f"P-value passed to Holm (largest Wilcoxon p over included cells): {p_holm:.3g}.")
    rep.say()
    rep.verdict(holds_sgd and holds_adam,
                f"SGD cells {int(sgd.passed.sum())}/{len(sgd)} pass, Adam cells {int(adam.passed.sum())}/{len(adam)} "
                f"pass; {'holds under SGD only' if holds_sgd and not holds_adam else 'pre-registered rule'}")

    # Descriptive: absorption against m, every eta, per cell.
    rep.say("### Descriptive: absorption by majority share and η")
    rep.say()
    table = rl.groupby(["arch", "rho", "optimizer", "update", "eta", "m"])[PERSIST].median().unstack("m").reset_index()
    rep.table(table, "by_eta")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    for ax, ((arch, rho), g) in zip(axes[0], rl[rl.optimizer == "sgd"].groupby(["arch", "rho"])):
        for m, gm in g[g["update"] == "expected_full"].groupby("m"):
            med = gm.groupby("eta")[PERSIST].median()
            ax.plot(med.index, med.values, "o-", label=f"m={m:.2f}")
        ax.set(xscale="log", title=f"SGD {arch} ρ={rho} (expected_full)", xlabel="η")
    for ax, ((arch, rho), g) in zip(axes[1], rl[rl.optimizer == "adam"].groupby(["arch", "rho"])):
        for m, gm in g[g["update"] == "expected_full"].groupby("m"):
            med = gm.groupby("eta")[PERSIST].median()
            ax.plot(med.index, med.values, "o-", label=f"m={m:.2f}")
        ax.set(xscale="log", title=f"Adam {arch} ρ={rho} (expected_full)", xlabel="η")
    axes[0, 0].set_ylabel("persistent absorption (median over tasks)")
    axes[1, 0].set_ylabel("persistent absorption (median over tasks)")
    axes[0, 0].legend()
    fig.tight_layout()
    rep.figure(fig, "absorption_vs_eta", "Persistent absorption against η, by majority share")

    rep.say("### Descriptive: who is absorbed")
    rep.say()
    ab = rl[rl.n_absorbed > 0]
    share = ab.groupby("m").agg(runs=("n_absorbed", "size"), absorbed=("n_absorbed", "sum"),
                                minority=("n_absorbed_minority", "sum"), minority_share=("minority_share", "mean"))
    share["absorbed_minority_frac"] = share.minority / share.absorbed
    rep.say("Pooled over all runs with any persistent absorption (all η, both optimizers and updates). "
            "`minority_share` is the share of prompts whose answer is a minority class.")
    rep.say()
    rep.table(share.reset_index(), "minority", show=True)
    inside = rl[rl.n_absorbed > 0].copy()
    inside["mean_ratio"] = inside[PERSIST] / inside.minority_share
    ratio = inside[inside.m > 0.34].groupby(["m", "optimizer"])["mean_ratio"].median().reset_index()
    rep.say("Median over runs with absorption of (absorbed fraction ÷ minority share); 1 means the whole "
            "minority set is absorbed.")
    rep.say()
    rep.table(ratio, "minority_ratio", show=True)

    sup = runs[runs["update"] == "supervised"].copy()
    sup["m"] = sup.majority.round(3)
    rep.say("### Descriptive: supervised control (cross-entropy, η = 1, 2,000 steps)")
    rep.say()
    rep.table(sup.groupby(["arch", "rho", "m"]).agg(persist=(PERSIST, "max"), final_mean=("final_mean", "median"))
              .reset_index(), "supervised", show=True)

    prompts = pd.read_csv(os.path.join(raw, "cache", "v3_t1_prompts.csv"))
    prompts["m"] = prompts.majority.round(3)
    prompts["group"] = np.where(prompts.cls == 0, "class 0", "minority")
    marg = prompts.groupby(["rho", "m", "group"]).margin.quantile([0.1, 0.5, 0.9]).unstack().reset_index()
    marg.columns = ["rho", "m", "class", "margin_q10", "margin_q50", "margin_q90"]
    rep.say("### Descriptive: per-class margins (top score minus second, over the 10 tasks)")
    rep.say()
    rep.table(marg, "margins", show=True)
    return {"P1": p_holm, "P1_pass": holds_sgd and holds_adam, "P1_sgd": holds_sgd}


# ---------------------------------------------------------------- P2 (T2)

def analyze_t2(runs, rep):
    rep.say("## P2 (T2): kick size, not response noise, at equal mean drift")
    rep.say()
    exp = runs[runs["update"] == "expected"]
    rows, counted = [], []
    for level, g in exp.groupby("level"):
        r, p = spearman(g.B, g[PERSIST], "less")
        med = g.groupby("B")[PERSIST].mean()
        rows.append(dict(level=level, **{f"B={b}": med.get(b, np.nan) for b in (8, 32, 128, 512)},
                         spearman=r, p=p, passed=p < 0.05))
    a = pd.DataFrame(rows)
    rep.say("(a) Kick size. Mean persistent absorption (τ = 1e-3) under `expected` at each B, pooled over "
            "the two η and 10 tasks; one-sided Spearman(B, absorption) < 0.")
    rep.say()
    rep.table(a, "kick", show=True)
    a_holds = int(a.passed.sum()) >= 3
    counted = a[a.passed]
    p_holm = float(counted.p.max()) if len(counted) else 1.0

    keys = ["level", "eta", "B", "task_seed"]
    paired = runs[runs["update"] == "sampled"].set_index(keys)[PERSIST].to_frame("sampled").join(
        exp.set_index(keys)[PERSIST].to_frame("expected"), how="inner").reset_index()
    paired["diff"] = paired.sampled - paired.expected
    rows = []
    for level, g in paired.groupby("level"):
        lo, hi = boot_tasks(g, "diff", np.mean)
        byB = g.groupby("B")["diff"].mean()
        rows.append(dict(level=level, mean_diff=g["diff"].mean(), ci_lo=lo, ci_hi=hi,
                         **{f"B={b}": byB.get(b, np.nan) for b in (8, 32, 128, 512)},
                         equivalent=(lo > -0.05) and (hi < 0.05)))
    b = pd.DataFrame(rows)
    rep.say("(b) Response noise. Mean of absorption(`sampled`) − absorption(`expected`) over tasks, η and B, "
            "with a 95% bootstrap CI over task seeds (10,000 draws); equivalence margin ±0.05.")
    rep.say()
    rep.table(b, "noise", show=True)
    b_holds = bool(b.equivalent.all())
    rep.say(f"P-value passed to Holm (largest (a) p among the levels counted): {p_holm:.3g}.")
    rep.say()
    rep.verdict(a_holds and b_holds, f"(a) {int(a.passed.sum())}/4 levels with Spearman < 0 at p < 0.05 "
                f"(need 3); (b) {int(b.equivalent.sum())}/4 levels inside ±0.05")

    full = runs[runs["update"] == "expected_full"].groupby(["level", "eta"])[PERSIST].mean().reset_index()
    rep.say("Descriptive: `expected_full` (kick → 0) at the same η, mean over tasks.")
    rep.say()
    rep.table(full, "full", show=True)
    by_eta = runs[runs["update"] != "expected_full"].groupby(["level", "eta", "update", "B"])[PERSIST].mean() \
        .unstack("B").reset_index()
    rep.table(by_eta, "by_eta", show=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for ax, (level, g) in zip(axes, runs[runs["update"] != "expected_full"].groupby("level")):
        for (eta, update), gu in g.groupby(["eta", "update"]):
            m = gu.groupby("B")[PERSIST].mean()
            ax.plot(m.index, m.values, "o-" if update == "expected" else "x--", label=f"η={eta:g} {update}")
        ax.set(xscale="log", title=level, xlabel="B")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("persistent absorption (mean over tasks)")
    rep.figure(fig, "kick", "Absorption against batch size: expected (solid) and sampled (dashed)")
    return {"P2": p_holm, "P2_pass": a_holds and b_holds}


# ---------------------------------------------------------------- T4

SAMPLES_PER_STEP = 256


def t4_best(g):
    """Best eta per G (medians over seeds), then the tie-broken best G, for one (level, p0, rule)."""
    per_g, edge = {}, {}
    grid = sorted(g.eta.unique())
    for G, gg in g.groupby("G"):
        med = gg.groupby("eta").step_to_half.median()  # NaN (never reached) beats nothing
        med = med.fillna(np.inf)
        best_eta = med.idxmin()
        per_g[G] = med.min() * SAMPLES_PER_STEP
        edge[G] = np.isfinite(med.min()) and best_eta in (grid[0], grid[-1])
    return per_g, edge


def analyze_t4(runs, rep):
    rep.say("## T4 (exploratory): H5 rerun, best group size with a deterministic tie-break")
    rep.say()
    rows = []
    rng = np.random.default_rng(0)
    for (level, p0, rule), g in runs.groupby(["level", "p0", "rule"]):
        per_g, edge = t4_best(g)
        best = best_by_tiebreak(per_g, SAMPLES_PER_STEP)
        seeds = g.seed.unique()
        draws = []
        for _ in range(1000):
            pick = rng.choice(seeds, size=len(seeds))
            gb = pd.concat([g[g.seed == s] for s in pick])
            draws.append(best_by_tiebreak(t4_best(gb)[0], SAMPLES_PER_STEP))
        finite = [v for v in per_g.values() if np.isfinite(v)]
        rows.append(dict(level=level, p0=p0, rule=rule, best_G=best,
                         ci_lo=np.nanquantile(draws, 0.025), ci_hi=np.nanquantile(draws, 0.975),
                         min_samples=min(finite) if finite else np.nan,
                         spread=(max(finite) / min(finite)) if finite else np.nan,
                         reached=f"{len(finite)}/{len(per_g)}",
                         edge_eta=",".join(str(G) for G, e in edge.items() if e)))
    t = pd.DataFrame(rows)
    rep.say("Samples to reach mean p ≥ 0.5 (step × 256, tracked every step), median over 5 seeds at each "
            "(G, η); best η per G; best G is the smallest G within 256 samples of the minimum. CI: 1,000 seed "
            "bootstraps. `spread` = slowest ÷ fastest G (at each G's best η). `edge_eta` lists G whose best η "
            "is at the grid edge.")
    rep.say()
    rep.table(t, "best_G", show=True)
    rep.say("### Prediction: in tabular GRPO the best G does not increase with p₀")
    rep.say()
    for level in ("tab", "lin", "m64"):
        s = t[(t.level == level) & (t.rule == "grpo")].sort_values("p0")
        r, p = spearman(s.p0, s.best_G, "greater") if s.best_G.notna().sum() > 2 else (np.nan, np.nan)
        rep.say(f"- {level} GRPO: best G by p₀ = {', '.join(f'{v:g}' for v in s.best_G)}; Spearman {r:.2f}.")
    rep.say()
    tab = t[(t.level == "tab") & (t.rule == "grpo")].sort_values("p0")
    r, _ = spearman(tab.p0, tab.best_G, "less")
    flat = t.pivot_table(index=["level", "p0"], columns="rule", values="spread").reset_index()
    rep.say("Spread across G (slowest ÷ fastest), mean rule against GRPO:")
    rep.say()
    rep.table(flat, "spread", show=True)
    flatter = float((flat["mean"] < flat["grpo"]).mean())
    # A constant best G has no rank correlation, and it does not increase, so the prediction holds.
    constant = tab.best_G.nunique() == 1
    holds = constant or (np.isfinite(r) and r <= 0)
    shown = "constant best G" if constant else f"Spearman(p₀, best G) = {r:.2f}"
    rep.say("With p₀ equal for every prompt, GRPO's normalisation only rescales each group, so a tuned η "
            "largely absorbs the difference between the rules; in tabular runs the two spreads coincide.")
    rep.say()
    rep.verdict(holds, f"tabular GRPO {shown} (prediction: does not increase); mean rule flatter than GRPO in "
                f"{flatter:.0%} of (level, p₀)")
    return {}


# ---------------------------------------------------------------- T6

T6_PREDICT = {"control": "no drops", "flip05": "drops", "flip20": "drops", "sign": "drops",
              "momentum": "no drops", "adam": "no drops", "gradclip": "no drops",
              "ppo4": "drops", "multi4": "drops", "kl01": "below p_ref", "kl1": "below p_ref"}


def analyze_t6(runs, rep):
    rep.say("## T6 (exploratory): ablations of the tabular proposition")
    rep.say()
    runs = runs.copy()
    runs["material"] = runs.max_drop > MATERIAL
    s = runs.groupby("ablation").agg(runs=("material", "size"), material_share=("material", "mean"),
                                     max_drop=("max_drop", "max"), mean_persist=(PERSIST, "mean"),
                                     max_persist=(PERSIST, "max")).reset_index()
    s["prediction"] = s.ablation.map(T6_PREDICT)
    s["consistent"] = [
        (m == 0) if pred == "no drops" else (m > 0) if pred == "drops" else True
        for m, pred in zip(s.material_share, s.prediction)
    ]
    rep.say("Material drop: max over prompts of (p_init − min p) > 1e-3. Absorption: persistent at τ = 1e-3.")
    rep.say()
    rep.table(s, "summary", show=True)
    by_eta = runs.pivot_table(index="ablation", columns="eta", values="material", aggfunc="mean").reset_index()
    by_eta.columns = [f"η={c:g}" if not isinstance(c, str) else c for c in by_eta.columns]
    rep.say("Share of runs with a material drop, by η:")
    rep.say()
    rep.table(by_eta, "material_by_eta", show=True)
    ab_eta = runs.pivot_table(index="ablation", columns="eta", values=PERSIST, aggfunc="mean").reset_index()
    ab_eta.columns = [f"η={c:g}" if not isinstance(c, str) else c for c in ab_eta.columns]
    rep.say("Mean persistent absorption, by η:")
    rep.say()
    rep.table(ab_eta, "absorption_by_eta", show=True)
    rep.say("KL conditions use the initial policy as the reference, so p_ref = p_init for every prompt and any "
            "drop of min p below p_init is a drop below p_ref; with only min p logged, a decline from above "
            "p_ref back towards it is not measurable. The KL prediction is therefore not testable as written.")
    rep.say()
    wrong = s[~s.consistent]
    rep.verdict(wrong.empty, "every condition matches its prediction" if wrong.empty
                else f"mismatch: {', '.join(wrong.ablation)}")
    return {}


# ---------------------------------------------------------------- T8

def analyze_t8(runs, rep):
    rep.say("## T8 (exploratory): KL bistability without noise")
    rep.say()
    rows = []
    for _, r in runs.iterrows():
        roots = fixed_points(r.p0, r.beta, int(r.G), "grpo")
        unstable = [p for p, st in roots if not st][0]
        rows.append(dict(p_ref=r.p0, beta=r.beta, eta=r.eta, offset=r.offset, p_unstable=unstable,
                         p_start=r.p_start, p_final=r.p_final, above=float(r.p_final > unstable)))
    t = pd.DataFrame(rows)
    t["predicted"] = np.where(t.offset > 0, 1.0, np.where(t.offset < 0, 0.0, np.nan))
    t["match"] = t.above == t.predicted
    rep.say("Tabular, G = 64, `expected_full` with the analytic KL gradient. `above` = final p above the "
            "unstable root; predicted 1 for offsets > 0 and 0 for offsets < 0 (logit offsets from the root).")
    rep.say()
    grid = t.pivot_table(index=["p_ref", "beta", "eta"], columns="offset", values="above").reset_index()
    grid.columns = [f"{c:+g}" if not isinstance(c, str) else c for c in grid.columns]
    grid = grid.astype({c: "Int64" for c in grid.columns if c not in ("p_ref", "beta", "eta")})
    rep.table(grid, "basin", show=True)
    low = t[(t.eta <= 3) & t.predicted.notna()]
    high = t[(t.eta == 10) & (t.offset > 0)]
    rep.say(f"At η = 10, {int((high.above == 0).sum())} of {len(high)} runs that start above the root end "
            f"below it, without any noise.")
    rep.say()
    rep.verdict(bool(low.match.all()), f"η ≤ 3: {int(low.match.sum())}/{len(low)} runs end in the predicted basin")
    rep.table(t, "runs")
    return {}


# ---------------------------------------------------------------- summary

def holm_section(primary, out):
    lines = ["## Primary hypotheses so far", "",
             "Holm–Bonferroni runs across P1–P5 once all five are in. P3–P5 are not run yet, so the table "
             "gives each raw p and the most conservative bound, p × 5 (Holm's adjusted p can only be "
             "smaller).", "",
             "| Hypothesis | Pre-registered rule | Raw p (to Holm) | Bound p × 5 |", "|---|---|---|---|"]
    for h in ("P1", "P2"):
        if h in primary:
            lines.append(f"| {h} | {'PASS' if primary[h + '_pass'] else 'FAIL'} | {primary[h]:.3g} | "
                         f"{min(1, 5 * primary[h]):.3g} |")
    with open(os.path.join(out, "sections", "holm.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


ANALYSES = {"v3_t1": analyze_t1, "v3_t2": analyze_t2, "v3_t4": analyze_t4, "v3_t6": analyze_t6,
            "v3_t8": analyze_t8}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("collect", "all") + TESTS)
    parser.add_argument("--raw", default="results/v3a")
    parser.add_argument("--out", default="analysis/v3a")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.what == "collect":
        collect(args.raw, args.jobs)
        return
    primary = {}
    for h in TESTS if args.what == "all" else (args.what,):
        rep = Report(args.out, h)
        runs = load(args.raw, h)
        primary.update((analyze_t1(runs, rep, args.raw) if h == "v3_t1" else ANALYSES[h](runs, rep)) or {})
        rep.save()
        print(f"{h}: done")
    if args.what == "all":
        holm_section(primary, args.out)
        assemble(args.out, ("holm",) + TESTS, "v3 batch 1 (pre-registered in PREREG_v3a.md)")
        path = os.path.join(args.out, "FINDINGS.md")
        text = open(path).read().replace("`experiments/analyze.py`", "`experiments/analyze_v3.py`")
        open(path, "w").write(text)


if __name__ == "__main__":
    main()
