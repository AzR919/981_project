"""
Analysis of v3 batch 2 (`v3b`), implementing analysis/review/PREREG_v3b.md.

usage:
    python experiments/analyze_v3b.py collect [--raw results/v3b] [--jobs 8]
    python experiments/analyze_v3b.py all [--raw results/v3b] [--v3a results/v3a] [--out analysis/v3b]

collect writes <raw>/cache/<test>.csv (one row per run) and, for T5,
t5_prompts.npz (per prompt: run row, label, p_init, step-0 drift, rarity).
Rarity is the share of a prompt's 10 nearest training prompts (Euclidean) with
another answer class; the task is rebuilt with SyntheticTask exactly as
run.build does (checked: identical features and answers).
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
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze import Report, assemble, markdown  # noqa: E402
from experiments.analyze_v3 import (CONFIG_KEYS, PERSIST, analyze_t1, best_by_tiebreak, persistent,  # noqa: E402
                                    t1_labels, task_key)

TESTS = ("v3_t5", "v3_t7", "v3_t9", "v3_t1s1", "v3_t2g", "v3_t6kl", "v3_t4e")
TAU = 1e-3
N_BOOT = 10_000


# ---------------------------------------------------------------- collect

def rarity(cfg, k=10):
    from synthetic_rl.synthetic_task import SyntheticTask

    task = SyntheticTask(num_tokens=cfg["K"], response_length=cfg["L"], num_features=cfg["d"] + 1,
                         seed=cfg["task_seed"], features=cfg["features"], rho=cfg["rho"], answers="rule",
                         class_shares=[cfg["majority"]] + [(1 - cfg["majority"]) / (cfg["K"] - 1)] * (cfg["K"] - 1))
    x, correct = task.generate_dataset(cfg["n_train"])
    cls = correct[:, 0]
    d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    near = np.argsort(d, axis=1)[:, :k]
    return (cls[near] != cls[:, None]).mean(axis=1), cls


LABELS = {}


def init_worker(labels):
    LABELS.update(labels)


def read_one(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    cfg = d["config"]
    h = cfg["hypothesis"]
    row = {k: cfg.get(k) for k in CONFIG_KEYS}
    if isinstance(row["p0"], str):
        row["p0"] = np.nan
    extra = None
    if cfg["mode"] == "gradcheck":
        boot = np.asarray(d["mean_z2_boot"])
        row.update(mean_z2=d["mean_z2"], rank=float((boot < d["mean_z2"]).mean()), rel_err=d["rel_err"])
        return h, row, extra
    p_init, tail = np.asarray(d["p_init"]), np.asarray(d["tail_max"])
    row[PERSIST] = persistent(p_init, tail, TAU)
    row["persist_0.01"] = persistent(p_init, tail, 1e-2)
    row["max_drop"] = float(np.max(p_init - np.asarray(d["min_p"])))
    row["drop_above"] = float(np.max(d["drop_above"]))
    row["step_to_half"] = np.nan if d["step_to_half"] is None else float(d["step_to_half"])
    snaps = d["snapshots"]
    row["final_mean"] = float(np.mean(snaps[str(max(int(s) for s in snaps))]))
    if cfg.get("majority") is not None and h in ("v3_t1s1", "v3_t9"):
        cls, _ = LABELS[task_key(cfg)]
        absorbed = (tail < TAU) & (p_init >= 10 * TAU)
        row.update(n_absorbed=int(absorbed.sum()), n_absorbed_minority=int((absorbed & (cls != 0)).sum()),
                   minority_share=float((cls != 0).mean()))
    if h == "v3_t5":
        extra = dict(label=((tail < TAU) & (p_init >= 10 * TAU)).astype(np.int8), eligible=p_init >= 10 * TAU,
                     p_init=p_init.astype(np.float32),
                     drift=np.asarray(d["checkpoints"]["0"]["drift"], dtype=np.float32),
                     key=(cfg["features"], cfg["rho"], cfg["task_seed"]))
    return h, row, extra


def collect(raw, jobs):
    paths = sorted(glob.glob(os.path.join(raw, "*", "results", "*", "*.json.gz")))
    # Minority labels for the T1-type tasks (T1 seed 1 and T9's minority task).
    first, t5_cfgs = {}, {}
    for p in paths:
        if "/v3_t1s1/" in p or "/v3_t9/" in p or "/v3_t5/" in p:
            with gzip.open(p, "rt") as f:
                cfg = json.load(f)["config"]
            if cfg["hypothesis"] == "v3_t5":
                t5_cfgs.setdefault((cfg["features"], cfg["rho"], cfg["task_seed"]), cfg)
            elif cfg.get("majority") is not None:
                first.setdefault(task_key(cfg), cfg)
    print(f"building {len(first)} T1-type tasks and {len(t5_cfgs)} T5 tasks")
    with Pool(jobs) as pool:
        labels = dict(zip(first, pool.map(t1_labels, first.values())))
        rar = dict(zip(t5_cfgs, pool.map(rarity, t5_cfgs.values())))
    print(f"reading {len(paths)} files")
    with Pool(jobs, initializer=init_worker, initargs=(labels,)) as pool:
        results = pool.map(read_one, paths, chunksize=32)
    cache = os.path.join(raw, "cache")
    os.makedirs(cache, exist_ok=True)
    for h in TESTS:
        items = sorted((r for r in results if r[0] == h), key=lambda r: r[1]["name"])
        pd.DataFrame([r[1] for r in items]).to_csv(os.path.join(cache, f"{h}.csv"), index=False)
        if h == "v3_t5":
            blocks = {k: [] for k in ("run", "label", "eligible", "p_init", "drift", "rarity")}
            for i, (_, _, e) in enumerate(items):
                n = len(e["label"])
                blocks["run"].append(np.full(n, i))
                for k in ("label", "eligible", "p_init", "drift"):
                    blocks[k].append(e[k])
                blocks["rarity"].append(rar[e["key"]][0].astype(np.float32))
            np.savez_compressed(os.path.join(cache, "t5_prompts.npz"),
                                **{k: np.concatenate(v) for k, v in blocks.items()})
        print(f"{h}: {len(items)} runs")


def load(raw, h):
    return pd.read_csv(os.path.join(raw, "cache", f"{h}.csv"))


# ---------------------------------------------------------------- T5 (P3)

def fit_logistic_fe(X, y, run):
    """Logistic regression with one intercept per run (fixed effects); returns slopes only."""
    runs, idx = np.unique(run, return_inverse=True)
    k = X.shape[1]

    def nll(w):
        z = X @ w[:k] + w[k:][idx]
        loss = np.sum(np.logaddexp(0, z) - y * z) + 1e-4 * np.sum(w[:k] ** 2)
        r = 1 / (1 + np.exp(-z)) - y
        grad = np.concatenate([X.T @ r + 2e-4 * w[:k], np.bincount(idx, weights=r, minlength=len(runs))])
        return loss, grad

    w = optimize.minimize(nll, np.zeros(k + len(runs)), jac=True, method="L-BFGS-B").x
    return w[:k]


def within_run_auc(score, y, run):
    """AUC over (positive, negative) pairs from the same run (run intercepts cancel)."""
    pos = neg = 0.0
    wins = 0.0
    for r in np.unique(run):
        m = run == r
        s, t = score[m], y[m]
        n1, n0 = t.sum(), (1 - t).sum()
        if n1 == 0 or n0 == 0:
            continue
        ranks = stats.rankdata(s)
        wins += ranks[t == 1].sum() - n1 * (n1 + 1) / 2
        pos, neg = pos + n1 * n0, neg
    return wins / pos if pos else np.nan


def analyze_t5(runs, rep, raw):
    rep.say("## P3 (T5): does the initial drift predict absorption on balanced tasks?")
    rep.say()
    P = dict(np.load(os.path.join(raw, "cache", "t5_prompts.npz")))
    keep = P["eligible"].astype(bool)
    run, y = P["run"][keep], P["label"][keep].astype(float)
    p = np.clip(P["p_init"][keep], 1e-6, 1 - 1e-6)
    base_X = np.column_stack([P["rarity"][keep], np.log(p / (1 - p))])
    full_X = np.column_stack([base_X, P["drift"][keep]])
    counts = pd.DataFrame({"run": run, "y": y}).groupby("run").y.agg(["sum", "size"])
    ok = counts.index[(counts["sum"] >= 20) & (counts["size"] - counts["sum"] >= 20)]
    runs = runs.reset_index(drop=True)
    rep.say(f"Runs with ≥ 20 absorbed and ≥ 20 non-absorbed eligible prompts: {len(ok)} of {len(runs)} "
            f"(by level: {runs.loc[ok].level.value_counts().to_dict()}).")
    rep.say()
    if len(ok) < 10:
        rep.verdict(False, "untestable: fewer than 10 qualifying runs (reported as untestable, not failed)")
        return {"P3": np.nan, "P3_pass": None}
    m = np.isin(run, ok)
    run, y, base_X, full_X = run[m], y[m], base_X[m], full_X[m]
    task = runs.task_seed.to_numpy()[run]
    # Standardise predictors (scale only; AUC is invariant).
    mu, sd = full_X.mean(0), full_X.std(0)
    full_X = (full_X - mu) / sd
    base_X = full_X[:, :2]
    s_base, s_full = np.zeros_like(y), np.zeros_like(y)
    for t in np.unique(task):
        tr, te = task != t, task == t
        s_base[te] = base_X[te] @ fit_logistic_fe(base_X[tr], y[tr], run[tr])
        s_full[te] = full_X[te] @ fit_logistic_fe(full_X[tr], y[tr], run[tr])
    auc_b, auc_f = within_run_auc(s_base, y, run), within_run_auc(s_full, y, run)
    delta = auc_f - auc_b
    rng = np.random.default_rng(0)
    tasks = np.unique(task)
    per_task = {t: (task == t) for t in tasks}
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(tasks, size=len(tasks))
        idx = np.concatenate([np.nonzero(per_task[t])[0] for t in pick])
        # Resampled tasks keep their own runs; relabel runs per draw so duplicates count separately.
        rr = np.concatenate([run[per_task[t]] * 1000 + j for j, t in enumerate(pick)])
        draws.append(within_run_auc(s_full[idx], y[idx], rr) - within_run_auc(s_base[idx], y[idx], rr))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    p_holm = max(float(np.mean(np.asarray(draws) <= 0)), 1e-4)
    single = {name: within_run_auc(col, y, run) for name, col in
              (("rarity", full_X[:, 0]), ("logit p_init", -full_X[:, 1]), ("drift", -full_X[:, 2]))}
    rep.say(f"Within-run AUC (leave-one-task-out): base (rarity + logit p_init) {auc_b:.3f}, full (+ drift) "
            f"{auc_f:.3f}; ΔAUC = {delta:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] (10,000 task bootstraps of the "
            f"held-out scores). Single predictors (sign chosen so higher = more absorption): "
            + ", ".join(f"{k} {v:.3f}" for k, v in single.items()) + ".")
    rep.say()
    by_level = []
    for level in runs.loc[ok].level.unique():
        lm = runs.level.to_numpy()[run] == level
        by_level.append(dict(level=level, runs=int(runs.loc[ok].level.eq(level).sum()),
                             auc_base=within_run_auc(s_base[lm], y[lm], run[lm]),
                             auc_full=within_run_auc(s_full[lm], y[lm], run[lm])))
    t = pd.DataFrame(by_level)
    t["delta"] = t.auc_full - t.auc_base
    rep.table(t, "by_level", show=True)
    passed = delta >= 0.03 and lo > 0
    rep.say(f"P-value passed to Holm: {p_holm:.3g}.")
    rep.say()
    rep.verdict(passed, f"ΔAUC {delta:.3f} (need ≥ 0.03) with CI lower bound {lo:.3f} (need > 0)")
    # Post-hoc sensitivity (not pre-registered): drift scales with eta, which spans 100x across runs, so one
    # pooled coefficient cannot weight it within each run. Standardise every predictor within its run.
    z = full_X.copy()
    for r in np.unique(run):
        m_r = run == r
        z[m_r] = (z[m_r] - z[m_r].mean(0)) / np.where(z[m_r].std(0) > 0, z[m_r].std(0), 1)
    z_base, z_full = np.zeros_like(y), np.zeros_like(y)
    for t in np.unique(task):
        tr, te = task != t, task == t
        z_base[te] = z[te, :2] @ fit_logistic_fe(z[tr, :2], y[tr], run[tr])
        z_full[te] = z[te] @ fit_logistic_fe(z[tr], y[tr], run[tr])
    zb, zf = within_run_auc(z_base, y, run), within_run_auc(z_full, y, run)
    zdraws = []
    for _ in range(2000):
        pick = rng.choice(tasks, size=len(tasks))
        idx = np.concatenate([np.nonzero(per_task[t])[0] for t in pick])
        rr = np.concatenate([run[per_task[t]] * 1000 + j for j, t in enumerate(pick)])
        zdraws.append(within_run_auc(z_full[idx], y[idx], rr) - within_run_auc(z_base[idx], y[idx], rr))
    zlo, zhi = np.quantile(zdraws, [0.025, 0.975])
    rep.say(f"**Post-hoc sensitivity (not pre-registered, does not change the verdict):** with every predictor "
            f"standardised within its run (drift scales with η, which spans 100× across runs), base AUC {zb:.3f}, "
            f"full {zf:.3f}, ΔAUC = {zf - zb:.3f}, 95% CI [{zlo:.3f}, {zhi:.3f}] (2,000 task bootstraps).")
    rep.say()
    rep.say("Implementation note: run fixed effects cannot be predicted for held-out tasks, so AUC is computed "
            "within runs (pairs from the same run), where the run intercept cancels; the bootstrap resamples "
            "tasks over held-out scores without refitting. Both are reported as clarifications of the "
            "pre-registration.")
    rep.say()
    return {"P3": p_holm, "P3_pass": passed}


# ---------------------------------------------------------------- T7

def analyze_t7(runs, rep):
    rep.say("## T7 (exploratory): calibrated H1 medium tier")
    rep.say()
    rows = []
    for level, g in runs.groupby("level"):
        ks = stats.kstest(g["rank"], "uniform")
        rows.append(dict(level=level, runs=len(g), median_rank=g["rank"].median(),
                         share_above_095=(g["rank"] > 0.95).mean(), ks_p=ks.pvalue))
    t = pd.DataFrame(rows)
    rep.say("Rank of the observed mean z² within its 200 bootstrap draws; under the exact identity the ranks are "
            "Uniform(0, 1) and 5% exceed 0.95.")
    rep.say()
    rep.table(t, "ranks", show=True)
    tab = t[t.level == "tab"].ks_p.iloc[0]
    rep.verdict(tab >= 0.05, f"tabular calibration KS p = {tab:.3g}; "
                + ("medium tier applied to lin and m64 above" if tab >= 0.05 else "medium tier dropped"))
    return {}


# ---------------------------------------------------------------- T9

def analyze_t9(runs, rep):
    rep.say("## T9 (exploratory): all-wrong-group fixes")
    rep.say()
    runs = runs.copy()
    runs["unit"] = runs.task_seed.fillna(runs.seed)
    runs["cell"] = runs.level + " / " + runs["update"].fillna("sampled") + " / η=" + runs.eta.map("{:g}".format)
    piv = runs.pivot_table(index=["cell", "unit"], columns="rule", values=PERSIST).reset_index()
    rows = []
    for cell, g in piv.groupby("cell"):
        row = {"cell": cell}
        for rule in ("grpo", "mean", "heckel", "virtual", "pm_alpha", "nsr"):
            row[rule] = g[rule].mean()
        d = g.heckel - g.grpo
        row["heckel<grpo p"] = (stats.wilcoxon(g.heckel, g.grpo, alternative="less").pvalue
                                if (d != 0).any() else 1.0)
        rows.append(row)
    t = pd.DataFrame(rows)
    rep.say("Mean persistent absorption (τ = 1e-3) per rule over 10 tasks or seeds; one-sided paired Wilcoxon "
            "for heckel < grpo.")
    rep.say()
    rep.table(t, "absorption", show=True)
    mino = t[t.cell.str.startswith("t1-minority")]
    heck = bool((mino["heckel<grpo p"] < 0.05).all())
    flat = bool(((mino.virtual - mino.grpo).abs() <= 0.02).all() and ((mino.pm_alpha - mino.grpo).abs() <= 0.02).all())
    notlower = bool(((mino.nsr - mino.grpo) >= -0.02).all() and ((mino["mean"] - mino.grpo) >= -0.02).all())
    m = runs[runs.level == "t1-minority"]
    share = m[m.n_absorbed > 0].groupby("rule").apply(
        lambda g: g.n_absorbed_minority.sum() / g.n_absorbed.sum()).rename("absorbed_minority_frac").reset_index()
    rep.table(share, "minority_frac", show=True)
    rep.verdict(heck and flat and notlower,
                f"t1-minority: heckel lower than grpo in every cell: {heck}; virtual and pm_alpha within ±0.02 of "
                f"grpo: {flat}; nsr and mean not lower than grpo: {notlower}")
    return {}


# ---------------------------------------------------------------- follow-ups

def analyze_t1s1(runs, rep, v3a):
    rep.say("## Follow-up (a): T1 MLP cells with init seed 1")
    rep.say()
    analyze_t1(runs, rep, v3a)
    return {}


def analyze_t2g(runs, rep, v3a):
    rep.say("## Follow-up (b): response-noise excess at B = 8 against G")
    rep.say()
    t2 = pd.read_csv(os.path.join(v3a, "cache", "v3_t2.csv"))
    both = pd.concat([runs, t2[(t2.B == 8) & t2["update"].isin(["expected", "sampled"])]])
    keys = ["level", "eta", "G", "task_seed"]
    w = both.pivot_table(index=keys, columns="update", values=PERSIST).reset_index()
    w["excess"] = w.sampled - w.expected
    r = stats.spearmanr(w.G, w.excess, alternative="less")
    t = w.groupby(["level", "G"]).excess.mean().unstack("G").reset_index()
    rep.table(t, "excess", show=True)
    rep.verdict(r.pvalue < 0.05, f"Spearman(G, excess) = {r.statistic:.2f}, one-sided p = {r.pvalue:.3g} "
                "(prediction: < 0)")
    return {}


def analyze_t6kl(runs, rep):
    rep.say("## Follow-up (c): T6 KL conditions with the decline above p_ref")
    rep.say()
    runs = runs.copy()
    runs["below"] = runs.max_drop > 1e-3
    runs["above"] = runs.drop_above > 1e-3
    t = runs.groupby(["ablation", "eta"]).agg(below_share=("below", "mean"), above_share=("above", "mean"),
                                              max_drop_above=("drop_above", "max"),
                                              persist=(PERSIST, "mean")).reset_index()
    rep.table(t, "kl", show=True)
    ok_low = bool((t[(t.eta <= 3) & t.ablation.isin(["kl01", "kl1"])].below_share == 0).all())
    ok_ctrl = bool((t[t.ablation == "control"].max_drop_above <= 1e-3).all())
    ok_high = bool((t[(t.eta < 30)].below_share == 0).all())
    rep.verdict(ok_low and ok_ctrl and ok_high,
                f"no material drop below p_init at η ≤ 3 for KL: {ok_low}; control has no decline from a peak: "
                f"{ok_ctrl}; drops below p_init only at η ≥ 30: {ok_high}")
    return {}


def analyze_t4e(runs, rep, v3a):
    rep.say("## Follow-up (d): T4 edge cells with η up to 10,000")
    rep.say()
    old = pd.read_csv(os.path.join(v3a, "cache", "v3_t4.csv"))
    old = old[old.level.isin(["tab", "lin"]) & old.p0.isin([0.01, 0.03, 0.1])]
    both = pd.concat([old, runs])
    rows = []
    for (level, p0, rule), g in both.groupby(["level", "p0", "rule"]):
        per_g, edge = {}, {}
        for G, gg in g.groupby("G"):
            med = gg.groupby("eta").step_to_half.median().fillna(np.inf)
            per_g[G] = med.min() * 256
            edge[G] = np.isfinite(med.min()) and med.idxmin() == max(med.index)
        o = old[(old.level == level) & (old.p0 == p0) & (old.rule == rule)]
        old_g = {}
        for G, gg in o.groupby("G"):
            old_g[G] = gg.groupby("eta").step_to_half.median().fillna(np.inf).min() * 256
        rows.append(dict(level=level, p0=p0, rule=rule, best_G_old=best_by_tiebreak(old_g, 256),
                         best_G_new=best_by_tiebreak(per_g, 256), min_samples=min(per_g.values()),
                         still_edge=",".join(str(G) for G, e in edge.items() if e)))
    t = pd.DataFrame(rows)
    rep.table(t, "best_G", show=True)
    rep.say("`still_edge`: G whose best η is now 10,000 (the new grid edge).")
    rep.say()
    return {}


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("collect", "all"))
    parser.add_argument("--raw", default="results/v3b")
    parser.add_argument("--v3a", default="results/v3a")
    parser.add_argument("--out", default="analysis/v3b")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.what == "collect":
        collect(args.raw, args.jobs)
        return
    primary = {}
    for h in TESTS:
        rep = Report(args.out, h)
        runs = load(args.raw, h)
        if h == "v3_t5":
            primary.update(analyze_t5(runs, rep, args.raw))
        elif h == "v3_t7":
            analyze_t7(runs, rep)
        elif h == "v3_t9":
            analyze_t9(runs, rep)
        elif h == "v3_t1s1":
            analyze_t1s1(runs, rep, args.v3a)
        elif h == "v3_t2g":
            analyze_t2g(runs, rep, args.v3a)
        elif h == "v3_t6kl":
            analyze_t6kl(runs, rep)
        else:
            analyze_t4e(runs, rep, args.v3a)
        rep.save()
        print(f"{h}: done")
    assemble(args.out, TESTS, "v3 batch 2 (pre-registered in PREREG_v3b.md)")
    path = os.path.join(args.out, "FINDINGS.md")
    text = open(path).read().replace("`experiments/analyze.py`", "`experiments/analyze_v3b.py`")
    open(path, "w").write(text)
    print(primary)


if __name__ == "__main__":
    main()
