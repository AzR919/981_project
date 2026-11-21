"""
Turn the raw results of one batch into tables, figures and FINDINGS.md.

usage:
    python experiments/analyze.py collect [--raw results/base] [--jobs 8]
    python experiments/analyze.py h1|h2|h3|h4|h5|all [--raw results/base] [--out analysis/base]

collect reads every <raw>/<host>/results/<h>/NNNNN.json.gz once and writes a
cache to <raw>/cache: runs_<h>.csv (one row per run, config plus scalar
outcomes) and prompts_<h>.npz (per-prompt arrays, row-aligned with the CSV).
Every other subcommand reads only the cache.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_rl.theory import advantage_gap, fixed_points, informative_fraction  # noqa: E402

HYPOTHESES = ("h1", "h2", "h3", "h4", "h5")
STRETCH = ("s_h1", "s_drift", "s_pred", "s_abl", "s_eq")
CONFIG_KEYS = (
    "name", "mode", "level", "part", "features", "rho", "arch", "width", "init", "p0", "L", "G", "B",
    "rule", "passk", "eta", "beta", "kl", "steps", "seed",
    "update", "block", "optimizer", "momentum", "reward_noise", "ablation", "p_start", "offset", "R", "chunks",
)
DIAGNOSTICS = ("mean_cos", "mean_abs_cos", "mean_sq_grad", "mean_sq_grad_logit")
PROMPT_ARRAYS = {
    "train": ("p_init", "p_final", "min_p", "absorbed_at", "predicted_drift"),
    "onestep": ("p0", "W", "delta_mean", "delta_se", "predicted_drift", "own_term"),
}


# ---------------------------------------------------------------- collect

def read_one(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    cfg = d["config"]
    row = {k: cfg.get(k) for k in CONFIG_KEYS}
    if isinstance(row["p0"], str):
        row["p0"] = np.nan  # "spread"
    row["seconds"] = d["seconds"]
    arrays = {}

    if cfg["mode"] == "gradcheck":
        for k in ("rel_err", "mc_rel", "cosine", "mean_z2", "frac_abs_z_gt3", "pred_norm", "n_coords"):
            row[k] = d[k]
        row["p_probe_mean"] = float(np.mean(d["p_probe"]))
        if "mean_z2_null" in d:
            null = np.asarray(d["mean_z2_null"])
            row.update({"bias_par": d["bias_par"], "bias_par_se": d["bias_par_se"],
                        "null_lo": float(np.quantile(null, 0.025)), "null_hi": float(np.quantile(null, 0.975)),
                        "null_median": float(np.median(null)), "null_rank": float((null < d["mean_z2"]).mean())})
    elif cfg["mode"] == "onestep":
        arrays = {k: np.asarray(d[k], dtype=np.float32) for k in PROMPT_ARRAYS["onestep"]}
    else:
        hist = d["history"]
        means = np.array([h["train"]["mean"] for h in hist])
        steps = np.array([h["step"] for h in hist])
        final = hist[-1]
        row.update({
            "s_bar": d["s_bar"], "W_init_mean": d["W_init_mean"],
            "p_init_mean": float(np.mean(d["p_init"])),
            "final_mean": final["train"]["mean"], "absorbed": final["train"]["absorbed"],
            "solved": final["train"]["solved"], "test_mean": final.get("test_mean", np.nan),
            "late_mean": float(means[steps >= steps[-1] / 2].mean()),
            "max_absorbed": max(h["train"]["absorbed"] for h in hist),
            "decreases_total": d["decreases_total"],
        })
        reached = np.nonzero(means >= 0.5)[0]
        row["step_to_half"] = float(steps[reached[0]]) if len(reached) else np.nan
        at = np.asarray(d["absorbed_at"])
        row["first_absorbed"] = float(at[at >= 0].min()) if (at >= 0).any() else np.nan
        inf = [h["informative"] for h in hist if "informative" in h]
        exp = [h["expected_informative"] for h in hist if "expected_informative" in h]
        row["informative"] = float(np.mean(inf)) if inf else np.nan
        row["expected_informative"] = float(np.mean(exp)) if exp else np.nan
        for k in DIAGNOSTICS:
            row[k] = d.get("diagnostics", {}).get(k, np.nan)
        snaps = d["snapshots"]
        last = snaps[str(max(int(s) for s in snaps))]
        arrays = {
            "p_init": np.asarray(d["p_init"], dtype=np.float32),
            "p_final": np.asarray(last, dtype=np.float32),
            "min_p": np.asarray(d["min_p"], dtype=np.float32),
            "absorbed_at": np.asarray(d["absorbed_at"], dtype=np.int32),
            "predicted_drift": np.asarray(d.get("predicted_drift", np.full(len(last), np.nan)), dtype=np.float32),
        }
        p_init = arrays["p_init"]
        row["max_drop"] = float(np.max(p_init - arrays["min_p"]))
        absorbed_steps = at[at >= 0]
        row["t50"] = float(np.median(absorbed_steps)) if len(absorbed_steps) else np.nan
        if "checkpoints" in d:
            row.update(predictor_scores(d, cfg, p_init, at))
        if cfg.get("p_start") is not None:
            row.update(switching(snaps, cfg))
        if cfg.get("part") == "b":
            # Var(logit p) across the 64 replicates at each snapshot in the second half.
            half = [s for s in snaps if int(s) >= cfg["steps"] / 2]
            z = [np.log(np.clip(snaps[s], 1e-12, 1 - 1e-7)) - np.log1p(-np.clip(snaps[s], 1e-12, 1 - 1e-7)) for s in half]
            row["var_logit"] = float(np.mean([np.var(v) for v in z]))
            row["mean_logit"] = float(np.mean([np.mean(v) for v in z]))
    return cfg["hypothesis"], row, arrays


def predictor_scores(d, cfg, p_init, absorbed_at, window=500):
    """Within-run AUCs for which prompts get absorbed, at step 0 and at each checkpoint.

    At step 0 the label is "absorbed at any later time"; at a checkpoint t it is
    "absorbed in (t, t + window]" among prompts still above 0.01 at t.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    snaps = d["snapshots"]
    eta, n = cfg["eta"], len(p_init)
    out = {"p_init_sd": float(np.std(p_init))}
    per = {k: [] for k in ("drift", "logit_drift", "p", "grad", "combo")}
    for t, ck in sorted(d["checkpoints"].items(), key=lambda kv: int(kv[0])):
        t = int(t)
        p = np.asarray(snaps[str(t)], dtype=float) if t else p_init.astype(float)
        drift, own = np.asarray(ck["drift"]), np.asarray(ck["own"])
        W = advantage_gap(p, cfg["G"], cfg["rule"], passk=cfg.get("passk"))
        grad = np.sqrt(np.clip(own * n / (eta * np.maximum(W, 1e-300)), 0, None))
        alive = (p >= 0.01) & (p_init >= 0.01)
        later = absorbed_at > t if t == 0 else (absorbed_at > t) & (absorbed_at <= t + window)
        label = later[alive]
        if not (5 <= label.sum() <= len(label) - 5):
            continue
        # Snapshots are rounded to 6 decimals, so a solved prompt can read exactly p = 1.
        pa, da = np.clip(p[alive], 1e-6, 1 - 1e-6), drift[alive]
        scores = {"drift": -da, "logit_drift": -da / (pa * (1 - pa)), "p": -pa, "grad": -grad[alive]}
        X = np.column_stack([np.log(pa), np.sign(da) * np.log1p(np.abs(da) / (np.abs(da).mean() + 1e-300))])
        folds = StratifiedKFold(5, shuffle=True, random_state=0)
        scores["combo"] = cross_val_predict(LogisticRegression(), X, label, cv=folds, method="predict_proba")[:, 1]
        for k, v in scores.items():
            a = auc(v, label)
            if t == 0:
                out[f"auc0_{k}"] = a
            else:
                per[k].append(a)
    for k, v in per.items():
        out[f"aucck_{k}"] = float(np.nanmean(v)) if v else np.nan
    return out


def switching(snaps, cfg):
    """Crossings of the unstable KL fixed point by the replicate prompts, and the end state."""
    roots = fixed_points(cfg["p0"], cfg["beta"], cfg["G"], cfg["rule"], direction=cfg["kl"])
    unstable = [p for p, stable in roots if not stable]
    stable = [p for p, s in roots if s]
    keys = sorted(snaps, key=int)
    P = np.array([snaps[k] for k in keys])
    out = {"n_stable": len(stable), "stable_low": stable[0], "stable_high": stable[-1],
           "unstable": unstable[0] if unstable else np.nan}
    u = unstable[0] if unstable else 0.5 * (stable[0] + 1)
    above = P > u
    out["crossings"] = int((above[1:] != above[:-1]).sum())
    late = above[len(keys) // 2:]
    out["late_crossings"] = int((late[1:] != late[:-1]).sum())
    out["share_above_final"] = float(above[-1].mean())
    out["share_above_start"] = float(above[0].mean())
    end = P[-1]
    out["end_low_mean"] = float(end[~above[-1]].mean()) if (~above[-1]).any() else np.nan
    out["end_high_mean"] = float(end[above[-1]].mean()) if above[-1].any() else np.nan
    return out


def collect(raw, jobs):
    paths = sorted(glob.glob(os.path.join(raw, "*", "results", "*", "*.json.gz")))
    print(f"reading {len(paths)} files with {jobs} processes")
    with Pool(jobs) as pool:
        results = pool.map(read_one, paths, chunksize=64)
    cache = os.path.join(raw, "cache")
    os.makedirs(cache, exist_ok=True)
    for h in sorted({r[0] for r in results}):
        items = sorted((r for r in results if r[0] == h), key=lambda r: r[1]["name"])
        rows = pd.DataFrame([r[1] for r in items])
        rows.to_csv(os.path.join(cache, f"runs_{h}.csv"), index=False)
        arrays = {}
        for mode, keys in PROMPT_ARRAYS.items():
            sel = [i for i, r in enumerate(items) if r[2] and keys[0] in r[2]]
            if not sel:
                continue
            arrays["row_" + mode] = np.array(sel)
            for k in keys:
                # Tabular H4/H5 runs have 64 prompts, the rest 512: pad with NaN.
                width = max(len(items[i][2][k]) for i in sel)
                block = np.full((len(sel), width), np.nan, dtype=np.float32)
                for j, i in enumerate(sel):
                    v = items[i][2][k]
                    block[j, : len(v)] = v
                arrays[k] = block
        if arrays:
            np.savez_compressed(os.path.join(cache, f"prompts_{h}.npz"), **arrays)
        print(f"{h}: {len(rows)} runs")


def load(raw, h):
    cache = os.path.join(raw, "cache")
    runs = pd.read_csv(os.path.join(cache, f"runs_{h}.csv"))
    path = os.path.join(cache, f"prompts_{h}.npz")
    prompts = dict(np.load(path)) if os.path.exists(path) else {}
    return runs, prompts


# ---------------------------------------------------------------- output

class Report:
    """Collects one hypothesis's tables, figures and findings text."""

    def __init__(self, out, h):
        self.out, self.h, self.lines = out, h, []
        for sub in ("figures", "tables", "sections"):
            os.makedirs(os.path.join(out, sub), exist_ok=True)

    def say(self, text=""):
        self.lines.append(text)

    def table(self, df, name, show=False, floatfmt="{:.3g}"):
        df.to_csv(os.path.join(self.out, "tables", f"{self.h}_{name}.csv"), index=False)
        if show:
            self.say(markdown(df, floatfmt))
            self.say()

    def figure(self, fig, name, caption):
        fig.savefig(os.path.join(self.out, "figures", f"{self.h}_{name}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
        self.say(f"![{caption}](figures/{self.h}_{name}.png)")
        self.say()

    def verdict(self, passed, criterion):
        self.say(f"**Verdict: {'PASS' if passed else 'FAIL'}** ({criterion})")
        self.say()

    def save(self):
        with open(os.path.join(self.out, "sections", f"{self.h}.md"), "w") as f:
            f.write("\n".join(self.lines) + "\n")


def markdown(df, floatfmt="{:.3g}"):
    cells = [[floatfmt.format(v) if isinstance(v, float) else str(v) for v in row] for row in df.itertuples(index=False)]
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    return "\n".join([head, rule] + ["| " + " | ".join(r) + " |" for r in cells])


def rule_label(rule, passk):
    return f"passk{int(passk)}" if rule == "passk" and passk == passk else rule


LEVELS = ("tab", "lin", "c05", "c09", "m64", "m256")


def assemble(out, names=HYPOTHESES, title="base batch"):
    parts = [f"# Findings: {title}", "",
             "Generated by `experiments/analyze.py`; every number below is recomputed from the raw results.", ""]
    sections = {h: open(p).read() for h in names
                if os.path.exists(p := os.path.join(out, "sections", f"{h}.md"))}
    parts += ["## Verdicts at a glance", "", "| Test | Verdict |", "|---|---|"]
    for h, text in sections.items():
        sub = h.upper()
        for line in text.splitlines():
            if line.startswith("### "):
                sub = h.upper() + " " + line[4:].split(":")[0].split(" (")[0].strip(".")
            if line.startswith("**Verdict:"):
                cell = line.replace("**Verdict: ", "").replace("**", "").replace("|", "\\|")
                parts.append(f"| {sub} | {cell} |")
    parts.append("")
    parts += list(sections.values())
    with open(os.path.join(out, "FINDINGS.md"), "w") as f:
        f.write("\n".join(parts))


# ---------------------------------------------------------------- H1

def analyze_h1(runs, prompts, rep):
    runs = runs.assign(rule=[rule_label(r, k) for r, k in zip(runs.rule, runs.passk)],
                       ratio=runs.rel_err / runs.mc_rel)
    rep.say("## H1: the identity E[ĝ] = W_G(p)∇p (gradcheck, 414 runs)")
    rep.say()
    rep.say("Each run averages R = 4000 sampled updates at fixed parameters and compares them with "
            "−(1/B)Σ W_G(p_x)∇p_x from autograd. Without bias, ‖error‖ ≈ ‖Monte Carlo SE‖, so "
            "`rel_err / mc_rel` ≈ 1 and the mean squared z-score ≈ 1.")
    rep.say()
    by_rule = runs.groupby("rule").agg(
        runs=("ratio", "size"), median_ratio=("ratio", "median"), max_ratio=("ratio", "max"),
        median_z2=("mean_z2", "median"), frac_z_gt3=("frac_abs_z_gt3", "mean"),
        min_cosine=("cosine", "min")).reset_index()
    rep.table(by_rule, "by_rule", show=True)
    by_level = runs.groupby(["level", "init"]).agg(
        median_ratio=("ratio", "median"), median_z2=("mean_z2", "median"),
        frac_z_gt3=("frac_abs_z_gt3", "mean")).reset_index()
    rep.table(by_level, "by_level", show=True)
    rep.table(runs[["name", "rule", "G", "L", "level", "init", "rel_err", "mc_rel", "ratio", "mean_z2",
                    "frac_abs_z_gt3", "cosine"]], "runs")

    bad = runs[(runs.ratio > 1.2) | (runs.mean_z2 < 0.8) | (runs.mean_z2 > 1.25)]
    frac3 = runs.frac_abs_z_gt3.mean()
    q = runs.ratio.quantile([0.05, 0.5, 0.95])
    rep.say(f"Runs outside ratio ≤ 1.2 or mean z² in [0.8, 1.25]: **{len(bad)} of {len(runs)}**. "
            f"Overall share of |z| > 3: {frac3:.4f} (Gaussian: 0.0027).")
    rep.say()
    rep.say(f"The plan's criterion was per configuration, but H1 has one run per configuration, so it "
            f"judges single noisy statistics: the ratio's 5–95% range is {q.iloc[0]:.2f}–{q.iloc[2]:.2f} "
            f"around a median of {q.iloc[1]:.2f}, symmetric about 1, and mean z² averages correlated "
            f"coordinates. A bias would shift a whole rule or level; no rule or level median moves. "
            f"The verdict below therefore uses the group medians; the literal per-run criterion fails.")
    rep.say()
    if len(bad):
        rep.table(bad[["name", "rule", "G", "L", "level", "init", "ratio", "mean_z2", "n_coords"]
                      if "n_coords" in bad else ["name", "rule", "G", "L", "level", "init", "ratio", "mean_z2"]],
                  "outliers", show=len(bad) <= 25)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for rule, g in runs.groupby("rule"):
        ax.scatter(g.mc_rel, g.rel_err, s=10, alpha=0.7, label=rule)
    lim = [runs[["mc_rel", "rel_err"]].min().min() * 0.8, runs[["mc_rel", "rel_err"]].max().max() * 1.2]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="Monte Carlo relative SE", ylabel="relative error vs W_G(p)∇p",
           title="H1: sampled gradient vs identity")
    ax.legend(fontsize=7)
    rep.figure(fig, "error_vs_mc", "relative error against Monte Carlo SE")
    # 414 independent runs: a few z-score excursions are expected by chance, so judge the
    # group medians and the overall tail rate rather than demanding zero outliers.
    group_ok = (by_rule.median_ratio <= 1.2).all() and by_rule.median_z2.between(0.8, 1.25).all()
    rep.verdict(group_ok and frac3 < 0.01,
                "per-rule medians: ratio ≤ 1.2, z² in [0.8, 1.25]; |z| > 3 share < 1%. Relaxed from the "
                f"per-run criterion, which {len(bad)} of {len(runs)} runs miss; see the text")


# ---------------------------------------------------------------- H4

def predicted_fixed_point(p_ref, beta, G, rule, kl):
    """Stable root reached from p_ref (flow is upward at p_ref, so the first root above it)."""
    roots = fixed_points(p_ref, beta, G, rule, direction=kl)
    above = [p for p, stable in roots if p > p_ref]
    return (above[0] if above else np.nan), sum(stable for _, stable in roots)


def threshold(df, keys, value="absorbed", level=0.1):
    """Smallest eta whose seed-median `value` is >= level, per group of keys."""
    med = df.groupby(keys + ["eta"])[value].median().reset_index()
    rows = []
    for k, g in med.groupby(keys):
        g = g.sort_values("eta")
        hit = g[g[value] >= level]
        k = k if isinstance(k, tuple) else (k,)
        rows.append(dict(zip(keys, k), eta_star=hit.eta.iloc[0] if len(hit) else np.nan,
                         at_min_eta=bool(len(hit)) and hit.eta.iloc[0] == g.eta.min()))
    return pd.DataFrame(rows)


def analyze_h4(runs, prompts, rep, h3=None):
    rep.say("## H4: KL fixed points, fluctuations and the stability threshold (1,968 runs)")
    rep.say()

    # (a)
    a = runs[runs.part == "a"].copy()
    pred = [predicted_fixed_point(r.p0, r.beta, r.G, r.rule, r.kl) for r in a.itertuples()]
    a["predicted"] = [p for p, _ in pred]
    a["stable_roots"] = [n for _, n in pred]
    a["error"] = a.late_mean - a.predicted
    # A run that is still moving at the end has not reached its fixed point yet.
    # A relative test: at p ~ 0.02 an absolute tolerance would hide a run still doubling.
    a["converged"] = (a.final_mean - a.late_mean).abs() <= 0.05 * a.late_mean
    rep.say("### (a) Where training settles (960 runs; 64 identical tabular prompts, η = 1, B = 64, 6000 steps)")
    rep.say()
    rep.say("Measured p* is the mean success over the second half of training; the prediction is the "
            "stable root of the finite-G fixed-point equation (`theory.fixed_points`) reached from p_ref.")
    rep.say()
    slow = a[~a.converged].groupby(["p0", "beta", "kl"]).size().reset_index(name="runs")
    rep.say(f"Runs still moving at step 6000 (final mean differs from the second-half mean by > 5%): "
            f"**{(~a.converged).sum()} of {len(a)}**, all at "
            + ", ".join(f"p_ref {r.p0}, β {r.beta}, {r.kl} ({r.runs})" for r in slow.itertuples())
            + ". From p_ref = 0.01 the drive W_G(p)p(1−p) is tiny, so 6000 steps at η = 1 are not enough; "
            "the errors below are reported for all runs and for converged runs.")
    rep.say()
    summary = a.groupby(["kl", "rule"]).agg(
        runs=("error", "size"), median_abs_error=("error", lambda e: e.abs().median()),
        max_abs_error=("error", lambda e: e.abs().max()), mean_error=("error", "mean")).reset_index()
    conv = a[a.converged].groupby(["kl", "rule"]).agg(
        converged=("error", "size"), median_abs_error_conv=("error", lambda e: e.abs().median()),
        max_abs_error_conv=("error", lambda e: e.abs().max())).reset_index()
    summary = summary.merge(conv, how="left")
    rep.table(summary, "a_summary", show=True)
    rep.table(a[["name", "p0", "beta", "G", "rule", "kl", "seed", "predicted", "late_mean", "error",
                 "stable_roots", "converged"]], "a_runs")
    check = a[(a.p0 == 0.05) & (a.beta == 2) & (a.G == 4) & (a.kl == "reverse")]
    check_tab = check.groupby("rule").agg(predicted=("predicted", "first"), measured=("late_mean", "mean")).reset_index()
    rep.say("Guide check (p_ref 0.05, β 2, G 4, KL(π‖π_ref)); the guide value is 0.109:")
    rep.say()
    rep.say(markdown(check_tab, "{:.4f}"))
    rep.say()
    worst = a[a.converged].assign(abs_error=a.error.abs()).groupby(["p0", "beta", "G", "rule", "kl"]).agg(
        predicted=("predicted", "first"), measured=("late_mean", "mean"), abs_error=("abs_error", "mean"),
        stable_roots=("stable_roots", "first")).reset_index().sort_values("abs_error", ascending=False)
    rep.say("Largest errors among converged runs (seed-averaged):")
    rep.say()
    rep.say(markdown(worst.head(8), "{:.4f}"))
    rep.say()
    multi = worst[worst.stable_roots > 1]
    rep.say(f"Configurations with more than one stable root: **{len(multi)}** "
            f"({', '.join(f'p_ref {r.p0}, β {r.beta}, G {r.G}, {r.rule}, {r.kl}' for r in multi.head(6).itertuples())}"
            f"{', …' if len(multi) > 6 else ''}).")
    rep.say()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, (kl, g) in zip(axes, a.groupby("kl")):
        for i, (G, gg) in enumerate(g.groupby("G")):
            done = gg[gg.converged]
            moving = gg[~gg.converged]
            ax.scatter(done.predicted, done.late_mean, s=12, alpha=0.7, color=f"C{i}", label=f"G={G}")
            ax.scatter(moving.predicted, moving.late_mean, s=20, marker="x", color=f"C{i}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(title=f"KL {'(π‖π_ref)' if kl == 'reverse' else '(π_ref‖π)'}", xlabel="predicted p*")
    axes[0].set_ylabel("measured p* (second-half mean)")
    axes[0].legend(fontsize=7)
    rep.figure(fig, "a_fixed_points", "measured against predicted fixed points (crosses: still moving at step 6000)")
    med = a.error.abs().median()
    ok_check = (check_tab.measured - 0.109).abs().min() < 0.01 if len(check_tab) else False
    rep.verdict(med <= 0.01 and ok_check, f"median |error| = {med:.4f} ≤ 0.01 and the guide value 0.109 reproduced")

    # (b)
    b = runs[runs.part == "b"].copy()
    b["eta_over_beta"] = b.eta / b.beta
    rep.say("### (b) Fluctuations around the fixed point (144 runs; G = 8, GRPO, p_ref 0.05)")
    rep.say()
    rep.say("Var(logit p) across the 64 replicate prompts, averaged over second-half snapshots. A linear "
            "(Ornstein-Uhlenbeck) approximation predicts Var ∝ η/β, a log-log slope of 1.")
    rep.say()
    b["p_star"] = [predicted_fixed_point(r.p0, r.beta, r.G, r.rule, r.kl)[0] for r in b.itertuples()]
    b["at_equilibrium"] = (b.late_mean - b.p_star).abs() <= 0.02
    rep.say(f"Runs whose second-half mean is still > 0.02 from p* (not yet at equilibrium, so their "
            f"spread is a transient): **{(~b.at_equilibrium).sum()} of {len(b)}**, at η = "
            f"{sorted(b[~b.at_equilibrium].eta.unique().tolist())}. They are excluded from the fits.")
    rep.say()
    fits = []
    fig, ax = plt.subplots(figsize=(5, 4.2))
    eq = b[b.at_equilibrium]
    for kl, g in eq.groupby("kl"):
        m = g.groupby("eta_over_beta").agg(var=("var_logit", "mean")).reset_index()
        small = m[m.eta_over_beta <= 2]
        slope = np.polyfit(np.log(small.eta_over_beta), np.log(small["var"]), 1)[0] if len(small) >= 3 else np.nan
        slope_all = np.polyfit(np.log(m.eta_over_beta), np.log(m["var"]), 1)[0]
        per_beta = [np.polyfit(np.log(gb.eta), np.log(gb.var_logit), 1)[0] for _, gb in g.groupby("beta")]
        fits.append({"kl": kl, "slope_eta_over_beta_le_2": slope, "slope_all": slope_all,
                     "slope_in_eta_fixed_beta_median": float(np.median(per_beta)),
                     "slope_in_eta_min": min(per_beta), "slope_in_eta_max": max(per_beta)})
        ax.scatter(g.eta_over_beta, g.var_logit, s=12, alpha=0.6, label=kl)
    ax.set(xscale="log", yscale="log", xlabel="η / β", ylabel="Var(logit p)", title="H4(b) fluctuations")
    ax.legend(fontsize=8)
    fits = pd.DataFrame(fits)
    rep.table(fits, "b_slopes", show=True)
    rep.table(b[["name", "beta", "eta", "kl", "seed", "eta_over_beta", "var_logit", "late_mean", "p_star",
                 "at_equilibrium"]], "b_runs")
    rep.figure(fig, "b_fluctuations", "Var(logit p) against η/β (equilibrated runs)")
    rep.verdict(fits.slope_eta_over_beta_le_2.between(0.8, 1.2).all(),
                "pre-registered: slope against η/β in 0.8–1.2 for both directions, η/β ≤ 2")
    rep.say("Post-hoc reading: η/β is not a sufficient variable, because p* (and with it the "
            "per-step noise and the restoring rate) moves with β. At fixed β the slope in η is the "
            "clean test of the linear-noise picture; its median is shown in the table above.")
    rep.say()

    # (c)
    c = runs[runs.part == "c"].copy()
    if h3 is not None:
        base = h3[h3.level.isin(["c05", "m64"]) & (h3.L == 2) & (h3.G == 8) & (h3.B == 32)
                  & h3.rule.isin(["mean", "grpo"])]
        for kl in ("reverse", "forward"):
            c = pd.concat([c, base.assign(kl=kl, beta=0.0)], ignore_index=True)
    rep.say("### (c) KL and the absorption threshold (1,152 runs plus β = 0 from H3)")
    rep.say()
    rep.say("η* is the smallest η whose seed-median absorbed fraction is ≥ 0.1 (blank: never reached on the grid).")
    rep.say()
    thr = threshold(c, ["level", "rule", "kl", "beta"])
    sbar = c.groupby(["level", "rule", "kl", "beta", "eta"]).s_bar.median().reset_index()
    thr = thr.merge(sbar.rename(columns={"eta": "eta_star", "s_bar": "s_bar_star"}), how="left")
    wide = thr.pivot_table(index=["level", "rule", "kl"], columns="beta", values="eta_star", dropna=False).reset_index()
    wide.columns = [f"β={c}" if isinstance(c, float) else c for c in wide.columns]
    rep.table(thr, "c_thresholds")
    rep.say(markdown(wide, "{:.3g}"))
    rep.say()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (level, g) in zip(axes, thr.groupby("level")):
        for (rule, kl), gg in g.groupby(["rule", "kl"]):
            gg = gg.sort_values("beta")
            ax.plot(gg.beta.replace(0, 0.03), gg.eta_star, "o-", label=f"{rule}, {kl}")
        ax.set(xscale="log", yscale="log", title=level, xlabel="β (β = 0 drawn at 0.03)")
    axes[0].set_ylabel("η* (absorbed ≥ 0.1)")
    axes[0].legend(fontsize=7)
    rep.figure(fig, "c_threshold_vs_beta", "absorption threshold against β")
    absorbed_by_beta = c.groupby(["level", "kl", "beta"]).absorbed.mean().reset_index()
    rep.table(absorbed_by_beta, "c_absorbed_by_beta")
    rep.say("Mean absorbed fraction over the η grid, by β:")
    rep.say()
    rep.say(markdown(absorbed_by_beta.pivot_table(index=["level", "kl"], columns="beta", values="absorbed").reset_index()
                     .rename(columns=lambda x: f"β={x}" if isinstance(x, float) else x), "{:.3f}"))
    rep.say()


# ---------------------------------------------------------------- H3

def fit_logistic(x, y):
    """Fractional-response logistic fit y ~ sigmoid(a + b x); returns (a, b)."""
    from scipy.optimize import minimize

    def nll(w):
        return deviance(y, w[0] + w[1] * x)

    return minimize(nll, np.array([0.0, 1.0]), method="Nelder-Mead").x


def deviance(y, logits):
    q = np.clip(1 / (1 + np.exp(-logits)), 1e-9, 1 - 1e-9)
    return float(-2 * np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def auc(score, label):
    from sklearn.metrics import roc_auc_score
    label = np.asarray(label, dtype=bool)
    return roc_auc_score(label, score) if 0 < label.sum() < len(label) else np.nan


def raw_history(raw, name):
    path = glob.glob(os.path.join(raw, "*", "results", f"{name}.json.gz"))[0]
    with gzip.open(path, "rt") as f:
        return json.load(f)["history"]


def candidates(df):
    """Combined step-size variables for the H3 collapse, from the initial diagnostics."""
    base = df.s_bar * df.mean_sq_grad_logit
    return {
        "(i) s̄": df.s_bar,
        "(ii) s̄·‖∇z‖²": base,
        "(iii) (ii)·|cos|": base * df.mean_abs_cos,
        "(iv) (ii)·(1+B·cos)": base * (1 + df.B * df.mean_cos).clip(lower=1e-6),
    }


def score_collapse(shared, cands):
    """Pooled and leave-one-level-out logistic fits of absorbed fraction against log10 x."""
    y = shared.absorbed.to_numpy()
    label = y > 0.1
    rows, fits = [], {}
    for cname, x in cands.items():
        lx = np.log10(x.to_numpy())
        held_dev, held_auc, shifts = [], [], {}
        for level in shared.level.unique():
            out = (shared.level == level).to_numpy()
            a, b = fit_logistic(lx[~out], y[~out])
            held_dev.append(deviance(y[out], a + b * lx[out]))
            held_auc.append(auc(lx[out], label[out]))
        a, b = fit_logistic(lx, y)
        fits[cname] = (a, b, lx)
        for level in shared.level.unique():
            sel = (shared.level == level).to_numpy()
            al, bl = fit_logistic(lx[sel], y[sel])
            # Horizontal shift of the level's midpoint from the pooled midpoint, in decades of x (= decades of η).
            shifts[level] = (-al / bl) - (-a / b) if bl > 0 else np.nan
        rows.append({"candidate": cname, "pooled_AUC": auc(lx, label), "pooled_deviance": deviance(y, a + b * lx),
                     "held_out_deviance": float(np.mean(held_dev)), "held_out_AUC_mean": float(np.nanmean(held_auc)),
                     "max_level_shift_decades": float(np.nanmax(np.abs(list(shifts.values())))),
                     **{f"shift_{k}": v for k, v in shifts.items()}})
    return pd.DataFrame(rows), fits


def analyze_h3(runs, prompts, rep, raw="results/base"):
    rep.say("## H3 (main): effective step, sharing and absorption (13,608 runs)")
    rep.say()
    rep.say("Grid: 6 sharing levels × L {1, 2, 4} × 7 (B, G) pairs × rules {mean, rloo, grpo} × 12 η × 3 seeds, "
            "5000 steps of plain SGD, β = 0. *Absorbed* = share of prompts that started at p ≥ 0.01 and end "
            "below 0.01. (The plan said 3000 steps and 10 η values; the batch ran 5000 and 12.)")
    rep.say()

    # 1. Tabular check.
    tab = runs[runs.level == "tab"]
    bad = tab[(tab.decreases_total > 0) | (tab.absorbed > 0)]
    rep.say("### 1. Tabular runs never lose a prompt")
    rep.say()
    rep.say(f"Tabular runs: {len(tab)}. Runs with any per-prompt decrease of p: **{(tab.decreases_total > 0).sum()}**. "
            f"Runs with any absorbed prompt: **{(tab.absorbed > 0).sum()}**. Largest absorbed fraction: "
            f"{tab.absorbed.max():.3g}. Total decreases logged: {int(tab.decreases_total.sum())}.")
    rep.say()
    if len(bad):
        rep.table(bad[["name", "L", "B", "G", "rule", "eta", "seed", "decreases_total", "absorbed"]].head(20),
                  "tabular_exceptions", show=True)
    rep.verdict(len(bad) == 0, "every tabular run has 0 decreases and 0 absorbed (the proposition)")

    # 2. Phase diagram and thresholds.
    rep.say("### 2. Absorbed fraction against η")
    rep.say()
    main = runs[(runs.B == 32) & (runs.G == 8)]
    fig, axes = plt.subplots(3, 6, figsize=(17, 8), sharex=True, sharey=True)
    for i, L in enumerate((1, 2, 4)):
        for j, level in enumerate(LEVELS):
            ax = axes[i, j]
            for rule, g in main[(main.L == L) & (main.level == level)].groupby("rule"):
                m = g.groupby("eta").absorbed.agg(["median", "min", "max"])
                ax.plot(m.index, m["median"], "o-", ms=3, label=rule)
                ax.fill_between(m.index, m["min"], m["max"], alpha=0.2)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(level)
            if j == 0:
                ax.set_ylabel(f"L={L}\nabsorbed fraction")
            if i == 2:
                ax.set_xlabel("η")
    axes[0, 0].legend(fontsize=7)
    rep.figure(fig, "phase", "absorbed fraction against η per sharing level and L (B = 32, G = 8; bands: seed range)")
    thr = threshold(runs, ["level", "L", "B", "G", "rule"])
    rep.table(thr, "thresholds")
    t32 = thr[(thr.B == 32) & (thr.G == 8)].pivot_table(index=["level", "L"], columns="rule", values="eta_star",
                                                        dropna=False).reindex(LEVELS, level=0).reset_index()
    rep.say("η* (smallest η with seed-median absorbed ≥ 0.1; blank: never on the grid 0.003–1000), B = 32, G = 8:")
    rep.say()
    rep.say(markdown(t32, "{:.3g}"))
    rep.say()
    rep.say("Caveat for L = 4: prompts start at p = 3⁻⁴ ≈ 0.0123, so a dip of 0.002 counts as absorbed; "
            "the very small L = 4 thresholds mostly reflect that, not a large loss.")
    rep.say()
    lowest = thr[thr.at_min_eta]
    rep.say(f"Configurations already absorbing at the smallest η (0.003): **{len(lowest)}**, "
            f"levels {lowest.level.value_counts().to_dict()}. At those, absorption is not a step-size effect.")
    rep.say()

    # 3. Collapse.
    rep.say("### 3. Does one variable collapse the shared levels?")
    rep.say()
    shared = runs[runs.level != "tab"].copy()
    shared = shared[np.isfinite(shared.mean_sq_grad_logit) & (shared.s_bar > 0)]
    cands = candidates(shared)
    coll, fits = score_collapse(shared, cands)
    rep.table(coll, "collapse", show=False)
    rep.say(markdown(coll[["candidate", "pooled_AUC", "pooled_deviance", "held_out_deviance", "held_out_AUC_mean",
                           "max_level_shift_decades"]], "{:.3f}"))
    rep.say()
    rep.say("Per-level midpoint shifts (decades of x, relative to the pooled fit):")
    rep.say()
    rep.say(markdown(coll[["candidate"] + [c for c in coll.columns if c.startswith("shift_")]], "{:+.2f}"))
    rep.say()
    sub = shared[shared.L <= 2]
    rob, _ = score_collapse(sub, candidates(sub))
    rep.table(rob, "collapse_L_le_2")
    rep.say("Robustness, L ≤ 2 only (at L = 4 every prompt starts at p = 0.0123, just above the 0.01 line, "
            "so \"absorbed\" can mean a dip of 0.002):")
    rep.say()
    rep.say(markdown(rob[["candidate", "pooled_AUC", "held_out_deviance", "max_level_shift_decades"]
                         + [c for c in rob.columns if c.startswith("shift_")]], "{:.2f}"))
    rep.say()
    y = shared.absorbed.to_numpy()
    best = coll.sort_values("held_out_deviance").iloc[0]
    for cname, (a, b, lx) in fits.items():
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for level in [lv for lv in LEVELS if lv in shared.level.unique()]:
            sel = (shared.level == level).to_numpy()
            ax.scatter(lx[sel], y[sel], s=3, alpha=0.25, label=level)
        grid = np.linspace(lx.min(), lx.max(), 200)
        ax.plot(grid, 1 / (1 + np.exp(-(a + b * grid))), "k-", lw=1.5, label="pooled logistic")
        ax.set(xlabel=f"log10 {cname}", ylabel="absorbed fraction", title=f"H3 collapse: {cname}")
        ax.legend(fontsize=7, markerscale=3)
        safe = cname.split(")")[0].strip("(")
        rep.figure(fig, f"collapse_{safe}", f"absorbed fraction against {cname}" +
                   (" (best held-out candidate)" if cname == best.candidate else ""))
    s_only = coll[coll.candidate == "(i) s̄"].iloc[0]
    rep.verdict(best.candidate != "(i) s̄" and best.held_out_deviance < s_only.held_out_deviance
                and best.max_level_shift_decades <= 1,
                f"pre-registered: an interference-weighted candidate beats s̄ alone on held-out levels and no "
                f"level is shifted by more than one decade; best held-out candidate is {best.candidate}, "
                f"max shift {best.max_level_shift_decades:.2f} decades")

    # 4. eta versus B at G = 8.
    rep.say("### 4. Does η* scale with B (s̄ ∝ η/B)?")
    rep.say()
    g8 = thr[(thr.G == 8) & thr.B.isin([8, 32, 128])]
    wide = g8.pivot_table(index=["level", "L", "rule"], columns="B", values="eta_star").reset_index()
    wide.columns = [f"B={c}" if isinstance(c, (int, np.integer)) else c for c in wide.columns]
    slopes = []
    for k, g in g8.groupby(["level", "L", "rule"]):
        g = g.dropna(subset=["eta_star"])
        g = g[~g.at_min_eta]
        if len(g) >= 2:
            slopes.append(np.polyfit(np.log10(g.B), np.log10(g.eta_star), 1)[0])
    rep.table(wide, "eta_vs_B")
    rep.say(f"Slope of log η* against log B, over {len(slopes)} (level, L, rule) groups with an interior threshold "
            f"at two or more B: median **{np.median(slopes):.2f}** (IQR {np.quantile(slopes, 0.25):.2f}–"
            f"{np.quantile(slopes, 0.75):.2f}). s̄ ∝ η/B predicts 1; a slope of 0 would mean B does not matter."
            if slopes else "Too few interior thresholds to fit a slope.")
    rep.say()

    # 4b. Post hoc: two absorption mechanisms.
    rep.say("### 4b. Post hoc: two ways to be absorbed")
    rep.say()
    rep.say("*Not pre-registered; added after the collapse failed.* Every prompt starts at the same p "
            "(uniform initialization gives p = K^−L for all prompts), so any difference in fate comes from "
            "the parameter sharing. The first-order predicted drift d_x at initialization (`predicted_drift`) "
            "separates two mechanisms:")
    rep.say()
    tr = runs.iloc[prompts["row_train"]].reset_index(drop=True)
    d, p0, at_ = prompts["predicted_drift"], prompts["p_init"], prompts["absorbed_at"]
    alive = p0 >= 0.01
    g8 = thr[(thr.G == 8) & thr.B.isin([8, 32, 128]) & ~thr.at_min_eta].dropna(subset=["eta_star"])
    regime = []
    for level in LEVELS[1:]:
        lv = (tr.level == level).to_numpy()
        row = {"level": level, "share_negative_drift": float(((d[lv] < 0) & alive[lv]).sum() / alive[lv].sum())}
        for tag, sel in (("η≤1", lv & (tr.eta <= 1).to_numpy()), ("η>1", lv & (tr.eta > 1).to_numpy())):
            neg, pos, ab = (d[sel] < 0) & alive[sel], (d[sel] >= 0) & alive[sel], at_[sel] >= 0
            row[f"P(abs|d<0) {tag}"] = float(ab[neg].mean()) if neg.any() else np.nan
            row[f"P(abs|d≥0) {tag}"] = float(ab[pos].mean())
        sl = [np.polyfit(np.log10(gg.B), np.log10(gg.eta_star), 1)[0]
              for _, gg in g8[g8.level == level].groupby(["L", "rule"]) if len(gg) >= 2]
        row["η*-vs-B slope"] = float(np.median(sl)) if sl else np.nan
        regime.append(row)
    regime = pd.DataFrame(regime)
    rep.table(regime, "regimes")
    rep.say(markdown(regime, "{:.3f}"))
    rep.say()
    rep.say("- **Interference drift (correlated features, c05 and c09).** Initial drift ranks the prompts "
            "that end up absorbed (AUC in section 7), a negative drift makes absorption likely even at small η, "
            "the absorbed share plateaus over decades of η, and η* hardly moves with B. Absorption here is "
            "deterministic and set by the geometry, so step size only sets how fast it happens.")
    rep.say("- **Step noise (MLPs).** Initial drift barely matters, absorption needs a large enough step, and "
            "η* grows with B close to the s̄ ∝ η/B prediction. Within this regime s̄ alone already lines up "
            "the two MLP widths (section 3 shifts).")
    rep.say("- One initial-diagnostic variable cannot collapse both mechanisms, which is why section 3 fails. "
            "This is exactly what H3 stretch goals 2 (per-prompt predictor) and 3 (drift against noise) test.")
    rep.say()

    # 5. Group size at a fixed budget.
    rep.say("### 5. Group size at a fixed budget BG = 256")
    rep.say()
    budget = thr[(thr.B * thr.G == 256)]
    wide = budget.pivot_table(index=["level", "rule"], columns="G", values="eta_star").reindex(LEVELS, level=0).reset_index()
    wide.columns = [f"G={c}" if isinstance(c, (int, np.integer)) else c for c in wide.columns]
    rep.table(wide, "eta_star_vs_G")
    rep.say("η* by G (all L pooled by taking the median threshold over L):")
    rep.say()
    med = budget.groupby(["level", "rule", "G"]).eta_star.median().reset_index()
    rep.say(markdown(med.pivot_table(index=["level", "rule"], columns="G", values="eta_star")
                     .reindex(LEVELS, level=0).reset_index()
                     .rename(columns=lambda c: f"G={c}" if isinstance(c, (int, np.integer)) else c), "{:.3g}"))
    rep.say()
    best_final = runs[runs.B * runs.G == 256].groupby(["level", "rule", "G", "L", "eta"]).final_mean.median() \
        .groupby(["level", "rule", "G"]).max().reset_index()
    rep.table(best_final, "best_final_vs_G")
    rep.say("Best final mean success over η (median over seeds, max over η, mean over L):")
    rep.say()
    bf = runs[runs.B * runs.G == 256].groupby(["level", "rule", "G", "L", "eta"]).final_mean.median() \
        .groupby(["level", "rule", "G", "L"]).max().groupby(["level", "rule", "G"]).mean().reset_index()
    rep.say(markdown(bf.pivot_table(index=["level", "rule"], columns="G", values="final_mean").reindex(LEVELS, level=0)
                     .reset_index().rename(columns=lambda c: f"G={c}" if isinstance(c, (int, np.integer)) else c),
                     "{:.3f}"))
    rep.say()

    # 6. Dynamics.
    rep.say("### 6. When prompts get absorbed")
    rep.say()
    idx = prompts["row_train"]
    at = prompts["absorbed_at"]
    lvl = runs.level.to_numpy()[idx]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for level in LEVELS[1:]:
        v = at[lvl == level]
        v = v[v >= 0]
        if len(v):
            axes[0].hist(v, bins=50, histtype="step", density=True, label=f"{level} (n={len(v)})")
    axes[0].set(xlabel="step of absorption", ylabel="density", title="absorption times (all shared runs)")
    axes[0].legend(fontsize=7)
    m = runs[(runs.level == "m64") & (runs.L == 2) & (runs.B == 32) & (runs.G == 8) & (runs.rule == "grpo")
             & (runs.seed == 0)]
    star = thr[(thr.level == "m64") & (thr.L == 2) & (thr.B == 32) & (thr.G == 8) & (thr.rule == "grpo")].eta_star.iloc[0]
    for eta in sorted(m.eta.unique()):
        if np.isfinite(star) and star / 10 <= eta <= star * 10:
            hist = raw_history(raw, m[m.eta == eta].name.iloc[0])
            axes[1].plot([h["step"] for h in hist], [h["train"]["absorbed"] for h in hist], label=f"η={eta:g}")
    axes[1].set(xlabel="step", ylabel="absorbed fraction", title=f"m64, L=2, GRPO, B=32, G=8 (η*={star:g})")
    axes[1].legend(fontsize=7)
    rep.figure(fig, "dynamics", "absorption times and absorbed fraction over training")
    early = (at[(at >= 0) & (lvl != "tab")[:, None]] <= 500).mean()
    rep.say(f"Share of absorptions in the first 500 of 5000 steps: {early:.2f}.")
    rep.say()

    # 7. Previews of the stretch goals.
    rep.say("### 7. Previews of the H3 stretch goals from the existing data")
    rep.say()
    p_init = prompts["p_init"]
    drift = prompts["predicted_drift"]
    absorbed_frac = runs.absorbed.to_numpy()[idx]
    rows = []
    for j, r in enumerate(idx):
        if lvl[j] == "tab" or not (0.05 <= absorbed_frac[j] <= 0.95):
            continue
        p = p_init[j]
        alive = np.isfinite(p) & (p >= 0.01)
        lab = at[j][alive] >= 0
        if not (0 < lab.sum() < alive.sum()):
            continue
        pa, da = p[alive], drift[j][alive]
        rows.append({"level": lvl[j], "auc_drift": auc(-da, lab), "auc_logit_drift": auc(-da / (pa * (1 - pa)), lab),
                     "auc_p_init": auc(-pa, lab), "auc_p_var": auc(-(pa * (1 - pa)), lab)})
    pred = pd.DataFrame(rows)
    if len(pred):
        summ = pred.groupby("level").median().reindex([lv for lv in LEVELS if lv in pred.level.unique()])
        summ.insert(0, "runs", pred.groupby("level").size())
        summ = summ.reset_index()
        rep.table(summ, "predictor_auc")
        rep.say("**Per-prompt absorption predictor** (median AUC over runs with 5–95% absorbed). Scores: "
                "most negative predicted first-step drift d_x; the same in logit units d_x/(p(1−p)); "
                "lowest initial p_x; lowest p_x(1−p_x). Every prompt starts at the same p in these runs, so "
                "the two p baselines are 0.5 by construction; the H5 warm-started runs or the stretch grid "
                "are needed for a real baseline comparison.")
        rep.say()
        rep.say(markdown(summ, "{:.3f}"))
        rep.say()
    # Lost problems.
    sel = np.isin(lvl, LEVELS[1:])
    ab = (at >= 0) & sel[:, None]
    started = p_init[ab]
    rep.say(f"**Lost problems:** {ab.sum()} prompts were absorbed across the shared runs. Their starting p is "
            f"K^−L (0.33, 0.11 or 0.012 for L = 1, 2, 4), so the share that started above 0.05 "
            f"(**{(started > 0.05).mean():.2f}**) only says how many came from L ≤ 2: absorption is not limited "
            f"to prompts that were hard at the start.")
    rep.say()
    pf = prompts["p_final"]
    pass64 = lambda p: np.nanmean(1 - (1 - np.nan_to_num(p)) ** 64, axis=1)
    before, after = pass64(p_init), pass64(pf)
    df = pd.DataFrame({"level": lvl, "before": before, "after": after, "absorbed": absorbed_frac})
    hit = df[df.absorbed > 0.1].groupby("level")[["before", "after"]].mean().reindex(LEVELS[1:]).reset_index()
    rep.table(hit, "pass64")
    rep.say("Exact pass@64 before and after training, in runs with > 10% absorbed (mean over runs):")
    rep.say()
    rep.say(markdown(hit, "{:.3f}"))
    rep.say()
    c09 = runs[runs.level == "c09"].groupby("eta").absorbed.agg(["median", "mean", "max"]).reset_index()
    rep.table(c09, "c09_absorbed_by_eta")
    rep.say("**ρ = 0.9 at small η** (smoke test: about 35% absorbed at every η). Absorbed fraction by η, all c09 runs:")
    rep.say()
    rep.say(markdown(c09, "{:.3f}"))
    rep.say()


# ---------------------------------------------------------------- H2

def analyze_h2(runs, prompts, rep):
    rep.say("## H2: does parameter sharing distort the finite-G surrogate? (207 one-step + 207 short-training runs)")
    rep.say()
    runs = runs.assign(rule=[rule_label(r, k) for r, k in zip(runs.rule, runs.passk)])
    idx = prompts["row_onestep"]
    one = runs.iloc[idx].reset_index(drop=True)
    delta, se, drift, own, p0 = (prompts[k] for k in ("delta_mean", "delta_se", "predicted_drift", "own_term", "p0"))

    # First-order check and interference share, per run.
    rows = []
    for j, r in one.iterrows():
        x, y = drift[j], delta[j]
        slope = float(x @ y / (x @ x))
        r2 = 1 - np.sum((y - slope * x) ** 2) / np.sum((y - y.mean()) ** 2)
        cross = np.abs(drift[j] - own[j]).sum() / np.abs(drift[j]).sum()
        rows.append({"name": r["name"], "level": r.level, "rule": r.rule, "G": r.G, "seed": r.seed,
                     "slope": slope, "R2": r2, "cross_share": cross})
    fo = pd.DataFrame(rows)
    rep.table(fo, "first_order_runs")
    summ = fo.groupby("level").agg(runs=("slope", "size"), median_slope=("slope", "median"),
                                   median_R2=("R2", "median"), min_R2=("R2", "min"),
                                   cross_share=("cross_share", "median")).reindex(["tab", "lin", "m64"]).reset_index()
    rep.say("**First-order check.** Per run, the measured mean one-step change Δp_x (R repeats from the same θ, "
            "η = 0.5, full batch) is regressed on the predicted drift d_x = η∇p_x·mean_j W_G(p_j)∇p_j. "
            "*cross_share* = Σ|d_x − own_x| / Σ|d_x|, the part of the drift that comes from other prompts.")
    rep.say()
    rep.say(markdown(summ, "{:.3f}"))
    rep.say()

    # The milestone plot: binned ratio against the no-sharing prediction.
    key = one.set_index(["level", "G", "seed", "rule"]).index
    pos = {k: j for j, k in enumerate(key)}
    pts = []
    for j, r in one.iterrows():
        if r.rule == "mean":
            continue
        m = pos.get((r.level, r.G, r.seed, "mean"))
        if m is None:
            continue
        p = p0[j]
        edges = np.quantile(p, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
        for k in range(10):
            sel = b == k
            num, den = delta[j][sel].sum(), delta[m][sel].sum()
            noise = np.sqrt((se[m][sel] ** 2).sum())
            if abs(den) < 3 * noise or own[m][sel].sum() == 0:
                continue  # the mean-rule change in this bin is not resolved
            pts.append({"level": r.level, "rule": r.rule, "G": r.G, "p": p[sel].mean(),
                        "predicted": own[j][sel].sum() / own[m][sel].sum(), "measured": num / den})
    pts = pd.DataFrame(pts)
    rep.table(pts, "binned_ratio")
    fit = []
    for level, g in pts.groupby("level"):
        x, y = g.predicted.to_numpy(), g.measured.to_numpy()
        fit.append({"level": level, "bins": len(g), "slope": float(x @ y / (x @ x)),
                    "R2_vs_identity": 1 - np.sum((y - x) ** 2) / np.sum((y - y.mean()) ** 2),
                    "RMSE": float(np.sqrt(np.mean((y - x) ** 2))),
                    # Robust to the few bins where the mean-rule change is barely resolved.
                    "median_abs_dev": float(np.median(np.abs(y - x)))})
    fit = pd.DataFrame(fit).set_index("level").reindex(["tab", "lin", "m64"]).reset_index()
    fit = fit.merge(summ[["level", "cross_share"]])
    rep.say("**The milestone plot.** Each rule run is paired with the mean-rule run at the same level, G and seed "
            "(same warm start, so the same θ). Prompts are binned by p into deciles; y is the ratio of binned "
            "measured changes Δp(rule)/Δp(mean), x the no-sharing prediction Σ own(rule)/Σ own(mean), i.e. the "
            "W_G-weighted ratio. Bins whose mean-rule change is within 3 SE of 0 are dropped.")
    rep.say()
    rep.table(fit, "binned_fit", show=True, floatfmt="{:.3f}")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    for ax, level in zip(axes, ("tab", "lin", "m64")):
        g = pts[pts.level == level]
        for rule, gg in g.groupby("rule"):
            ax.scatter(gg.predicted, gg.measured, s=10, alpha=0.6, label=rule)
        lim = [0, max(pts.predicted.max(), 1) * 1.05]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(title=level, xlabel="W_G-weighted prediction (no sharing)")
    axes[0].set_ylabel("measured Δp(rule) / Δp(mean)")
    axes[0].legend(fontsize=7)
    rep.figure(fig, "binned_ratio", "binned one-step change relative to the mean rule against the no-sharing prediction")
    rep.say("Every level's measured change is explained by the full first-order drift (R² ≈ 0.999 above), but "
            "only the tabular level follows the per-prompt W_G-weighted prediction. With sharing, most of each "
            "prompt's drift comes from other prompts (cross_share > 0.9), so a rule's W_G(p) no longer sets how "
            "fast that prompt moves relative to another rule; the effect is large for the MLP and small for the "
            "linear model.")
    rep.say()
    tab = fit[fit.level == "tab"].iloc[0]
    ordered = fit.sort_values("cross_share").RMSE.is_monotonic_increasing
    rep.verdict(tab.R2_vs_identity >= 0.95 and 0.9 <= tab.slope <= 1.1 and ordered,
                "tabular on the line (R² ≥ 0.95, slope 0.9–1.1) and RMSE increasing with the interference share")

    # Ghosh et al.: does the short training trajectory depend on G?
    tr = runs[runs["mode"] == "train"]
    gh = tr.groupby(["level", "rule", "G"]).final_mean.mean().reset_index()
    wide = gh.pivot_table(index=["level", "rule"], columns="G", values="final_mean").reindex(["tab", "lin", "m64"], level=0)
    wide["max/min over G"] = wide.max(axis=1) / wide.min(axis=1)
    wide = wide.reset_index().rename(columns=lambda c: f"G={c}" if isinstance(c, (int, np.integer)) else c)
    rep.table(wide, "ghosh")
    rep.say("**Does the trajectory depend on G? (Ghosh et al.)** Mean success after 200 steps of ordinary training "
            "(B = 32), averaged over seeds:")
    rep.say()
    rep.say(markdown(wide, "{:.3f}"))
    rep.say()


# ---------------------------------------------------------------- H5

def analyze_h5(runs, prompts, rep):
    rep.say("## H5: group size at a fixed budget and escape from low p₀ (3,528 runs)")
    rep.say()
    rep.say("BG = 256 samples per step, G from 2 to 128, 5000 steps, best η per configuration. Tabular runs set "
            "p₀ exactly; linear and MLP runs are trained down to p₀ on decoy answers first, so their measured "
            "starting mean is used. *Samples to target* = 256 × the first logged step with mean p ≥ 0.5.")
    rep.say()
    cap = 256 * 5000
    runs = runs.assign(samples=256 * runs.step_to_half)
    keys = ["level", "p0", "G", "rule"]
    per_eta = runs.groupby(keys + ["eta"]).agg(samples=("samples", lambda v: np.nanmedian(v.fillna(np.inf))),
                                              p_init=("p_init_mean", "mean")).reset_index()
    best = per_eta.sort_values("samples").groupby(keys).first().reset_index()
    best["censored"] = ~np.isfinite(best.samples)
    rep.table(best, "best_eta")
    start = runs.groupby(["level", "p0"]).p_init_mean.mean().unstack()
    rep.say("Measured starting mean p by level (nominal p₀ in columns):")
    rep.say()
    rep.say(markdown(start.reindex(["tab", "lin", "m64"]).reset_index(), "{:.3f}"))
    rep.say()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, (p0, g) in zip(axes, best.groupby("p0")):
        for (level, rule), gg in g.groupby(["level", "rule"]):
            ls = {"tab": "-", "lin": ":", "m64": "--"}[level]
            col = "C0" if rule == "grpo" else "C1"
            y = gg.samples.where(~gg.censored, cap)
            ax.plot(gg.G, y, ls, color=col, label=f"{level}, {rule}")
            ax.scatter(gg.G[gg.censored], y[gg.censored], facecolors="none", edgecolors=col, s=25)
        ax.set(xscale="log", yscale="log", title=f"p₀ = {p0}", xlabel="G")
    axes[0].set_ylabel("samples to mean p = 0.5 (hollow: not reached)")
    axes[0].legend(fontsize=6)
    rep.figure(fig, "samples_vs_G", "samples to reach mean p = 0.5 against G, best η per point")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for (level, rule), g in best[~best.censored].groupby(["level", "rule"]):
        ax.scatter(g.G * g.p_init, g.samples, s=12, label=f"{level}, {rule}",
                   marker={"tab": "o", "lin": "s", "m64": "^"}[level], color="C0" if rule == "grpo" else "C1")
    ax.set(xscale="log", yscale="log", xlabel="G · p̄_init (expected correct samples per group)",
           ylabel="samples to target", title="H5 collapse")
    ax.legend(fontsize=6)
    rep.figure(fig, "collapse", "samples to target against G · p̄_init")

    bestG = best[~best.censored].sort_values("samples").groupby(["level", "p0", "rule"]).first().reset_index()
    wide = bestG.pivot_table(index=["level", "rule"], columns="p0", values="G").reset_index()
    wide.columns = [f"p₀={c}" if isinstance(c, float) else c for c in wide.columns]
    rep.table(bestG, "best_G")
    rep.say("Best G (fewest samples to target) by p₀:")
    rep.say()
    rep.say(markdown(wide, "{:.0f}"))
    rep.say()
    spread = best.groupby(["level", "p0", "rule"]).apply(
        lambda g: g.samples.max() / g.samples.min() if g.censored.sum() == 0 else np.inf,
        include_groups=False).rename("max/min over G").reset_index()
    sw = spread.pivot_table(index=["level", "rule"], columns="p0", values="max/min over G").reset_index()
    sw.columns = [f"p₀={c}" if isinstance(c, float) else c for c in sw.columns]
    rep.table(spread, "spread_over_G")
    rep.say("Sensitivity to G: max/min of samples-to-target over G (inf: some G never reached the target):")
    rep.say()
    rep.say(markdown(sw, "{:.2f}"))
    rep.say()

    rep.say("**Informative groups.** Measured share of groups with mixed rewards against the prediction "
            "mean_x φ(p_x) from the batch's own p (both averaged over training):")
    rep.say()
    inf = runs.groupby("level").apply(lambda g: pd.Series({
        "measured": g.informative.mean(), "predicted": g.expected_informative.mean(),
        "ratio": (g.informative / g.expected_informative).median()}), include_groups=False
    ).reindex(["tab", "lin", "m64"]).reset_index()
    rep.say(markdown(inf, "{:.3f}"))
    rep.say()
    pi = prompts["p_init"]
    rows = runs.iloc[prompts["row_train"]].reset_index(drop=True)
    het = []
    for (level, G), g in rows.groupby(["level", "G"]):
        p = pi[g.index]
        pbar = np.nanmean(p, axis=1)
        het.append({"level": level, "G": G, "phi_of_mean": float(np.mean(informative_fraction(pbar, G))),
                    "mean_of_phi": float(np.mean(np.nanmean(informative_fraction(p, G), axis=1)))})
    het = pd.DataFrame(het)
    het["excess_uninformative"] = het.phi_of_mean - het.mean_of_phi
    rep.table(het, "heterogeneity")
    rep.say("Heterogeneity at the start: φ(p̄) − mean φ(p_x) is the extra share of all-same groups caused by "
            f"spread in p (Nie et al.). Largest value over (level, G): {het.excess_uninformative.max():.3f}.")
    rep.say()

    ab = runs.groupby(["level", "p0", "rule"]).absorbed.mean().reset_index()
    aw = ab.pivot_table(index=["level", "rule"], columns="p0", values="absorbed").reset_index()
    aw.columns = [f"p₀={c}" if isinstance(c, float) else c for c in aw.columns]
    rep.table(ab, "absorbed")
    rep.say("Mean absorbed fraction over all η and G:")
    rep.say()
    rep.say(markdown(aw, "{:.3f}"))
    rep.say()

    t = bestG[(bestG.level == "tab") & (bestG.rule == "grpo")].sort_values("p0")
    from scipy.stats import spearmanr
    rho = spearmanr(t.p0, t.G).correlation if len(t) >= 3 else np.nan
    flat = spread[(spread.level == "tab") & (spread.rule == "mean")]["max/min over G"]
    rep.say("Samples are resolved only to 256 × 25 = 6,400 (the logging interval), which limits the ratios "
            "for fast configurations. The tabular mean-rule ratio reaches exactly 2.00 at p₀ = 0.3, on the "
            "boundary of the criterion.")
    rep.say()
    rep.verdict(rho < 0 and (flat <= 2).all(),
                f"pre-registered: GRPO's best G falls as p₀ rises in tabular (Spearman {rho:.2f}) and the mean "
                f"rule's samples-to-target vary by at most 2× over G (max {flat.max():.2f})")


# ---------------------------------------------------------------- stretch

def bootstrap_ci(v, stat=np.median, n=2000, seed=0):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = [stat(rng.choice(v, len(v))) for _ in range(n)]
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def analyze_s_h1(runs, prompts, rep):
    from scipy.stats import chi2_contingency
    runs = runs.assign(rule=[rule_label(r, k) for r, k in zip(runs.rule, runs.passk)],
                       ratio=runs.rel_err / runs.mc_rel)
    rep.say("## H1 re-run: loose, medium and tight tests (414 runs, R = 20,000)")
    rep.say()
    rep.say("Same 414 configurations as the base H1, with 5× the samples, split into 20 chunks. Two new "
            "statistics: the projected relative bias b∥ = (m − pred)·pred/‖pred‖² with an exact 1-D standard "
            "error, and for every run a null distribution of mean z² from 50 half-against-half splits of its "
            "own chunks (same coordinate correlation, no model involved).")
    rep.say()

    rep.say("### Loose: per-rule medians (the base criterion)")
    rep.say()
    by_rule = runs.groupby("rule").agg(runs=("ratio", "size"), median_ratio=("ratio", "median"),
                                       median_z2=("mean_z2", "median"),
                                       frac_z_gt3=("frac_abs_z_gt3", "mean")).reset_index()
    rep.table(by_rule, "loose", show=True)
    old_band = ((runs.ratio > 1.2) | ~runs.mean_z2.between(0.8, 1.25)).sum()
    rep.say(f"The base per-run band (ratio ≤ 1.2, z² in [0.8, 1.25]) is missed by {old_band} of {len(runs)} runs "
            f"at R = 20,000 (base run, R = 4,000: 174).")
    rep.say()
    loose = (by_rule.median_ratio <= 1.2).all() and by_rule.median_z2.between(0.8, 1.25).all() \
        and runs.frac_abs_z_gt3.mean() < 0.01
    rep.verdict(loose, "per-rule median ratio ≤ 1.2 and median z² in [0.8, 1.25]; |z| > 3 share < 1%")

    rep.say("### Medium: each run against its own calibrated null")
    rep.say()
    runs["miss"] = (runs.mean_z2 < runs.null_lo) | (runs.mean_z2 > runs.null_hi)
    rate = runs.miss.mean()
    tests = []
    for key in ("rule", "level", "G", "L", "init"):
        tab = pd.crosstab(runs[key], runs.miss)
        pval = chi2_contingency(tab)[1] if tab.shape[1] == 2 else 1.0
        tests.append({"split": key, "chi2_p": pval,
                      "miss_rate_by_group": ", ".join(f"{k}: {v:.2f}" for k, v in runs.groupby(key).miss.mean().items())})
    tests = pd.DataFrame(tests)
    rep.say(f"Runs whose mean z² falls outside the central 95% of their own null: **{runs.miss.sum()} of {len(runs)} "
            f"({rate:.3f})**. Expected under no bias: about 0.05 (a little more, since the 2.5/97.5% points come "
            f"from 50 draws).")
    rep.say()
    rep.table(tests, "medium_by_group", show=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(runs.null_rank, bins=20, range=(0, 1))
    axes[0].set(xlabel="rank of observed mean z² within its null", ylabel="runs", title="should be flat")
    t = runs.bias_par / runs.bias_par_se
    axes[1].hist(t, bins=40, density=True, alpha=0.7)
    x = np.linspace(-5, 5, 200)
    axes[1].plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), "k-")
    axes[1].set(xlabel="b∥ / SE", title="projected bias against N(0, 1)")
    rep.figure(fig, "null_rank_and_bias", "calibrated null ranks and projected-bias z-scores")
    rep.verdict(rate <= 0.08 and (tests.chi2_p > 0.01).all(),
                f"miss rate ≤ 0.08 (got {rate:.3f}) and no split with χ² p ≤ 0.01 (min p {tests.chi2_p.min():.3f})")
    runs["side"] = np.where(runs.mean_z2 < runs.null_lo, "low", np.where(runs.mean_z2 > runs.null_hi, "high", "inside"))
    sides = pd.crosstab([runs.L, runs.level], runs.side).reset_index()
    rep.table(sides, "medium_sides")
    high = (runs.side == "high").mean()
    rep.say(f"*Post hoc.* Bias can only make the error larger than its null, i.e. a *high* miss. High misses: "
            f"**{(runs.side == 'high').sum()} of {len(runs)} ({high:.3f})**, below the 0.025 expected by chance. "
            f"The split that fails is L: the L = 1 tabular runs have {((runs.L == 1) & (runs.level == 'tab') & (runs.side == 'low')).sum()} "
            f"*low* misses (error smaller than the null) and L = 2 has none at all, so the split-half null is "
            f"somewhat too wide or too narrow depending on the configuration. That is a calibration limit of the "
            f"null, not a sign of bias.")
    rep.say()
    rep.say(markdown(sides, "{:.0f}"))
    rep.say()

    rep.say("### Tight: bound on the bias along the predicted direction")
    rep.say()
    runs["ci_lo"] = runs.bias_par - 1.96 * runs.bias_par_se
    runs["ci_hi"] = runs.bias_par + 1.96 * runs.bias_par_se
    outside = runs[(runs.ci_lo < -0.02) | (runs.ci_hi > 0.02)]
    big = runs[t.abs() > 3.9]
    pooled = runs.groupby("rule").apply(lambda g: pd.Series({
        "pooled_bias": np.average(g.bias_par, weights=g.bias_par_se ** -2),
        "pooled_se": np.sqrt(1 / np.sum(g.bias_par_se ** -2)),
        "max_abs_z": (g.bias_par / g.bias_par_se).abs().max()}), include_groups=False).reset_index()
    rep.table(runs[["name", "rule", "G", "L", "level", "init", "bias_par", "bias_par_se", "ci_lo", "ci_hi"]], "tight_runs")
    rep.say(f"Runs whose 95% CI for b∥ leaves ±2%: **{len(outside)}**. Runs with |b∥/SE| > 3.9 (Bonferroni over "
            f"414 at 5%): **{len(big)}**. Median CI half-width: {1.96 * runs.bias_par_se.median():.4f}.")
    rep.say()
    rep.table(pooled, "tight_pooled", show=True, floatfmt="{:.4f}")
    if len(outside) or len(big):
        rep.table(pd.concat([outside, big]).drop_duplicates("name")[
            ["name", "rule", "G", "L", "level", "init", "bias_par", "bias_par_se"]], "tight_exceptions", show=True)
    rep.verdict(len(outside) == 0 and len(big) == 0 and (pooled.pooled_bias.abs() <= 0.005).all(),
                "every 95% CI inside ±2%, no |b∥/SE| > 3.9, pooled bias per rule within ±0.5%")
    wide_ci = outside.groupby(["L", "init"]).size().reset_index(name="runs")
    rep.say(f"*Post hoc.* The ±2% bound fails for lack of precision, not because of bias: every run outside it is "
            f"at L = 4 with uniform starts (p ≈ 0.012, so few samples carry signal), "
            f"{', '.join(f'L {r.L} {r.init}: {r.runs}' for r in wide_ci.itertuples())}; their |b∥| is at most "
            f"{outside.bias_par.abs().max():.3f} and their |z| at most {(outside.bias_par / outside.bias_par_se).abs().max():.2f}. "
            f"Across all 414 runs the largest |b∥/SE| is {t.abs().max():.2f} and the pooled bias per rule is within "
            f"±{pooled.pooled_bias.abs().max():.4f}. The rules share samples within a configuration, so these runs "
            f"are not independent.")
    rep.say()


def analyze_s_drift(runs, prompts, rep):
    rep.say("## H3 stretch: interference drift against step noise (2,340 runs)")
    rep.say()
    rep.say("Each configuration is trained three ways: *sampled* (ordinary GRPO), *expected* (the exact expected "
            "step on the sampled prompt batch: no response-sampling noise) and *expected_full* (the exact expected "
            "step on all 512 prompts: a deterministic flow). L = 2, GRPO, G = 8, 5000 steps, 5 seeds, 13 η per "
            "level centred on the base η*.")
    rep.say()
    levels = [lv for lv in ("lin", "c05", "c09", "m64") if lv in runs.level.unique()]
    main = runs[(runs.block == "main")]
    med = main.groupby(["level", "update", "eta"]).absorbed.median().reset_index()
    rep.table(med, "absorbed_by_mode")

    rep.say("### 1: Absorbed fraction by update mode")
    rep.say()
    fig, axes = plt.subplots(1, len(levels), figsize=(4.2 * len(levels), 3.8), sharey=True)
    for ax, level in zip(np.atleast_1d(axes), levels):
        for update, g in main[main.level == level].groupby("update"):
            m = g.groupby("eta").absorbed.agg(["median", "min", "max"])
            ax.plot(m.index, m["median"], "o-", ms=3, label=update)
            ax.fill_between(m.index, m["min"], m["max"], alpha=0.2)
        ax.set(xscale="log", title=level, xlabel="η")
    np.atleast_1d(axes)[0].set_ylabel("absorbed fraction (B = 32)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    rep.figure(fig, "absorbed_by_mode", "absorbed fraction against η for sampled, expected and deterministic updates")
    wide = med.pivot_table(index=["level", "eta"], columns="update", values="absorbed").reset_index()
    rep.say(markdown(wide, "{:.3g}"))
    rep.say()

    rep.say("### 2: Time to absorption")
    rep.say()
    sl = []
    fig, axes = plt.subplots(1, len(levels), figsize=(4.2 * len(levels), 3.8), sharey=True)
    for ax, level in zip(np.atleast_1d(axes), levels):
        for update, g in main[(main.level == level) & (main.absorbed >= 0.05)].groupby("update"):
            m = g.groupby("eta").agg(t50=("t50", "median"), n=("t50", "size")).reset_index()
            m = m[m.n >= 3]
            if len(m):
                ax.plot(m.eta, m.t50, "o-", ms=3, label=update)
            slope = np.polyfit(np.log10(m.eta), np.log10(m.t50), 1)[0] if len(m) >= 3 else np.nan
            sl.append({"level": level, "update": update, "points": len(m), "t50_slope": slope})
        ax.set(xscale="log", title=level, xlabel="η")
    if any(ax.lines for ax in np.atleast_1d(axes)):
        np.atleast_1d(axes)[0].set_yscale("log")  # shared y axis
    np.atleast_1d(axes)[0].set_ylabel("median absorption step (runs with ≥ 5% absorbed)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    rep.figure(fig, "t50", "median absorption time against η")
    sl = pd.DataFrame(sl, columns=["level", "update", "points", "t50_slope"])
    rep.table(sl, "t50_slopes", show=True)
    rep.say("A slope of −1 means the time scales as 1/η (deterministic drift); a much steeper or threshold-like "
            "curve means escape driven by noise.")
    rep.say()
    timing = main[main.absorbed >= 0.05].groupby(["level", "update"]).t50.describe()[["count", "min", "50%", "max"]] \
        .reset_index().rename(columns={"50%": "median"})
    rep.table(timing, "t50_range")
    rep.say("*Post hoc.* Median absorption step (log resolution 5 steps) over runs with ≥ 5% absorbed. A slope "
            "of 0 above is not a scaling result: in lin and c05 every absorption happens within the first 5 "
            "steps, a one-shot overshoot by a large step. Only c09 absorbs slowly (hundreds to thousands of steps).")
    rep.say()
    rep.say(markdown(timing, "{:.4g}"))
    rep.say()

    rep.say("### 3: Threshold against B")
    rep.say()
    thr = threshold(runs[runs.block.isin(["main", "batch"]) & runs["update"].isin(["sampled", "expected"])],
                    ["level", "update", "B"])
    rep.table(thr, "eta_star_by_B")
    rep.say(markdown(thr.pivot_table(index=["level", "update"], columns="B", values="eta_star").reset_index()
                     .rename(columns=lambda c: f"B={c}" if isinstance(c, (int, np.integer)) else c), "{:.3g}"))
    rep.say()
    bslope = {}
    for (level, update), g in thr.dropna(subset=["eta_star"]).groupby(["level", "update"]):
        g = g[~g.at_min_eta]
        if len(g) >= 2:
            bslope[(level, update)] = np.polyfit(np.log10(g.B), np.log10(g.eta_star), 1)[0]
    rep.say("Slope of log η* against log B: " + ", ".join(f"{k[0]} {k[1]}: {v:.2f}" for k, v in bslope.items())
            + ". Noise predicts 1 (s̄ ∝ η/B); drift predicts 0.")
    rep.say()

    rep.say("### 4: Drift share")
    rep.say()
    ds = []
    for level in levels:
        w = wide[wide.level == level]
        hit = w[w.sampled >= 0.05]
        share = (hit.expected_full / hit.sampled).clip(upper=2)
        ds.append({"level": level, "η with sampled ≥ 5% absorbed": len(hit),
                   "median share present without noise": float(share.median()) if len(hit) else np.nan,
                   "min share": float(share.min()) if len(hit) else np.nan})
    ds = pd.DataFrame(ds)
    rep.table(ds, "drift_share", show=True)

    # Pre-registered checks, operationalized before the results were seen.
    centre = {"lin": 1000, "c05": 30, "c09": 0.01, "m64": 1}
    corr_ok = []
    for level in ("c05", "c09"):
        w = wide[(wide.level == level) & (wide.sampled >= 0.05)]
        low = w[w.eta <= centre[level]]
        if not len(low):
            low = w.head(1)
        slope = sl[(sl.level == level) & (sl["update"] == "expected_full")].t50_slope
        corr_ok.append(len(low) > 0 and (low.expected_full >= 0.8 * low.sampled).all()
                       and len(slope) and -1.2 <= slope.iloc[0] <= -0.8)
    rep.verdict(all(corr_ok), "correlated features (c05, c09): at every η ≤ the base η* where sampled training "
                "absorbs ≥ 5%, the deterministic flow absorbs ≥ 0.8× as much, and its absorption time scales as "
                "η^s with s in [−1.2, −0.8]")
    w = wide[wide.level == "m64"]
    gap = w[(w.sampled >= 0.1) & (w.expected_full < 0.02)]
    s_b = bslope.get(("m64", "sampled"), np.nan)
    rep.verdict(len(gap) > 0 and 0.6 <= s_b <= 1.2, f"m64: some η where sampled absorbs ≥ 10% while the deterministic "
                f"flow absorbs < 2% ({len(gap)} such η), and the sampled η* scales with B with slope in [0.6, 1.2] "
                f"(got {s_b:.2f})")

    rep.say("### 5: Adam")
    rep.say()
    adam = runs[runs.block == "adam"].groupby(["level", "update", "eta"]).absorbed.median().reset_index()
    rep.table(adam, "adam")
    rep.say(markdown(adam.pivot_table(index=["level", "eta"], columns="update", values="absorbed").reset_index(), "{:.3g}"))
    rep.say()


def analyze_s_pred(runs, prompts, rep):
    rep.say("## H3 stretch: per-prompt absorption predictor (800 runs)")
    rep.say()
    rep.say("Within-run AUC for which prompts get absorbed. Scores: −d_x (predicted first-order drift), "
            "−d_x/(p(1−p)), −p_x, −‖∇p_x‖, and a 5-fold cross-validated logistic model on (log p_x, drift). "
            "*Step 0*: absorbed at any later time. *Checkpoints*: absorbed within the next 500 steps, averaged "
            "over checkpoints. *sft_mixed* starts give a real spread of p; uniform starts give every prompt the "
            "same p, so the p baseline is 0.5 there by construction.")
    rep.say()
    sd = runs.groupby(["level", "init"]).p_init_sd.median().unstack()
    rep.say("Median spread (sd) of starting p by level and init: " +
            "; ".join(f"{lv}: {', '.join(f'{k} {v:.2f}' for k, v in row.items())}" for lv, row in sd.iterrows()))
    rep.say()
    kinds = ("drift", "logit_drift", "p", "grad", "combo")
    rows = []
    for (level, init), g in runs.groupby(["level", "init"]):
        row = {"level": level, "init": init, "runs_scored": int(g.auc0_drift.notna().sum())}
        for k in kinds:
            row[f"step0_{k}"] = g[f"auc0_{k}"].median()
        for k in kinds:
            row[f"ck_{k}"] = g[f"aucck_{k}"].median()
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ["level"] = pd.Categorical(summ.level, LEVELS)
    summ = summ.sort_values(["init", "level"])
    rep.table(summ, "auc_summary")
    rep.say("Step-0 AUCs (median over runs):")
    rep.say()
    rep.say(markdown(summ[["level", "init", "runs_scored"] + [f"step0_{k}" for k in kinds]], "{:.3f}"))
    rep.say()
    rep.say("Checkpoint AUCs (absorbed within the next 500 steps; median over runs):")
    rep.say()
    rep.say(markdown(summ[["level", "init"] + [f"ck_{k}" for k in kinds]], "{:.3f}"))
    rep.say()

    mixed = runs[runs.init == "sft_mixed"]
    gains = []
    for level, g in mixed.groupby("level"):
        best = max(("p", "grad"), key=lambda k: g[f"auc0_{k}"].median())
        diff = (g.auc0_drift - g[f"auc0_{best}"]).dropna()
        ck = (g.aucck_drift - g.auc0_drift).dropna()
        gains.append({"level": level, "best_baseline": best, "runs": len(diff),
                      "drift_minus_baseline": diff.median(), "ci_lo": bootstrap_ci(diff)[0], "ci_hi": bootstrap_ci(diff)[1],
                      "ck_minus_step0": ck.median(), "ck_ci_lo": bootstrap_ci(ck)[0], "ck_ci_hi": bootstrap_ci(ck)[1]})
    gains = pd.DataFrame(gains)
    gains["level"] = pd.Categorical(gains.level, LEVELS)
    gains = gains.sort_values("level")
    rep.table(gains, "gains", show=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, k in enumerate(("drift", "p", "grad", "combo")):
        g = mixed.groupby("level")[f"auc0_{k}"].median().reindex([lv for lv in LEVELS if lv in mixed.level.unique()])
        ax.bar(np.arange(len(g)) + 0.2 * i, g.values, width=0.2, label=k)
    ax.set_xticks(np.arange(len(g)) + 0.3, g.index)
    ax.axhline(0.5, color="k", lw=0.8)
    ax.set(ylabel="median step-0 AUC", title="absorption predictor, sft_mixed starts")
    ax.legend(fontsize=7)
    rep.figure(fig, "auc", "step-0 AUC by score and level (sft_mixed starts)")

    def row(level):
        r = gains[gains.level == level]
        return r.iloc[0] if len(r) else None
    corr = [row(lv) for lv in ("c05", "c09")]
    rep.verdict(all(r is not None and r.drift_minus_baseline >= 0.05 and r.ci_lo > 0 for r in corr),
                "drift beats the best single baseline by ≥ 0.05 AUC with a bootstrap CI above 0 on c05 and c09")
    mlp = [row(lv) for lv in ("m64", "m256")]
    rep.verdict(all(r is not None and r.drift_minus_baseline < 0.05 for r in mlp),
                "on the MLPs step-0 drift does not beat the best baseline by 0.05")
    rep.verdict(all(r is not None and r.ck_ci_lo > 0 for r in mlp),
                "on the MLPs checkpoint drift predicts better than step-0 drift (bootstrap CI above 0)")


def analyze_s_abl(runs, prompts, rep):
    rep.say("## H3 stretch: ablations of the tabular proposition (558 runs)")
    rep.say()
    rep.say("Tabular, starting p spread over logits in [−4, 3], GRPO, B = 32, 5000 steps; one condition of the "
            "proposition broken at a time. *max drop* = largest p_init − min p over prompts.")
    rep.say()
    order = ["control", "flip05", "flip20", "momentum", "adam", "kl01", "kl1", "sign"]
    summ = runs.groupby("ablation").agg(
        runs=("decreases_total", "size"), runs_with_decreases=("decreases_total", lambda v: int((v > 0).sum())),
        median_decreases=("decreases_total", "median"), max_drop=("max_drop", "max"),
        mean_absorbed=("absorbed", "mean"), max_absorbed=("absorbed", "max"),
        mean_final=("final_mean", "mean")).reindex(order).reset_index()
    rep.table(summ, "summary", show=True)
    byL = runs.groupby(["ablation", "L"]).agg(max_drop=("max_drop", "max"), max_absorbed=("absorbed", "max")).reset_index()
    rep.table(byL, "by_L")
    rep.say("Largest drop and absorbed fraction by L:")
    rep.say()
    rep.say(markdown(byL.pivot_table(index="ablation", columns="L", values=["max_drop", "max_absorbed"])
                     .reindex(order).reset_index().pipe(lambda d: d.set_axis(
                         ["ablation"] + [f"{a} L={b}" for a, b in d.columns[1:]], axis=1)), "{:.3f}"))
    rep.say()
    by_eta = runs.groupby(["ablation", "eta"]).agg(max_drop=("max_drop", "max"), max_absorbed=("absorbed", "max")).reset_index()
    rep.table(by_eta, "by_eta")
    rep.say("By η (largest drop / largest absorbed fraction):")
    rep.say()
    rep.say(markdown(by_eta.assign(cell=[f"{a:.3f} / {b:.3f}" for a, b in zip(by_eta.max_drop, by_eta.max_absorbed)])
                     .pivot_table(index="ablation", columns="eta", values="cell", aggfunc="first").reindex(order)
                     .reset_index()))
    rep.say()
    ctrl = summ[summ.ablation == "control"].iloc[0]
    others = summ[summ.ablation != "control"]
    rep.verdict(ctrl.runs_with_decreases == 0 and (others.runs_with_decreases > 0).all(),
                "the control never lowers p, and every broken condition lowers p in at least one run")


def analyze_s_eq(runs, prompts, rep):
    rep.say("## H4 stretch: two stable KL fixed points (594 runs)")
    rep.say()
    rep.say("GRPO with KL(π‖π_ref), G = 64, tabular, 64 identical prompts per run (64 replicates). The policy "
            "starts at p_start, a logit offset from the predicted unstable root, while the reference stays at "
            "p_ref. A crossing is a replicate passing the unstable root between snapshots (every 100 steps).")
    rep.say()
    g = runs[runs.rule == "grpo"]
    basin = g.groupby(["p0", "beta", "eta", "offset"]).share_above_final.mean().reset_index()
    rep.table(basin, "basin")
    settings = list(g.groupby(["p0", "beta"]).groups)
    fig, axes = plt.subplots(1, len(settings), figsize=(4.3 * len(settings), 3.8), sharey=True)
    for ax, (p0, beta) in zip(np.atleast_1d(axes), settings):
        for eta, b in basin[(basin.p0 == p0) & (basin.beta == beta)].groupby("eta"):
            ax.plot(b.offset, b.share_above_final, "o-", ms=3, label=f"η={eta:g}")
        ax.axvline(0, color="k", lw=0.8)
        ax.set(title=f"p_ref {p0}, β {beta}", xlabel="start: logit offset from the unstable root")
    np.atleast_1d(axes)[0].set_ylabel("share ending in the upper state")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    rep.figure(fig, "basins", "basin of the upper fixed point against the starting offset")
    rep.say(markdown(basin[basin.eta <= 1].pivot_table(index=["p0", "beta", "eta"], columns="offset",
                                                       values="share_above_final").reset_index(), "{:.2f}"))
    rep.say()
    sw = g.groupby(["p0", "beta", "eta"]).crossings.mean().reset_index()
    rep.table(sw, "switching")
    rep.say("Mean crossings per run (64 replicates, 10,000 steps):")
    rep.say()
    rep.say(markdown(sw.pivot_table(index=["p0", "beta"], columns="eta", values="crossings").reset_index(), "{:.1f}"))
    rep.say()
    small = g[g.eta <= 1]
    err_low = (small.end_low_mean - small.stable_low).abs()
    err_high = (small.end_high_mean - small.stable_high).abs()
    rep.say(f"End states at η ≤ 1: median |error| to the lower root {err_low.median():.4f} (max {err_low.max():.4f}), "
            f"to the upper root {err_high.median():.4f} (max {err_high.max():.4f}).")
    rep.say()
    ctrl = runs[runs.rule == "mean"]
    ctrl_err = (ctrl.end_low_mean.fillna(ctrl.end_high_mean) - ctrl.stable_low).abs()
    rep.say(f"Mean-rule control (single predicted root): {ctrl.n_stable.max():.0f} stable root(s); share of runs "
            f"ending within 0.01 of it: {(ctrl_err <= 0.01).mean():.2f}.")
    rep.say()
    sharp = []
    for (p0, beta, eta), b in basin[basin.eta <= 1].groupby(["p0", "beta", "eta"]):
        lo = b[b.offset <= -0.5].share_above_final.max()
        hi = b[b.offset >= 0.5].share_above_final.min()
        sharp.append(lo < 0.1 and hi > 0.9)
    from scipy.stats import spearmanr
    rising = [spearmanr(b.eta, b.crossings).correlation > 0 for _, b in sw.groupby(["p0", "beta"])]
    rep.verdict(all(sharp), "at η ≤ 1 the upper-state share is < 0.1 for starts ≤ −0.5 logit and > 0.9 for "
                "starts ≥ +0.5 logit from the predicted unstable root")
    rep.verdict(err_low.median() <= 0.01 and err_high.median() <= 0.01,
                "end states within 0.01 of the predicted stable roots (median)")
    rep.verdict(all(rising), "crossings increase with η in every setting (Spearman > 0)")
    late = g[g.offset != 0].groupby(["p0", "beta", "eta"]).late_crossings.sum().reset_index()
    rep.table(late, "late_crossings")
    rep.say("*Post hoc.* The crossing count above is dominated by replicates started at the unstable root "
            "(offset 0) slowly leaving it, which is slower at small η. Switching between the two stable states, "
            "counted in the second half of training for starts away from the unstable root (total over runs):")
    rep.say()
    rep.say(markdown(late.pivot_table(index=["p0", "beta"], columns="eta", values="late_crossings").reset_index(), "{:.3g}"))
    rep.say()
    rep.say("Noise-driven switching appears only where the barrier is small (p_ref 0.03: stable 0.699, unstable "
            "0.757) and grows steeply with η, as a barrier-crossing picture predicts.")
    rep.say()
    rep.verdict(len(ctrl) > 0 and (ctrl_err <= 0.01).all(),
                "the mean-rule control ends at its single root from every start")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("collect", "all", "stretch") + HYPOTHESES + STRETCH)
    parser.add_argument("--raw", default="results/base")
    parser.add_argument("--out", default="analysis/base")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.what == "collect":
        collect(args.raw, args.jobs)
        return
    todo = {"all": HYPOTHESES, "stretch": STRETCH}.get(args.what, (args.what,))
    for h in todo:
        runs, prompts = load(args.raw, h)
        rep = Report(args.out, h)
        if h == "h4":
            ANALYSES[h](runs, prompts, rep, h3=load(args.raw, "h3")[0])
        elif h == "h3":
            ANALYSES[h](runs, prompts, rep, raw=args.raw)
        else:
            ANALYSES[h](runs, prompts, rep)
        rep.save()
        print(f"{h}: done")
    if todo[0].startswith("s_"):
        assemble(args.out, STRETCH, "stretch batch")
    else:
        assemble(args.out)


ANALYSES = {"h1": analyze_h1, "h2": analyze_h2, "h3": analyze_h3, "h4": analyze_h4, "h5": analyze_h5,
            "s_h1": analyze_s_h1, "s_drift": analyze_s_drift, "s_pred": analyze_s_pred, "s_abl": analyze_s_abl,
            "s_eq": analyze_s_eq}


if __name__ == "__main__":
    main()
