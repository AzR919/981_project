"""
Batch v3r (T30-T33), implementing PREREG_v3d_addendum11.md with its Amendment 1.

usage:
    python experiments/analyze_v3r.py baseline [--raw ~/rlvr-data/v3r] [--out analysis/v3r]
    python experiments/analyze_v3r.py collect  [--raw ~/rlvr-data/v3r] [--jobs 3]
    python experiments/analyze_v3r.py analyze  [--raw ~/rlvr-data/v3r] [--out analysis/v3r]

baseline (T32, Amendment 1 item 1) compares the recomputed GRPO flows (<raw>/flows/t32/grpo) with the saved T25 flows
(~/rlvr-data/v3m) and writes analysis/v3r/t32_baseline.md and .csv. It opens no pass@k flow. collect runs baseline
first, and only then reads the pass@k flows (t32/rloo_k2, t32/rloo_k4); then the T30 flows, then the runs of the three
grids (v3_t30, v3_t31, v3_t33) and, for T33(a), the existing sampled runs of T16 (~/rlvr-data/v3f) and A3
(~/rlvr-data/v3o, grid v3_a3). Everything is written to <raw>/cache: one CSV per grid, the trajectories (logged
class means and mean pass@1) in one .npz per grid, and the flow tables.

Definitions (from the earlier scripts unless the pre-registration fixes them):
- class 0 is the majority; the "minority class mean" is the mean success over the prompts of classes 1..K-1 pooled,
  as the simulator logs it (history class_mean[1]; analyze_v3f, analyze_v3g, check_permanence);
- absorbed (end of a run): tail max of p over the last 20% of training below tau, among prompts starting at or
  above 10 tau (analyze_v3.persistent, analyze_v3g); the absorbed share is over the minority prompts, ineligible
  prompts counting as not absorbed (analyze_v3g.t17_row). At the uniform start every prompt is eligible;
- pass@k = mean over all prompts of 1 - (1 - p)^k from the per-prompt p (analyze_harm.pass_at_k); mean pass@1 at a
  logged step is the logged mean success (history train.mean);
- the median of 1/p over minority prompts is reported as its log10, with p clipped at the float32 floor
  (analyze_harm).
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
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze import markdown  # noqa: E402
from experiments.analyze_harm import F32_FLOOR, pass_at_k  # noqa: E402

LN10 = np.log(10.0)
T25_TAUS = np.logspace(-1, 14, 80)
T32_RULES = ("grpo", "rloo_k2", "rloo_k4")
T32_K = {"grpo": 1, "rloo_k2": 2, "rloo_k4": 4}
T25_TASKS = [(rho, m, t) for rho in (0.3, 0.5, 0.7, 0.9, 0.95) for m in (0.7, 0.9) for t in range(10)]
T30_SCALES = (0, 0.5, 1, 2, 4, 8)
T30_MS = (1 / 3, 0.7, 0.9)
SEEDS = list(range(30, 50))
GRIDS = {"v3_t30": 1520, "v3_t31": 1300, "v3_t33": 120}
END = 100_000           # T31 and T33(b): last step


# ---------------------------------------------------------------- pure helpers (tests/test_analyze_v3r.py)

def log_class_mean(logp, mask):
    """log of the mean success over the prompts in mask, per row of logp (n_times x n), without underflow."""
    return logsumexp(logp[:, mask], axis=1) - np.log(mask.sum())


def first_at_or_above(steps, values, level):
    """First step at which values >= level; None if never."""
    hit = np.nonzero(np.asarray(values) >= level)[0]
    return None if not len(hit) else steps[hit[0]]


def largest_fall(values):
    """Largest fall below the running maximum (the first value included); 0 if the series never falls."""
    v = np.asarray(values, float)
    return float((np.maximum.accumulate(v) - v).max())


def stays_above_from(steps, a, b):
    """First step from which a > b at every later logged step (that step included); None if a <= b at the end."""
    above = np.asarray(a) > np.asarray(b)
    if not above[-1]:
        return None
    below = np.nonzero(~above)[0]
    return steps[0] if not len(below) else steps[below[-1] + 1]


def recovery_log10_time(log10_tau, mean, level=0.5):
    """log10 of the flow time from which mean stays above level to the end of the grid (the last upward crossing),
    linearly interpolated in log10 time between the two grid points around it. +inf if mean ends at or below level;
    log10_tau[0] if it is above level from the first point."""
    mean = np.asarray(mean, float)
    if mean[-1] <= level:
        return np.inf
    below = np.nonzero(mean <= level)[0]
    if not len(below):
        return float(log10_tau[0])
    i = below[-1]
    t0, t1, y0, y1 = log10_tau[i], log10_tau[i + 1], mean[i], mean[i + 1]
    return float(t0 + (level - y0) / (y1 - y0) * (t1 - t0))


def absorbed_share(p_init, tail, cls, tau):
    """Share of the minority prompts (class != 0) with tail < tau among those starting at or above 10 tau
    (ineligible prompts count as not absorbed, as analyze_v3g.t17_row)."""
    p_init, tail = np.asarray(p_init, float), np.asarray(tail, float)
    absorbed = (tail < tau) & (p_init >= 10 * tau)
    return float(absorbed[np.asarray(cls) != 0].mean())


def protected_and_learned(minority, majority, absorbed_1e2):
    """T31's per-task rule: final minority mean >= 0.9, final majority mean >= 0.9, no minority prompt absorbed at
    1e-2."""
    return bool(minority >= 0.9 and majority >= 0.9 and absorbed_1e2 == 0)


def p5_verdict(counts):
    """T31 prediction 5: 'protects' if some cell has c >= 16, 'fails' if every cell has c <= 4, else 'partial'."""
    counts = list(counts)
    if max(counts) >= 16:
        return "protects"
    if max(counts) <= 4:
        return "fails"
    return "partial"


def rerun_cell(cells):
    """The cell to rerun on seeds 50-69: the largest c, then the higher median final minority mean, then the smaller
    reset_every. cells: DataFrame with columns beta, reset_every, c, minority_final."""
    order = cells.sort_values(["c", "minority_final", "reset_every"], ascending=[False, False, True])
    return order.iloc[0]


def interp_log_time(taus, values, at):
    """values (one per tau) linearly interpolated in log10 tau at the flow times `at`."""
    return np.interp(np.log10(at), np.log10(taus), values)


def block_labels(cfg):
    """Correct class of each prompt of a block-feature task: prompts are ordered by class (block_dataset)."""
    from synthetic_rl.synthetic_task import class_counts
    return np.repeat(np.arange(cfg["K"]), class_counts(cfg["n_train"], float(cfg["majority"]), cfg["K"]))


def fmt_range(s, f="{:.3g}"):
    s = pd.Series(s).astype(float)
    return f"{f.format(s.median())} ({f.format(s.min())}–{f.format(s.max())})"


def count(mask):
    mask = np.asarray(mask, bool)
    return f"{int(mask.sum())}/{len(mask)}"


# ---------------------------------------------------------------- T32, step 1: the recomputed GRPO baseline

def t32_name(rho, m, t):
    return f"rho{rho}_m{m}_t{t}"


def load_flow(path):
    z = np.load(path)
    if "status" in z.files and str(z["status"]) != "ok":
        return None
    return dict(taus=z["taus"], logp=z["logp"].astype(float), cls=z["cls"],
                solver=str(z["solver"]) if "solver" in z.files else "")


def flow_task_row(f):
    """Per-task measures of one T32 flow: minority and majority class means over the grid."""
    cls = f["cls"]
    lmin = log_class_mean(f["logp"], cls != 0)
    maj = np.exp(log_class_mean(f["logp"], cls == 0))
    mino = np.exp(lmin)
    l10 = np.log10(f["taus"])
    row = dict(minority_final=float(mino[-1]), recovered=bool(mino[-1] > 0.5),
               log10_lowest_minority=float(lmin.min() / LN10), log10_recovery_time=recovery_log10_time(l10, mino),
               solver=f["solver"])
    for at, v in zip((1e2, 1e3, 1e4), interp_log_time(f["taus"], maj, [1e2, 1e3, 1e4])):
        row[f"majority_at_{at:.0e}"] = float(v)
    return row


def t32_baseline(raw, saved, out):
    """Recomputed GRPO flows against the saved T25 flows. Reads only t32/grpo and ~/rlvr-data/v3m."""
    gamma = pd.read_csv(os.path.join("analysis", "v3m", "runs.csv"))
    rows, diffs, mdiffs = [], [], []
    for rho, m, t in T25_TASKS:
        name = t32_name(rho, m, t)
        new = load_flow(os.path.join(raw, "flows", "t32", "grpo", name + ".npz"))
        old = load_flow(os.path.join(saved, name + ".npz"))
        assert np.allclose(np.log(new["taus"]), np.log(old["taus"])) and np.array_equal(new["cls"], old["cls"])
        d = np.abs(new["logp"] - old["logp"]) / LN10                   # per prompt and time, decades
        mino = old["cls"] != 0
        dm = np.abs(log_class_mean(new["logp"], mino) - log_class_mean(old["logp"], mino)) / LN10
        diffs.append(d.ravel())
        mdiffs.append(dm)
        rn, ro = flow_task_row(new), flow_task_row(old)
        g = gamma[(gamma.rho == rho) & (gamma.m == m) & (gamma.task == t)].iloc[0]
        rows.append(dict(rho=rho, m=m, task=t, solver_new=new["solver"], gamma_min=g.gamma_min,
                         max_dlog10_prompt=float(d.max()), median_dlog10_prompt=float(np.median(d)),
                         max_dlog10_minority_mean=float(dm.max()),
                         minority_final_saved=ro["minority_final"], minority_final_new=rn["minority_final"],
                         recovered_saved=ro["recovered"], recovered_new=rn["recovered"],
                         recovered_t25_csv=bool(g.recovered_class),
                         log10_lowest_saved=ro["log10_lowest_minority"], log10_lowest_new=rn["log10_lowest_minority"]))
    df = pd.DataFrame(rows)
    alld, allm = np.concatenate(diffs), np.concatenate(mdiffs)
    worst = df.loc[df.max_dlog10_prompt.idxmax()]
    agree = int((df.recovered_saved == df.recovered_new).sum())
    summary = dict(max_prompt=float(alld.max()), median_prompt=float(np.median(alld)),
                   p99_prompt=float(np.quantile(alld, 0.99)), max_mean=float(allm.max()),
                   median_mean=float(np.median(allm)), agree=agree, n=len(df),
                   recovered_saved=int(df.recovered_saved.sum()), recovered_new=int(df.recovered_new.sum()),
                   worst=f"{t32_name(worst.rho, worst.m, worst.task)}", solvers=df.solver_new.value_counts().to_dict())
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "t32_baseline.csv"), index=False)
    dis = df[df.recovered_saved != df.recovered_new]
    big = df.sort_values("max_dlog10_prompt", ascending=False).head(8)
    lines = ["# T32 step 1: recomputed GRPO flows against the saved T25 flows", "",
             "Generated by `experiments/analyze_v3r.py baseline` (Amendment 1, item 1). This step reads only the "
             "recomputed GRPO flows (`~/rlvr-data/v3r/flows/t32/grpo`) and the saved T25 flows (`~/rlvr-data/v3m`); "
             "no pass@k flow (`rloo_k2`, `rloo_k4`) is opened before it has been written.", "",
             f"- Tasks: {len(df)}; recomputed solver: {summary['solvers']} (T25: 98 LSODA, one Radau, one BDF).",
             f"- |Δ log10 success| per prompt over all tasks × 80 time points × 512 prompts: largest "
             f"{summary['max_prompt']:.3g} decades (task {summary['worst']}), median {summary['median_prompt']:.3g}, "
             f"99th percentile {summary['p99_prompt']:.3g}.",
             f"- |Δ log10 minority class mean| over tasks × time points: largest {summary['max_mean']:.3g} decades, "
             f"median {summary['median_mean']:.3g}.",
             f"- Task-level outcome (minority class mean > 0.5 at 1e14): {agree} of {len(df)} tasks agree; "
             f"recovered {summary['recovered_saved']} (saved) and {summary['recovered_new']} (recomputed).", ""]
    stiff = df[[(r.rho, r.m, r.task) in ((0.7, 0.9, 9), (0.9, 0.7, 5)) for r in df.itertuples()]]
    lines += ["The two tasks T25 finished with Radau (`rho0.7_m0.9_t9`) and BDF at tolerance 1e-6 (`rho0.9_m0.7_t5`), "
              "recomputed with LSODA here:", "",
              markdown(stiff[["rho", "m", "task", "max_dlog10_prompt", "max_dlog10_minority_mean",
                              "minority_final_saved", "minority_final_new"]]), ""]
    if len(dis):
        lines += ["Tasks whose outcome differs:", "", markdown(dis[["rho", "m", "task", "gamma_min", "solver_new",
                                                                    "minority_final_saved", "minority_final_new"]]), ""]
    lines += ["Largest per-prompt differences:", "",
              markdown(big[["rho", "m", "task", "gamma_min", "solver_new", "max_dlog10_prompt", "median_dlog10_prompt",
                            "max_dlog10_minority_mean", "minority_final_saved", "minority_final_new"]]), ""]
    with open(os.path.join(out, "t32_baseline.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))
    return summary


def collect_t32(raw, cache):
    """Per task and rule (grpo = the recomputed baseline, then rloo_k2, rloo_k4). Called after t32_baseline."""
    gamma = pd.read_csv(os.path.join("analysis", "v3m", "runs.csv"))
    rows = []
    for rule in T32_RULES:
        for rho, m, t in T25_TASKS:
            f = load_flow(os.path.join(raw, "flows", "t32", rule, t32_name(rho, m, t) + ".npz"))
            g = gamma[(gamma.rho == rho) & (gamma.m == m) & (gamma.task == t)].iloc[0]
            row = dict(rule=rule, k=T32_K[rule], rho=rho, m=m, task=t, gamma_min=g.gamma_min)
            if f is None:       # not integrable with any solver: counts as not recovered (none occurred)
                row.update(status="failed", recovered=False)
            else:
                row.update(status="ok", **flow_task_row(f))
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(cache, "t32_flows.csv"), index=False)
    print(f"t32: {len(rows)} flows", flush=True)


# ---------------------------------------------------------------- T30 flows: max_drop, first step, R, gamma_min

def t30_flow_one(path):
    from experiments.check_late_time import l2_max_margin
    from experiments.check_odds_law import predict
    from experiments.exact_flow import task_arrays

    z = np.load(path)
    cfg = json.loads(str(z["config"]))
    logp, cls, taus = z["logp"].astype(float), z["cls"], z["taus"]
    mino = cls != 0
    lmin = log_class_mean(logp, mino)
    row = dict(s=cfg["shared_scale"], m=round(cfg["majority"], 3), task_seed=cfg["task_seed"], status=str(z["status"]),
               solver=str(z["solver"]), max_drop=float(z["max_drop"].max()),
               prompts_drop_1e6=int((z["max_drop"] > 1e-6).sum()),
               first_tau=float(taus[0]), first_change=float(np.exp(lmin[0]) - 1 / 3),
               minority_lowest=float(np.exp(lmin.min())), minority_final=float(np.exp(lmin[-1])),
               absorbed_flow_end=float((logp[-1, mino] < np.log(1e-3)).mean()))
    for at, v in zip((1e2, 1e4), interp_log_time(taus, np.exp(lmin), [1e2, 1e4])):
        row[f"minority_at_{at:.0e}"] = float(v)
    pr = predict({k: v for k, v in cfg.items() if k not in ("diagnostics",)})
    row.update(R=float(-pr["pred_d_b_a"] / pr["pred_d_b_b"]), pred_d_b_mixed=pr["pred_d_b_mixed"])
    phi, cls2, _, _ = task_arrays(cfg)
    assert np.array_equal(cls2, cls)
    u = l2_max_margin(phi[cls == 0])
    row["gamma_min"] = float((phi[mino] @ u).min())
    row["gamma_median"] = float(np.median(phi[mino] @ u))
    row["support_margin"] = float((phi[cls == 0] @ u).min())
    return row


def collect_t30_flows(raw, cache, jobs):
    paths = sorted(glob.glob(os.path.join(raw, "flows", "t30", "*.npz")))
    with Pool(jobs) as pool:
        rows = pool.map(t30_flow_one, paths, chunksize=4)
    pd.DataFrame(rows).to_csv(os.path.join(cache, "t30_flows.csv"), index=False)
    print(f"t30 flows: {len(rows)}", flush=True)


# ---------------------------------------------------------------- runs

LABELS = {}


def history_arrays(d):
    h = d["history"]
    steps = np.array([r["step"] for r in h])
    cm = np.array([r["class_mean"] for r in h], float)
    mean = np.array([r["train"]["mean"] for r in h], float)
    return steps, cm[:, 0], cm[:, 1], mean


def snapshot(d, step):
    return np.asarray(d["snapshots"][str(step)], float)


def p_measures(p, cls, prefix):
    out = {f"{prefix}pass{k}": pass_at_k(p, k) for k in (1, 4, 8, 16)}
    out[f"{prefix}log10_median_inv_min"] = float(np.log10(np.median(1.0 / np.maximum(p[cls != 0], F32_FLOOR))))
    for c in np.unique(cls):
        out[f"{prefix}class{c}"] = float(p[cls == c].mean())
    return out


def arm_of(cfg):
    """(grid tag, arm label, arm id) from the job's level and settings."""
    level = cfg["level"]
    for tag in ("linear-exact-switch", "linear-exact", "mlp-exact", "linear-sampled"):
        if level.startswith(tag):
            label = level[len(tag) + 1:] if tag != "linear-exact-switch" else "switch"
            break
    if tag == "linear-exact-switch":
        tag = f"switch{level.split('switch')[1]}"
    parts = [label]
    if cfg.get("beta"):
        parts.append(f"β={cfg['beta']:g}")
    if cfg.get("reset_every"):
        parts.append(f"N={cfg['reset_every']}")
    if cfg.get("passk") and label.startswith("passk"):
        parts.append(f"k={cfg['passk']}")
    if label == "passk-matched":
        parts.append(f"η={cfg['eta']:.3g}")
    if cfg.get("switch_passk"):
        parts.append(f"k={cfg['switch_passk']}")
    return tag, label, " ".join(parts)


def t30_row(d):
    cfg = d["config"]
    cls = block_labels(cfg)
    p_init, tail = np.asarray(d["p_init"], float), np.asarray(d["tail_max"], float)
    steps, maj, mino, mean = history_arrays(d)
    pf = snapshot(d, cfg["steps"])
    row = dict(name=cfg["name"], arch=cfg["arch"], update=cfg["update"], s=cfg["shared_scale"],
               m=round(cfg["majority"], 3), eta_tilde=cfg["eta_tilde"], eta=cfg["eta"], task_seed=cfg["task_seed"],
               absorbed_1e3=absorbed_share(p_init, tail, cls, 1e-3), absorbed_1e2=absorbed_share(p_init, tail, cls, 1e-2),
               absorbed_all_1e3=float(((tail < 1e-3) & (p_init >= 1e-2)).mean()),
               majority_final=float(maj[-1]), minority_final=float(mino[-1]), minority_lowest=float(mino.min()),
               first_change=float(mino[1] - mino[0]), first_step=int(steps[1]),
               log_check=float(abs(pf[cls == 0].mean() - maj[-1]) + abs(pf[cls != 0].mean() - mino[-1])))
    for c in range(cfg["K"]):
        row[f"class{c}_final"] = float(pf[cls == c].mean())
    return row, None


def yardstick(d, cls):
    """T31's measures at step 2,000 and at the end (100,000), the same for every arm."""
    cfg = d["config"]
    steps, maj, mino, mean = history_arrays(d)
    p_init, tail = np.asarray(d["p_init"], float), np.asarray(d["tail_max"], float)
    row = {}
    hit = first_at_or_above(steps, maj, 0.99)
    row["maj99_step"] = np.nan if hit is None else int(hit)
    for w, step in (("2k_", 2000), ("end_", cfg["steps"])):
        i = int(np.nonzero(steps == step)[0][0])
        p = snapshot(d, step)
        row.update({f"{w}minority": float(mino[i]), f"{w}minority_lowest": float(mino[:i + 1].min()),
                    f"{w}majority": float(maj[i])})
        row.update(p_measures(p, cls, w))
        if w == "end_":
            a3, a2 = absorbed_share(p_init, tail, cls, 1e-3), absorbed_share(p_init, tail, cls, 1e-2)
        else:          # no per-prompt history before step 2,000: the snapshot value there
            a3, a2 = absorbed_share(p_init, p, cls, 1e-3), absorbed_share(p_init, p, cls, 1e-2)
        row[f"{w}absorbed_1e3"], row[f"{w}absorbed_1e2"] = a3, a2
        row[f"{w}protected"] = protected_and_learned(row[f"{w}minority"], row[f"{w}majority"], a2)
    row["majority_largest_fall"] = largest_fall(maj)
    return row


def t31_row(d):
    cfg = d["config"]
    cls = LABELS[cfg["task_seed"]]
    tag, label, arm = arm_of(cfg)
    row = dict(name=cfg["name"], tag=tag, label=label, arm=arm, arch=cfg["arch"], update=cfg["update"],
               rule=cfg["rule"], passk=cfg.get("passk"), kl=cfg.get("kl"), beta=cfg.get("beta", 0),
               reset_every=cfg.get("reset_every"), eta=cfg["eta"], task_seed=cfg["task_seed"],
               resets=d.get("resets"))
    row.update(yardstick(d, cls))
    return row, history_arrays(d)


def t33_row(d):
    cfg = d["config"]
    cls = LABELS[cfg["task_seed"]]
    tag, label, arm = arm_of(cfg)
    sw = int(d["switch_step"])
    steps, maj, mino, mean = history_arrays(d)
    i = int(np.nonzero(steps == sw)[0][0])
    after = steps > sw
    p_sw = snapshot(d, sw)
    row = dict(name=cfg["name"], tag=tag, arm=arm, k=cfg["switch_passk"], switch_step=sw, task_seed=cfg["task_seed"],
               majority_at_switch=float(maj[i]), minority_at_switch=float(mino[i]), pass1_at_switch=float(mean[i]),
               pass8_at_switch=pass_at_k(p_sw, 8), majority_fall=float(max(0.0, maj[i] - maj[after].min())),
               snapshot_check=float(abs(p_sw.mean() - mean[i])))
    row["majority_fell_005"] = bool(row["majority_fall"] >= 0.05)
    hits = []
    for step in (2000, END):
        p = snapshot(d, step)
        row[f"pass1_{step}"], row[f"pass8_{step}"] = float(p.mean()), pass_at_k(p, 8)
        hits.append(step > sw and p.mean() < row["pass1_at_switch"] and row[f"pass8_{step}"] > row["pass8_at_switch"])
    row["pass1_down_pass8_up"] = bool(any(hits))
    row["pass1_lowest_after"] = float(mean[after].min())
    row.update(yardstick(d, cls))
    return row, (steps, maj, mino, mean)


def t33a_row(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    cfg = d["config"]
    steps, maj, mino, mean = history_arrays(d)
    k = cfg.get("passk") or 1
    rule = cfg["rule"] if cfg.get("passk") is None else f"{cfg['rule']}{cfg['passk']}"
    src = "T16" if cfg["hypothesis"] == "v3_t16" else "A3"
    row = dict(name=f"{src}/{cfg['name']}", source=src, rule=rule, family=cfg["rule"], k=k, G=cfg["G"],
               eta=cfg["eta"], task_seed=cfg["task_seed"], init=cfg.get("init"), update=cfg["update"],
               majority_largest_fall=largest_fall(maj), majority_final=float(maj[-1]),
               minority_final=float(mino[-1]))
    return row, (steps, maj, mino, mean)


def read_run(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    h = d["config"]["hypothesis"]
    return h, {"v3_t30": t30_row, "v3_t31": t31_row, "v3_t33": t33_row}[h](d)


def init_worker(labels):
    LABELS.update(labels)


def t31_labels(seed):
    from experiments.analyze_v3 import t1_labels
    from experiments.grids import t31_base
    return t1_labels(t31_base("linear", seed))[0]


def save_traj(path, rows, trajs):
    arrays = {}
    for r, (steps, maj, mino, mean) in zip(rows, trajs):
        key = r["name"].replace("/", "__")
        arrays[key] = np.stack([steps, maj, mino, mean]).astype(float)
    np.savez_compressed(path, **arrays)


def load_traj(path):
    z = np.load(path)
    return {k.replace("__", "/"): z[k] for k in z.files}


def collect_runs(raw, cache, jobs, raw_f, raw_o):
    paths = sorted(glob.glob(os.path.join(raw, "*", "results", "*", "*.json.gz")))
    with Pool(jobs) as pool:
        labels = dict(zip(SEEDS, pool.map(t31_labels, SEEDS)))
    print(f"reading {len(paths)} v3r runs", flush=True)
    with Pool(jobs, initializer=init_worker, initargs=(labels,)) as pool:
        res = pool.map(read_run, paths, chunksize=16)
    for h in GRIDS:
        got = [r for g, r in res if g == h]
        rows = [r for r, _ in got]
        pd.DataFrame(rows).sort_values("name").to_csv(os.path.join(cache, f"{h}.csv"), index=False)
        if h != "v3_t30":
            save_traj(os.path.join(cache, f"traj_{h}.npz"), rows, [t for _, t in got])
        print(f"{h}: {len(rows)} runs", flush=True)
    old = sorted(glob.glob(os.path.join(raw_f, "*", "results", "v3_t16", "*.json.gz"))
                 + glob.glob(os.path.join(raw_o, "*", "results", "v3_a3", "*.json.gz")))
    with Pool(jobs) as pool:
        got = pool.map(t33a_row, old, chunksize=16)
    rows = [r for r, _ in got]
    pd.DataFrame(rows).to_csv(os.path.join(cache, "t33a.csv"), index=False)
    save_traj(os.path.join(cache, "traj_t33a.npz"), rows, [t for _, t in got])
    print(f"t33a: {len(rows)} runs", flush=True)


def collect(raw, out, jobs, saved, raw_f, raw_o):
    cache = os.path.join(raw, "cache")
    os.makedirs(cache, exist_ok=True)
    t32_baseline(raw, saved, out)          # Amendment 1: before any pass@k flow is opened
    collect_t32(raw, cache)
    collect_t30_flows(raw, cache, jobs)
    collect_runs(raw, cache, jobs, raw_f, raw_o)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("baseline", "collect", "analyze", "all"))
    ap.add_argument("--raw", default=os.path.expanduser("~/rlvr-data/v3r"))
    ap.add_argument("--saved", default=os.path.expanduser("~/rlvr-data/v3m"), help="the saved T25 flows")
    ap.add_argument("--raw-f", default=os.path.expanduser("~/rlvr-data/v3f"))
    ap.add_argument("--raw-o", default=os.path.expanduser("~/rlvr-data/v3o"))
    ap.add_argument("--out", default="analysis/v3r")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    if a.what == "baseline":
        t32_baseline(a.raw, a.saved, a.out)
    if a.what in ("collect", "all"):
        collect(a.raw, a.out, a.jobs, a.saved, a.raw_f, a.raw_o)
    if a.what in ("analyze", "all"):
        analyze(a.raw, a.out)


if __name__ == "__main__":
    main()
