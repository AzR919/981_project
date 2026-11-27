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


# ---------------------------------------------------------------- verdict rows

VERDICTS = []


def verdict(test, pred, claim, rule, result, ok):
    VERDICTS.append(dict(test=test, prediction=pred, claim=claim, rule=rule, result=result,
                         verdict="holds" if ok else "fails"))


def reported(test, pred, claim, result):
    VERDICTS.append(dict(test=test, prediction=pred, claim=claim, rule="reported", result=result, verdict="reported"))


def med_inf(x):
    """Median with NaN read as 'never' (+inf)."""
    x = np.where(np.isnan(np.asarray(x, float)), np.inf, np.asarray(x, float))
    return float(np.median(x))


def fmt_step(x):
    return "never" if not np.isfinite(x) else f"{x:,.0f}"


# ---------------------------------------------------------------- T30

def t30(cache, out):
    flows = pd.read_csv(os.path.join(cache, "t30_flows.csv"))
    runs = pd.read_csv(os.path.join(cache, "v3_t30.csv"))
    lin = runs[runs.arch == "linear"]
    lines = ["## T30: non-negative interactions at full size (block features)", ""]

    f0 = flows[flows.s == 0]
    ok1 = f0.max_drop <= 1e-6
    verdict("T30", "1 (confirmatory)", "s = 0, exact flow: no prompt's success falls more than 1e-6 below its running "
            "maximum", "60 of 60 tasks", f"{count(ok1)} tasks; largest max_drop {f0.max_drop.max():.3g} "
            f"(over 512 prompts × 221 flow times to 1e8)", ok1.all() and len(f0) == 60)

    r0 = lin[(lin["update"] == "expected_full") & (lin.s == 0)]
    ok2 = (r0.absorbed_1e3 == 0) & (r0[["class0_final", "class1_final", "class2_final"]] > 0.99).all(axis=1)
    worst = r0[["class0_final", "class1_final", "class2_final"]].min(axis=1)
    verdict("T30", "2", "s = 0, discrete exact runs: no minority prompt absorbed and every class mean ends above 0.99",
            "≥ 95% of runs", f"{count(ok2)} runs ({ok2.mean():.1%}); absorbed in {int((r0.absorbed_1e3 > 0).sum())}; "
            f"lowest final class mean {worst.min():.4f}", ok2.mean() >= 0.95)

    cell = (lin.groupby(["update", "eta_tilde", "m", "s"]).agg(
        median=("absorbed_1e3", "median"), mean=("absorbed_1e3", "mean"), lo=("absorbed_1e3", "min"),
        hi=("absorbed_1e3", "max"), tasks=("absorbed_1e3", "size"),
        median_1e2=("absorbed_1e2", "median"), minority_final=("minority_final", "median"),
        majority_final=("majority_final", "median")).reset_index())
    cell.to_csv(os.path.join(out, "t30_cells.csv"), index=False)

    ex = lin[(lin["update"] == "expected_full") & (lin.m == 0.9) & (lin.eta_tilde == 1.5)]
    w = ex.pivot_table(index="task_seed", columns="s", values="absorbed_1e3")
    a0 = (w[0] == 0)
    up8 = w[8] > w[0]
    curve = w.median()
    first = next((s for s in T30_SCALES if curve[s] >= 0.05), None)
    verdict("T30", "3", "m = 0.9, exact, η̃ = 1.5: absorbed share 0 at s = 0; at s = 8 above the s = 0 value with a "
            "median ≥ 0.5", "s = 0: 20 of 20 at 0; s = 8: ≥ 18 of 20 above, median ≥ 0.5",
            f"s = 0: {count(a0)} at 0; s = 8: {count(up8)} above, median {curve[8]:.3f}",
            a0.sum() == 20 and up8.sum() >= 18 and curve[8] >= 0.5)
    reported("T30", "3 (reported)", "dose–response at m = 0.9 (exact, η̃ = 1.5): median absorbed share by s; smallest "
             "s with median ≥ 0.05", ", ".join(f"s = {s:g}: {curve[s]:.3f}" for s in T30_SCALES)
             + f"; smallest s with median ≥ 0.05: {first if first is None else f'{first:g}'}")

    m3 = cell[cell.m == 0.333]
    worst4 = m3.loc[m3["median"].idxmax()]
    verdict("T30", "4", "m = 1/3: median absorbed share ≤ 0.05 at every s",
            "every (update, η̃, s) cell of the linear grid",
            (f"largest cell median {worst4['median']:.3f} ({worst4['update']}, η̃ = {worst4.eta_tilde:g}, "
             f"s = {worst4.s:g}); " if worst4["median"] > 0 else "every cell median 0 (largest share in any run "
             f"{m3.hi.max():.3g}); ") +
            f"{int((m3['median'] <= 0.05).sum())}/{len(m3)} cells pass", (m3["median"] <= 0.05).all())

    sm = cell[(cell["update"] == "sampled") & (cell.m == 0.9)]
    parts, ok5 = [], True
    for e, g in sm.groupby("eta_tilde"):
        a, b = float(g[g.s == 0]["median"].iloc[0]), float(g[g.s == 8]["median"].iloc[0])
        ok5 &= a <= 0.02 and b >= 0.5
        parts.append(f"η̃ = {e:g}: s = 0 {a:.3f}, s = 8 {b:.3f}")
    verdict("T30", "5", "sampled GRPO, m = 0.9: median absorbed share ≤ 0.02 at s = 0 and ≥ 0.5 at s = 8",
            "both thresholds, at each η̃", "; ".join(parts), ok5)

    fell = flows.first_change < 0
    agree6 = fell == (flows.R > 1)
    near = flows[~agree6]
    reported("T30", "6", "first-step sign of the minority class mean (exact flow at τ = 1e-3) against R > 1",
             f"agrees in {count(agree6)} tasks" + (f"; disagreements at R = {', '.join(f'{r:.3f}' for r in near.R)}"
                                                    if len(near) else ""))
    ex1 = lin[lin["update"] == "expected_full"].merge(flows[["s", "m", "task_seed", "R"]], on=["s", "m", "task_seed"])
    agree_run = (ex1.first_change < 0) == (ex1.R > 1)

    # Per task at m = 0.9: R(s), gamma_min(s), absorbed share.
    tasks = flows[flows.m == 0.9][["s", "task_seed", "R", "gamma_min", "gamma_median", "support_margin", "max_drop",
                                   "minority_lowest", "minority_final", "absorbed_flow_end"]].merge(
        ex[["s", "task_seed", "absorbed_1e3"]].rename(columns={"absorbed_1e3": "absorbed_exact_eta1.5"}),
        on=["s", "task_seed"], how="left")
    tasks.sort_values(["s", "task_seed"]).to_csv(os.path.join(out, "t30_tasks_m0.9.csv"), index=False)
    flows.to_csv(os.path.join(out, "t30_flows.csv"), index=False)
    by_s = tasks.groupby("s").agg(R=("R", fmt_range), gamma_min=("gamma_min", fmt_range),
                                  tasks_gamma_ge_1=("gamma_min", lambda g: count(g >= 1)),
                                  tasks_R_gt_1=("R", lambda r: count(r > 1)),
                                  flow_max_drop=("max_drop", "median"),
                                  flow_minority_at_1e8=("minority_final", "median"),
                                  absorbed=("absorbed_exact_eta1.5", fmt_range)).reset_index()

    lines += [f"Flows: {len(flows)} tasks, solver {flows.solver.value_counts().to_dict()}; runs: {len(runs)} "
              f"({len(lin)} linear, {len(runs) - len(lin)} MLP).", "",
              "**Dose–response** (median absorbed share of minority prompts at 1e-3 over 20 tasks; columns are s):", ""]
    dose = cell.pivot_table(index=["update", "eta_tilde", "m"], columns="s", values="median").reset_index()
    dose.columns = [c if isinstance(c, str) else f"s={c:g}" for c in dose.columns]
    anyab = cell.assign(n=lin.groupby(["update", "eta_tilde", "m", "s"]).absorbed_1e3.apply(
        lambda a: int((a > 0).sum())).values).pivot_table(index=["update", "eta_tilde", "m"], columns="s",
                                                           values="n").reset_index()
    anyab.columns = [c if isinstance(c, str) else f"s={c:g}" for c in anyab.columns]
    lines += [markdown(dose), "", f"Every cell's shares lie in [{cell.lo.min():g}, {cell.hi.max():g}]; cells are "
              f"either all 0 or all 1 except {int(((cell.lo < cell.hi)).sum())} of {len(cell)}. Tasks (of 20) with at "
              "least one minority prompt absorbed:", "", markdown(anyab), "",
              "The discrete runs stop at flow time η̃ × 20,000 / (5 + s²): 2,000 (η̃ = 0.5) and 6,000 (η̃ = 1.5) at "
              "s = 0, but 145 and 435 at s = 8. The flows below show where the same tasks go by 1e8.", "",
              "**Per task at m = 0.9** (R from `check_odds_law.predict`, γ_min from `check_late_time.l2_max_margin` "
              "as in `check_permanence`; median (range) over 20 tasks; flow columns from the exact flow to 1e8; "
              "absorbed from the exact runs at η̃ = 1.5). Per task: `t30_tasks_m0.9.csv`.", "", markdown(by_s), "",
              f"The l2 max-margin solve has a smallest majority margin of {flows.support_margin.min():.4f} (it should "
              f"be 1; L-BFGS-B tolerance), so γ_min is accurate to about {1 - flows.support_margin.min():.1g}.", "",
              f"As a check only: in the discrete exact runs the sign at the first logged step (step 100, flow time "
              f"2–30, past the first-order regime) agrees with R > 1 in {count(agree_run)} runs; the disagreements are "
              f"all at R between {ex1[~agree_run].R.min():.2f} and {ex1[~agree_run].R.max():.2f}, where the minority "
              "dips and is already back above 1/3 by step 100.", ""]
    mlp = runs[runs.arch == "mlp"].groupby(["update", "s"]).agg(
        runs=("absorbed_1e3", "size"), absorbed_median=("absorbed_1e3", "median"),
        absorbed_mean=("absorbed_1e3", "mean"), minority_final=("minority_final", "median"),
        majority_final=("majority_final", "median")).reset_index()
    lines += ["**Exploratory MLP** (width 64, m = 0.9, η̃ = 0.5; the theorem does not cover it):", "", markdown(mlp), ""]
    flowcell = flows.groupby(["m", "s"]).agg(max_drop=("max_drop", "max"), R=("R", "median"),
                                             gamma_min=("gamma_min", "median"),
                                             minority_lowest=("minority_lowest", "median"),
                                             minority_at_1e4=("minority_at_1e+04", "median"),
                                             minority_at_1e8=("minority_final", "median"),
                                             absorbed_at_1e8=("absorbed_flow_end", "median")).reset_index()
    lines += ["**Exact flows by cell** (medians over 20 tasks; max_drop is the largest over tasks; absorbed at 1e8: "
              "share of minority prompts below 1e-3 at flow time 1e8):", "", markdown(flowcell), ""]
    runs.to_csv(os.path.join(out, "t30_runs.csv"), index=False)
    return lines


# ---------------------------------------------------------------- T31

YCOLS = ["minority", "minority_lowest", "majority", "pass1", "pass4", "pass16", "log10_median_inv_min",
         "absorbed_1e3", "absorbed_1e2"]


def arm_order():
    from experiments.grids import t31_arms
    names = []
    for label, s in t31_arms():
        cfg = dict(level=f"linear-exact-{label}", **s)
        cfg.setdefault("eta", 0.1)
        names.append(arm_of(cfg)[2])
    return names


def yard_table(df, w, order):
    rows = []
    for arm in order:
        g = df[df.arm == arm]
        if not len(g):
            continue
        hit = g.maj99_step.where(g.maj99_step <= (2000 if w == "2k_" else END))
        rows.append({"arm": arm, "tasks": len(g),
                     "minority": fmt_range(g[f"{w}minority"]), "lowest minority": fmt_range(g[f"{w}minority_lowest"]),
                     "majority": fmt_range(g[f"{w}majority"]),
                     "step majority ≥ 0.99 (reached)": f"{fmt_step(med_inf(hit))} ({int(hit.notna().sum())})",
                     "pass@1": fmt_range(g[f"{w}pass1"]), "pass@4": f"{g[f'{w}pass4'].median():.3g}",
                     "pass@16": f"{g[f'{w}pass16'].median():.3g}",
                     "log10 median 1/p (min.)": f"{g[f'{w}log10_median_inv_min'].median():.3g}",
                     "absorbed 1e-3": fmt_range(g[f"{w}absorbed_1e3"]), "absorbed 1e-2": fmt_range(g[f"{w}absorbed_1e2"]),
                     "protected and learned": int(g[f"{w}protected"].sum())})
    return pd.DataFrame(rows)


def start_row():
    p = np.full(512, 1 / 3)
    return {"arm": "start policy", "tasks": "", "minority": "0.333", "lowest minority": "", "majority": "0.333",
            "step majority ≥ 0.99 (reached)": "", "pass@1": f"{pass_at_k(p, 1):.3g}", "pass@4": f"{pass_at_k(p, 4):.3g}",
            "pass@16": f"{pass_at_k(p, 16):.3g}", "log10 median 1/p (min.)": f"{np.log10(3):.3g}",
            "absorbed 1e-3": "0", "absorbed 1e-2": "0", "protected and learned": ""}


def traj_diff(traj, a, b):
    """Largest |difference| of the logged class means (both classes) between runs a and b."""
    x, y = traj[a], traj[b]
    assert np.array_equal(x[0], y[0])
    return float(np.abs(x[1:3] - y[1:3]).max())


def t31(cache, out):
    runs = pd.read_csv(os.path.join(cache, "v3_t31.csv"))
    traj = load_traj(os.path.join(cache, "traj_v3_t31.npz"))
    lin = runs[runs.tag == "linear-exact"]
    order = arm_order()
    assert set(lin.arm) == set(order), set(lin.arm) ^ set(order)
    lines = ["## T31: prevention with one knob changed (linear, ρ = 0.9, m = 0.9, exact update, SGD η = 0.1)", ""]
    end_tab = pd.DataFrame([start_row()] + yard_table(lin, "end_", order).to_dict("records"))
    k2_tab = pd.DataFrame([start_row()] + yard_table(lin, "2k_", order).to_dict("records"))
    end_tab.to_csv(os.path.join(out, "t31_arms_end.csv"), index=False)
    k2_tab.to_csv(os.path.join(out, "t31_arms_2000.csv"), index=False)
    # numeric medians per arm, both times
    num = lin.groupby("arm")[[f"{w}{c}" for w in ("2k_", "end_") for c in YCOLS] + ["maj99_step"]].median()
    num["protected_2k"] = lin.groupby("arm")["2k_protected"].sum()
    num["protected_end"] = lin.groupby("arm")["end_protected"].sum()
    num.reindex(order).reset_index().to_csv(os.path.join(out, "t31_arms.csv"), index=False)

    key = lambda g: g.set_index("task_seed")  # noqa: E731
    plain = key(lin[lin.arm == "plain"])

    # P1: reset_every = 1 against plain
    d1 = []
    for beta in (0.3, 1, 3):
        r = key(lin[(lin.label == "reset-reverse") & (lin.beta == beta) & (lin.reset_every == 1)])
        d1 += [traj_diff(traj, r.loc[t, "name"], plain.loc[t, "name"]) for t in SEEDS]
    d1 = np.array(d1)
    rm = key(lin[(lin.label == "reset-reverse-mean") & (lin.reset_every == 1)])
    pm = key(lin[lin.arm == "plain-mean"])
    d1m = np.array([traj_diff(traj, rm.loc[t, "name"], pm.loc[t, "name"]) for t in SEEDS])
    verdict("T31", "1 (identity)", "reset_every = 1 reproduces plain GRPO", "class means within 1e-6 at every logged "
            "step (β = 0.3, 1, 3; 20 tasks each)", f"{count(d1 <= 1e-6)} runs; largest difference {d1.max():.2g} "
            f"(mean rule, β = 1, against plain mean rule: largest {d1m.max():.2g})", (d1 <= 1e-6).all())

    p2 = plain.end_minority < 1e-3
    verdict("T31", "2", "plain GRPO: the minority mean ends below 1e-3", "≥ 19 of 20 tasks",
            f"{count(p2)} tasks; median final minority mean {plain.end_minority.median():.2g} "
            f"(largest {plain.end_minority.max():.2g})", p2.sum() >= 19)

    fr = lin[lin.label == "fixed-reverse"].groupby("beta")
    c3 = fr.end_protected.sum()
    verdict("T31", "3", "fixed reverse KL: no β is protected and learned in ≥ 16 of 20 tasks", "every β has c < 16",
            ", ".join(f"β = {b:g}: c = {int(c)}" for b, c in c3.items())
            + "; final majority / minority medians "
            + ", ".join(f"{b:g}: {g.end_majority.median():.3f} / {g.end_minority.median():.3g}" for b, g in fr),
            (c3 < 16).all())

    ff = lin[lin.label == "fixed-forward"].groupby("beta")
    n4 = ff.end_absorbed_1e3.apply(lambda a: int((a == 0).sum()))
    c4 = ff.end_protected.sum()
    verdict("T31", "4", "fixed forward KL: no minority prompt absorbed at 1e-3 at β ≥ 0.03, but protected and learned "
            "in ≤ 4 of 20 tasks at every β", "≥ 18 of 20 tasks with none absorbed, and c ≤ 4, at every β",
            ", ".join(f"β = {b:g}: none absorbed {int(n4[b])}/20, c = {int(c4[b])}" for b in n4.index),
            (n4 >= 18).all() and (c4 <= 4).all())

    rr = lin[lin.label == "reset-reverse"]
    grid = rr.groupby(["beta", "reset_every"]).agg(c=("end_protected", "sum"),
                                                   minority_final=("end_minority", "median"),
                                                   majority_final=("end_majority", "median"),
                                                   absorbed_1e2=("end_absorbed_1e2", "median"),
                                                   minority_lowest=("end_minority_lowest", "median"),
                                                   c_2000=("2k_protected", "sum"),
                                                   minority_ge_09=("end_minority", lambda x: int((x >= 0.9).sum())),
                                                   majority_ge_09=("end_majority", lambda x: int((x >= 0.9).sum())),
                                                   none_absorbed_1e2=("end_absorbed_1e2",
                                                                      lambda x: int((x == 0).sum()))).reset_index()
    grid["reset_every"] = grid.reset_every.astype(int)
    grid.to_csv(os.path.join(out, "t31_p5_grid.csv"), index=False)
    v5 = p5_verdict(grid.c)
    best = rerun_cell(grid)
    rerun = (f"rerun cell (largest c, then higher median final minority mean, then smaller reset_every): β = "
             f"{best.beta:g}, reset_every = {int(best.reset_every)} (c = {int(best.c)})")
    VERDICTS.append(dict(test="T31", prediction="5 (two-sided)", claim="reverse-KL resets: c(β, N) = tasks protected "
                         "and learned", rule="protects: some cell c ≥ 16; fails: every cell c ≤ 4; else partial",
                         result=f"largest c = {int(grid.c.max())} (β = {best.beta:g}, N = {int(best.reset_every)}); "
                                f"cells with c ≥ 16: {int((grid.c >= 16).sum())}, with c ≤ 4: {int((grid.c <= 4).sum())} "
                                f"of {len(grid)}", verdict=v5))
    cgrid = grid.pivot_table(index="beta", columns="reset_every", values="c").astype(int).reset_index()
    cgrid.columns = [c if isinstance(c, str) else f"N={c}" for c in cgrid.columns]

    b1 = rr[rr.beta == 1]
    n10, n1000 = key(b1[b1.reset_every == 10]), key(b1[b1.reset_every == 1000])
    up6 = n1000.end_minority > n10.end_minority
    verdict("T31", "6", "β = 1 reverse resets: final minority mean at reset_every = 1000 above that at 10",
            "≥ 16 of 20 tasks", f"{count(up6)} tasks; medians {n1000.end_minority.median():.3g} (N = 1000) and "
            f"{n10.end_minority.median():.3g} (N = 10)", up6.sum() >= 16)

    pk = lin[lin.label == "passk"]
    res7, ok7 = [], True
    for k in (2, 4):
        g = key(pk[pk.passk == k])
        n = int((g.end_minority >= 0.5).sum())
        ok7 &= n >= 18
        res7.append(f"k = {k}: minority ≥ 0.5 in {n}/20 (median {g.end_minority.median():.3f})")
    pinf = plain.maj99_step.fillna(np.inf)
    for k in (2, 4, 8):
        g = key(pk[pk.passk == k])
        later = g.maj99_step.fillna(np.inf)[plain.index] > pinf
        later &= np.isfinite(pinf)
        ok7 &= later.sum() == 20
        res7.append(f"k = {k}: majority reaches 0.99 later in {int(later.sum())}/20 (median step "
                    f"{fmt_step(med_inf(g.maj99_step))} against {fmt_step(med_inf(plain.maj99_step))})")
    verdict("T31", "7", "pass@k (unscaled): k ∈ {2, 4} final minority mean ≥ 0.5; every k: majority reaches 0.99 later "
            "than plain GRPO", "≥ 18 of 20 tasks for k = 2, 4; 20 of 20 tasks for every k", "; ".join(res7), ok7)

    rf = lin[lin.label == "reset-forward"].groupby(["beta", "reset_every"]).agg(
        c=("end_protected", "sum"), minority_final=("end_minority", "median"),
        majority_final=("end_majority", "median"), absorbed_1e3=("end_absorbed_1e3", "median"),
        absorbed_1e2=("end_absorbed_1e2", "median"),
        minority_ge_09=("end_minority", lambda x: int((x >= 0.9).sum()))).reset_index()
    rf["reset_every"] = rf.reset_every.astype(int)
    rf.to_csv(os.path.join(out, "t31_reset_forward.csv"), index=False)
    mean_rows = lin[lin.label.isin(["plain-mean", "reset-reverse-mean"])]
    reported("T31", "8", "forward resets: c(β, N)", "; ".join(f"β = {r.beta:g}, N = {int(r.reset_every)}: c = {int(r.c)}"
                                                           for r in rf.itertuples()))
    mr = mean_rows.groupby("arm").agg(c=("end_protected", "sum"), minority=("end_minority", "median")).reindex(
        [a for a in order if a in set(mean_rows.arm)])
    reported("T31", "8", "mean-rule reverse resets (β = 1): c and median final minority mean",
             "; ".join(f"{a}: c = {int(r.c)}, minority {r.minority:.3g}" for a, r in mr.iterrows()))

    # Robustness rows
    rob = runs[runs.tag != "linear-exact"]
    rob_order = [a for a in order if a in set(rob.arm)]
    rob_tabs = {tag: yard_table(rob[rob.tag == tag], "end_", rob_order) for tag in ("mlp-exact", "linear-sampled")}
    for tag, t in rob_tabs.items():
        t.to_csv(os.path.join(out, f"t31_robust_{tag}.csv"), index=False)
    reported("T31", "8", "robustness rows: protected and learned (of 20) per arm",
             "; ".join(f"{tag}: " + ", ".join(f"{r['arm']} {r['protected and learned']}" for _, r in t.iterrows())
                       for tag, t in rob_tabs.items()))

    lines += ["Every arm on the same yardstick. Medians over the 20 tasks, ranges in brackets for the headline "
              "columns. pass@k from the per-prompt success in the snapshot; log10 median 1/p over the minority "
              "prompts; absorbed shares over the minority prompts (at the end: tail max over the last 20% of steps; "
              "at step 2,000: the snapshot value); 'step majority ≥ 0.99': median first logged step (never = ∞) and "
              "the number of tasks that reach it by that time.", "", "**At the end (step 100,000, flow time 1e4):**", "",
              markdown(end_tab), "", "**At step 2,000 (flow time 200):**", "", markdown(k2_tab), "",
              "**Prediction 5: c(β, N), tasks (of 20) protected and learned, reverse-KL resets:**", "", markdown(cgrid),
              "", f"Verdict: **{v5}**. " + (rerun + "." if v5 == "protects" else f"(For the record, the {rerun}.)"), "",
              "Per cell (medians over tasks; the last three columns count the tasks meeting each part of the rule):",
              "", markdown(grid), "",
              "**Forward resets** (reported):", "", markdown(rf), "",
              "**Robustness, MLP with the exact update** (end of training):", "", markdown(rob_tabs["mlp-exact"]), "",
              "**Robustness, linear with sampled updates (B = 32)** (end of training):", "",
              markdown(rob_tabs["linear-sampled"]), ""]
    runs.to_csv(os.path.join(out, "t31_runs.csv"), index=False)
    return lines, dict(grid=grid, v5=v5, best=best, plain=plain, lin=lin, traj=traj)


# ---------------------------------------------------------------- T32

def t32(cache, out):
    f = pd.read_csv(os.path.join(cache, "t32_flows.csv"))
    f.to_csv(os.path.join(out, "t32_tasks.csv"), index=False)
    w = {r: g.set_index(["rho", "m", "task"]) for r, g in f.groupby("rule")}
    g0, g2, g4 = w["grpo"], w["rloo_k2"], w["rloo_k4"]
    lines = ["## T32: pass@k over long times (exact flow to 1e14)", "",
             f"Status: {({r: dict(g.status.value_counts()) for r, g in f.groupby('rule')})}; solvers: "
             f"{({r: dict(g.solver.value_counts()) for r, g in f.groupby('rule')})}. The GRPO baseline is the "
             "recomputed flow (`t32_baseline.md`, written before any pass@k flow was read).", ""]
    r2, r4 = int(g2.recovered.sum()), int(g4.recovered.sum())
    verdict("T32", "1", "tasks recovered at 1e14 (minority class mean > 0.5)", "≥ 85 of 100 (pass@2), ≥ 95 of 100 "
            "(pass@4)", f"pass@2 {r2}/100, pass@4 {r4}/100 (GRPO {int(g0.recovered.sum())}/100)", r2 >= 85 and r4 >= 95)
    hi = g0.gamma_min >= 1
    verdict("T32", "2", "tasks with γ_min ≥ 1: recovered under pass@2", "≥ 24 of them",
            f"{int(g2.recovered[hi].sum())} of {int(hi.sum())} (GRPO: {int(g0.recovered[hi].sum())} of {int(hi.sum())})",
            g2.recovered[hi].sum() >= 24)
    res3, ok3 = [], True
    for k, g in ((2, g2), (4, g4)):
        up = g.log10_lowest_minority > g0.log10_lowest_minority
        ok3 &= up.sum() >= 95
        res3.append(f"pass@{k}: {int(up.sum())}/100 (median lowest 10^{g.log10_lowest_minority.median():.2f} against "
                    f"10^{g0.log10_lowest_minority.median():.2f})")
    verdict("T32", "3", "the lowest minority class mean is higher than under GRPO", "≥ 95 of 100 tasks, each k",
            "; ".join(res3), ok3)
    both = g2.recovered & g0.recovered
    earlier = g2.log10_recovery_time[both] < g0.log10_recovery_time[both]
    i2 = np.searchsorted(np.log10(T25_TAUS), g2.log10_recovery_time[both] - 1e-12)   # first grid point at or after
    i0 = np.searchsorted(np.log10(T25_TAUS), g0.log10_recovery_time[both] - 1e-12)
    verdict("T32", "4", "tasks recovering under both pass@2 and GRPO: pass@2 recovers earlier", "≥ 90%",
            f"{int(earlier.sum())} of {int(both.sum())} ({earlier.mean():.1%}); median recovery time 10^"
            f"{g2.log10_recovery_time[both].median():.2f} against 10^{g0.log10_recovery_time[both].median():.2f}. "
            f"Read on T25's grid points alone (no interpolation): earlier in {int((i2 < i0).sum())}, same grid point "
            f"in {int((i2 == i0).sum())}, later in {int((i2 > i0).sum())}", earlier.mean() >= 0.9)
    tabs = []
    for k, g in ((1, g0), (2, g2), (4, g4)):
        name = f"{'GRPO' if k == 1 else f'pass@{k}'} (k = {k})"
        t = pd.DataFrame([{name: lab, "recovered": int((mask & g.recovered).sum()),
                           "not recovered": int((mask & ~g.recovered).sum())}
                          for lab, mask in ((f"γ_min < {k}", g.gamma_min < k), (f"γ_min ≥ {k}", g.gamma_min >= k))])
        tabs.append(t)
    reported("T32", "5", "2 × 2 tables (γ_min < k against recovered)",
             "; ".join(f"k = {k}: " + ", ".join(f"{r.iloc[0]}: {r.recovered} recovered, {r['not recovered']} not"
                                                 for _, r in t.iterrows()) for k, t in zip((1, 2, 4), tabs)))
    cost = f.groupby("rule")[["majority_at_1e+02", "majority_at_1e+03", "majority_at_1e+04"]].agg(fmt_range)
    cost_m = f.groupby(["rule", "m"])[["majority_at_1e+02", "majority_at_1e+03", "majority_at_1e+04"]].median()
    reported("T32", "5", "majority class mean at flow times 1e2, 1e3, 1e4 (median over 100 tasks)",
             "; ".join(f"{r}: " + ", ".join(f"{v.split(' ')[0]}" for v in row.values) for r, row in cost.iterrows()))
    cell = f.groupby(["rule", "rho", "m"]).agg(recovered=("recovered", "sum"),
                                               lowest=("log10_lowest_minority", "median"),
                                               recovery=("log10_recovery_time", lambda x: med_inf(x)),
                                               minority_final=("minority_final", "median")).reset_index()
    rec = cell.pivot_table(index=["rho", "m"], columns="rule", values="recovered").reset_index()
    low = cell.pivot_table(index=["rho", "m"], columns="rule", values="lowest").reset_index()
    lines += ["**2 × 2 tables** (γ_min from `analysis/v3m/runs.csv`):", ""] + sum(([markdown(t), ""] for t in tabs), [])
    lines += ["**Majority class mean at flow times 1e2, 1e3, 1e4** (median and range over 100 tasks; linear "
              "interpolation in log10 time on T25's grid):", "", markdown(cost.reset_index()), "",
              "By majority share (medians):", "", markdown(cost_m.reset_index()), "",
              "**Tasks recovered (of 10) by cell:**", "", markdown(rec), "",
              "**Median log10 of the lowest minority class mean by cell:**", "", markdown(low), ""]
    return lines


# ---------------------------------------------------------------- T33

def t33a(cache, out, t31x):
    runs = pd.read_csv(os.path.join(cache, "t33a.csv"))
    traj = load_traj(os.path.join(cache, "traj_t33a.npz"))
    lines = ["## T33(a): pass@k training measured on pass@1, existing sampled runs (T16 in v3f, A3 in v3o)", "",
             f"Runs: {runs.source.value_counts().to_dict()}; start: {runs.init.value_counts().to_dict()}.", ""]
    pk = runs[runs.family.isin(["passk", "rloo_k"]) & runs.k.isin([2, 4, 8])]
    ok1 = pk.majority_largest_fall <= 0.05
    pk = pk.assign(max_minus_final=[traj[n][1].max() - traj[n][1][-1] for n in pk.name])
    by = pk.groupby(["source", "rule"]).agg(
        runs=("majority_largest_fall", "size"), within_005=("majority_largest_fall", lambda x: int((x <= 0.05).sum())),
        median_fall=("majority_largest_fall", "median"), largest_fall=("majority_largest_fall", "max"),
        median_max_minus_final=("max_minus_final", "median")).reset_index()
    by_eta = pk.groupby(["G", "eta"]).agg(
        runs=("majority_largest_fall", "size"), within_005=("majority_largest_fall", lambda x: int((x <= 0.05).sum())),
        median_fall=("majority_largest_fall", "median"), median_max_minus_final=("max_minus_final", "median"),
        majority_final=("majority_final", "median")).reset_index()
    grpo_fall = runs[runs.rule == "grpo"].majority_largest_fall
    lin = t31x["lin"]
    ex = lin[lin.label == "passk"].groupby("passk").majority_largest_fall
    exm = lin[lin.label == "passk-matched"].groupby("passk").majority_largest_fall
    ex_ok = ex.apply(lambda x: int((x <= 0.02).sum()))
    plain_fall = lin[lin.arm == "plain"].majority_largest_fall
    verdict("T33", "1", "from a uniform start pass@k slows the majority and does not reverse it: the majority class "
            "mean never falls more than 0.05 below its running maximum (sampled runs); 0.02 in T31's exact pass@k arms",
            "≥ 90% of the sampled pass@k runs (k = 2, 4, 8, both forms, pooled); ≥ 18 of 20 tasks per k (exact)",
            f"sampled: {count(ok1)} ({ok1.mean():.1%}), largest fall {pk.majority_largest_fall.max():.3f}; exact: "
            + ", ".join(f"k = {int(k)}: {int(n)}/20 (largest {ex.max()[k]:.2g})" for k, n in ex_ok.items()),
            ok1.mean() >= 0.9 and (ex_ok >= 18).all())

    grpo = runs[runs.rule == "grpo"].set_index(["source", "task_seed", "G", "eta"])
    pairs = []
    for r in pk.itertuples():
        g = grpo.loc[(r.source, r.task_seed, r.G, r.eta)]
        a, b = traj[r.name], traj[g["name"]]
        assert np.array_equal(a[0], b[0])
        steps = a[0]
        hit = first_at_or_above(steps, b[1], 0.99)
        row = dict(source=r.source, rule=r.rule, family=r.family, k=r.k, G=r.G, eta=r.eta, task_seed=r.task_seed,
                   grpo_maj99_step=np.nan if hit is None else hit)
        if hit is not None:
            i = int(np.nonzero(steps == hit)[0][0])
            row.update(passk_pass1=a[3][i], grpo_pass1=b[3][i], below=bool(a[3][i] < b[3][i]))
        st = stays_above_from(steps, a[3], b[3])
        row["stays_above_from"] = np.nan if st is None else st
        pairs.append(row)
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(os.path.join(out, "t33a_pairs.csv"), index=False)
    ok2, res2 = True, []
    for k in (2, 4, 8):
        p = pairs[(pairs.k == k) & pairs.grpo_maj99_step.notna()]
        frac_below = p.below.astype(bool).mean()
        n_all = int((pairs.k == k).sum())
        res2.append(f"k = {k}: {int(p.below.sum())}/{len(p)} ({frac_below:.1%}; {n_all - len(p)} pairs whose GRPO run "
                    "never reaches 0.99)")
        if k in (4, 8):
            ok2 &= frac_below >= 0.9
    verdict("T33", "2", "at the first logged step where GRPO's majority mean is ≥ 0.99 (same task, G, η), mean pass@1 "
            "under pass@k is below GRPO's", "≥ 90% of pairs, for k = 4 and k = 8 (k = 2 reported)",
            "; ".join(res2), ok2)
    c3 = pairs.groupby(["k", "family", "G", "eta"]).agg(
        pairs=("task_seed", "size"), with_step=("stays_above_from", lambda x: int(x.notna().sum())),
        median_step=("stays_above_from", lambda x: med_inf(x))).reset_index()
    c3["median_step"] = c3.median_step.map(fmt_step)
    c3.to_csv(os.path.join(out, "t33a_p3_cells.csv"), index=False)
    reported("T33", "3", "first step from which mean pass@1 under pass@k stays above GRPO's (median over tasks per k, "
             "G, η; never = ∞)", f"pairs with such a step: {int(pairs.stays_above_from.notna().sum())} of "
             f"{len(pairs)}; per k: " + ", ".join(f"k = {k}: {int(g.stays_above_from.notna().sum())}/{len(g)}"
                                                    for k, g in pairs.groupby("k")) + " (table in FINDINGS)")
    lines += ["**Prediction 1**, largest fall of the majority below its running maximum, by source and rule:", "",
              markdown(by), "", "By (G, η), pooled over rules (`max_minus_final`: the majority's largest logged "
              "value minus its final value):", "", markdown(by_eta), "",
              f"GRPO runs for comparison: largest fall {grpo_fall.max():.2g} over {len(grpo_fall)} runs.", "",
              "T31's exact pass@k arms (100,000 steps): largest fall over the 20 tasks, unscaled "
              + ", ".join(f"k = {int(k)}: {v:.2g}" for k, v in ex.max().items()) + "; matched step "
              + ", ".join(f"k = {int(k)}: {v:.2g}" for k, v in exm.max().items())
              + f"; plain GRPO: {plain_fall.max():.2g}.", "",
              "**Prediction 3**, first logged step from which mean pass@1 under pass@k stays above GRPO's:", "",
              markdown(c3), ""]
    runs.to_csv(os.path.join(out, "t33a_runs.csv"), index=False)
    return lines


def t33b(cache, out, t31x):
    sw = pd.read_csv(os.path.join(cache, "v3_t33.csv"))
    traj31 = t31x["traj"]
    plain = t31x["plain"]
    ctrl = []
    for t in SEEDS:
        x = traj31[plain.loc[t, "name"]]
        for s in (100, 1000):
            i = int(np.nonzero(x[0] == s)[0][0])
            ctrl.append(dict(task_seed=t, switch_step=s, majority_fall=max(0.0, x[1][i] - x[1][i + 1:].min()),
                             minority_final=x[2][-1], pass1_lowest_after=x[3][i + 1:].min(), pass1_at=x[3][i]))
    ctrl = pd.DataFrame(ctrl)
    lines = ["## T33(b): switching to pass@k after a collapse (the T31 cell; control: T31's plain arm)", ""]
    g = lambda k, s: sw[(sw.k == k) & (sw.switch_step == s)].set_index("task_seed")  # noqa: E731
    a = g(8, 100)
    verdict("T33", "4", "k = 8, switch at 100: the majority class mean falls at least 0.05 below its value at the "
            "switch at some later logged step", "≥ 16 of 20 tasks", f"{count(a.majority_fell_005)} tasks; median "
            f"largest fall {a.majority_fall.median():.3g} (plain GRPO after step 100: "
            f"{ctrl[ctrl.switch_step == 100].majority_fall.median():.3g})", a.majority_fell_005.sum() >= 16)
    med = {s: [g(k, s).majority_fall.median() for k in (2, 4, 8)] for s in (100, 1000)}
    m1 = med[100]
    verdict("T33", "5", "the median largest fall of the majority is ordered k = 2 < k = 4 < k = 8 (switch at 100)",
            "strict order", f"switch 100: {', '.join(f'{v:.3g}' for v in m1)}; switch 1,000 (reported): "
            f"{', '.join(f'{v:.3g}' for v in med[1000])}", m1[0] < m1[1] < m1[2])
    verdict("T33", "6", "k = 8, switch at 100: mean pass@1 below its value at the switch while mean pass@8 is above its "
            "value at the switch, at some logged step", "≥ 16 of 20 tasks",
            f"{count(a.pass1_down_pass8_up)} tasks (checked at steps 2,000 and 100,000, the logged steps with "
            f"per-prompt success); mean pass@1 ever below its switch value at a logged step: "
            f"{count(a.pass1_lowest_after < a.pass1_at_switch)}", a.pass1_down_pass8_up.sum() >= 16)
    b = g(4, 100)
    up = b.end_minority > 0.5
    verdict("T33", "7", "rescue: k = 4, switch at 100, minority class mean back above 0.5 by the end", "≥ 16 of 20 tasks",
            f"{count(up)} tasks; median final minority mean {b.end_minority.median():.3g} (plain GRPO: "
            f"{plain.end_minority.median():.2g}, above 0.5 in {int((plain.end_minority > 0.5).sum())}/20)",
            up.sum() >= 16)
    cells = sw.groupby(["switch_step", "k"]).agg(
        majority_at_switch=("majority_at_switch", "median"), minority_at_switch=("minority_at_switch", "median"),
        majority_fall=("majority_fall", "median"), fell_005=("majority_fell_005", "sum"),
        pass1_down_pass8_up=("pass1_down_pass8_up", "sum"), minority_final=("end_minority", "median"),
        minority_above_05=("end_minority", lambda x: int((x > 0.5).sum())), majority_final=("end_majority", "median"),
        pass1_final=("end_pass1", "median"), pass8_final=("end_pass8", "median"),
        protected=("end_protected", "sum")).reset_index()
    cc = ctrl.groupby("switch_step").agg(majority_fall=("majority_fall", "median"),
                                         minority_final=("minority_final", "median")).reset_index()
    reported("T33", "7 (reported)", "final minority class mean above 0.5, other k and switch at 1,000",
             "; ".join(f"switch {int(r.switch_step)}, k = {int(r.k)}: {int(r.minority_above_05)}/20 (median "
                       f"{r.minority_final:.3g})" for r in cells.itertuples()))
    cells.to_csv(os.path.join(out, "t33b_cells.csv"), index=False)
    sw.to_csv(os.path.join(out, "t33b_switch_runs.csv"), index=False)
    lines += ["Per (switch step, k), medians over 20 tasks or counts of tasks:", "", markdown(cells), "",
              "Control, T31's plain arm (no switch): largest fall of the majority after step 100 / 1,000 and final "
              "minority mean (medians):", "", markdown(cc), ""]
    return lines


# ---------------------------------------------------------------- report

CHOICES = [
    "Minority class mean: the mean over the prompts of classes 1..K−1 pooled, as the simulator logs it "
    "(`history.class_mean[1]`; analyze_v3f, analyze_v3g, check_permanence). For block tasks the same.",
    "Absorbed share: absorbed minority prompts ÷ minority prompts, absorbed = tail max over the last 20% of steps below "
    "τ among prompts starting at or above 10τ (analyze_v3.persistent, analyze_v3g.t17_row). At m = 1/3 the "
    "'minority' is still classes 1 and 2.",
    "T30 prediction 2: 'every class mean' is each of the three classes' mean success in the final snapshot "
    "(step 20,000), over the 120 exact runs at s = 0 (both η̃, all three m).",
    "T30 prediction 4: read over every (update, η̃) cell of the linear grid at m = 1/3, the strictest reading.",
    "T30 prediction 5: required at each of the two η̃ values.",
    "T30 prediction 6 and R(s): R = −Δ_A/Δ_B from `check_odds_law.predict`, the first-order step-0 change of the pooled "
    "minority mean (the odds law; it applies unchanged to block features). The first-step sign is read from the exact "
    "flow at its first time point τ = 1e-3; the discrete runs' first logged step (step 100) is given as a check.",
    "T30 γ_min(s): T25's definition (`check_permanence`): the smallest projection of a minority prompt's features on the "
    "majority's l2 max-margin direction (`check_late_time.l2_max_margin`), computed from the task's block features. "
    "The theory behind it (the scalar late model) was derived for the class-label toy; it is reported, not tested.",
    "T31 at step 2,000: the absorbed shares use the per-prompt snapshot at step 2,000 (no tail window exists there), "
    "and 'protected and learned' at step 2,000 uses those. At the end they use the tail max.",
    "T31 'first logged step at which the majority reaches 0.99': over the whole run; in the step-2,000 table only "
    "steps ≤ 2,000 count. Medians treat 'never' as infinite.",
    "T31 prediction 7: a task where the pass@k arm never reaches 0.99 counts as 'later' if plain GRPO reaches it.",
    "T31 prediction 1 compares the logged class means (majority and pooled minority) at all 1,001 logged steps.",
    "T32 recovered: minority class mean > 0.5 at 1e14 as registered (T25's table used ≥ 0.5; no task is at 0.5). "
    "Lowest minority class mean: the smallest value on T25's 80-point grid, compared in log space.",
    "T32 prediction 4: recovery time is the flow time from which the minority class mean stays above 0.5 to 1e14 (the "
    "last upward crossing), interpolated linearly in log10 time between grid points; 'earlier' is strict.",
    "T32 majority class mean at 1e2, 1e3, 1e4: linear interpolation in log10 time on T25's grid.",
    "T33(a) prediction 1: the 0.9 share is over all sampled pass@k runs pooled (both sources, both forms, k = 2, 4, 8). "
    "The exact part uses T31's three unscaled pass@k arms (the registered ones); the matched-step arms are reported.",
    "T33(a) prediction 2: pairs whose GRPO run never reaches 0.99 are left out (their number is given). Mean pass@1 "
    "at a logged step is the logged mean success over all prompts.",
    "T33(b) prediction 5: read at the switch at step 100 (the step of predictions 4, 6 and 7); step 1,000 reported.",
    "T33(b) prediction 6: per-prompt success after the switch is stored only at steps 2,000 and 100,000, so pass@8 "
    "(and the conjunction) is checked at those two steps.",
]


def analyze(raw, out):
    cache = os.path.join(raw, "cache")
    os.makedirs(out, exist_ok=True)
    VERDICTS.clear()
    counts = {h: len(pd.read_csv(os.path.join(cache, f"{h}.csv"))) for h in GRIDS}
    assert counts == GRIDS, counts
    base = open(os.path.join(out, "t32_baseline.md")).read().split("\n")
    l30 = t30(cache, out)
    l31, x31 = t31(cache, out)
    l32 = t32(cache, out)
    l33a = t33a(cache, out, x31)
    l33b = t33b(cache, out, x31)
    res = pd.DataFrame(VERDICTS)
    res.to_csv(os.path.join(out, "results.csv"), index=False)
    lines = ["# Findings: v3r, T30–T33 (PREREG_v3d addendum 11 with Amendment 1)", "",
             f"Generated by `experiments/analyze_v3r.py` from {sum(counts.values()):,} runs "
             f"({', '.join(f'{h} {n:,}' for h, n in counts.items())}), 360 T30 flows and 300 T32 flows, plus 840 "
             "existing sampled runs for T33(a). Task seeds 30–49 unless stated.", "",
             "## Verdicts", "", markdown(res[["test", "prediction", "claim", "rule", "result", "verdict"]]), "",
             "## T32 step 1 (done first): the recomputed GRPO baseline", ""] + base[2:] + [""]
    lines += l30 + l31 + l32 + l33a + l33b
    lines += ["## Choices not fixed by the pre-registration", ""] + [f"{i}. {c}" for i, c in enumerate(CHOICES, 1)] + [""]
    with open(os.path.join(out, "FINDINGS.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(res[["test", "prediction", "result", "verdict"]].to_string(index=False))


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
