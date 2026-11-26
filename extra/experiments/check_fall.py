"""
Check the reduced ODE of the fall (extra/analysis/theory/fall.md, extra/experiments/fall_ode.py) against the simulator runs.

usage:
    python extra/experiments/check_fall.py predict [--procs 4]   # model outputs per run; reads only config and p_init
    python extra/experiments/check_fall.py compare               # measured values, then the registered comparisons

`predict` writes extra/analysis/theory/fall_predictions.csv (run level) and the per-prompt test-particle predictions (and
class labels) to ~/rlvr-data/analysis/fall_tp.npz. `compare` writes extra/analysis/theory/fall_runs.csv and fall_compare.md.
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
from scipy.stats import spearmanr

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # repo root
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]  # extra/

from experiments.fall_ode import ABSORB, DEFAULTS, features, predict  # noqa: E402

RAW = os.path.expanduser("~/rlvr-data")
BATCHES = {"v3e": ("v3_t15",), "v3g": ("v3_t19",), "v3j": ("v3_t22",), "v3k": ("v3_t23",), "v3l": ("v3_t24",)}
OUT = "extra/analysis/theory"
TAG = ""                 # "_heldout" for the held-out test (PREREG_fall_heldout.md): separate output files
LINEAR_SGD = False       # held-out test: only linear, exact-update, SGD, beta = 0 runs (the model's scope)


def out_path(name, ext):
    return os.path.join(OUT, f"{name}{TAG}.{ext}")


def tp_path():
    return os.path.join(RAW, "analysis", f"fall_tp{TAG}.npz")
CELL = ["batch", "arch", "update", "rule", "G", "K", "rho", "m", "eta", "steps", "init", "p0"]


def paths():
    out = []
    for batch, hyps in BATCHES.items():
        for hyp in hyps:
            for p in sorted(glob.glob(os.path.join(RAW, batch, "*", "results", hyp, "*.json.gz"))):
                if LINEAR_SGD:
                    c = load(p)["config"]
                    if not (c["arch"] == "linear" and c["update"] == "expected_full" and c["optimizer"] == "sgd"
                            and not c.get("beta")):
                        continue
                out.append((f"{batch}-{hyp}" if len(hyps) > 1 else batch, p))
    return out


def small(cfg):
    return {k: v for k, v in cfg.items() if k in DEFAULTS or k in ("task_seed", "sft_lr")}


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


TASK = ("K", "rho", "majority", "used_classes", "task_seed", "arch", "init", "p0", "seed")


def predict_one(item):
    batch, path = item
    d = load(path)
    cfg = d["config"]
    out = predict(small(cfg))
    p0 = np.asarray(d["p_init"])
    cls = features(small(cfg))[1]               # memoised: the task was built by predict()
    alive = p0[cls != 0] >= 1e-2
    tp_min, tp_tail = out.pop("tp_log10_min"), out.pop("tp_log10_tail_max")
    out["tp_absorbed_share"] = float((tp_tail[alive] < np.log10(ABSORB)).mean()) if alive.any() else np.nan
    out.pop("K"), out.pop("U")
    row = dict(batch=batch, name=cfg["name"], arch=cfg["arch"], update=cfg["update"], rule=cfg["rule"], G=cfg["G"],
               K=cfg["K"], used=cfg.get("used_classes") or cfg["K"], rho=cfg["rho"], m=cfg["majority"], eta=cfg["eta"],
               steps=cfg["steps"], init=cfg["init"], p0=cfg.get("p0") if cfg["init"] != "uniform" else "uniform",
               task_seed=cfg["task_seed"], **out)
    key = batch + "__" + cfg["name"].replace("/", "__")
    return row, {key: tp_min.astype(np.float32), key + "__cls": cls.astype(np.int8)}


def predict_group(group):
    """All runs that share one task (so the task is built once, in memory)."""
    return [predict_one(item) for item in group]


def run_predict(procs):
    groups = {}
    for batch, p in paths():
        cfg = load(p)["config"]
        groups.setdefault(json.dumps({k: cfg.get(k) for k in TASK}, sort_keys=True), []).append((batch, p))
    print(len(groups), "tasks", sum(map(len, groups.values())), "runs", flush=True)
    res = []
    with Pool(procs) as pool:
        for i, out in enumerate(pool.imap_unordered(predict_group, list(groups.values()))):
            res += out
            if i % 50 == 0:
                print("tasks done", i, "of", len(groups), flush=True)
    rows = pd.DataFrame([r for r, _ in res]).sort_values(["batch", "name"])
    rows.to_csv(out_path("fall_predictions", "csv"), index=False)
    arrays = {}
    for _, a in res:
        arrays.update(a)
    np.savez_compressed(tp_path(), **arrays)
    print(len(rows), "runs ->", out_path("fall_predictions", "csv"))


# ---------------------------------------------------------------- measured values

def measure_one(item):
    batch, path, cls = item
    d = load(path)
    cfg = d["config"]
    mino = cls != 0
    p0, tail, pmin = (np.asarray(d[k], dtype=float) for k in ("p_init", "tail_max", "min_p"))
    alive = p0[mino] >= 1e-2
    absorbed = tail[mino][alive] < ABSORB
    means = np.array([h["class_mean"] for h in d["history"]])
    lmin = np.log10(np.clip(pmin[mino], 1e-320, None))
    return dict(batch=batch, name=cfg["name"], meas_log10_min_median=float(np.median(lmin)),
                meas_log10_classmean_min=float(np.log10(max(means[:, 1].min(), 1e-320))),
                meas_absorbed_share=float(absorbed.mean()) if alive.any() else np.nan,
                meas_log10_end_median=float(np.median(np.log10(np.clip(
                    np.asarray(d["snapshots"][str(cfg["steps"])], dtype=float)[mino], 1e-320, None))))), \
        (batch + "__" + cfg["name"].replace("/", "__"), lmin.astype(np.float32))


def spear(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b).correlation)


def scores(g, model):
    """The registered statistics for one group of runs and one model (sym, cls or tp)."""
    pred = g[f"{model}_log10_min"] if model != "tp" else g["tp_log10_min_median"]
    meas = g.meas_log10_min_median
    pa = g[f"{model}_absorbed"] if model != "tp" else g.tp_absorbed_share >= 0.5
    ma = g.meas_absorbed_share >= 0.5
    within = [spear(c[pred.name].to_numpy(), c.meas_log10_min_median.to_numpy())
              for _, c in g.groupby([k for k in CELL if k in g.columns], dropna=False) if len(c) >= 5]
    within = [w for w in within if not np.isnan(w)]
    return dict(runs=len(g), spearman=spear(pred.to_numpy(), meas.to_numpy()),
                within_cell_spearman=float(np.median(within)) if within else np.nan, cells=len(within),
                median_log_ratio=float(np.median(pred - meas)), mad_log_ratio=float(np.median(np.abs(pred - meas))),
                accuracy=float((pa.astype(bool) == ma).mean()), measured_absorbed=int(ma.sum()),
                predicted_absorbed=int(pa.astype(bool).sum()),
                false_pos=int((pa.astype(bool) & ~ma).sum()), false_neg=int((~pa.astype(bool) & ma).sum()))


def threshold(g, col):
    """Smallest grid m whose median over tasks of `col` is above 0.5 (v3e's absorption_from)."""
    med = g.groupby("m")[col].median().sort_index()
    above = med[med > 0.5]
    return float(above.index.min()) if len(above) else np.nan


def run_compare(procs):
    pred = pd.read_csv(out_path("fall_predictions", "csv"))
    tp = np.load(tp_path())
    items = []
    for batch, p in paths():
        name = load(p)["config"]["name"]
        items.append((batch, p, tp[batch + "__" + name.replace("/", "__") + "__cls"]))
    with Pool(procs) as pool:
        res = pool.map(measure_one, items, chunksize=8)
    meas = pd.DataFrame([r for r, _ in res])
    runs = pred.merge(meas, on=["batch", "name"])
    for model in ("sym", "cls"):
        runs[f"{model}_absorbed"] = runs[f"{model}_absorbed"].astype(bool)
    runs["meas_absorbed"] = runs.meas_absorbed_share >= 0.5
    runs.to_csv(out_path("fall_runs", "csv"), index=False)

    # per-prompt: test-particle minimum against the measured per-prompt minimum, within each run
    per_prompt = {k: v for _, (k, v) in res}
    runs["tp_prompt_spearman"] = [spear(tp[f"{b}__{n.replace('/', '__')}"], per_prompt[f"{b}__{n.replace('/', '__')}"])
                                  for b, n in zip(runs.batch, runs.name)]
    runs.to_csv(out_path("fall_runs", "csv"), index=False)

    lines = ["# Fall ODE against the simulator (generated by `extra/experiments/check_fall.py compare`)", ""]
    for scope, sel in (("linear, exact update (in scope)", (runs.arch == "linear") & (runs["update"] == "expected_full")),
                       ("linear, sampled update", (runs.arch == "linear") & (runs["update"] == "sampled")),
                       ("MLP, exact update (out of scope)", (runs.arch == "mlp") & (runs["update"] == "expected_full")),
                       ("MLP, sampled update (out of scope)", (runs.arch == "mlp") & (runs["update"] == "sampled"))):
        rows = []
        for batch, g in runs[sel].groupby("batch"):
            for model in ("sym", "cls", "tp"):
                rows.append(dict(batch=batch, model=model, **scores(g, model)))
        if not rows:
            continue
        lines += [f"## {scope}", "", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3g"), ""]
        fell = runs[sel & (runs.meas_log10_min_median < -2)]
        pp = fell.groupby("batch").tp_prompt_spearman.median()
        lines += ["Per-prompt test particles, runs whose median minority prompt fell below 1e-2: median within-run "
                  "Spearman of predicted against measured per-prompt minimum: "
                  + ", ".join(f"{b} {v:.2f} ({(fell.batch == b).sum()} runs)" for b, v in pp.items()), ""]

    # v3e threshold in m
    rows = []
    for (K, arch, rho), g in runs[runs.batch == "v3e"].groupby(["K", "arch", "rho"]):
        rows.append(dict(K=K, arch=arch, rho=rho, measured=threshold(g, "meas_absorbed_share"),
                         sym=threshold(g.assign(x=g.sym_absorbed.astype(float)), "x"),
                         cls=threshold(g.assign(x=g.cls_absorbed.astype(float)), "x"),
                         tp=threshold(g, "tp_absorbed_share")))
    th = pd.DataFrame(rows, columns=["K", "arch", "rho", "measured", "sym", "cls", "tp"])
    lines += ["## v3e: absorption threshold in m (smallest grid m with median absorption above 0.5)", "",
              th.to_markdown(index=False, floatfmt=".3g"), ""]
    if len(th):
        th.to_csv(out_path("fall_thresholds", "csv"), index=False)
    with open(out_path("fall_compare", "md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["predict", "compare"])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--batches", help="e.g. v3o=v3_a1,v3_a2,v3_a5 (default: the in-sample batches)")
    ap.add_argument("--tag", default="", help="output suffix, e.g. heldout")
    ap.add_argument("--linear-sgd", action="store_true", help="only linear, exact-update, SGD, beta = 0 runs")
    a = ap.parse_args()
    if a.batches:
        BATCHES = {b: tuple(h.split(",")) for b, h in (x.split("=") for x in a.batches.split(";"))}
    TAG = f"_{a.tag}" if a.tag else ""
    LINEAR_SGD = a.linear_sgd
    run_predict(a.procs) if a.stage == "predict" else run_compare(a.procs)
