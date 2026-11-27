"""
Many exact flows (extra/experiments/exact_flow.py) in parallel, one .npz per task (PREREG_v3d addendum 11, T30 and T32).

usage:
    python extra/experiments/flow_batch.py t30 [--out DIR] [--workers 8] [--limit N] [--seeds 30-49]
    python extra/experiments/flow_batch.py t32 [--out DIR] [--workers 8] [--limit N] [--rules grpo,rloo_k2,rloo_k4]

t30  block tasks (s in {0, 0.5, 1, 2, 4, 8} x m in {1/3, 0.7, 0.9} x task seeds 30-49; the tasks of grids.v3_t30),
     linear policy, GRPO G = 8, to flow time 1e8 on a grid of 20 points per decade. Saved per task: taus, logp
     (n_taus x n, float32), cls, and max_drop: per prompt the largest fall of success (not log-success) below
     its running maximum over the grid, the start (p = 1/K) included, computed in float64.
t32  the 100 tasks of T25 (rho in {0.3, 0.5, 0.7, 0.9, 0.95}, m in {0.7, 0.9}, task seeds 0-9; configs as in
     check_permanence.py on analysis-v3) under GRPO (T25's rule, the recomputed baseline), rloo_k with k = 2 and
     k = 4 (G = 8), to 1e14 on T25's time grid (np.logspace(-1, 14, 80)). Saved in T25's format (taus, logp
     float32, cls), one directory per rule, with the same file names as ~/rlvr-data/v3m.

Every flow uses stable_tail=True and T25's tolerances (flow's defaults). Each task tries LSODA with a time limit
(--minutes, default 30), then Radau, then BDF, each with the same limit; the file records the solver that
finished ("solver"), the wall time of each attempt ("attempts", JSON) and "status" ("ok" or "failed"; a failed
task has no logp). Tasks whose output exists are skipped.
"""

import argparse
import json
import os
import signal
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (this script is in extra/experiments/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path[1:1] = [os.path.join(ROOT, "extra"), os.path.join(ROOT, "extra", "experiments")]  # extra/ layout

from experiments.exact_flow import flow  # noqa: E402
from grids import REGISTERED_SEEDS, T30_SCALES, t30_base  # noqa: E402

T25_TAUS = np.logspace(-1, 14, 80)
T30_TAUS = np.logspace(-3, 8, 11 * 20 + 1)
T32_RULES = {"grpo": dict(rule="grpo"), "rloo_k2": dict(rule="rloo_k", passk=2),
             "rloo_k4": dict(rule="rloo_k", passk=4)}
SOLVERS = ("LSODA", "Radau", "BDF")


def t25_config(rho, m, task_seed):
    """The T25 task (check_permanence.config on analysis-v3)."""
    return dict(arch="linear", width=64, features="correlated", rho=rho, majority=m, L=1, K=3, G=8, B=32,
                rule="grpo", task_seed=task_seed, seed=0, diagnostics=False)


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def solve(cfg, taus, seconds):
    """flow() with LSODA, then Radau, then BDF, each stopped after `seconds`. Returns (out or None, attempts)."""
    attempts = []
    for method in SOLVERS:
        t0 = time.time()
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            out = flow(cfg, taus, method=method, stable_tail=True)
            status = "ok"
        except Timeout:
            out, status = None, "timeout"
        except Exception as e:  # noqa: BLE001 (a solver failure moves on to the next solver)
            out, status = None, f"error: {e}"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
        attempts.append({"solver": method, "status": status, "seconds": round(time.time() - t0, 2)})
        if out is not None:
            return out, attempts
    return None, attempts


def save(path, **arrays):
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def run_job(job):
    path, cfg, taus, seconds, kind = job
    if os.path.exists(path):
        return path, "exists", 0.0
    out, attempts = solve(cfg, taus, seconds)
    meta = dict(attempts=json.dumps(attempts), config=json.dumps(cfg))
    if out is None:
        save(path, status="failed", solver="", taus=taus, **meta)
        return path, "failed", sum(a["seconds"] for a in attempts)
    logp = out["log_success"]
    extra = {}
    if kind == "t30":
        p = np.exp(logp)
        start = np.full((1, p.shape[1]), 1.0 / cfg["K"])
        run_max = np.maximum.accumulate(np.concatenate([start, p]), axis=0)[1:]
        extra["max_drop"] = (run_max - p).max(axis=0)
    save(path, status="ok", solver=attempts[-1]["solver"], taus=out["taus"], logp=logp.astype(np.float32),
         cls=out["cls"], **extra, **meta)
    return path, attempts[-1]["solver"], sum(a["seconds"] for a in attempts)


def seed_list(text):
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(v) for v in text.split(",")]


def jobs_t30(out, seeds, seconds):
    os.makedirs(os.path.join(out, "t30"), exist_ok=True)
    for s in T30_SCALES:
        for m in (1 / 3, 0.7, 0.9):
            for task_seed in seeds:
                path = os.path.join(out, "t30", f"s{s}_m{m:.3g}_t{task_seed}.npz")
                yield path, t30_base("linear", s, m, task_seed), T30_TAUS, seconds, "t30"


def jobs_t32(out, rules, seconds, limit):
    tasks = [(rho, m, t) for rho in (0.3, 0.5, 0.7, 0.9, 0.95) for m in (0.7, 0.9) for t in range(10)]
    for rho, m, t in tasks[:limit]:
        for rule in rules:
            os.makedirs(os.path.join(out, "t32", rule), exist_ok=True)
            path = os.path.join(out, "t32", rule, f"rho{rho}_m{m}_t{t}.npz")
            yield path, dict(t25_config(rho, m, t), **T32_RULES[rule]), T25_TAUS, seconds, "t32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", choices=("t30", "t32"))
    ap.add_argument("--out", default=os.path.expanduser("~/rlvr-data/v3r/flows"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks (smoke tests)")
    ap.add_argument("--minutes", type=float, default=30.0, help="time limit per solver attempt")
    ap.add_argument("--seeds", default=None, help="t30 task seeds, e.g. 30-49 (default) or 0,1")
    ap.add_argument("--rules", default="grpo,rloo_k2,rloo_k4", help="t32 rules")
    a = ap.parse_args()
    seconds = 60 * a.minutes
    if a.batch == "t30":
        seeds = seed_list(a.seeds) if a.seeds else REGISTERED_SEEDS
        jobs = list(jobs_t30(a.out, seeds, seconds))[:a.limit]
    else:
        jobs = list(jobs_t32(a.out, a.rules.split(","), seconds, a.limit))
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (path, solver, sec) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {os.path.relpath(path, a.out)}: {solver} ({sec:.1f} s)", flush=True)
    print(f"{len(jobs)} tasks in {time.time() - t0:.0f} s; output in {a.out}")


if __name__ == "__main__":
    main()
