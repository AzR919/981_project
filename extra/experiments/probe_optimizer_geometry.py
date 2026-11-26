"""S1: does the l-infinity max-margin geometry predict the late sinking rate under Adam, and l2 under SGD?"""
import glob, gzip, json, os, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.stats import spearmanr
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # repo root
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]  # extra/
from experiments.run import DEFAULTS, build


def l2_margin(phi_a):
    k = phi_a @ phi_a.T
    f = lambda a: (0.5 * a @ k @ a - a.sum(), k @ a - 1)  # noqa: E731
    res = minimize(f, np.full(len(k), 1e-3), jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(k),
                   options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12})
    return phi_a.T @ res.x


def linf_margin(phi_a):
    """min t s.t. -t <= u_i <= t, phi_a u >= 1. Variables (u, t)."""
    n, d = phi_a.shape
    c = np.zeros(d + 1); c[-1] = 1
    a_ub = np.vstack([np.hstack([-phi_a, np.zeros((n, 1))]),            # -phi u <= -1
                      np.hstack([np.eye(d), -np.ones((d, 1))]),           # u - t <= 0
                      np.hstack([-np.eye(d), -np.ones((d, 1))])])         # -u - t <= 0
    b_ub = np.concatenate([-np.ones(n), np.zeros(2 * d)])
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=[(None, None)] * d + [(0, None)], method="highs")
    return res.x[:d]


def one(path):
    d = json.load(gzip.open(path, "rt")); c = d["config"]
    if c["arch"] != "linear":
        return None
    small = {k: v for k, v in c.items() if k in DEFAULTS or k == "task_seed"}
    _, train, _, _ = build(dict(DEFAULTS, **small))
    phi = np.asarray(train[0], dtype=float); a = train[1][:, 0] == 0
    u2, ui = l2_margin(phi[a]), linf_margin(phi[a])
    g2, gi = phi[~a] @ u2, phi[~a] @ ui
    return dict(rho=c["rho"], m=c["majority"], opt=c["optimizer"], eta=c["eta"], update=c["update"], task=c["task_seed"],
                g2_min=float(g2.min()), g2_med=float(np.median(g2)), ginf_min=float(gi.min()), ginf_med=float(np.median(gi)),
                margin2=float((phi[a] @ u2).min()), margininf=float((phi[a] @ ui).min()))


if __name__ == "__main__":
    ps = sorted(glob.glob(os.path.expanduser("~/rlvr-data/v3d/*/results/v3_t12/*.json.gz")))
    with Pool(8) as pool:
        rows = [r for r in pool.map(one, ps, chunksize=4) if r]
    run = pd.DataFrame(rows)
    cache = pd.read_csv(os.path.expanduser("~/rlvr-data/v3d/cache/runs.csv"))
    cache = cache[cache.name.str.startswith("v3_t12") & (cache.arch == "linear")].rename(
        columns={"majority": "m", "optimizer": "opt", "task_seed": "task"})
    j = run.merge(cache[["rho", "m", "opt", "eta", "update", "task", "margin_growth", "recovered"]],
                  on=["rho", "m", "opt", "eta", "update", "task"])
    j.to_csv("extra/analysis/theory/optimizer_geometry_runs.csv", index=False)
    print("runs:", len(j), "| support margins (should be 1):", round(j.margin2.min(), 4), round(j.margininf.min(), 4))
    for (opt, upd), g in j.groupby(["opt", "update"]):
        out = []
        for col in ("g2_min", "ginf_min", "g2_med", "ginf_med"):
            overall = spearmanr(g[col], g.margin_growth, nan_policy="omit").statistic
            within = [spearmanr(h[col], h.margin_growth, nan_policy="omit").statistic
                      for _, h in g.groupby(["rho", "m", "eta"]) if h.margin_growth.notna().sum() >= 6]
            out.append(f"{col}: {overall:.2f} (within-cell median {np.nanmedian(within):.2f})")
        print(f"{opt:4s} {upd:13s} " + " | ".join(out))
