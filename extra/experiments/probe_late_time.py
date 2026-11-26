"""Quick probe of the late-time prediction on T12 (linear policy). gamma_y = u . phi_y with u the min-norm vector
with u . phi_x >= 1 on every majority prompt (hard-margin direction of the majority)."""
import glob, gzip, json, os, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.optimize import minimize
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # repo root
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]  # extra/
from experiments.run import DEFAULTS, build


def max_margin(phi_a):
    k = phi_a @ phi_a.T
    f = lambda a: (0.5 * a @ k @ a - a.sum(), k @ a - 1)  # noqa: E731
    res = minimize(f, np.full(len(k), 1e-3), jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(k),
                   options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12})
    u = phi_a.T @ res.x
    return u, float((phi_a @ u).min())


def one(path):
    d = json.load(gzip.open(path, "rt")); c = d["config"]
    if c["arch"] != "linear":
        return None
    small = {k: v for k, v in c.items() if k in DEFAULTS or k == "task_seed"}
    _, train, _, pol = build(dict(DEFAULTS, **small))
    phi = np.asarray(train[0], dtype=float)
    cls = train[1][:, 0]; a = cls == 0
    u, minmarg = max_margin(phi[a])
    g = phi[~a] @ u
    p0 = np.asarray(d["p_init"])[~a]; pf = np.asarray(d["snapshots"][str(c["steps"])])[~a]; pmin = np.asarray(d["min_p"])[~a]
    k = phi @ phi.T
    return [dict(rho=c["rho"], m=c["majority"], opt=c["optimizer"], eta=c["eta"], update=c["update"], task=c["task_seed"],
                 gamma=gi, block=float(k[~a][:, a].mean() / k[a][:, a].mean()), p_min=float(pm), p_final=float(pfi),
                 support_margin=minmarg) for gi, pm, pfi in zip(g, pmin, pf)]


if __name__ == "__main__":
    ps = sorted(glob.glob(os.path.expanduser("~/rlvr-data/v3d/*/results/v3_t12/*.json.gz")))
    with Pool(8) as pool:
        rows = [r for out in pool.map(one, ps, chunksize=4) if out for r in out]
    df = pd.DataFrame(rows)
    df.to_csv("extra/analysis/theory/late_time_probe_rows.csv", index=False)
    print("prompts:", len(df), "runs:", len(df.groupby(["rho", "m", "opt", "eta", "update", "task"])))
    df["fell"] = df.p_min < 1e-3
    df["recovered"] = df.fell & (df.p_final > 0.1)
    fell = df[df.fell]
    for (opt, upd), g in fell.groupby(["opt", "update"]):
        lo, hi = g[g.gamma < 1], g[g.gamma >= 1]
        print(f"{opt:4s} {upd:13s} fell {len(g):6d} | gamma<1: {len(lo):5d} prompts, recovered {lo.recovered.mean():.3f} | gamma>=1: {len(hi):5d}, recovered {hi.recovered.mean():.3f}")
    print(df.groupby(["rho", "m"]).gamma.describe()[["min", "25%", "50%", "75%", "max"]].round(3))
