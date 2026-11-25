"""
T25: permanent or plateau? Exact long-time flow of the linear policy, implementing PREREG_v3d_addendum7.md.

usage:
    python experiments/check_permanence.py [--data ~/rlvr-data/v3m] [--out analysis/v3m]

For 100 tasks (rho in {0.3, 0.5, 0.7, 0.9, 0.95}, m in {0.7, 0.9}, task seeds 0-9) the exact gradient flow is
integrated to flow time 1e14 (experiments/exact_flow.py). From each minority prompt's state at tau0 (majority
class mean first >= 0.99) the exact solution of dL/dtau = gamma/tau - c e^{-L} predicts when (or whether) it
recovers; gamma_y = u.phi_y with u the majority's l2 max-margin direction, c_y from the kernel and W_G(0).
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze import markdown  # noqa: E402
from experiments.check_late_time import l2_max_margin  # noqa: E402
from experiments.exact_flow import flow, task_arrays  # noqa: E402
from synthetic_rl.theory import advantage_gap  # noqa: E402

TAUS = np.logspace(-1, 14, 80)
LOG_HALF = np.log(0.5)


def config(rho, m, task_seed):
    return dict(arch="linear", width=64, features="correlated", rho=rho, majority=m, L=1, K=3, G=8, B=32,
                rule="grpo", task_seed=task_seed, seed=0, diagnostics=False)


def predicted_log_tau(gamma, L0, c, log_tau0):
    """log of the predicted recovery time from the exact scalar solution; +inf if never."""
    out = np.full_like(gamma, np.inf)
    lo = gamma < 1 - 1e-9
    hi = gamma > 1 + 1e-9
    mid = ~(lo | hi)
    g = gamma[lo]
    logA = np.logaddexp((1 - g) * log_tau0, np.log(1 - g) + L0[lo] - g * log_tau0 - np.log(c[lo]))
    out[lo] = logA / (1 - g)
    g = gamma[hi]
    r = (g - 1) * np.exp(np.clip(L0[hi], None, 700)) / (c[hi] * np.exp(log_tau0))
    ok = r < 1
    vals = np.full_like(g, np.inf)
    vals[ok] = log_tau0 + np.log1p(-r[ok]) / (1 - g[ok])
    out[hi] = vals
    out[mid] = log_tau0 + np.exp(np.clip(L0[mid], None, 700)) / (c[mid] * np.exp(log_tau0))
    return out


def one(args):
    rho, m, task_seed, data = args
    cfg = config(rho, m, task_seed)
    phi, cls, K, _ = task_arrays(cfg)
    path = os.path.join(data, f"rho{rho}_m{m}_t{task_seed}.npz")
    if os.path.exists(path):                       # reuse a finished integration
        saved = np.load(path)
        out = {"taus": saved["taus"], "log_success": saved["logp"].astype(float)}
    else:
        out = flow(cfg, TAUS)
        np.savez_compressed(path, taus=out["taus"], logp=out["log_success"].astype(np.float32), cls=cls)
    logp = out["log_success"]
    maj, mino = cls == 0, cls != 0
    maj_mean = np.exp(logp[:, maj]).mean(axis=1)
    run = dict(rho=rho, m=m, task=task_seed)
    reach = np.nonzero(maj_mean >= 0.99)[0]
    if not len(reach):
        run["status"] = "majority never reached 0.99"
        return run, None
    i0 = reach[0]
    log_tau0 = np.log(out["taus"][i0])
    u = l2_max_margin(phi[maj])
    gamma = phi[mino] @ u
    k = phi[mino] @ phi[mino].T
    cm = cls[mino]
    s = np.where(cm[:, None] == cm[None, :], 2.0, 1.0)
    w0 = float(advantage_gap(np.array([1e-12]), 8, "grpo")[0])
    c = w0 / len(cls) * (k * s).sum(axis=1)
    L0 = -logp[i0, mino]
    pred = predicted_log_tau(gamma, L0, c, log_tau0)
    lm = logp[:, mino]
    fallen = lm[i0] < LOG_HALF
    rec_idx = np.array([next((j for j in range(i0, len(TAUS)) if lm[j, y] >= LOG_HALF), -1) for y in range(lm.shape[1])])
    obs = np.where(rec_idx >= 0, np.log(out["taus"][np.maximum(rec_idx, 0)]), np.inf)
    # Run-level late exponent of the minority class mean.
    pbar = np.exp(lm).mean(axis=1)
    lt = np.log(out["taus"])
    after = np.arange(len(TAUS)) >= i0
    rise = np.nonzero(after & (pbar >= 1e-3) & (np.arange(len(TAUS)) > i0))[0]
    end = rise[0] - 1 if len(rise) else len(TAUS) - 1
    start_t = lt[end] - 2 * np.log(10)
    j = np.searchsorted(lt, start_t)
    exponent = np.nan
    if j >= i0 and end > j and (pbar[j:end + 1] < 1e-3).all():
        exponent = float(-(np.log(pbar[end]) - np.log(pbar[j])) / (lt[end] - lt[j]))
    run.update(status="ok", tau0=float(np.exp(log_tau0)), gamma_min=float(gamma.min()), exponent=exponent,
               fallen=int(fallen.sum()), recovered_by_end=int((fallen & (rec_idx >= 0)).sum()),
               minority_final=float(pbar[-1]), recovered_class=bool(pbar[-1] >= 0.5))
    prompts = pd.DataFrame(dict(rho=rho, m=m, task=task_seed, gamma=gamma, L0=L0, c=c, pred_log_tau=pred,
                                obs_log_tau=obs, fallen=fallen))
    return run, prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/rlvr-data/v3m"))
    ap.add_argument("--out", default="analysis/v3m")
    ap.add_argument("--skip", default="", help="tasks the solver could not integrate, e.g. 0.9:0.7:5;...")
    a = ap.parse_args()
    skip = {tuple(float(v) for v in x.split(":")) for x in a.skip.split(";") if x}
    os.makedirs(a.data, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    jobs = [(rho, m, t, a.data) for rho in (0.3, 0.5, 0.7, 0.9, 0.95) for m in (0.7, 0.9) for t in range(10)
            if (rho, m, float(t)) not in skip]
    with Pool(8) as pool:
        res = pool.map(one, jobs, chunksize=1)
    runs = pd.DataFrame([r for r, _ in res])
    pr = pd.concat([p for _, p in res if p is not None], ignore_index=True)
    runs.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    pr.to_csv(os.path.join(a.out, "prompts.csv"), index=False)
    f = pr[pr.fallen]
    never = f[np.isinf(f.pred_log_tau)]
    early = f[f.pred_log_tau < np.log(1e12)]
    both = f[np.isfinite(f.obs_log_tau) & np.isfinite(f.pred_log_tau)]
    p1 = float(np.isfinite(never.obs_log_tau).mean()) if len(never) else np.nan
    p2 = float(np.isfinite(early.obs_log_tau).mean()) if len(early) else np.nan
    p3s = float(spearmanr(both.pred_log_tau, both.obs_log_tau).statistic) if len(both) > 3 else np.nan
    p3m = float(np.median(np.abs(both.pred_log_tau - both.obs_log_tau)) / np.log(10)) if len(both) else np.nan
    ok = runs[(runs.status == "ok") & runs.exponent.notna()]
    p4 = float((np.abs(ok.exponent - ok.gamma_min) <= 0.1).mean()) if len(ok) else np.nan
    cell = runs.groupby(["rho", "m"]).agg(tasks=("task", "size"), gamma_min=("gamma_min", "median"),
                                          exponent=("exponent", "median"),
                                          minority_recovered_by_1e14=("recovered_class", "sum"),
                                          fallen=("fallen", "sum"), recovered=("recovered_by_end", "sum")).reset_index()
    lines = ["# Findings: v3m, T25 permanent or plateau (exact long-time flow; PREREG_v3d addendum 7)", "",
             f"Generated by `experiments/check_permanence.py`: {len(runs)} tasks integrated to flow time 1e14; "
             f"{len(f)} minority prompts below 0.5 at τ0."
             + (f" Not integrable (solver did not finish, left out): {sorted(skip)}." if skip else ""), "",
             f"1. **Permanence** (predicted never to recover; at most 5% recovered by 1e14): {p1:.3f} of {len(never)} prompts recovered.",
             f"2. **Recovery** (predicted before 1e12; at least 80% recovered by 1e14): {p2:.3f} of {len(early)} prompts.",
             f"3. **Timing** (Spearman ≥ 0.7, median |log10 ratio| ≤ 1): Spearman {p3s:.3f}, median {p3m:.2f} decades, "
             f"over {len(both)} prompts.",
             f"4. **Run level** (late exponent within 0.1 of γ_min in at least 80% of qualifying runs): {p4:.3f} of {len(ok)} runs.",
             "5. **By cell** (tasks whose minority class mean is at least 0.5 at 1e14):", "", markdown(cell), ""]
    with open(os.path.join(a.out, "FINDINGS.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
