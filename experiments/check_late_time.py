"""
T24: the late phase, prompt by prompt. Implements the prediction rule of PREREG_v3d_addendum6.md.

usage:
    python experiments/check_late_time.py [--raw ~/rlvr-data/v3l] [--out analysis/v3l]

For each run: the late phase starts at tau0, the earliest stored step with majority class mean >= 0.99. From each
minority prompt's stored success at tau0, its wrong-side logit L_y is integrated forward with
    dL_y/dtau = gamma_y dM/dtau - (1/n) sum_{y'} k(y, y') W_G(p_y') p_y' s(y, y') (1 - p_y'),
where gamma_y = u.phi_y (u: the majority's l2 max-margin direction, unit margin on its closest prompts), M is the
measured majority margin log(2 p_A / (1 - p_A)), k the linear kernel, s = 2 for classmates and 1 for the other
minority class. Compared with the stored success at later steps, against baselines B1-B4.
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
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze import markdown  # noqa: E402
from experiments.run import DEFAULTS, build  # noqa: E402
from synthetic_rl.theory import advantage_gap  # noqa: E402

TINY = 1e-300


def l2_max_margin(phi_a):
    """min |u| s.t. phi_a u >= 1, through its dual (nonnegative quadratic program)."""
    k = phi_a @ phi_a.T
    f = lambda a: (0.5 * a @ k @ a - a.sum(), k @ a - 1)  # noqa: E731
    res = minimize(f, np.full(len(k), 1e-3), jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(k),
                   options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12})
    return phi_a.T @ res.x


def logit_wrong(p):
    p = np.clip(p, TINY, 1 - 1e-16)
    return np.log1p(-p) - np.log(p)


def success(L):
    return 1.0 / (1.0 + np.exp(np.clip(L, -700, 700)))


def integrate(L0, taus, M_tau, M_val, gamma, k_bb, s_bb, n, restore=True):
    """Integrate the minority model from taus[0] to each later tau; returns L at those times."""
    def rhs(t, L):
        i = min(max(np.searchsorted(M_tau, t, side="right") - 1, 0), len(M_tau) - 2)
        dM = (M_val[i + 1] - M_val[i]) / (M_tau[i + 1] - M_tau[i])
        out = gamma * dM
        if restore is not False:
            p = success(L)
            w = advantage_gap(p, 8, "grpo") * p * (1 - p)
            r = (k_bb * s_bb) @ w if restore is True else np.diag(k_bb * s_bb) * w
            out = out - r / n
        return out
    sol = solve_ivp(rhs, (taus[0], taus[-1]), L0, t_eval=taus, method="LSODA", rtol=1e-6, atol=1e-6)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def one(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    cfg = d["config"]
    small = {k: v for k, v in cfg.items() if k in DEFAULTS or k == "task_seed"}
    _, train, _, _ = build(dict(DEFAULTS, **small))
    phi = np.asarray(train[0], dtype=float)
    cls = train[1][:, 0]
    a = cls == 0
    eta, n = cfg["eta"], len(cls)
    steps = np.array([h["step"] for h in d["history"]])
    pa = np.array([h["class_mean"][0] for h in d["history"]])
    pb_mean = np.array([h["class_mean"][1] for h in d["history"]])
    snaps = {int(s): np.asarray(v, dtype=float) for s, v in d["snapshots"].items()}
    info = dict(rho=cfg["rho"], m=cfg["majority"], eta=eta, task=cfg["task_seed"], n_minority=int((~a).sum()))
    # Late margin growth over the last decade (as in M2), for prediction 5.
    tenth = int(np.searchsorted(steps, steps[-1] / 10))
    den = -np.log(np.clip(1 - pa[-1], TINY, 1)) + np.log(np.clip(1 - pa[tenth], TINY, 1))
    info["margin_growth"] = float((-np.log(max(pb_mean[-1], TINY)) + np.log(max(pb_mean[tenth], TINY))) / den) if den > 0 else np.nan
    u = l2_max_margin(phi[a])
    gamma = phi[~a] @ u
    info["gamma_min"], info["gamma_median"] = float(gamma.min()), float(np.median(gamma))
    candidates = [s for s in sorted(snaps) if s > 0 and pa[np.searchsorted(steps, s)] >= 0.99 and steps[np.searchsorted(steps, s)] == s]
    if not candidates:
        info["status"] = "majority never reached 0.99"
        return info, None
    s0 = candidates[0]
    later = [s for s in sorted(snaps) if s > s0]
    if not later:
        info["status"] = "no stored step after tau0"
        return info, None
    taus = eta * np.array([s0] + later, dtype=float)
    keep = steps >= s0
    M_tau = eta * steps[keep].astype(float)
    M_val = np.log(2 * np.clip(pa[keep], TINY, 1)) - np.log(np.clip(1 - pa[keep], TINY, 1))
    pb0 = snaps[s0][~a]
    info.update(tau0_step=s0, below_half_at_tau0=int((pb0 < 0.5).sum()))
    L0 = logit_wrong(pb0)
    k = phi @ phi.T
    k_bb = k[~a][:, ~a]
    cb = cls[~a]
    s_bb = np.where(cb[:, None] == cb[None, :], 2.0, 1.0)
    sim = snaps[later[-1]][~a]
    preds = {
        "model": integrate(L0, taus, M_tau, M_val, gamma, k_bb, s_bb, n, restore=True)[-1],
        "B1 persistence": L0,
        "B2 push only": integrate(L0, taus, M_tau, M_val, gamma, k_bb, s_bb, n, restore=False)[-1],
        "B3 self only": integrate(L0, taus, M_tau, M_val, gamma, k_bb, s_bb, n, restore="self")[-1],
        "B4 one gamma": integrate(L0, taus, M_tau, M_val, np.full_like(gamma, gamma.mean()), k_bb, s_bb, n, restore=True)[-1],
    }
    log_sim = np.log10(np.clip(sim, TINY, 1))
    for name, L in preds.items():
        p = success(L)
        lp = np.log10(np.clip(p, TINY, 1))
        info[f"mae[{name}]"] = float(np.mean(np.abs(lp - log_sim)))
        info[f"spearman[{name}]"] = float(spearmanr(lp, log_sim).statistic) if np.ptp(log_sim) > 0 else np.nan
        info[f"recovered[{name}]"] = int((p > 0.1).sum())
    info["recovered[simulated]"] = int((sim > 0.1).sum())
    info["status"] = "ok"
    prompts = pd.DataFrame({"gamma": gamma, "p_tau0": pb0, "p_final_sim": sim, "p_final_model": success(preds["model"]),
                            "p_final_self": success(preds["B3 self only"]), "class": cb})
    for key in ("rho", "m", "eta", "task"):
        prompts[key] = info[key]
    return info, prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.expanduser("~/rlvr-data/v3l"))
    ap.add_argument("--out", default="analysis/v3l")
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.raw, "*", "results", "*", "*.json.gz")))
    with Pool(8) as pool:
        out = pool.map(one, paths, chunksize=2)
    runs = pd.DataFrame([r for r, _ in out])
    prompts = pd.concat([p for _, p in out if p is not None], ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    runs.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    prompts.to_csv(os.path.join(a.out, "prompts.csv"), index=False)
    ok = runs[(runs.status == "ok") & (runs.below_half_at_tau0 >= 20)]
    rec = ok[ok["recovered[simulated]"] >= 10]
    beats = lambda base, frame: float((frame["mae[model]"] < frame[f"mae[{base}]"]).mean())  # noqa: E731
    within = [spearmanr(g.gamma_min, g.margin_growth, nan_policy="omit").statistic
              for _, g in runs.groupby(["rho", "m", "eta"]) if g.margin_growth.notna().sum() >= 6]
    count_ok = ((rec["recovered[model]"] - rec["recovered[simulated]"]).abs() <= 0.25 * rec["recovered[simulated]"]).mean() if len(rec) else np.nan
    lines = ["# Findings: v3l, T24 the late phase prompt by prompt (PREREG_v3d addendum 6)", "",
             f"Generated by `experiments/check_late_time.py` from {len(runs)} runs; {int((runs.status == 'ok').sum())} reach the "
             f"late phase; {len(ok)} have at least 20 minority prompts below 0.5 at τ0 (the runs that count); {len(rec)} of those "
             "have at least 10 recovering prompts.", "",
             f"1. **The model tracks each prompt** (median Spearman, predicted vs simulated log p at the last step, at least "
             f"0.8): {ok['spearman[model]'].median():.3f}.",
             f"2. **It beats the baselines** (smaller error than B1, B2 and B4 in at least 80% of runs): B1 "
             f"{beats('B1 persistence', ok):.2f}, B2 {beats('B2 push only', ok):.2f}, B4 {beats('B4 one gamma', ok):.2f}.",
             f"3. **Recovery is collective** (beats B3, self-only restoring, in at least 70% of runs with at least 10 "
             f"recoveries): {beats('B3 self only', rec) if len(rec) else float('nan'):.2f} over {len(rec)} runs.",
             f"4. **Recovery counts within 25%** (at least 70% of runs with at least 10 recoveries): {count_ok:.2f}.",
             f"5. **Probe replication** (within-cell Spearman of the smallest γ against the late margin growth, median at "
             f"least 0.6): {np.nanmedian(within):.2f} over {len(within)} cells.", "",
             "## Error and correlation by cell (medians over the runs that count)", ""]
    cols = [c for c in ok.columns if c.startswith("mae[") or c.startswith("spearman[")]
    lines += [markdown(ok.groupby(["rho", "m", "eta"])[cols].median().round(3).reset_index()), "",
              "## Recovery counts by cell (sums)", "",
              markdown(ok.groupby(["rho", "m", "eta"])[[c for c in ok.columns if c.startswith("recovered[")]].sum().reset_index()), "",
              "## Runs that do not count", "", markdown(runs[runs.status != "ok"][["rho", "m", "eta", "task", "status"]]), ""]
    with open(os.path.join(a.out, "FINDINGS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
