"""
The two-kind minimal example of minority collapse, tabulated.

Two prompt kinds, K = 2 answers, the linear policy from a uniform start (W = 0, both kinds at success 0.5):
    majority  phi_a = (1, 0),      answer 0, share m
    minority  phi_b = (gamma, 0.5), answer 1, share 1 - m
gamma is the overlap of the minority's features with the majority's. At the start the majority's update lowers
the minority's success and the minority's own update raises it; their ratio is

    R = gamma m / ((1 - m) |phi_b|^2),   |phi_b|^2 = gamma^2 + 1/4,

so the minority starts to fall when R > 1 (gamma > 0 with m large enough).

usage:
    python experiments/minimal_example.py [--out analysis/theory/minimal_example] [--workers 8] [--only switch]

Writes regimes.csv, passk.csv, resets.csv, sampled.csv, switch.csv and agreement.csv to --out and prints a short
markdown summary of each. Flow time is tau = eta * steps (the convention of tests/test_exact_flow.py).
"""

import argparse
import csv
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import run  # noqa: E402
from experiments.exact_flow import flow  # noqa: E402
from synthetic_rl.theory import advantage_gap  # noqa: E402

COUNTS = {0.9: [9, 1], 0.7: [7, 3], 0.5: [1, 1]}
LN10 = math.log(10)
TAUS = np.logspace(-5, 14, 19 * 40 + 1)      # 40 points per decade up to flow time 1e14

GAMMAS = [-0.5, 0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0]
SHARES = [0.5, 0.7, 0.9]
RULES = [("mean", 8), ("grpo", 2), ("grpo", 8), ("grpo", "inf")]   # mean: W = 1 for every G
# Pass@k: the unbiased form is rule "rloo_k" (RLOO_K of Thrampoulidis et al.); "passk" is their biased GRPO_K.
# Both need k <= G; rloo_k's expected weight does not depend on G. Checked numerically: for G in (8, 16) and
# k in (1, 2, 4, 8), advantage_gap(p, G, "rloo_k", passk=k) equals (1 - p)^(k - 1) to 1e-15 on a p grid in
# (0, 1), i.e. (1/k) d pass@k / dp, NOT k (1 - p)^(k - 1): flow times under rloo_k are k times longer than
# under the k (1 - p)^(k - 1) normalisation. The biased "passk" form matches neither.
PASSK_KS = [1, 2, 4, 8]
PASSK_GAMMAS = [0.9, 1.2, 1.5, 2.5, 3.0, 5.0]

RESET_GAMMAS = [0.9, 1.2]
RESET_RULES = [("mean", 8), ("grpo", 8)]
BETAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
RESETS = [None, 1, 3, 10, 30, 100, 300, 1000, 3000]
RESET_ETA, RESET_STEPS = 0.5, 60_000

SAMPLED_GAMMAS = [-0.5, 0.5, 1.2]
SAMPLED_SEEDS = 20
SAMPLED_ETA, SAMPLED_STEPS = 0.5, 5_000       # flow time 2.5e3 (expected step = eta x the full expected step)

SWITCH_GAMMAS = [1.2, 1.5, 2.5, 3.0, 5.0, 6.0]
SWITCH_FIRST = [("mean", 8), ("grpo", 8)]
SWITCH_KS = [2, 4, 8]
SWITCH_DEPTHS = [1e-3, 1e-6]
SWITCH_HORIZON = 1e14                          # flow time after the switch

AGREE_GAMMAS = [0.5, 1.2]
AGREE_ETA, AGREE_STEPS = 0.01, 50_000         # flow time up to 500


def example(gamma, m, **kw):
    """Config of the two-kind example (the keys run.DEFAULTS does not already set right)."""
    return dict(K=2, L=1, arch="linear", features="explicit", phi=[[1.0, 0.0], [float(gamma), 0.5]],
                labels=[0, 1], counts=COUNTS[m], init="uniform", diagnostics=False, **kw)


def push_ratio(gamma, m):
    return gamma * m / ((1 - m) * (gamma**2 + 0.25))


def init_worker():
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    # Argument checks only; the numbers are unchanged.
    torch.distributions.Distribution.set_default_validate_args(False)


# ---------------------------------------------------------------- exact flow tables

def flow_row(job):
    gamma, m, rule, G, k = job
    cfg = example(gamma, m, rule=rule, G=G, passk=k)
    out = flow(cfg, TAUS, stable_tail=True)
    b = COUNTS[m][0]                                   # first minority prompt
    log_b = out["log_success"][:, b] / LN10
    log_a = out["log_success"][:, 0] / LN10
    i = int(np.argmin(log_b))
    back = np.nonzero(log_b[i:] > math.log10(0.5))[0]
    row = {"gamma": gamma, "m": m, "rule": rule, "G": G}
    if k is not None:
        row["k"] = k
    never_fell = bool(10 ** log_b[i] >= 0.5 - 1e-9)
    row.update({
        "R": push_ratio(gamma, m),
        "min_success": 10 ** log_b[i], "log10_min_success": log_b[i], "tau_min": TAUS[i],
        "log10_final_success": log_b[-1], "final_majority_success": 10 ** log_a[-1],
        # First grid point (40 per decade) after the minimum with the minority back above 0.5.
        "recovery_tau": TAUS[i + back[0]] if len(back) and not never_fell else "",
        "never_fell": never_fell,
    })
    return row


# ---------------------------------------------------------------- KL resets: batched exact updates

def batched_expected_sgd(gamma, m, rule, G, runs, eta, steps, record_every=0):
    """The update of run_train(update="expected_full", optimizer="sgd") for many (kl, beta, reset_every) runs of
    one task at once, in numpy (run_train costs about 1 ms a step; this about 0.1 ms for all runs together).
    runs: list of (kl, beta, reset_every). Returns per run the exact per-step minimum and the final log10
    success of the minority, the final majority success, and (record_every > 0) the minority's log10 success
    every record_every steps. Checked against run_train itself in check_batched()."""
    phi = np.array([[1.0, 0.0], [gamma, 0.5]])
    c = np.array([0, 1])
    weight = np.array(COUNTS[m], dtype=float) / sum(COUNTS[m])          # the mean over the n prompts
    R = len(runs)
    beta = np.array([r[1] for r in runs])[:, None, None]
    forward = np.array([r[0] == "forward" for r in runs])[:, None, None]
    every = np.array([r[2] or 0 for r in runs])
    W = np.zeros((R, 2, 2))                                            # (run, answer, feature)
    onehot = np.eye(2)[c][None]

    def log_pi(W):
        z = np.einsum("rkd,xd->rxk", W, phi)
        return z - np.logaddexp(z[..., 0], z[..., 1])[..., None]

    ref = log_pi(W)
    low = np.zeros(R) + math.log(0.5)
    trace = []
    for step in range(1, steps + 1):
        hit = (every > 0) & (step > 1) & ((step - 1) % np.maximum(every, 1) == 0)
        if hit.any():
            ref[hit] = log_pi(W[hit])
        lp = log_pi(W)
        pi = np.exp(lp)
        p = pi[:, [0, 1], c]
        w = advantage_gap(p, G, rule)
        g = -(w * p)[..., None] * (onehot - pi)                       # d loss / d z, reward part
        kl_rev = (pi * (lp - ref)).sum(-1, keepdims=True)
        g_kl = np.where(forward, pi - np.exp(ref), pi * (lp - ref - kl_rev))
        g = g + beta * g_kl
        W -= eta * np.einsum("x,rxk,xd->rkd", weight, g, phi)
        lb = log_pi(W)[:, 1, 1]
        np.minimum(low, lb, out=low)
        if record_every and step % record_every == 0:
            trace.append(lb / LN10)
    last = log_pi(W)
    return {"log10_min": low / LN10, "log10_final": last[:, 1, 1] / LN10, "final_a": np.exp(last[:, 0, 0]),
            "trace": np.array(trace).T if trace else None}


def reset_group(job):
    gamma, rule, G = job
    runs = [(kl, beta, r) for kl in ("reverse", "forward") for beta in BETAS for r in RESETS] + [("reverse", 0.0, None)]
    res = batched_expected_sgd(gamma, 0.9, rule, G, runs, RESET_ETA, RESET_STEPS)
    rows = []
    for j, (kl, beta, r) in enumerate(runs):
        rows.append({"gamma": gamma, "m": 0.9, "rule": rule, "G": G, "kl": kl if beta > 0 else "",
                     "beta": beta, "reset_every": "" if r is None else r,
                     "log10_min_success": res["log10_min"][j], "log10_final_success": res["log10_final"][j],
                     "final_majority_success": res["final_a"][j]})
    return rows


CHECK_RUNS = [(1.2, "mean", 8, "reverse", 0.1, None), (1.2, "grpo", 8, "reverse", 0.3, 100),
              (0.9, "grpo", 8, "forward", 1.0, 1000), (0.9, "mean", 8, "forward", 0.03, 3),
              (1.2, "grpo", 8, "forward", 3.0, None), (0.9, "grpo", 8, "reverse", 0.01, 3000),
              (1.2, "mean", 8, "reverse", 1.0, 30), (0.9, "mean", 8, "reverse", 0.0, None)]
CHECK_EVERY, CHECK_STEPS = 50, 15_000   # the check covers 5 periods of the longest reset_every


def run_train_check(job):
    """One reset run through run_train itself; returns the minority's log10 success every CHECK_EVERY steps,
    from the logged class means (float64)."""
    gamma, rule, G, kl, beta, r = job
    cfg = dict(run.DEFAULTS, name="check", **example(gamma, 0.9, rule=rule, G=G, kl=kl, beta=beta, reset_every=r,
                                                       update="expected_full", optimizer="sgd", eta=RESET_ETA,
                                                       steps=CHECK_STEPS, log_every=CHECK_EVERY))
    out = run.run_train(cfg)
    return np.log10([h["class_mean"][1] for h in out["history"][1:]]), out["history"][-1]["class_mean"][0]


# ---------------------------------------------------------------- sampled GRPO

def sampled_row(job):
    gamma, seed = job
    cfg = dict(run.DEFAULTS, name="sampled", **example(gamma, 0.9, rule="grpo", G=8, B=32, update="sampled",
                                                         optimizer="sgd", eta=SAMPLED_ETA, steps=SAMPLED_STEPS,
                                                         log_every=SAMPLED_STEPS, track_every=10, seed=seed))
    out = run.run_train(cfg)
    # min_p is stored as float32 (it underflows below ~1e-45); the final value comes from the float64 class mean.
    return gamma, seed, float(out["min_p"][-1]), out["history"][-1]["class_mean"][1]


def sampled_rows(results):
    rows = []
    for gamma in SAMPLED_GAMMAS:
        mine = [r for r in results if r[0] == gamma]
        final = np.array([r[3] for r in mine])
        low = np.array([r[2] for r in mine])
        rows.append({"gamma": gamma, "m": 0.9, "rule": "grpo", "G": 8, "B": 32, "eta": SAMPLED_ETA,
                     "steps": SAMPLED_STEPS, "seeds": len(mine), "R": push_ratio(gamma, 0.9),
                     "share_final_below_1e-3": float(np.mean(final < 1e-3)),
                     "median_min_success": float(np.median(low)), "median_final_success": float(np.median(final))})
    return rows


# ---------------------------------------------------------------- switch to pass@k after a collapse

def switch_rows(job):
    """Collapse under the first rule until the minority's success first falls to `depth` (first point of the
    40-per-decade grid at or below it), then continue the exact flow from those weights under rloo_k."""
    gamma, rule, G = job
    first = example(gamma, 0.9, rule=rule, G=G)
    b = COUNTS[0.9][0]
    before = flow(first, TAUS, stable_tail=True)["log_success"][:, b] / LN10
    after_taus = np.logspace(-5, math.log10(SWITCH_HORIZON), int(40 * (math.log10(SWITCH_HORIZON) + 5)) + 1)
    rows = []
    for depth in SWITCH_DEPTHS:
        hit = np.nonzero(before <= math.log10(depth))[0]
        for k in SWITCH_KS:
            row = {"gamma": gamma, "first_rule": rule, "first_G": G, "k": k, "target_depth": depth}
            if not len(hit):
                row.update(tau_switch="", log10_success_at_switch="", log10_min_after="", log10_final="",
                           final_majority_success="", min_majority_after="", recovered="not reached",
                           recovery_tau_after="", log10_change_last_2_decades="", horizon=SWITCH_HORIZON)
                rows.append(row)
                continue
            tau_s = TAUS[hit[0]]
            start = flow(first, [tau_s], stable_tail=True)
            out = flow(example(gamma, 0.9, rule="rloo_k", G=8, passk=k), after_taus, stable_tail=True,
                       start=start["W"])
            lb, la = out["log_success"][:, b] / LN10, out["log_success"][:, 0] / LN10
            back = np.nonzero(lb > math.log10(0.5))[0]
            j = int(np.searchsorted(after_taus, SWITCH_HORIZON / 100))
            row.update(tau_switch=tau_s, log10_success_at_switch=start["log_success"][-1, b] / LN10,
                       log10_min_after=lb.min(), log10_final=lb[-1], final_majority_success=10 ** la[-1],
                       min_majority_after=10 ** la.min(), recovered="yes" if len(back) else "no",
                       recovery_tau_after=after_taus[back[0]] if len(back) else "",
                       log10_change_last_2_decades=lb[-1] - lb[j], horizon=SWITCH_HORIZON)
            rows.append(row)
    return rows


# ---------------------------------------------------------------- exact flow against small-step SGD

def agreement_row(job):
    gamma, rule, G = job
    snaps = sorted({int(s) for s in np.logspace(1, math.log10(AGREE_STEPS), 41)})
    cfg = dict(run.DEFAULTS, name="agree", **example(gamma, 0.9, rule=rule, G=G, update="expected_full",
                                                       optimizer="sgd", eta=AGREE_ETA, steps=AGREE_STEPS,
                                                       log_every=AGREE_STEPS, snapshot_at=snaps))
    sim = run.run_train(cfg)                        # with snapshot_at the snapshots are float64
    taus = AGREE_ETA * np.array(snaps, dtype=float)
    fl = flow(example(gamma, 0.9, rule=rule, G=G), taus, stable_tail=True)
    want = np.log10(np.array([sim["snapshots"][str(s)] for s in snaps]))
    diff = np.abs(fl["log_success"] / LN10 - want)
    j = int(np.argmax(diff.max(axis=1)))
    return {"gamma": gamma, "m": 0.9, "rule": rule, "G": G, "eta": AGREE_ETA, "max_tau": taus[-1],
            "max_abs_diff_log10": float(diff.max()), "tau_of_max_diff": taus[j],
            "log10_minority_at_max_tau": float(want[-1, COUNTS[0.9][0]])}


# ---------------------------------------------------------------- output

def write(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "analysis", "theory", "minimal_example"))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--only", choices=("switch",), help="write only this table")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    init_worker()

    flow_jobs = [(g, m, rule, G, None) for g in GAMMAS for m in SHARES for rule, G in RULES]
    passk_jobs = [(g, 0.9, "rloo_k", 8, k) for k in PASSK_KS for g in PASSK_GAMMAS]
    reset_jobs = [(g, rule, G) for g in RESET_GAMMAS for rule, G in RESET_RULES]
    sampled_jobs = [(g, s) for g in SAMPLED_GAMMAS for s in range(SAMPLED_SEEDS)]
    agree_jobs = [(g, rule, G) for g in AGREE_GAMMAS for rule, G in (("mean", 8), ("grpo", 8))]

    if args.only == "switch":
        with Pool(args.workers, initializer=init_worker) as pool:
            print_switch(args.out, pool.map(switch_rows, [(g, r, G) for g in SWITCH_GAMMAS for r, G in SWITCH_FIRST]))
        print(f"\n{time.time() - t0:.0f} s; CSV in {args.out}")
        return
    with Pool(args.workers, initializer=init_worker) as pool:
        # The slow run_train jobs go first so the cheap ones fill the gaps.
        switch = pool.map_async(switch_rows, [(g, r, G) for g in SWITCH_GAMMAS for r, G in SWITCH_FIRST])
        checks = pool.map_async(run_train_check, CHECK_RUNS)
        agree = pool.map_async(agreement_row, agree_jobs)
        sampled = pool.map_async(sampled_row, sampled_jobs)
        resets = pool.map_async(reset_group, reset_jobs)
        regimes = pool.map_async(flow_row, flow_jobs)
        passk = pool.map_async(flow_row, passk_jobs)

        regimes, passk = regimes.get(), passk.get()
        write(os.path.join(args.out, "regimes.csv"), regimes)
        write(os.path.join(args.out, "passk.csv"), passk)
        print("## regimes.csv (exact flow to tau = 1e14; m = 0.9 shown)\n")
        print(table([r for r in regimes if r["m"] == 0.9],
                    ["gamma", "rule", "G", "R", "log10_min_success", "tau_min", "log10_final_success",
                     "final_majority_success", "recovery_tau", "never_fell"]))
        print("\n## passk.csv (unbiased pass@k, rule rloo_k, G = 8, m = 0.9)\n")
        print(table(passk, ["k", "gamma", "R", "log10_min_success", "tau_min", "log10_final_success",
                            "recovery_tau"]))

        resets = [row for rows in resets.get() for row in rows]
        write(os.path.join(args.out, "resets.csv"), resets)
        print(f"\n## resets.csv (expected_full SGD, eta = {RESET_ETA}, {RESET_STEPS} steps; log10 final minority"
              " success / final majority success)\n")
        for g in RESET_GAMMAS:
            for rule, G in RESET_RULES:
                base = [r for r in resets if r["gamma"] == g and r["rule"] == rule and r["beta"] == 0][0]
                print(f"gamma = {g}, {rule} G = {G}; beta = 0: {base['log10_final_success']:.2f} / "
                      f"{base['final_majority_success']:.4f}\n")
                for kl in ("reverse", "forward"):
                    cols = ["beta"] + [("fixed" if r is None else str(r)) for r in RESETS]
                    rows = []
                    for beta in BETAS:
                        row = {"beta": beta}
                        for r in RESETS:
                            x = [y for y in resets if y["gamma"] == g and y["rule"] == rule and y["kl"] == kl
                                 and y["beta"] == beta and y["reset_every"] == ("" if r is None else r)][0]
                            row["fixed" if r is None else str(r)] = (f"{x['log10_final_success']:.1f} / "
                                                                     f"{x['final_majority_success']:.3f}")
                        rows.append(row)
                    print(f"{kl} KL, columns reset_every:\n")
                    print(table(rows, cols) + "\n")
        worst = check_batched_from(checks.get())
        print(f"Batched updates against run_train on {len(CHECK_RUNS)} settings: largest |difference| "
              f"{worst:.2e} (log10 minority success every {CHECK_EVERY} steps to step {CHECK_STEPS}, and final "
              "majority success).")

        sampled = sampled_rows(sampled.get())
        write(os.path.join(args.out, "sampled.csv"), sampled)
        print("\n## sampled.csv\n")
        print(table(sampled, ["gamma", "R", "seeds", "share_final_below_1e-3", "median_min_success",
                              "median_final_success"]))

        print_switch(args.out, switch.get())

        agree = agree.get()
        write(os.path.join(args.out, "agreement.csv"), agree)
        print("\n## agreement.csv (exact flow against expected_full SGD at eta = 0.01, tau = eta x steps)\n")
        print(table(agree, ["gamma", "rule", "G", "max_tau", "max_abs_diff_log10", "tau_of_max_diff",
                            "log10_minority_at_max_tau"]))
    print(f"\n{time.time() - t0:.0f} s; CSVs in {args.out}")


def print_switch(out, groups):
    rows = [row for group in groups for row in group]
    write(os.path.join(out, "switch.csv"), rows)
    print(f"\n## switch.csv (collapse under the first rule to the target depth, then rloo_k; exact flow to "
          f"{SWITCH_HORIZON:.0e} after the switch; m = 0.9)\n")
    print(table(rows, ["gamma", "first_rule", "k", "target_depth", "log10_success_at_switch", "log10_min_after",
                       "log10_final", "log10_change_last_2_decades", "min_majority_after", "recovered",
                       "recovery_tau_after"]))


def check_batched_from(got):
    worst = 0.0
    for job, (want, want_a) in zip(CHECK_RUNS, got):
        gamma, rule, G, kl, beta, r = job
        res = batched_expected_sgd(gamma, 0.9, rule, G, [(kl, beta, r)], RESET_ETA, CHECK_STEPS,
                                   record_every=CHECK_EVERY)
        worst = max(worst, float(np.max(np.abs(res["trace"][0] - want))), abs(float(res["final_a"][0]) - want_a))
    return worst


if __name__ == "__main__":
    main()
