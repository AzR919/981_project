"""
Numerical checks of the derivations in extra/analysis/theory/minimal_example.md (the two-kind example of
extra/experiments/minimal_example.py), and the phase table under the mean rule.

usage:
    python extra/experiments/check_minimal_example.py [--out extra/analysis/theory] [--workers 4]

Writes extra/analysis/theory/minimal_example/phase.csv and extra/analysis/theory/minimal_example_check.md. Reads the driver's
tables (passk.csv, resets.csv, switch.csv) from extra/analysis/theory/minimal_example/. Nothing is fitted.

Notation (as in the note): a = (1, 0) majority features (answer 0, share m), b = (gamma, 0.5) minority features
(answer 1), w = W_0 - W_1, u = w.a (majority margin), v = w.b (minority deficit, s = -v its margin). The simulator's
flow time tau is half the time t of the direct logistic model (both rows of W move), so dw/dtau = 2 dw/dt.
"""

import argparse
import csv
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit, log_expit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (this script is in extra/experiments/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path[1:1] = [os.path.join(ROOT, "extra"), os.path.join(ROOT, "extra", "experiments")]  # extra/ layout

from experiments.exact_flow import flow  # noqa: E402
from synthetic_rl.theory import advantage_gap, fixed_points  # noqa: E402
import minimal_example as mx  # noqa: E402

LN10 = math.log(10)
TAUS = mx.TAUS                                    # 40 points per decade, 1e-5 to 1e14
PHASE_GAMMAS = np.round(np.linspace(-0.5, 3.0, 71), 4)
PHASE_SHARES = np.round(np.arange(50, 96) / 100, 2)   # 0.50, 0.51, ..., 0.95 (46 values)
W_MEAN = float(advantage_gap(np.array([0.5]), 8, "mean")[0])   # 1 - 1/G = 0.875, the same at every p


def counts_for(m):
    """Smallest integer counts [n_A, n_B] with n_A / (n_A + n_B) = m (m on a grid of 0.01)."""
    n_a = int(round(100 * m))
    g = math.gcd(n_a, 100 - n_a)
    return [n_a // g, (100 - n_a) // g]


def two_kind(gamma, m, **kw):
    cfg = mx.example(gamma, 0.9, **kw)
    cfg["counts"] = counts_for(m)
    return cfg


def w_margins(W, gamma):
    """(u, v) = (w.a, w.b) from K x d weights, w = W_0 - W_1 (exact; log-success loses u, v beyond about 35)."""
    w = W[0] - W[1]
    return float(w[0]), float(gamma * w[0] + 0.5 * w[1])


def start_weights(gamma, u, v):
    """K x d weights with w.a = u and w.b = v (row 1 zero)."""
    return np.array([[u, 2.0 * (v - gamma * u)], [0.0, 0.0]])


def tail_constants(rule, G=8, k=None):
    """(alpha, beta, C_A, C_B) with h_A ~ C_A e^{-alpha u}, h_B ~ C_B e^{-beta v}."""
    if rule == "rloo_k":
        return float(k), 1.0, 1.0, 1.0
    c = float(advantage_gap(np.array([1e-300]), G, rule)[0])
    return 1.0, 1.0, c, c


def log_rstar(gamma, m, alpha, beta, CA, CB, b2=None):
    """log of rho* = zeta*/xi*: the separatrix beta v - alpha u = log rho* of the tail system (|a| = 1)."""
    b2 = gamma**2 + 0.25 if b2 is None else b2
    return math.log((1 - m) * CB * (beta * b2 - alpha * gamma) / (m * CA * (beta * gamma - alpha)))


# ---------------------------------------------------------------- phase table (mean rule, G = 8)

def phase_row(job):
    gamma, m = job
    cfg = two_kind(gamma, m, rule="mean", G=8)
    n_a = cfg["counts"][0]
    out = flow(cfg, TAUS, stable_tail=True)
    logp = out["log_success"]
    lb = logp[:, n_a]
    i = int(np.argmin(lb))
    never = bool(lb[i] >= math.log(0.5) - 1e-12)
    back = np.nonzero(lb[i:] > math.log(0.5))[0]
    if never:
        outcome = "never falls"
    elif len(back):
        outcome = "falls and recovers"
    else:
        outcome = "stays down"
    u, v = w_margins(out["W"], gamma)
    if gamma > 1 and outcome == "stays down":
        tail = float(v - u - log_rstar(gamma, m, 1, 1, W_MEAN, W_MEAN))
    else:
        tail = ""
    return {"gamma": float(gamma), "m": float(m), "n_A": cfg["counts"][0], "n_B": cfg["counts"][1],
            "R": mx.push_ratio(gamma, m), "first_step": "down" if lb[0] < math.log(0.5) else "up",
            "outcome": outcome, "log10_min_success": lb[i] / LN10, "tau_min": TAUS[i],
            "recovery_tau": TAUS[i + back[0]] if (len(back) and not never) else "",
            "log10_final_success": lb[-1] / LN10,
            # Largest one-grid-step decrease of log10 success over both prompts (0 when nothing ever decreases).
            "max_decrease_log10": float(max(0.0, -np.diff(logp[:, [0, n_a]], axis=0).min() / LN10)),
            # beta v - alpha u - log rho* at tau = 1e14 (> 0: beyond the separatrix of the tail system).
            "tail_margin_at_end": tail}


# ---------------------------------------------------------------- claim 1: the sufficient condition

def softmax_linear_checks(rng, draws=300):
    """Random softmax-linear policies with all <d(x, y), d(x', y')> >= 0 (d >= 0 entrywise): the gradient formula
    against central differences, kappa >= 0, and dp_x/dt >= 0 under each rule's weight."""
    rules = [("mean", 8, None), ("rloo", 8, None), ("grpo", 2, None), ("grpo", 8, None), ("grpo", "inf", None),
             ("rloo_k", 8, 2), ("rloo_k", 8, 4), ("passk", 8, 4)]
    worst_grad, worst_kappa, worst_dp = 0.0, np.inf, np.inf
    K, n, dim = 3, 6, 5
    for _ in range(draws):
        d = rng.exponential(size=(n, K, dim))           # d(x, y) >= 0 entrywise
        star = rng.integers(K, size=n)
        d[np.arange(n), star] = 0.0                      # d(x, y*) = 0
        phi = -d                                         # phi(x, y*) = 0, phi(x, y) = -d(x, y)
        theta = rng.normal(scale=3.0, size=dim)

        def success(th):
            z = phi @ th
            z = z - z.max(axis=1, keepdims=True)
            pi = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
            return pi, pi[np.arange(n), star]

        pi, p = success(theta)
        grad = p[:, None] * np.einsum("xy,xyd->xd", pi, d)
        h = 1e-6
        fd = np.stack([(success(theta + h * e)[1] - success(theta - h * e)[1]) / (2 * h) for e in np.eye(dim)], 1)
        worst_grad = max(worst_grad, float(np.abs(grad - fd).max() / np.abs(grad).max()))
        kappa = grad @ grad.T
        worst_kappa = min(worst_kappa, float(kappa.min()))
        for rule, G, k in rules:
            W = advantage_gap(p, G, rule, passk=k)
            worst_dp = min(worst_dp, float((kappa @ W).min()))
    return worst_grad, worst_kappa, worst_dp


def rule_weights_nonnegative():
    p = np.concatenate([np.logspace(-12, -1, 200), np.linspace(0.1, 0.9, 200), 1 - np.logspace(-1, -12, 200)])
    rules = [("mean", 2, None), ("mean", 8, None), ("rloo", 8, None), ("grpo", 2, None), ("grpo", 8, None),
             ("grpo", 32, None), ("grpo", "inf", None), ("rloo_k", 8, 2), ("rloo_k", 8, 4), ("rloo_k", 8, 8),
             ("passk", 8, 4)]
    return min(float(advantage_gap(p, G, r, passk=k).min()) for r, G, k in rules)


def nonneg_flow_monotone(rng):
    """One random K = 3 nonneg-interaction problem integrated under GRPO G = 8 to t = 1e8: largest decrease of any
    prompt's log success between output times."""
    K, n, dim = 3, 6, 5
    d = rng.exponential(size=(n, K, dim))
    star = rng.integers(K, size=n)
    d[np.arange(n), star] = 0.0
    phi = -d

    def rhs(s, th):
        z = phi @ th
        z = z - z.max(axis=1, keepdims=True)
        pi = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        p = pi[np.arange(n), star]
        W = advantage_gap(p, 8, "grpo")
        return math.exp(s) * ((W * p)[:, None] * np.einsum("xy,xyd->xd", pi, d)).mean(axis=0)

    ts = np.linspace(math.log(1e-3), math.log(1e8), 400)
    sol = solve_ivp(rhs, (ts[0], ts[-1]), np.zeros(dim), t_eval=ts, method="LSODA", rtol=1e-10, atol=1e-12)
    logp = []
    for th in sol.y.T:
        z = phi @ th
        logp.append((z - np.logaddexp.reduce(z, axis=1, keepdims=True))[np.arange(n), star])
    return float(max(0.0, -np.diff(np.array(logp), axis=0).min()))


def converse_k2():
    """K = 2, uniform start: prompt x (features b, answer 1) with n' copies of x' (features a, answer 0) falls iff
    n' > |d_x|^2 / |<d_x, d_x'>| = |b|^2 / gamma. Checked on the simulator's first step for n' around the threshold."""
    rows = []
    for gamma in (0.1, 0.3, 0.6, 1.5):
        thr = (gamma**2 + 0.25) / gamma
        for n_copies in sorted({max(1, math.floor(thr)), math.floor(thr) + 1, math.floor(thr) + 2}):
            cfg = mx.example(gamma, 0.9, rule="mean", G=8)
            cfg["counts"] = [n_copies, 1]
            lb = flow(cfg, [1e-3], stable_tail=True)["log_success"][0, n_copies]
            rows.append((gamma, thr, n_copies, bool(lb < math.log(0.5)), n_copies > thr))
    return rows


# ---------------------------------------------------------------- claim 3: tails

def local_exponents():
    """-d log h / dz at large z on the majority side (u) and the minority side (v), from advantage_gap itself."""
    def slopes(W_of_p, z1, z2):
        out = []
        for side in ("A", "B"):
            vals = []
            for z in (z1, z2):
                p = expit(z) if side == "A" else expit(-z)
                vals.append(math.log(float(W_of_p(np.array([p]))[0])) + float(log_expit(z) + log_expit(-z)))
            out.append(-(vals[1] - vals[0]) / (z2 - z1))
        return out
    rows = []
    for name, fn in [("mean G=8", lambda p: advantage_gap(p, 8, "mean")),
                     ("grpo G=2", lambda p: advantage_gap(p, 2, "grpo")),
                     ("grpo G=8", lambda p: advantage_gap(p, 8, "grpo")),
                     ("grpo G=inf, eps=0", lambda p: 1 / np.sqrt(p * (1 - p))),
                     ("rloo_k k=2", lambda p: advantage_gap(p, 8, "rloo_k", passk=2)),
                     ("rloo_k k=4", lambda p: advantage_gap(p, 8, "rloo_k", passk=4)),
                     ("rloo_k k=8", lambda p: advantage_gap(p, 8, "rloo_k", passk=8)),
                     ("supervised CE (W = 1/p)", lambda p: 1 / p)]:
        a30, b30 = slopes(fn, 25.0, 30.0)
        rows.append((name, a30, b30))
    # GRPO G = inf as coded (eps = 1e-5): 1/2 while p(1-p) >> eps, then 1.
    fn = lambda p: advantage_gap(p, "inf", "grpo")  # noqa: E731
    rows.append(("grpo G=inf, eps=1e-5, z in [3, 6]", *slopes(fn, 3.0, 6.0)))
    rows.append(("grpo G=inf, eps=1e-5, z in [25, 30]", *slopes(fn, 25.0, 30.0)))
    return rows


def slope_job(job):
    """Late slopes d log10 p / d log10 tau of the minority and of 1 - p_A over the last two decades before 1e14."""
    kind, gamma, rule, G, k = job
    cfg = mx.example(gamma, 0.9, rule=rule, G=G, passk=k)
    start = None
    if kind == "tail":                     # start beyond the separatrix of rule (rloo_k): deep in the tail
        al, be, CA, CB = tail_constants(rule, G, k)
        u0 = 24.0 / al                     # e^{alpha u0} = e^24 for every k
        start = start_weights(gamma, u0, (al * u0 + log_rstar(gamma, 0.9, al, be, CA, CB) + 2.0) / be)
    ends = []
    for tau in (1e12, 1e14):
        u, v = w_margins(flow(cfg, np.logspace(-5, math.log10(tau), 200), stable_tail=True, start=start)["W"], gamma)
        ends.append((float(log_expit(-v)) / LN10, float(log_expit(-u)) / LN10))   # log10 p_B, log10 (1 - p_A)
    return {"start": kind, "gamma": gamma, "rule": rule, "G": G, "k": k if k else "",
            "minority_slope": (ends[1][0] - ends[0][0]) / 2, "majority_slope": (ends[1][1] - ends[0][1]) / 2,
            "log10_final": ends[1][0]}


def logistic_flow(gamma, m, log_h, taus, beta_kl=0.0, kl="reverse", w0=(0.0, 0.0), z_ref=(0.0, 0.0)):
    """The direct logistic model in the simulator's flow time: dw/dtau = 2 [m F_A(u) a - (1 - m) F_B(s) b], with
    F(z) = h(z) = exp(log_h(z)) = W(sigma(z)) sigma(z) sigma(-z) minus the KL force to a fixed reference with margins
    z_ref = (u_ref, s_ref) (default: the uniform start): beta sigma(z) sigma(-z) (z - z_ref) for reverse KL and
    beta (sigma(z) - sigma(z_ref)) for forward KL. Starts at w0; returns u and s at each tau after the start."""
    a, b = np.array([1.0, 0.0]), np.array([gamma, 0.5])

    def force(z, zr):
        f = np.exp(log_h(z))
        if beta_kl > 0:
            ss = np.exp(log_expit(z) + log_expit(-z))
            f = f - beta_kl * (ss * (z - zr) if kl == "reverse" else expit(z) - expit(zr))
        return f

    def rhs(x, w):
        u, s = w @ a, -(w @ b)
        return math.exp(x) * 2 * (m * force(u, z_ref[0]) * a - (1 - m) * force(s, z_ref[1]) * b)

    xs = np.log(np.asarray(taus, dtype=float))
    tau0 = 1e-6
    w_start = np.asarray(w0, dtype=float) + tau0 * rhs(0.0, np.asarray(w0, dtype=float))
    sol = solve_ivp(rhs, (math.log(tau0), xs[-1]), w_start, t_eval=xs, method="LSODA", rtol=1e-10, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T @ a, -(sol.y.T @ b)


def anchored_after_collapse(gamma=1.2, depth=1e-6):
    """KL to a reference taken after the collapse: run the mean rule until the minority first reaches `depth`, then
    continue from there with a fixed KL to that state, to tau = 1e14 after the anchor. Returns rows."""
    taus = TAUS
    u, s = logistic_flow(gamma, 0.9, log_h_mean, taus)
    i = int(np.nonzero(log_expit(s) <= math.log(depth))[0][0])
    w0 = np.array([u[i], (-s[i] - gamma * u[i]) / 0.5])
    rows = []
    for kl in ("forward", "reverse"):
        for beta in (0.1, 1.0, 10.0):
            uu, ss = logistic_flow(gamma, 0.9, log_h_mean, taus, beta_kl=beta, kl=kl, w0=w0, z_ref=(u[i], s[i]))
            lb = log_expit(ss) / LN10
            # The minority's own one-prompt fixed point relative to its anchored reference (majority push ignored).
            p_ref = float(expit(s[i]))
            roots = [r for r, stable in fixed_points(p_ref, beta, 8, "mean", direction=kl, z_range=(-40.0, 40.0),
                                                     num=160001) if stable]
            rows.append((kl, beta, float(taus[i]), math.log10(p_ref), float(lb.max()), float(lb[-1]),
                         math.log10(roots[0]) if len(roots) == 1 else str(roots)))
    return rows


def log_h_mean(z):
    return math.log(W_MEAN) + log_expit(z) + log_expit(-z)


def separatrix_job(job):
    rule, G, k, gamma, u0, delta = job
    al, be, CA, CB = tail_constants(rule, G, k)
    lr = log_rstar(gamma, 0.9, al, be, CA, CB)
    v0 = (al * u0 + lr + delta) / be
    cfg = mx.example(gamma, 0.9, rule=rule, G=G, passk=k)
    res = flow(cfg, TAUS, stable_tail=True, start=start_weights(gamma, u0, v0))
    out = res["log_success"]
    u, v = w_margins(res["W"], gamma)
    back = bool((out[:, 9] > math.log(0.5)).any())
    return {"rule": rule if k is None else f"rloo_k k={k}", "gamma": gamma, "u0": u0, "v0": v0, "delta": delta,
            "end_margin": float(be * v - al * u - lr), "recovered": back,
            "log10_final": out[-1, 9] / LN10}


# ---------------------------------------------------------------- claim 4: levers

def switch_prediction(job):
    """At each switch point of switch.csv: beta v - k u - log rho*_k, the side of pass@k's separatrix."""
    gamma, rule, G, tau_s, k = job
    u, v = w_margins(flow(mx.example(gamma, 0.9, rule=rule, G=G), [tau_s], stable_tail=True)["W"], gamma)
    if gamma <= k:
        return ""
    return float(v - k * u - log_rstar(gamma, 0.9, k, 1.0, 1.0, 1.0))


def kl_job(job):
    """Fixed KL to the start, mean rule G = 8, gamma, m = 0.9, to tau = 1e14 in the direct model."""
    kl, beta, gamma = job
    taus = TAUS
    u, s = logistic_flow(gamma, 0.9, log_h_mean, taus, beta_kl=beta, kl=kl)
    lb = log_expit(s) / LN10
    i = int(np.argmin(lb))
    back = np.nonzero(lb[i:] > math.log10(0.5))[0]
    row = {"kl": kl, "beta": beta, "gamma": gamma, "log10_min": float(lb[i]), "tau_min": float(taus[i]),
           "u_at_min": float(u[i]), "recovery_tau": float(taus[i + back[0]]) if len(back) else "",
           "final_pA": float(expit(u[-1])), "final_pB": float(expit(s[-1]))}
    roots = [r for r, stable in fixed_points(0.5, beta, 8, "mean", direction=kl, z_range=(-40.0, 200.0), num=96001)
             if stable]
    row["fixed_point"] = roots[-1] if roots else ""
    if kl == "reverse" and len(back):
        # Bottom-dominated estimate of the time to climb out: e^{v} / (2 (1 - m) |b|^2 (W + beta v)) at the minimum.
        vmin = -s[i]
        row["climb_estimate"] = math.exp(vmin) / (2 * 0.1 * (gamma**2 + 0.25) * (W_MEAN + beta * vmin))
    else:
        row["climb_estimate"] = ""
    return row


def inner_solve_check():
    """The exact solution of one reset's inner problem, max_w sum_x share_x [F(p_x) - beta KL(pi_x || pi_old,x)] (F the
    rule's surrogate, F' = W), reached by integrating its gradient flow in w (not in the margins) to tau = 1e14,
    from a non-uniform state (u0 = 0.8, v0 = 1.1),
    against each prompt's own one-prompt root z - z0 = W(sigma(z)) / beta. (The flow gets there slowly once a prompt
    is nearly solved, since the KL curvature is p(1 - p); hence the long horizon.)"""
    worst, n = 0.0, 0
    for rule in ("mean", "grpo"):
        for gamma in (0.9, 1.2, 2.0):
            for m in (0.7, 0.9):
                for beta in (0.3, 1.0, 3.0):
                    a, b = np.array([1.0, 0.0]), np.array([gamma, 0.5])
                    w_old = np.array([0.8, 2.0 * (1.1 - gamma * 0.8)])
                    z0 = np.array([w_old @ a, -(w_old @ b)])

                    def rhs(x, w):          # gradient flow of the inner objective in log time x = log tau
                        z = np.array([w @ a, -(w @ b)])
                        p = expit(z)
                        f = (advantage_gap(p, 8, rule) - beta * (z - z0)) * p * (1 - p)
                        return math.exp(x) * 2 * (m * f[0] * a - (1 - m) * f[1] * b)

                    sol = solve_ivp(rhs, (math.log(1e-6), math.log(1e14)), w_old, method="LSODA", rtol=1e-12,
                                    atol=1e-12)
                    w = sol.y[:, -1]
                    gain = np.array([w @ a, -(w @ b)]) - z0
                    want = []
                    for zz in z0:               # the one-prompt root, by bisection
                        lo, hi = zz, zz + 50.0
                        for _ in range(200):
                            mid = 0.5 * (lo + hi)
                            if mid - zz - float(advantage_gap(np.array([expit(mid)]), 8, rule)[0]) / beta < 0:
                                lo = mid
                            else:
                                hi = mid
                        want.append(lo - zz)
                    worst = max(worst, float(np.abs(gain - np.array(want)).max()))
                    n += 1
    return n, worst


def reset_thresholds(path):
    """Per (gamma, rule, kl, beta): the largest reset_every with the minority still down at the end
    (log10 final < -1) and the smallest with it back (log10 final > -1), from resets.csv."""
    rows = list(csv.DictReader(open(path)))
    out = []
    for gamma in mx.RESET_GAMMAS:
        for rule, G in mx.RESET_RULES:
            for kl in ("reverse", "forward"):
                for beta in mx.BETAS:
                    sel = [r for r in rows if float(r["gamma"]) == gamma and r["rule"] == rule and r["kl"] == kl
                           and float(r["beta"]) == beta and r["reset_every"] != ""]
                    sel.sort(key=lambda r: int(r["reset_every"]))
                    down = [int(r["reset_every"]) for r in sel if float(r["log10_final_success"]) < -1]
                    up = [int(r["reset_every"]) for r in sel if float(r["log10_final_success"]) > -1]
                    monotone = not down or not up or max(down) < min(up)
                    out.append({"gamma": gamma, "rule": rule, "kl": kl, "beta": beta,
                                "last_down": max(down) if down else "", "first_up": min(up) if up else "",
                                "monotone": monotone})
    return out


def lambda_uniform(gamma, m=0.9):
    """2 x smallest eigenvalue of sum_x share_x p_x (1 - p_x) phi_x phi_x^T at the uniform start (p = 1/2)."""
    a, b = np.array([1.0, 0.0]), np.array([gamma, 0.5])
    M = 0.25 * (m * np.outer(a, a) + (1 - m) * np.outer(b, b))
    return 2 * float(np.linalg.eigvalsh(M)[0])


# ---------------------------------------------------------------- output

def write(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def g3(x):
    return f"{x:.3g}" if isinstance(x, float) else str(x)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(g3(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "extra", "analysis", "theory"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tables = os.path.join(args.out, "minimal_example")
    t0 = time.time()
    rng = np.random.default_rng(0)
    lines = ["# Checks of the minimal example", "",
             "Generated by `extra/experiments/check_minimal_example.py`. Exact flows use `exact_flow.flow(stable_tail=True)`; "
             "τ is the simulator's flow time (η × steps). Nothing is fitted. Pass/fail is stated for each check against "
             "the claim as written in `minimal_example.md`.", ""]

    with Pool(args.workers, initializer=mx.init_worker) as pool:
        phase_async = pool.map_async(phase_row, [(g, m) for m in PHASE_SHARES for g in PHASE_GAMMAS], chunksize=8)
        slope_jobs = ([("uniform", g, "mean", 8, None) for g in (1.2, 1.5, 2.0, 3.0)]
                      + [("uniform", g, "grpo", 8, None) for g in (1.5, 3.0)]
                      + [("uniform", g, "grpo", "inf", None) for g in (1.5, 3.0)]
                      + [("tail", g, "rloo_k", 8, 2) for g in (3.0, 5.0)]
                      + [("tail", g, "rloo_k", 8, 4) for g in (6.0, 8.0)])
        slopes_async = pool.map_async(slope_job, slope_jobs)
        sep_jobs = [(rule, G, k, g, u0, d) for rule, G, k, g, u0 in
                    [("mean", 8, None, 1.2, 15.0), ("mean", 8, None, 2.0, 15.0), ("grpo", 8, None, 1.5, 15.0),
                     ("rloo_k", 8, 2, 3.0, 10.0), ("rloo_k", 8, 4, 6.0, 8.0)]
                    for d in (-0.5, -0.1, 0.1, 0.5)]
        sep_async = pool.map_async(separatrix_job, sep_jobs)
        kl_async = pool.map_async(kl_job, [("reverse", b, 1.2) for b in (0.01, 0.03, 0.1, 0.3, 1.0)]
                                  + [("forward", b, 1.2) for b in (0.001, 0.01, 0.03, 0.1)])
        switch_rows = list(csv.DictReader(open(os.path.join(tables, "switch.csv"))))
        sw_jobs = [(float(r["gamma"]), r["first_rule"], int(r["first_G"]), float(r["tau_switch"]), int(r["k"]))
                   for r in switch_rows]
        sw_async = pool.map_async(switch_prediction, sw_jobs)

        # ---- claim 1
        lines += ["## Claim 1: non-negative interactions", ""]
        wmin = rule_weights_nonnegative()
        g_err, k_min, dp_min = softmax_linear_checks(rng)
        mono = nonneg_flow_monotone(rng)
        lines += [f"- Rule weights W(p) ≥ 0 on a grid p ∈ [1e-12, 1 − 1e-12] for mean (G = 2, 8), RLOO, GRPO "
                  f"(G = 2, 8, 32, ∞), unbiased pass@k (k = 2, 4, 8) and biased pass@4: smallest value {wmin:.3g}. "
                  f"**{'Pass' if wmin >= 0 else 'Fail'}.**",
                  f"- 300 random softmax-linear policies (K = 3, 6 prompts, 5 parameters, d(x, y) ≥ 0 entrywise, "
                  f"θ ~ N(0, 9)): ∇p_x = p_x Σ_y π(y|x) d(x, y) against central differences, largest relative error "
                  f"{g_err:.1e}; smallest κ(x, x′) {k_min:.3g}; smallest dp_x/dt under the 8 rules {dp_min:.3g}. "
                  f"**{'Pass' if g_err < 1e-6 and k_min >= 0 and dp_min >= 0 else 'Fail'}.**",
                  f"- One such problem integrated under GRPO (G = 8) to t = 1e8: largest decrease of any prompt's log "
                  f"success between 400 output times {mono:.1e}. **{'Pass' if mono < 1e-9 else 'Fail'}.**"]
        conv = converse_k2()
        ok = all(fell == pred for *_, fell, pred in conv)
        lines += [f"- K = 2 converse at the uniform start (simulator's first step, one prompt of features b against "
                  f"n′ copies of a): falls exactly when n′ > |b|²/γ in {sum(f == p for *_, f, p in conv)} of {len(conv)} "
                  f"cases (γ ∈ {{0.1, 0.3, 0.6, 1.5}}, n′ around the threshold). **{'Pass' if ok else 'Fail'}.**", ""]

        phase = phase_async.get()
        write(os.path.join(tables, "phase.csv"), phase)
        nonpos = [r for r in phase if r["gamma"] <= 0]
        lines += [f"- Phase table, γ ≤ 0 ({len(nonpos)} runs, m from 0.50 to 0.95, mean rule to τ = 1e14): outcome "
                  f"'never falls' in {sum(r['outcome'] == 'never falls' for r in nonpos)}; largest one-step decrease of "
                  f"log10 success of either prompt {max(r['max_decrease_log10'] for r in nonpos):.1e}.", ""]

        # ---- claim 2
        lines += ["## Claim 2: the first step", ""]
        edge = [r for r in phase if abs(r["R"] - 1) < 1e-9]          # R = 1 up to rounding: first order is 0
        off = [r for r in phase if abs(r["R"] - 1) >= 1e-9]
        first_ok = sum((r["first_step"] == "down") == (r["R"] > 1) for r in off)
        never_ok = sum((r["outcome"] == "never falls") == (r["R"] <= 1) for r in off)
        lines += [f"- Sign of the minority's change at τ = 1e-5 against R > 1: {first_ok} of {len(off)} phase runs with "
                  f"R ≠ 1 agree. **{'Pass' if first_ok == len(off) else 'Fail'}.** "
                  f"{len(edge)} run(s) have R = 1 to rounding ((γ, m) = "
                  + ", ".join(f"({r['gamma']}, {r['m']}): first step {r['first_step']}, {r['outcome']}" for r in edge)
                  + "); there the first-order change is zero and the sign is not predicted.",
                  f"- 'Never falls' (minimum ≥ 0.5 up to 1e14) against R ≤ 1: {never_ok} of {len(off)} runs with R ≠ 1 "
                  "agree: on this grid a minority that does not start to fall never falls later.",
                  "- R = c·m/(1 − m) with c = k_BA/k_BB = γ/|b|² is the odds law for K = 2 (identity; SymPy test "
                  "`test_first_step_push_ratio`; the simulator's first-step ratio is tested in "
                  "`extra/tests/test_minimal_example.py::test_first_step_ratio`).", ""]

        # ---- claim 3
        lines += ["## Claim 3: tails", "", "Local tail exponents −d log h/dz between z = 25 and z = 30 (from the "
                  "code's weights):", ""]
        lines += [md_table(["rule", "α (majority, u)", "β (minority, v)"], local_exponents()), ""]
        lines += ["Saddle (SymPy, `MinimalExampleTests`): the fixed point of the tail system in log time has "
                  "eigenvalues −1 (along the ray, a time shift) and (βg − α|a|²)(β|b|² − αg)/(αβ|a|²|b_⊥|²) > 0 on the "
                  "branch, and Y/X obeys dρ/dt = [mC_A(βg − α|a|²)ρ − (1 − m)C_B(β|b|² − αg)]/X, so the stable "
                  "manifold is the ray ρ = ρ* = ζ*/ξ* exactly: the separatrix is the line βv − αu = log ρ*.", ""]
        sl = slopes_async.get()
        rows = []
        for r in sl:
            al = tail_constants(r["rule"], r["G"], r["k"] or None)[0]
            rows.append((r["start"], r["rule"], r["G"], r["k"], r["gamma"], -r["gamma"] / al, r["minority_slope"],
                         -1 / al, r["majority_slope"], r["log10_final"]))
        lines += ["Late slopes over τ ∈ [1e12, 1e14] (d log10 p_B/d log10 τ for the minority, d log10(1 − p_A)/d log10 τ "
                  "for the majority). 'uniform' starts at W = 0; 'tail' starts at αu = 24 beyond pass@k's separatrix "
                  "(βv − αu = log ρ* + 2). GRPO at G = ∞ runs with the code's ε = 1e-5, which makes α = β = 1 deep in "
                  "the tail (prediction −γ, not −2γ):", ""]
        lines += [md_table(["start", "rule", "G", "k", "γ", "predicted −γ/α", "minority slope", "predicted −1/α",
                            "majority slope", "log10 p_B at 1e14"], rows), ""]
        # GRPO G = inf without eps, direct model.
        eps0 = []
        for g in (1.5, 3.0):
            taus = np.logspace(-5, 14, 761)
            u, s = logistic_flow(g, 0.9, lambda z: 0.5 * (log_expit(z) + log_expit(-z)), taus)
            lb = log_expit(s) / LN10
            i = np.searchsorted(taus, 1e12)
            eps0.append((g, -2 * g, (lb[-1] - lb[i]) / 2.0, float(lb[-1])))
        lines += ["GRPO at G = ∞ without ε (direct logistic model, W = 1/√(p(1 − p)), α = β = 1/2):", "",
                  md_table(["γ", "predicted −γ/α = −2γ", "minority slope", "log10 p_B at 1e14"], eps0), ""]
        sep = sep_async.get()
        ok = sum((r["delta"] > 0) == (r["end_margin"] > r["delta"] and not r["recovered"]) for r in sep)
        lines += ["Separatrix against the exact flow (m = 0.9): start at u0 with βv0 − αu0 = log ρ* + δ and integrate to "
                  "τ = 1e14 after the start. Predicted: δ > 0 moves away upward (stays down), δ < 0 recovers.", "",
                  md_table(["rule", "γ", "u0", "δ", "βv − αu − log ρ* at 1e14", "back above 0.5", "log10 p_B at 1e14"],
                           [(r["rule"], r["gamma"], r["u0"], r["delta"], r["end_margin"], r["recovered"],
                             r["log10_final"]) for r in sep]), "",
                  f"**{ok} of {len(sep)} as predicted.**", ""]
        down = [r for r in phase if r["outcome"] == "stays down"]
        beyond = [r for r in down if r["tail_margin_at_end"] != "" and r["tail_margin_at_end"] > 0]
        low = [r for r in down if r["gamma"] <= 1]
        rest = [r for r in down if r["gamma"] > 1 and not (r["tail_margin_at_end"] != "" and r["tail_margin_at_end"] > 0)]
        lines += [f"Phase table: {len(down)} runs stay down to 1e14. {len(beyond)} of them are beyond the separatrix at "
                  f"1e14 (v − u > log ρ*; the tail system keeps them down for ever). {len(low)} have γ ≤ 1 ("
                  + ", ".join(f"γ = {r['gamma']}, m = {r['m']}" for r in low)
                  + "), where there is no collapsed branch (at γ = 1, dρ/dt < 0 for every ρ: a recovery slower than "
                  "any power of τ); "
                  + ", ".join(f"γ = {r['gamma']}, m = {r['m']} (margin {r['tail_margin_at_end']:.2f})" for r in rest)
                  + " is on the recovering side. These are predicted to recover after 1e14.", ""]

        lines += ["Phase table by m (γ grid step 0.05 on [−0.5, 3]): the γ range that falls at all, and the γ range "
                  "that stays down to 1e14:", ""]
        rows = []
        for m in PHASE_SHARES:
            sel = [r for r in phase if r["m"] == m]
            sd = [r["gamma"] for r in sel if r["outcome"] == "stays down"]
            fell = [r["gamma"] for r in sel if r["outcome"] != "never falls"]
            rows.append((m, f"{min(fell)}–{max(fell)}" if fell else "none",
                         f"{min(sd)}–{max(sd)}" if sd else "none", len(sd)))
        pick = [r for r in rows if round(r[0] * 100) % 5 == 0 or r[0] in (0.66, 0.67, 0.68, 0.76)]
        lines += [md_table(["m", "γ that falls", "γ that stays down", "runs down"], pick), ""]

        # ---- claim 4
        lines += ["## Claim 4: levers", ""]
        passk = list(csv.DictReader(open(os.path.join(tables, "passk.csv"))))
        lines += ["pass@k from the uniform start (`passk.csv`, m = 0.9, τ = 1e14): stays down (final < 1e-3) against "
                  "γ > k:", ""]
        lines += [md_table(["k", "γ", "γ > k", "log10 min", "stays down"],
                           [(int(r["k"]), float(r["gamma"]), float(r["gamma"]) > int(r["k"]),
                             float(r["log10_min_success"]), float(r["log10_final_success"]) < -3) for r in passk]), ""]
        swp = sw_async.get()
        agree = sum((r["recovered"] == "no") == (float(r["gamma"]) > int(r["k"])) for r in switch_rows)
        lines += [f"Switch to pass@k after a collapse (`switch.csv`): 'stays down' equals γ > k in {agree} of "
                  f"{len(switch_rows)} rows.",
                  "Side of pass@k's separatrix at the switch (v − ku − log ρ*_k, rows with γ > k): "
                  + ", ".join(f"γ = {r['gamma']}, k = {r['k']}, depth {r['target_depth']}: {g3(p)}"
                              for r, p in zip(switch_rows, swp) if p != "" and r["first_rule"] == "mean") + ".", ""]
        kl = kl_async.get()
        lines += ["Fixed KL to the start (direct logistic model, mean rule, γ = 1.2, m = 0.9, to τ = 1e14; the "
                  "one-prompt fixed point from `theory.fixed_points`; the majority's fixed margin is u* = W/β for "
                  "reverse KL):", "",
                  md_table(["KL", "β", "log10 min p_B", "τ at min", "u at min", "u* (reverse)", "recovery τ",
                            "climb estimate", "(recovery τ − τ at min) / estimate", "final p_A", "final p_B",
                            "fixed point"],
                           [(r["kl"], r["beta"], r["log10_min"], r["tau_min"], r["u_at_min"],
                             W_MEAN / r["beta"] if r["kl"] == "reverse" else "", r["recovery_tau"], r["climb_estimate"],
                             (r["recovery_tau"] - r["tau_min"]) / r["climb_estimate"] if r["climb_estimate"] != "" else "",
                             r["final_pA"], r["final_pB"], r["fixed_point"]) for r in kl]), "",
                  "'climb estimate' is e^{v_min}/(2(1 − m)|b|²(W + βv_min)), the time to climb out if the majority no "
                  "longer pushes and the minority's pull is p_B(W + β log(1/p_B)) at the bottom.", ""]
        # validation of the direct model
        taus = np.logspace(-2, 9, 23)
        u, s = logistic_flow(1.2, 0.9, log_h_mean, taus)
        ref = flow(mx.example(1.2, 0.9, rule="mean", G=8), taus, stable_tail=True)["log_success"][:, 9]
        diff0 = float(np.abs(log_expit(s) - ref).max() / LN10)
        res = mx.batched_expected_sgd(1.2, 0.9, "mean", 8, [("reverse", 0.1, None), ("forward", 0.1, None)], 0.01,
                                      3000, record_every=3000)
        diffs = []
        for j, klname in enumerate(("reverse", "forward")):
            _, s = logistic_flow(1.2, 0.9, log_h_mean, [30.0], beta_kl=0.1, kl=klname)
            diffs.append(abs(float(log_expit(s[-1]) / LN10) - float(res["trace"][j, -1])))
        anch = anchored_after_collapse()
        lines += ["KL anchored after the collapse (mean rule, γ = 1.2, m = 0.9; reference = the state where the minority "
                  "first reaches 1e-6; fixed KL to it from there, to τ = 1e14 after the anchor):", "",
                  md_table(["KL", "β", "τ of the anchor", "log10 p_B at the anchor", "highest log10 p_B after",
                            "log10 p_B at 1e14", "log10 of the one-prompt fixed point for p_ref = p_B at the anchor"],
                           anch), "",
                  "For the mean rule (W = 0.875) the one-prompt fixed point is σ(logit p_ref + W/β) for reverse KL; for "
                  "forward KL it stays near p_ref·β/(β − W) when β > W and is near 1 − β/W when β < W.", ""]
        lines += [f"Direct model validated: against `exact_flow` (no KL, τ up to 1e9) largest |Δ log10 p_B| "
                  f"{diff0:.1e}; against the driver's batched exact SGD at η = 0.01 (β = 0.1, τ = 30) "
                  f"{diffs[0]:.1e} (reverse) and {diffs[1]:.1e} (forward).", ""]
        n_inner, inner = inner_solve_check()
        lines += [f"Resets, the exact inner step: the end point of the gradient flow in w of the rule's surrogate − β KL(π‖π_old) (to τ = 1e14) "
                  f"from a non-uniform state, against each prompt's own root z − z0 = W(σ(z))/β: largest |gain error| over "
                  f"{n_inner} cases (mean and GRPO G = 8; γ ∈ {{0.9, 1.2, 2}}; m ∈ {{0.7, 0.9}}; β ∈ {{0.3, 1, 3}}) "
                  f"{inner:.1e}. **{'Pass' if inner < 1e-6 else 'Fail'}.**", ""]
        thr = reset_thresholds(os.path.join(tables, "resets.csv"))
        rows = []
        for r in thr:
            lam = lambda_uniform(r["gamma"])
            pred = 1 / (mx.RESET_ETA * r["beta"] * lam)
            lo, hi = r["last_down"], r["first_up"]
            rows.append((r["gamma"], r["rule"], r["kl"], r["beta"], lo, hi, pred,
                         (mx.RESET_ETA * r["beta"] * lam * lo) if lo != "" and hi != "" else "",
                         (mx.RESET_ETA * r["beta"] * lam * hi) if lo != "" and hi != "" else "", r["monotone"]))
        lines += ["Reset thresholds from `resets.csv` (η = 0.5, 60,000 steps, τ = 3e4; 'down' = final log10 p_B < −1). "
                  "λ = 2 λ_min(Σ_x share_x p_x(1 − p_x) φ_x φ_xᵀ) at the uniform start "
                  f"(γ = 0.9: {lambda_uniform(0.9):.4f}; γ = 1.2: {lambda_uniform(1.2):.4f}). The guess N* ≈ 1/(ηβλ) "
                  "holds if ηβλN* ≈ 1 inside the bracket:", "",
                  md_table(["γ", "rule", "KL", "β", "largest N down", "smallest N up", "1/(ηβλ)", "ηβλ·(N down)",
                            "ηβλ·(N up)", "monotone in N"], rows), ""]

    lines += [f"Run time {time.time() - t0:.0f} s."]
    with open(os.path.join(args.out, "minimal_example_check.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
