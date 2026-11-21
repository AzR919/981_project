"""
Write the job lists for each experiment batch.

usage:
    python experiments/grids.py base > jobs_base.jsonl

Each line is one config for experiments/run.py; its "name" is
<hypothesis>/<index>, and the full config is saved inside the result.
"""

import itertools
import json
import sys

# Sharing levels, from no shared parameters to a wide MLP. Correlated
# features keep rule answers so every level is realizable.
SHARING = {
    "tab": {"features": "onehot"},
    "lin": {"arch": "linear"},
    "c05": {"arch": "linear", "features": "correlated", "rho": 0.5},
    "c09": {"arch": "linear", "features": "correlated", "rho": 0.9},
    "m64": {"arch": "mlp", "width": 64},
    "m256": {"arch": "mlp", "width": 256},
}
ETAS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
SEEDS = [0, 1, 2]


def rules_for(G):
    """The rules H1 and H2 compare; pass@k needs k <= G."""
    rules = [("mean", None), ("rloo", None), ("grpo", None), ("maxrl", None)]
    rules += [("passk", k) for k in (2, 4) if k <= G]
    return rules


def h1():
    for L, level, init, G in itertools.product((1, 2, 4), ("tab", "lin", "m64"), ("uniform", "warm"), (2, 4, 8, 32)):
        for rule, k in rules_for(G):
            cfg = dict(SHARING[level], mode="gradcheck", L=L, G=G, rule=rule, passk=k, R=4000, probe=32, level=level)
            if init == "warm":
                cfg.update(init="tabular", p0="spread") if level == "tab" else cfg.update(init="sft_up", p0=0.3)
            yield cfg


def h2():
    for level, G, seed in itertools.product(("tab", "lin", "m64"), (2, 4, 16, 64), SEEDS):
        warm = dict(init="tabular", p0="spread") if level == "tab" else dict(init="sft_up", p0=0.4)
        for rule, k in rules_for(G):
            base = dict(SHARING[level], **warm, L=1, G=G, rule=rule, passk=k, seed=seed, level=level)
            yield dict(base, mode="onestep", eta=0.5, R=1000 if G == 64 else 2000)
            # A short window of ordinary training in the drift regime.
            eta = {"tab": 3.0, "lin": 0.3, "m64": 0.03}[level]
            yield dict(base, mode="train", eta=eta, B=32, steps=200, log_every=20, snapshot_every=20)


def h3():
    pairs = [(128, 2), (64, 4), (32, 8), (16, 16), (4, 64), (8, 8), (128, 8)]
    for level, L, (B, G), rule, eta, seed in itertools.product(
        SHARING, (1, 2, 4), pairs, ("mean", "rloo", "grpo"), ETAS, SEEDS
    ):
        yield dict(SHARING[level], L=L, B=B, G=G, rule=rule, eta=eta, seed=seed, steps=5000, log_every=50, level=level)


def h4():
    # (a) where training settles: 64 identical tabular prompts per run.
    for p_ref, beta, G, rule, kl, seed in itertools.product(
        (0.01, 0.05, 0.2, 0.5), (0.5, 1, 2, 4), (2, 4, 8, 16, 64), ("mean", "grpo"), ("reverse", "forward"), SEEDS
    ):
        yield dict(part="a", features="onehot", n_train=64, init="tabular", p0=p_ref, L=1, G=G, B=64,
                   rule=rule, eta=1.0, beta=beta, kl=kl, seed=seed, steps=6000, log_every=100,
                   snapshot_every=1000, diagnostics=False, level="tab")
    # (b) fluctuations around the fixed point against eta / beta.
    for beta, eta, kl, seed in itertools.product((0.5, 1, 2, 4), (0.1, 0.3, 1, 3, 10, 30), ("reverse", "forward"), SEEDS):
        yield dict(part="b", features="onehot", n_train=64, init="tabular", p0=0.05, L=1, G=8, B=64,
                   rule="grpo", eta=eta, beta=beta, kl=kl, seed=seed, steps=6000, log_every=100,
                   snapshot_every=100, diagnostics=False, level="tab")
    # (c) KL and the stability threshold (beta = 0 comes from H3).
    for level, eta, beta, kl, rule, seed in itertools.product(
        ("c05", "m64"), ETAS, (0.1, 0.5, 2), ("reverse", "forward"), ("mean", "grpo"), SEEDS
    ):
        yield dict(SHARING[level], part="c", L=2, G=8, B=32, rule=rule, eta=eta, beta=beta, kl=kl,
                   seed=seed, steps=5000, log_every=50, level=level)


def h5():
    for level, p0, G, rule, eta, seed in itertools.product(
        ("tab", "lin", "m64"), (0.01, 0.03, 0.1, 0.3), (2, 4, 8, 16, 32, 64, 128), ("mean", "grpo"),
        (0.01, 0.03, 0.1, 0.3, 1, 3, 10), SEEDS
    ):
        if level == "tab":
            init = dict(features="onehot", n_train=64, init="tabular", p0=p0)
        else:
            init = dict(SHARING[level], init="sft_down", p0=p0)
        yield dict(init, L=1, G=G, B=256 // G, rule=rule, eta=eta, seed=seed, steps=5000, log_every=25,
                   snapshot_every=1000, level=level)


# ---------------------------------------------------------------- stretch

def around(centre, decades=1.5, per_decade=4):
    """Log-spaced eta values within +-decades of centre."""
    k = int(round(decades * per_decade))
    return [float(f"{centre * 10 ** (i / per_decade):.4g}") for i in range(-k, k + 1)]


# Base-batch absorption thresholds (B = 32, G = 8, GRPO) used to centre the eta grids.
BASE_ETA_STAR = {
    ("lin", 1): 1000, ("lin", 2): 1000, ("c05", 1): 10, ("c05", 2): 30, ("c09", 1): 10, ("c09", 2): 0.01,
    ("m64", 1): 3, ("m64", 2): 1, ("m256", 1): 3, ("m256", 2): 1,
}
STRETCH_SEEDS = [0, 1, 2, 3, 4]


def s_h1():
    """H1 again with 5x the samples and chunked statistics for the calibrated tests."""
    for cfg in h1():
        yield dict(cfg, R=20000, chunks=20)


def s_drift():
    """Drift against noise: the same runs with sampled, expected and full expected updates."""
    common = dict(L=2, G=8, rule="grpo", steps=5000, log_every=5, snapshot_every=1000)
    for level in ("lin", "c05", "c09", "m64"):
        etas = around(BASE_ETA_STAR[(level, 2)])
        for eta, seed in itertools.product(etas, STRETCH_SEEDS):
            base = dict(SHARING[level], **common, eta=eta, seed=seed, level=level)
            for update in ("sampled", "expected", "expected_full"):
                yield dict(base, block="main", update=update, B=32)
            for update, B in itertools.product(("sampled", "expected"), (8, 128)):
                yield dict(base, block="batch", update=update, B=B)
        for eta, seed in itertools.product(around(10 ** -2.5), STRETCH_SEEDS):
            for update in ("sampled", "expected_full"):
                yield dict(SHARING[level], **common, block="adam", optimizer="adam", update=update, B=32,
                           eta=eta, seed=seed, level=level)


def s_pred():
    """Per-prompt absorption predictor, with spread starting points and drift checkpoints."""
    for level, L, rule, init, seed in itertools.product(
        ("lin", "c05", "c09", "m64", "m256"), (1, 2), ("grpo", "mean"), ("sft_mixed", "uniform"), STRETCH_SEEDS
    ):
        for f in (0.3, 1, 3, 10):
            yield dict(SHARING[level], L=L, rule=rule, init=init, eta=BASE_ETA_STAR[(level, L)] * f, B=32, G=8,
                       seed=seed, steps=5000, log_every=25, snapshot_every=500, drift_every=500, level=level)


ABLATIONS = {
    "control": {},
    "flip05": {"reward_noise": 0.05},
    "flip20": {"reward_noise": 0.2},
    "momentum": {"momentum": 0.9},
    "kl01": {"beta": 0.1},
    "kl1": {"beta": 1.0},
    "sign": {"rule": "sign"},
}


def s_abl():
    """Break one condition of the tabular monotonicity proposition at a time."""
    common = dict(features="onehot", init="tabular", p0="spread", G=8, B=32, rule="grpo", steps=5000,
                  log_every=50, diagnostics=False, level="tab")
    for L, G, seed in itertools.product((1, 2, 4), (4, 8), SEEDS):
        for name, change in ABLATIONS.items():
            for eta in (0.3, 3, 30, 300):
                yield dict(common, **change, ablation=name, L=L, G=G, eta=eta, seed=seed)
        for eta in (1e-3, 1e-2, 1e-1):
            yield dict(common, ablation="adam", optimizer="adam", L=L, G=G, eta=eta, seed=seed)


def s_eq():
    """Bistable KL fixed points: start on both sides of the unstable root."""
    import math
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from synthetic_rl.theory import fixed_points

    common = dict(features="onehot", n_train=64, init="tabular", L=1, G=64, B=64, kl="reverse", steps=10000,
                  log_every=100, snapshot_every=100, diagnostics=False, level="tab")
    for p_ref, beta in ((0.01, 0.5), (0.03, 0.5), (0.2, 1.0)):
        roots = fixed_points(p_ref, beta, 64, "grpo")
        unstable = [p for p, stable in roots if not stable][0]
        z_u = math.log(unstable / (1 - unstable))
        for offset in (-3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3):
            p_start = 1 / (1 + math.exp(-(z_u + offset)))
            for seed in SEEDS:
                for eta in (0.1, 0.3, 1, 3, 10):
                    yield dict(common, p0=p_ref, beta=beta, rule="grpo", p_start=p_start, offset=offset,
                               eta=eta, seed=seed)
                yield dict(common, p0=p_ref, beta=beta, rule="mean", p_start=p_start, offset=offset,
                           eta=1.0, seed=seed)


# ---------------------------------------------------------------- v3 batch 1 (plan v3, T1 T2 T4 T6 T8)

TASK_SEEDS = list(range(10))
MAJORITY = [1 / 3, 0.5, 0.7, 0.9]


def flow_steps(eta, flow=100.0, low=1000, high=20000):
    """Steps for a fixed flow time eta * t, clipped to a practical range."""
    return int(min(max(flow / eta, low), high))


def v3_t1():
    """P1: absorption against majority share at fixed feature geometry."""
    for arch, rho, m, task_seed in itertools.product(("linear", "mlp"), (0.5, 0.9), MAJORITY, TASK_SEEDS):
        base = dict(arch=arch, width=64, features="correlated", rho=rho, majority=m, L=1, K=3, G=8, B=32,
                    rule="grpo", task_seed=task_seed, seed=0, track_every=5, diagnostics=False,
                    level=f"{arch}-rho{rho}")
        for eta in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30):
            steps = flow_steps(eta)
            for update in ("expected_full", "sampled"):
                yield dict(base, update=update, eta=eta, steps=steps, log_every=max(steps // 20, 1),
                           snapshot_every=steps, optimizer="sgd")
        for eta in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
            for update in ("expected_full", "sampled"):
                yield dict(base, update=update, eta=eta, steps=5000, log_every=250, snapshot_every=5000,
                           optimizer="adam")
        yield dict(base, update="supervised", eta=1.0, steps=2000, log_every=100, snapshot_every=2000,
                   optimizer="sgd")


def v3_t2():
    """P2: per-kick size eta/B against response noise, at fixed eta (equal mean drift)."""
    cells = [("lin", dict(SHARING["lin"], L=2), (300, 1000)),
             ("c05", dict(SHARING["c05"], L=2), (10, 30)),
             ("m64", dict(SHARING["m64"], L=2), (0.3, 1)),
             ("bal", dict(arch="linear", features="correlated", rho=0.9, majority=1 / 3, L=1, K=3), (3, 10))]
    for (level, setup, etas), eta, task_seed in itertools.product(cells, (0, 1), TASK_SEEDS):
        base = dict(setup, G=8, rule="grpo", eta=etas[eta], task_seed=task_seed, seed=0, steps=3000, log_every=50,
                    snapshot_every=3000, track_every=5, diagnostics=False, level=level)
        yield dict(base, update="expected_full", B=32)
        for update, B in itertools.product(("expected", "sampled"), (8, 32, 128, 512)):
            yield dict(base, update=update, B=B, replace=B < 512)


def v3_t4():
    """H5 rerun: wider eta grid, exact step to mean p = 0.5, 5 seeds."""
    for level, p0, G, rule, eta, seed in itertools.product(
        ("tab", "lin", "m64"), (0.01, 0.03, 0.1, 0.3), (2, 4, 8, 16, 32, 64, 128), ("mean", "grpo"),
        (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000), range(5)
    ):
        init = (dict(features="onehot", n_train=64, init="tabular", p0=p0) if level == "tab"
                else dict(SHARING[level], init="sft_down", p0=p0))
        yield dict(init, L=1, G=G, B=256 // G, rule=rule, eta=eta, seed=seed, task_seed=seed, steps=5000,
                   log_every=250, snapshot_every=5000, track_every=1, diagnostics=False, level=level)


T6_CONDITIONS = {
    "control": {}, "flip05": {"reward_noise": 0.05}, "flip20": {"reward_noise": 0.2},
    "momentum": {"momentum": 0.9}, "kl01": {"beta": 0.1}, "kl1": {"beta": 1.0}, "sign": {"rule": "sign"},
    "ppo4": {"ppo_epochs": 4, "clip_eps": 0.2}, "multi4": {"ppo_epochs": 4}, "gradclip": {"grad_clip": 1.0},
}


def v3_t6():
    """Ablations of the tabular proposition, including realistic-training conditions."""
    common = dict(features="onehot", init="tabular", p0="spread", B=32, rule="grpo", steps=5000, log_every=50,
                  snapshot_every=5000, track_every=5, diagnostics=False, level="tab")
    for L, G, seed in itertools.product((1, 2, 4), (4, 8), range(10)):
        for name, change in T6_CONDITIONS.items():
            for eta in (0.3, 3, 30, 300):
                yield dict(common, **change, ablation=name, L=L, G=G, eta=eta, seed=seed)
        for eta in (1e-3, 1e-2, 1e-1):
            yield dict(common, ablation="adam", optimizer="adam", L=L, G=G, eta=eta, seed=seed)


def v3_t8():
    """KL bistability without noise: the exact expected update with the KL term."""
    import math
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from synthetic_rl.theory import fixed_points

    common = dict(features="onehot", n_train=64, init="tabular", L=1, G=64, B=64, kl="reverse", rule="grpo",
                  steps=10000, log_every=100, snapshot_every=100, diagnostics=False, level="tab",
                  update="expected_full")
    for p_ref, beta in ((0.01, 0.5), (0.03, 0.5), (0.2, 1.0)):
        unstable = [p for p, stable in fixed_points(p_ref, beta, 64, "grpo") if not stable][0]
        z_u = math.log(unstable / (1 - unstable))
        for offset in (-3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3):
            for eta in (1, 3, 10):
                yield dict(common, p0=p_ref, beta=beta, p_start=1 / (1 + math.exp(-(z_u + offset))),
                           offset=offset, eta=eta, seed=0)


BATCHES = {
    "base": [("h1", h1), ("h2", h2), ("h3", h3), ("h4", h4), ("h5", h5)],
    "stretch": [("s_h1", s_h1), ("s_drift", s_drift), ("s_pred", s_pred), ("s_abl", s_abl), ("s_eq", s_eq)],
    "v3a": [("v3_t1", v3_t1), ("v3_t2", v3_t2), ("v3_t4", v3_t4), ("v3_t6", v3_t6), ("v3_t8", v3_t8)],
}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "base"
    for prefix, grid in BATCHES[tag]:
        for i, cfg in enumerate(grid()):
            cfg = {k: v for k, v in cfg.items() if v is not None}
            print(json.dumps(dict(cfg, name=f"{prefix}/{i:05d}", hypothesis=prefix)))


if __name__ == "__main__":
    main()
