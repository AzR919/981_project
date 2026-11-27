"""
Write the job lists for each experiment batch.

usage:
    python experiments/grids.py base > jobs_base.jsonl

Each line is one config for experiments/run.py; its "name" is
<hypothesis>/<index>, and the full config is saved inside the result.
"""

import itertools
import json
import os
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


T9_RULES = ("grpo", "mean", "heckel", "virtual", "pm_alpha", "nsr")


# T5 warm-starts chosen (before launch) so that p_init has sd >= 0.25 on task seeds 0-2.
T5_SFT = {"lin": dict(sft_steps=1000, sft_lr=8.0), "c05": dict(sft_steps=1000, sft_lr=2.0),
          "c09": dict(sft_steps=500, sft_lr=1.0), "m64": dict(sft_steps=200, sft_lr=0.5)}


def v3_t5():
    """P3: does the initial drift predict absorbed prompts on balanced tasks, beyond rarity and p_init?"""
    for level, task_seed in itertools.product(("lin", "c05", "c09", "m64"), TASK_SEEDS):
        for f in (0.3, 1, 3, 10, 30):
            yield dict(SHARING[level], **T5_SFT[level], majority=1 / 3, L=1, K=3, init="sft_mixed", rule="grpo",
                       G=8, B=32,
                       eta=BASE_ETA_STAR[(level, 1)] * f, task_seed=task_seed, seed=0, steps=5000, log_every=250,
                       snapshot_every=500, drift_every=500, track_every=1, diagnostics=False, level=level)


def v3_t7():
    """H1 medium tier: bootstrap null for mean z^2 (the identity is exact, so ranks should be uniform)."""
    for cfg in h1():
        if cfg["rule"] == "grpo" and cfg["L"] in (1, 2) and cfg["G"] in (4, 8):
            for seed in range(10):
                yield dict(cfg, boot=200, seed=seed)


def v3_t9():
    """All-wrong-group fixes on the absorbing configurations of T1 (minority), T2 (kick) and T6 (noisy verifier)."""
    for rule, task_seed in itertools.product(T9_RULES, TASK_SEEDS):
        minority = dict(arch="linear", width=64, features="correlated", rho=0.9, majority=0.9, L=1, K=3, G=8,
                        B=32, rule=rule, task_seed=task_seed, seed=0, track_every=5, diagnostics=False,
                        optimizer="sgd", level="t1-minority")
        for eta, update in itertools.product((0.1, 1), ("expected_full", "sampled")):
            steps = flow_steps(eta)
            yield dict(minority, eta=eta, update=update, steps=steps, log_every=max(steps // 20, 1),
                       snapshot_every=steps)
        kick = dict(SHARING["lin"], L=2, G=8, B=8, rule=rule, eta=300, task_seed=task_seed, seed=0, steps=3000,
                    log_every=50, snapshot_every=3000, track_every=5, diagnostics=False, level="t2-kick")
        for update in ("expected", "sampled"):
            yield dict(kick, update=update)
        for eta in (30, 300):
            yield dict(features="onehot", init="tabular", p0="spread", L=1, G=8, B=32, rule=rule, eta=eta,
                       reward_noise=0.2, seed=task_seed, steps=5000, log_every=50, snapshot_every=5000,
                       track_every=5, diagnostics=False, level="t6-flip20")


def v3_t1_seed1():
    """Follow-up (a): T1 MLP cells again with a second initialisation seed."""
    for cfg in v3_t1():
        if cfg["arch"] == "mlp":
            yield dict(cfg, seed=1)


def v3_t2_groups():
    """Follow-up (b): T2 at B = 8 with G in {4, 16, 64}: does response noise shrink as G grows?"""
    for cfg in v3_t2():
        if cfg["B"] == 8 and cfg["update"] in ("expected", "sampled"):
            for G in (4, 16, 64):
                yield dict(cfg, G=G)


def v3_t6_kl():
    """Follow-up (c): T6 KL conditions with the decline above p_ref tracked (drop_above)."""
    for cfg in v3_t6():
        if cfg["ablation"] in ("kl01", "kl1", "control"):
            yield dict(cfg, track_every=1)


def v3_t4_edge():
    """Follow-up (d): T4 with eta extended to 3000 and 10000, stopping at mean p = 0.5."""
    for level, p0, G, rule, eta, seed in itertools.product(
        ("tab", "lin"), (0.01, 0.03, 0.1), (2, 4, 8, 16, 32, 64, 128), ("mean", "grpo"), (3000, 10000), range(5)
    ):
        init = (dict(features="onehot", n_train=64, init="tabular", p0=p0) if level == "tab"
                else dict(SHARING[level], init="sft_down", p0=p0))
        yield dict(init, L=1, G=G, B=256 // G, rule=rule, eta=eta, seed=seed, task_seed=seed, steps=5000,
                   log_every=250, snapshot_every=5000, track_every=1, stop_at_half=True, diagnostics=False,
                   level=level)


PASSK_RULES = [("grpo", None), ("mean", None)] + [(r, k) for r in ("passk", "rloo_k") for k in (2, 4, 8)]


def v3_t10():
    """Shelf 2: pass@k training rules (biased GRPO_k, unbiased RLOO_k) against grpo and mean."""
    levels = {
        "tab": (dict(features="onehot", init="tabular", p0="spread"), (3, 30, 300)),
        "lin": (dict(SHARING["lin"], **T5_SFT["lin"], init="sft_mixed", majority=1 / 3), (100, 300, 1000)),
        "m64": (dict(SHARING["m64"], **T5_SFT["m64"], init="sft_mixed", majority=1 / 3), (0.3, 1, 3)),
        "minority": (dict(arch="linear", features="correlated", rho=0.9, majority=0.9), (0.03, 0.3, 1)),
    }
    for (level, (setup, etas)), (rule, k), G, eta, seed in itertools.product(
        levels.items(), PASSK_RULES, (8, 32), (0, 1, 2), range(10)
    ):
        unit = dict(seed=seed) if level == "tab" else dict(task_seed=seed, seed=0)
        yield dict(setup, **unit, L=1, K=3, G=G, B=256 // G, rule=rule, passk=k, eta=etas[eta], steps=5000,
                   log_every=250, snapshot_every=5000, track_every=5, diagnostics=False, level=level)


def t1_base(arch, rho, m, task_seed):
    return dict(arch=arch, width=64, features="correlated", rho=rho, majority=m, L=1, K=3, G=8, B=32, rule="grpo",
                task_seed=task_seed, seed=0, diagnostics=False, level=f"{arch}-rho{rho}")


def v3_t11():
    """Four-branch update test: A-only, B-only and mixed steps from the same state, on the T1 tasks."""
    arms = [("sgd", 0.01, "expected_full", 0), ("sgd", 0.01, "sampled", 16),
            ("adam", 1e-3, "expected_full", 0), ("adam", 1e-4, "expected_full", 0)]
    for arch, rho, m, task_seed in itertools.product(("linear", "mlp"), (0.5, 0.9), (1 / 3, 0.7, 0.9), TASK_SEEDS):
        for optimizer, eta, update, samples in arms:
            steps = flow_steps(eta) if optimizer == "sgd" else 5000
            yield dict(t1_base(arch, rho, m, task_seed), optimizer=optimizer, eta=eta, update=update, steps=steps,
                       log_every=max(steps // 50, 1), snapshot_every=steps, track_every=5,
                       branch_at=[0, 0.1, 0.5, 0.9], branch_samples=samples)


def v3_t12():
    """Plateau or permanent: the T1 minority tasks at 10x the T1 horizon; linear Adam at the smallest step."""
    arms = [("sgd", 0.01, 100_000), ("sgd", 0.1, 10_000), ("sgd", 1, 10_000),
            ("adam", 1e-4, 50_000), ("adam", 1e-3, 50_000)]
    for arch, rho, m, task_seed in itertools.product(("linear", "mlp"), (0.5, 0.9), (0.7, 0.9), TASK_SEEDS):
        for (optimizer, eta, steps), update in itertools.product(arms, ("expected_full", "sampled")):
            yield dict(t1_base(arch, rho, m, task_seed), optimizer=optimizer, eta=eta, update=update, steps=steps,
                       log_every=steps // 200, snapshot_every=steps, track_every=50)


def v3_t13():
    """Projection control: remove the majority step's component against the minority's success gradient."""
    for arch, rho, m, task_seed in itertools.product(("linear", "mlp"), (0.5, 0.9), (0.7, 0.9), TASK_SEEDS):
        for eta, update in itertools.product((0.01, 0.1), ("expected_full", "projected", "shortened")):
            steps = flow_steps(eta)
            yield dict(t1_base(arch, rho, m, task_seed), optimizer="sgd", eta=eta, update=update, steps=steps,
                       log_every=max(steps // 50, 1), snapshot_every=steps, track_every=5)


def v3_t15():
    """Threshold in class share: where does the mixed step start to lower the minority, for K = 2, 3, 5?"""
    offsets = (-0.05, 0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.2, 0.35)
    for K, arch, rho, task_seed in itertools.product((2, 3, 5), ("linear", "mlp"), (0.5, 0.9), TASK_SEEDS):
        for off in offsets:
            yield dict(t1_base(arch, rho, round(1 / K + off, 4), task_seed), K=K, optimizer="sgd", eta=0.01,
                       update="expected_full", steps=10_000, log_every=200, snapshot_every=10_000, track_every=5,
                       branch_at=[0], level=f"{arch}-rho{rho}-K{K}")


def v3_t16():
    """pass@k rules at matched progress: the minority task at 4x the v3c horizon, with class means logged."""
    rules = [("grpo", None), ("mean", None)] + [(r, k) for r in ("passk", "rloo_k") for k in (2, 4, 8)]
    for (rule, k), G, eta, task_seed in itertools.product(rules, (8, 32), (0.03, 0.3, 1), TASK_SEEDS):
        yield dict(arch="linear", features="correlated", rho=0.9, majority=0.9, L=1, K=3, G=G, B=256 // G, rule=rule,
                   passk=k, eta=eta, task_seed=task_seed, seed=0, steps=20_000, log_every=100, snapshot_every=20_000,
                   track_every=5, diagnostics=False, level="minority")


def v3_t17():
    """KL anchor on the minority task: does anchoring to the start policy protect the minority, and at what cost?"""
    arms = [(0.0, "reverse")] + [(beta, kl) for kl in ("reverse", "forward") for beta in (0.01, 0.03, 0.1, 0.3, 1, 3)]
    for arch, update, eta, (beta, kl), task_seed in itertools.product(
            ("linear", "mlp"), ("expected_full", "sampled"), (0.1, 1), arms, TASK_SEEDS):
        yield dict(t1_base(arch, 0.9, 0.9, task_seed), optimizer="sgd", eta=eta, update=update, beta=beta, kl=kl,
                   steps=20_000, log_every=100, snapshot_every=20_000, track_every=5, level=f"{arch}-{update}")


def v3_t18():
    """Release test: collapse the minority at m = 0.9 with beta = 0, then stop rewarding the majority.
    After the release: no KL, or a reverse or forward KL anchor to the start policy. Control: no release."""
    after = [(0.0, "reverse")] + [(beta, kl) for kl in ("reverse", "forward") for beta in (0.01, 0.1)]
    for arch, task_seed in itertools.product(("linear", "mlp"), TASK_SEEDS):
        base = dict(t1_base(arch, 0.9, 0.9, task_seed), optimizer="sgd", eta=0.1, update="expected_full",
                    steps=20_000, log_every=100, snapshot_every=20_000, track_every=5)
        yield dict(base, level=f"{arch}-control")
        for release_at, (beta, kl) in itertools.product((0.005, 0.05, 0.25), after):
            yield dict(base, release_at=release_at, release_beta=beta, release_kl=kl, level=f"{arch}-release")


def v3_t19():
    """Group size and the collapse threshold: the T15 share grid at K = 3 for several G."""
    offsets = (-0.05, 0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.2, 0.35, 0.45, 0.55)
    for arch, rho, task_seed, off in itertools.product(("linear", "mlp"), (0.5, 0.9), TASK_SEEDS, offsets):
        base = dict(t1_base(arch, rho, round(1 / 3 + off, 4), task_seed), K=3, optimizer="sgd", eta=0.01,
                    steps=10_000, log_every=200, snapshot_every=10_000, track_every=5)
        for G in (2, 4, 8, 16, 64):
            yield dict(base, G=G, update="expected_full", branch_at=[0], level=f"{arch}-rho{rho}-grpo-exact")
            if G in (2, 64):
                yield dict(base, G=G, rule="mean", update="expected_full", branch_at=[0],
                           level=f"{arch}-rho{rho}-mean-exact")
            if rho == 0.9:
                yield dict(base, G=G, B=256 // G, update="sampled", level=f"{arch}-rho{rho}-grpo-sampled")


def v3_t20():
    """Correct minority in the enumerable transformer environment of Li, Pandey and Chen (extra/experiments/enum_env.py)."""
    tasks = [(3, n, "random") for n in (22, 32, 40, 48, 56, 60)] + [(2, n, "random") for n in (32, 40, 48, 56, 60)]
    tasks.append((2, 48, "parity"))
    steppers = [("sgd", 0.3), ("sgd", 1), ("sgd", 3), ("adam", 1e-3), ("adam", 3e-3)]
    common = dict(mode="enum", steps=2000, log_every=20, G=8)
    for (K, n_major, partition), (optimizer, eta), seed in itertools.product(tasks, steppers, range(10)):
        job = dict(common, K=K, n_major=n_major, partition=partition, optimizer=optimizer, eta=eta, seed=seed)
        for algo in ("pg", "grpo"):
            yield dict(job, enum_policy="shared", algo=algo, level=f"shared-{algo}")
        yield dict(job, enum_policy="tabular", algo="pg", level="tabular-pg")
    for (K, n_major, partition), seed in itertools.product([(3, 22, "random"), (3, 56, "random"), (2, 48, "parity")], range(3)):
        yield dict(common, K=K, n_major=n_major, partition=partition, optimizer="adam", eta=1e-3, seed=seed,
                   enum_policy="copies", algo="pg", level="copies-pg")


def v3_t21():
    """Prevention levers in the enumerable transformer environment: KL anchors and pass@k weighting."""
    arms = [dict()] + [dict(enum_beta=b, enum_kl=d) for d in ("reverse", "forward") for b in (0.03, 0.3)]
    arms += [dict(enum_passk=k) for k in (2, 4)]
    tasks = [(3, 56, "random"), (2, 48, "parity")]
    for (K, n_major, partition), algo, (optimizer, eta), arm, seed in itertools.product(
            tasks, ("pg", "grpo"), (("adam", 1e-3), ("sgd", 0.3)), arms, range(10)):
        yield dict(mode="enum", steps=2000, log_every=20, G=8, K=K, n_major=n_major, partition=partition,
                   enum_policy="shared", algo=algo, optimizer=optimizer, eta=eta, seed=seed, level=f"shared-{algo}", **arm)


def v3_a4():
    """A4: T20's share effect under Adam (lr 1e-3, shared transformer, exact) on fresh partition seeds 10-29."""
    tasks = [(3, n, "random") for n in (22, 32, 40, 48, 56, 60)] + [(2, n, "random") for n in (32, 40, 48, 56, 60)]
    for (K, n_major, partition), algo, seed in itertools.product(tasks, ("pg", "grpo"), range(10, 30)):
        yield dict(mode="enum", steps=2000, log_every=20, G=8, K=K, n_major=n_major, partition=partition,
                   optimizer="adam", eta=1e-3, seed=seed, enum_policy="shared", algo=algo, level=f"shared-{algo}")


def v3_t29():
    """J: the T20 share grid under SGD at small step sizes (flow time matched to SGD 0.3 x 2,000 steps)."""
    tasks = [(3, n, "random") for n in (22, 32, 40, 48, 56, 60)]
    for (K, n_major, partition), (eta, steps), algo, seed in itertools.product(
            tasks, ((0.03, 20_000), (0.1, 6_000)), ("pg", "grpo"), range(10)):
        yield dict(mode="enum", steps=steps, log_every=max(steps // 100, 1), G=8, K=K, n_major=n_major,
                   partition=partition, optimizer="sgd", eta=eta, seed=seed, enum_policy="shared", algo=algo,
                   level=f"shared-{algo}-sgd{eta}")


def v3_t22():
    """Junk answers: two answer classes with prompts, and K - 2 tokens that are never correct."""
    for K, arch, update, task_seed, m in itertools.product(
            (2, 3, 6, 20), ("linear", "mlp"), ("expected_full", "sampled"), TASK_SEEDS,
            (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98)):
        job = dict(t1_base(arch, 0.9, m, task_seed), K=K, used_classes=2, optimizer="sgd", eta=0.03, update=update,
                   steps=10_000, log_every=200, snapshot_every=10_000, track_every=5, level=f"{arch}-K{K}-{update}")
        if update == "expected_full":
            job["branch_at"] = [0]
        yield job


def v3_t23():
    """Warm start: the T1 minority task from a policy that already answers every class correctly with about p0."""
    starts = [dict(init="uniform")] + [dict(init="sft_balanced", p0=p0) for p0 in (0.5, 0.7, 0.9)]
    for arch, m, update, start, task_seed in itertools.product(
            ("linear", "mlp"), (0.7, 0.9), ("expected_full", "sampled"), starts, TASK_SEEDS):
        job = dict(t1_base(arch, 0.9, m, task_seed), optimizer="sgd", eta=0.1, update=update, steps=10_000,
                   log_every=100, snapshot_every=10_000, track_every=5, level=f"{arch}-{update}", **start)
        if update == "expected_full":
            job["branch_at"] = [0]
        yield job


T24_SNAPSHOTS = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000]


def v3_t24():
    """Late phase, prompt by prompt: per-prompt snapshots at log-spaced steps, linear policy, SGD, exact update."""
    for rho, m, eta, task_seed in itertools.product((0.5, 0.7, 0.9), (0.7, 0.9), (0.1, 1), TASK_SEEDS):
        yield dict(t1_base("linear", rho, m, task_seed), optimizer="sgd", eta=eta, update="expected_full", steps=100_000,
                   log_every=100, snapshot_every=100_000, track_every=10, snapshot_at=T24_SNAPSHOTS,
                   level=f"linear-rho{rho}-eta{eta}")


def v3_t26():
    """Shared skills: 4-token answers, k procedure positions (the answer class) and 4 - k shared positions."""
    for arch, rho, m, k, task_seed in itertools.product(("linear", "mlp"), (0.5, 0.9), (0.7, 0.9), (1, 2, 3, 4), TASK_SEEDS):
        yield dict(t1_base(arch, rho, m, task_seed), L=4, differing_positions=k, optimizer="sgd", eta=0.1,
                   update="expected_full", steps=10_000, log_every=100, snapshot_every=10_000, track_every=5,
                   branch_at=[0, 0.1, 0.5], level=f"{arch}-rho{rho}-k{k}")


REPLICATION_SEEDS = list(range(10, 30))


def fresh(jobs, keep=lambda job: True):
    """The task-0 jobs of a grid that pass `keep`, repeated on task seeds 10-29 (fresh tasks)."""
    for job in jobs:
        if job["task_seed"] == 0 and keep(job):
            for task_seed in REPLICATION_SEEDS:
                yield dict(job, task_seed=task_seed)


def v3_a1():
    """A1: P1 (absorption rises with the majority share) on fresh tasks."""
    yield from fresh(v3_t1())


def v3_a2():
    """A2: forward-KL floor on fresh tasks (no anchor and forward KL at beta 0.03, 0.1, 0.3)."""
    yield from fresh(v3_t17(), lambda j: j["beta"] == 0 or (j["kl"] == "forward" and j["beta"] in (0.03, 0.1, 0.3)))


def v3_a3():
    """A3: pass@4 protection at matched progress on fresh tasks (GRPO, pass@4 and its unbiased form)."""
    yield from fresh(v3_t16(), lambda j: j["rule"] == "grpo" or j["passk"] == 4)


def v3_a5():
    """A5: warm start p0 = 0.5 against the uniform start at m = 0.7, on fresh tasks."""
    yield from fresh(v3_t23(), lambda j: j["majority"] == 0.7 and (j["init"] == "uniform" or j.get("p0") == 0.5))


def v3_t27():
    """G: forward KL toward a collapsed reference. The anchor is the policy at 0.5%, 5% or 25% of training;
    controls: no anchor, and forward KL toward the start policy (as in T17)."""
    for arch, task_seed in itertools.product(("linear", "mlp"), TASK_SEEDS):
        base = dict(t1_base(arch, 0.9, 0.9, task_seed), optimizer="sgd", eta=0.1, update="expected_full",
                    steps=20_000, log_every=100, snapshot_every=20_000, track_every=5)
        yield dict(base, level=f"{arch}-control")
        for beta in (0.01, 0.1, 1):
            yield dict(base, beta=beta, kl="forward", level=f"{arch}-start-anchor")
            for anchor_at in (0.005, 0.05, 0.25):
                yield dict(base, anchor_at=anchor_at, anchor_beta=beta, anchor_kl="forward", level=f"{arch}-late-anchor")


def v3_t28():
    """H: Adam's epsilon. Large epsilon removes the per-coordinate normalisation (Adam tends to momentum SGD
    with step eta/eps), so the learning rate grows with epsilon to keep progress comparable. Control: SGD."""
    arms = [("adam", max(1e-3, 0.1 * eps), eps) for eps in (1e-8, 1e-5, 1e-3, 1e-2, 1e-1, 1.0)] + [("sgd", 0.1, 1e-8)]
    for arch, rho, m, (optimizer, eta, eps), task_seed in itertools.product(
            ("linear", "mlp"), (0.5, 0.9), (0.7, 0.9), arms, TASK_SEEDS):
        yield dict(t1_base(arch, rho, m, task_seed), optimizer=optimizer, eta=eta, adam_eps=eps,
                   update="expected_full", steps=10_000, log_every=100, snapshot_every=10_000, track_every=5,
                   level=f"{arch}-rho{rho}-{optimizer}")


REGISTERED_SEEDS = list(range(30, 50))   # v3r (PREREG_v3d addendum 11): never run before
CONFIRM_SEEDS = list(range(50, 70))      # reserved for the T31 confirmation rerun of the best reset cell
T30_SCALES = (0, 0.5, 1, 2, 4, 8)


def t30_base(arch, s, m, task_seed):
    return dict(arch=arch, width=64, features="block", block_dim=5, shared_scale=s, majority=m, L=1, K=3,
                n_train=512, G=8, B=32, rule="grpo", optimizer="sgd", task_seed=task_seed, seed=0, diagnostics=False)


def v3_t30():
    """T30: non-negative interactions at full size. Block features; the shared coordinate s breaks the condition.
    eta = eta_tilde / (5 + s^2), the same step relative to the feature norm at every s. Linear grid, then the
    exploratory MLP rows (s in {0, 8}, m = 0.9, eta_tilde = 0.5)."""
    rows = [("linear", s, m, eta_tilde, update)
            for s, m, eta_tilde, update in itertools.product(T30_SCALES, (1 / 3, 0.7, 0.9), (0.5, 1.5),
                                                             ("expected_full", "sampled"))]
    rows += [("mlp", s, 0.9, 0.5, update) for s in (0, 8) for update in ("expected_full", "sampled")]
    for (arch, s, m, eta_tilde, update), task_seed in itertools.product(rows, REGISTERED_SEEDS):
        yield dict(t30_base(arch, s, m, task_seed), update=update, eta=eta_tilde / (5 + s**2), eta_tilde=eta_tilde,
                   steps=20_000, log_every=100, track_every=5, snapshot_every=20_000, snapshot_at=[20_000],
                   level=f"{arch}-{update}-s{s}")


def t31_base(arch, task_seed, update="expected_full"):
    """The T31 standard cell: linear, rho = 0.9, m = 0.9, SGD eta = 0.1, 100,000 steps (flow time 1e4)."""
    return dict(t1_base(arch, 0.9, 0.9, task_seed), optimizer="sgd", eta=0.1, update=update, steps=100_000,
                log_every=100, track_every=5, snapshot_every=100_000, snapshot_at=[2_000, 100_000])


def t31_arms():
    """(label, settings) of the 49 registered T31 arms; every arm differs from plain GRPO only in these."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from synthetic_rl.theory import advantage_gap

    arms = [("plain", {}), ("plain-mean", dict(rule="mean"))]
    arms += [("fixed-reverse", dict(kl="reverse", beta=b)) for b in (0.03, 0.1, 0.3, 1, 3)]
    arms += [("fixed-forward", dict(kl="forward", beta=b)) for b in (0.03, 0.1, 0.3, 1)]
    arms += [("reset-reverse", dict(kl="reverse", beta=b, reset_every=n))
             for b in (0.3, 1, 3) for n in (1, 10, 100, 1000, 5000)]
    arms += [("reset-forward", dict(kl="forward", beta=b, reset_every=n))
             for b in (0.1, 0.3, 1) for n in (10, 100, 1000, 5000)]
    arms += [("passk", dict(rule="rloo_k", passk=k)) for k in (2, 4, 8)]
    arms += [("reset-reverse-mean", dict(rule="mean", kl="reverse", beta=1, reset_every=n))
             for n in (1, 10, 100, 1000, 5000)]
    # Amendment 1: pass@k with the initial step matched to GRPO's at the uniform start (p = 1/K = 1/3).
    w_grpo = float(advantage_gap(1 / 3, 8, "grpo"))
    arms += [("passk-matched", dict(rule="rloo_k", passk=k,
                                    eta=0.1 * w_grpo / float(advantage_gap(1 / 3, 8, "rloo_k", passk=k))))
             for k in (2, 4, 8)]
    return arms


T31_ROBUST = ([("plain", {}), ("fixed-reverse", dict(kl="reverse", beta=0.3)),
               ("fixed-reverse", dict(kl="reverse", beta=1)), ("fixed-forward", dict(kl="forward", beta=0.1))]
              + [("reset-reverse", dict(kl="reverse", beta=1, reset_every=n)) for n in (10, 100, 1000)]
              + [("passk", dict(rule="rloo_k", passk=4))])


def v3_t31():
    """T31: prevention with one knob changed, on the standard cell; then the robustness rows (MLP with the exact
    update, linear with sampled updates)."""
    for task_seed, (label, arm) in itertools.product(REGISTERED_SEEDS, t31_arms()):
        yield dict(t31_base("linear", task_seed), **arm, level=f"linear-exact-{label}")
    for (arch, update, tag), task_seed, (label, arm) in itertools.product(
            (("mlp", "expected_full", "mlp-exact"), ("linear", "sampled", "linear-sampled")), REGISTERED_SEEDS,
            T31_ROBUST):
        yield dict(t31_base(arch, task_seed, update), **arm, level=f"{tag}-{label}")


def v3_t31_confirm(beta, reset_every):
    """T31 prediction 5: the best reverse-reset cell and the plain arm, rerun on seeds 50-69. Not in a batch: write
    it with the chosen cell once T31 is analysed."""
    for task_seed in CONFIRM_SEEDS:
        yield dict(t31_base("linear", task_seed), level="linear-exact-plain")
        yield dict(t31_base("linear", task_seed), kl="reverse", beta=beta, reset_every=reset_every,
                   level="linear-exact-reset-reverse")


def v3_t33():
    """T33(b): plain GRPO up to step 100 or 1,000, then unbiased pass@k (rloo_k); every prompt's success is stored
    at the switch step (Amendment 1, item 4). The control is T31's plain arm."""
    for task_seed, switch_at, k in itertools.product(REGISTERED_SEEDS, (0.001, 0.01), (2, 4, 8)):
        step = int(round(switch_at * 100_000))
        yield dict(t31_base("linear", task_seed), switch_at=switch_at, switch_rule="rloo_k", switch_passk=k,
                   snapshot_at=[step, 2_000, 100_000], level=f"linear-exact-switch{step}")


BATCHES = {
    "base": [("h1", h1), ("h2", h2), ("h3", h3), ("h4", h4), ("h5", h5)],
    "stretch": [("s_h1", s_h1), ("s_drift", s_drift), ("s_pred", s_pred), ("s_abl", s_abl), ("s_eq", s_eq)],
    "v3a": [("v3_t1", v3_t1), ("v3_t2", v3_t2), ("v3_t4", v3_t4), ("v3_t6", v3_t6), ("v3_t8", v3_t8)],
    "v3b": [("v3_t5", v3_t5), ("v3_t7", v3_t7), ("v3_t9", v3_t9), ("v3_t1s1", v3_t1_seed1),
            ("v3_t2g", v3_t2_groups), ("v3_t6kl", v3_t6_kl), ("v3_t4e", v3_t4_edge)],
    "v3c": [("v3_t10", v3_t10)],
    "v3d": [("v3_t11", v3_t11), ("v3_t12", v3_t12), ("v3_t13", v3_t13)],
    "v3e": [("v3_t15", v3_t15)],
    "v3f": [("v3_t16", v3_t16)],
    "v3g": [("v3_t17", v3_t17), ("v3_t18", v3_t18), ("v3_t19", v3_t19)],
    "v3h": [("v3_t20", v3_t20)],
    "v3i": [("v3_t21", v3_t21)],
    "v3q": [("v3_a4", v3_a4), ("v3_t29", v3_t29)],
    "v3j": [("v3_t22", v3_t22)],
    "v3k": [("v3_t23", v3_t23)],
    "v3l": [("v3_t24", v3_t24)],
    "v3n": [("v3_t26", v3_t26)],
    "v3o": [("v3_a1", v3_a1), ("v3_a2", v3_a2), ("v3_a3", v3_a3), ("v3_a5", v3_a5)],
    "v3p": [("v3_t27", v3_t27), ("v3_t28", v3_t28)],
    "v3r": [("v3_t30", v3_t30), ("v3_t31", v3_t31), ("v3_t33", v3_t33)],
}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "base"
    for prefix, grid in BATCHES[tag]:
        for i, cfg in enumerate(grid()):
            cfg = {k: v for k, v in cfg.items() if v is not None}
            print(json.dumps(dict(cfg, name=f"{prefix}/{i:05d}", hypothesis=prefix)))


if __name__ == "__main__":
    main()
