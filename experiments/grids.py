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


BATCHES = {"base": [("h1", h1), ("h2", h2), ("h3", h3), ("h4", h4), ("h5", h5)]}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "base"
    for prefix, grid in BATCHES[tag]:
        for i, cfg in enumerate(grid()):
            cfg = {k: v for k, v in cfg.items() if v is not None}
            print(json.dumps(dict(cfg, name=f"{prefix}/{i:05d}", hypothesis=prefix)))


if __name__ == "__main__":
    main()
