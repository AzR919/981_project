"""
Run one experiment configuration and write its result.

usage:
    python experiments/run.py --jobs jobs_<tag>.jsonl --index 17 --out results/raw
    python experiments/run.py --config '{"mode": "train", ...}' --out results/raw

Each config line is a JSON object. Its "name" (for example "h3/abc123")
becomes the output path <out>/<name>.json.gz; an existing output is skipped,
so a batch can be relaunched after a crash or a reboot.

Modes:
    train      train with plain SGD (or the optimizer named) and log exact
               per-prompt success
    gradcheck  H1: average many sampled gradients at fixed parameters and
               compare with the predicted mean -(1/B) sum_x W_G(p_x) grad p_x
    onestep    H2: average the change in every p_x over many independent
               one-step updates from the same parameters (full batch)
"""

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_rl.diagnostics import (  # noqa: E402
    effective_step,
    interference,
    predicted_drift,
    success_jacobian,
)
from synthetic_rl.policy import Policy  # noqa: E402
from synthetic_rl.synthetic_task import SyntheticTask, verify  # noqa: E402
from synthetic_rl.theory import advantage_gap  # noqa: E402
from synthetic_rl.trainer import Trainer, calculate_advantages  # noqa: E402
from synthetic_rl.warmstart import set_tabular_success, supervised_warmstart  # noqa: E402

DEFAULTS = {
    "mode": "train",
    "K": 3, "L": 1, "d": 16,
    "features": "gaussian", "rho": 0.0, "answers": "rule",
    "n_train": 512, "n_test": 256,
    "arch": "mlp", "width": 64,
    "init": "uniform", "p0": None, "spread": [-4.0, 3.0],
    "rule": "grpo", "passk": None,
    "G": 8, "B": 32, "eta": 1.0,
    "beta": 0.0, "kl": "reverse",
    "optimizer": "sgd", "momentum": 0.0,
    "reward_noise": 0.0,
    "steps": 3000, "log_every": 50, "snapshot_every": 500,
    "diagnostics": True, "subset": 128,
    "R": 2000, "probe": 32,
    "seed": 0,
}

ABSORBED, SOLVED = 0.01, 0.99


def build(cfg):
    """Return (task, train_data, test_data, policy) with the base policy set."""
    seed = cfg["seed"]
    tabular = cfg["features"] == "onehot"
    d = cfg["n_train"] if tabular else cfg["d"]
    answers = "random" if tabular else cfg["answers"]

    task = SyntheticTask(
        num_tokens=cfg["K"], response_length=cfg["L"], num_features=d,
        seed=cfg.get("task_seed", seed), features=cfg["features"],
        rho=cfg["rho"], answers=answers,
    )
    train = task.generate_dataset(cfg["n_train"])
    test = None if tabular else task.generate_dataset(cfg["n_test"])

    torch.manual_seed(seed)
    policy = Policy(
        num_features=d, num_tokens=cfg["K"], response_length=cfg["L"],
        hidden_size=cfg["width"], arch="linear" if tabular else cfg["arch"],
    ).double()

    init = cfg["init"]
    rng = np.random.default_rng(seed + 10_000)
    if init == "tabular":
        if cfg["p0"] == "spread":
            z = rng.uniform(*cfg["spread"], size=cfg["n_train"])
            p0 = 1 / (1 + np.exp(-z))
        else:
            p0 = float(cfg["p0"])
        set_tabular_success(policy, train[1], p0)
    elif init in ("sft_up", "sft_down"):
        if init == "sft_up":
            targets, direction = train[1], "up"
        else:
            # Decoy answers: shift every correct token by one.
            targets, direction = (train[1] + 1) % cfg["K"], "down"
        steps, reached = supervised_warmstart(
            policy, train[0], targets, float(cfg["p0"]), train[1],
            lr=cfg.get("sft_lr", 0.5), direction=direction,
        )
        cfg["_warmstart"] = {"steps": steps, "mean_success": reached}
    elif init != "uniform":
        raise ValueError(f"unknown init {init}")

    return task, train, test, policy


def tensors(policy, data):
    prompts, correct = data
    return (
        torch.as_tensor(prompts, dtype=next(policy.parameters()).dtype),
        torch.as_tensor(correct, dtype=torch.long),
    )


@torch.no_grad()
def success(policy, data):
    prompts, correct = tensors(policy, data)
    return policy.success_probability(prompts, correct).numpy()


def make_optimizer(cfg, policy):
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(policy.parameters(), lr=cfg["eta"], momentum=cfg["momentum"])
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(policy.parameters(), lr=cfg["eta"])
    raise ValueError(f"unknown optimizer {cfg['optimizer']}")


def summarize(p, p_init):
    started_alive = p_init >= ABSORBED
    return {
        "mean": float(p.mean()),
        "absorbed": float(((p < ABSORBED) & started_alive).sum() / max(started_alive.sum(), 1)),
        "solved": float((p > SOLVED).mean()),
        "quantiles": [float(v) for v in np.quantile(p, [0.05, 0.25, 0.5, 0.75, 0.95])],
    }


def run_train(cfg):
    task, train, test, policy = build(cfg)
    p_init = success(policy, train)
    out = {"p_init": p_init.astype(np.float32).tolist()}

    prompts_t, correct_t = tensors(policy, train)
    if cfg["diagnostics"]:
        t0 = time.time()
        out["diagnostics"] = interference(policy, prompts_t, correct_t, subset=cfg["subset"], seed=cfg["seed"])
        drift = predicted_drift(policy, prompts_t, correct_t, cfg["rule"], cfg["G"], cfg["eta"], passk=cfg["passk"])
        out["predicted_drift"] = drift.numpy().astype(np.float32).tolist()
        out["diagnostics"]["seconds"] = time.time() - t0
    out["s_bar"] = effective_step(p_init, cfg["G"], cfg["rule"], cfg["eta"], cfg["B"], passk=cfg["passk"])
    out["W_init_mean"] = float(np.mean(advantage_gap(p_init, cfg["G"], cfg["rule"], passk=cfg["passk"])))

    trainer = Trainer(
        policy=policy, optimizer=make_optimizer(cfg, policy),
        group_size=cfg["G"], advantage=cfg["rule"], beta=cfg["beta"],
        kl_direction=cfg["kl"], seed=cfg["seed"], passk=cfg["passk"],
        reward_noise=cfg["reward_noise"],
    )

    n = len(train[0])
    absorbed_at = np.full(n, -1, dtype=np.int64)
    min_p = p_init.copy()
    history, snapshots = [], {}
    informative, expected = [], []

    def log(step):
        p = success(policy, train)
        np.minimum(min_p, p, out=min_p)
        newly = (absorbed_at < 0) & (p < ABSORBED) & (p_init >= ABSORBED)
        absorbed_at[newly] = step
        record = {"step": step, "train": summarize(p, p_init), "decreases": trainer.decreases}
        if test is not None:
            record["test_mean"] = float(success(policy, test).mean())
        if informative:
            record["informative"] = float(np.mean(informative))
            record["expected_informative"] = float(np.mean(expected))
            informative.clear()
            expected.clear()
        if cfg["beta"] > 0 or step == 0 or step == cfg["steps"]:
            ev = trainer.evaluate(*train)
            record["entropy"], record["kl"] = ev["entropy"], ev["kl"]
        history.append(record)
        if step % cfg["snapshot_every"] == 0 or step == cfg["steps"]:
            snapshots[str(step)] = np.round(p, 6).astype(np.float32).tolist()

    log(0)
    for step in range(1, cfg["steps"] + 1):
        idx = trainer.rng.integers(n, size=cfg["B"])
        metrics = trainer.train_step(train[0][idx], train[1][idx])
        informative.append(metrics["informative_before"])
        expected.append(metrics["expected_informative_before"])
        if step % cfg["log_every"] == 0 or step == cfg["steps"]:
            log(step)

    out.update({
        "history": history,
        "snapshots": snapshots,
        "absorbed_at": absorbed_at.tolist(),
        "min_p": min_p.astype(np.float32).tolist(),
        "decreases_total": trainer.decreases,
    })
    return out


def flat_grad(loss, policy):
    grads = torch.autograd.grad(loss, list(policy.parameters()))
    return torch.cat([g.reshape(-1) for g in grads])


def run_gradcheck(cfg):
    """H1: sampled mean gradient against -(1/B) sum_x W_G(p_x) grad p_x."""
    task, train, test, policy = build(cfg)
    prompts, correct = tensors(policy, train)
    prompts, correct = prompts[: cfg["probe"]], correct[: cfg["probe"]]

    p, J = success_jacobian(policy, prompts, correct)
    W = torch.as_tensor(advantage_gap(p.numpy(), cfg["G"], cfg["rule"], passk=cfg["passk"]), dtype=p.dtype)
    predicted = -(W[:, None] * J).mean(dim=0)

    torch.manual_seed(cfg["seed"] + 1)
    total = torch.zeros_like(predicted)
    total_sq = torch.zeros_like(predicted)
    R = cfg["R"]
    for _ in range(R):
        responses = policy.sample(prompts, cfg["G"])
        rewards = torch.as_tensor(verify(responses.numpy(), correct.numpy()), dtype=prompts.dtype)
        A = calculate_advantages(rewards, kind=cfg["rule"], passk=cfg["passk"])
        loss = -(A * policy.log_prob(prompts, responses)).mean()
        g = flat_grad(loss, policy)
        total += g
        total_sq += g * g

    mean = total / R
    se = ((total_sq / R - mean**2).clamp(min=0) / (R - 1)).sqrt()
    norm = predicted.norm().item()
    mask = se > 0
    z = (mean - predicted)[mask] / se[mask]
    return {
        "p_probe": p.numpy().astype(np.float32).tolist(),
        "pred_norm": norm,
        "rel_err": (mean - predicted).norm().item() / norm,
        "mc_rel": se.norm().item() / norm,
        "cosine": torch.nn.functional.cosine_similarity(mean, predicted, dim=0).item(),
        "mean_z2": (z**2).mean().item(),
        "frac_abs_z_gt3": (z.abs() > 3).float().mean().item(),
        "n_coords": int(mask.sum().item()),
    }


def run_onestep(cfg):
    """H2: expected one-step change of every p_x over R full-batch updates."""
    task, train, test, policy = build(cfg)
    prompts, correct = tensors(policy, train)
    p0 = policy.success_probability(prompts, correct).detach()
    _, J = success_jacobian(policy, prompts, correct)
    W = advantage_gap(p0.numpy(), cfg["G"], cfg["rule"], passk=cfg["passk"])
    drift = predicted_drift(policy, prompts, correct, cfg["rule"], cfg["G"], cfg["eta"], passk=cfg["passk"])
    own = cfg["eta"] / len(p0) * W * (J * J).sum(dim=1).numpy()

    start = {k: v.clone() for k, v in policy.state_dict().items()}
    trainer = Trainer(
        policy=policy, optimizer=torch.optim.SGD(policy.parameters(), lr=cfg["eta"]),
        group_size=cfg["G"], advantage=cfg["rule"], seed=cfg["seed"], passk=cfg["passk"],
    )
    total = np.zeros(len(p0))
    total_sq = np.zeros(len(p0))
    R = cfg["R"]
    for _ in range(R):
        policy.load_state_dict(start)
        trainer.train_step(*train)
        with torch.no_grad():
            delta = (policy.success_probability(prompts, correct) - p0).numpy()
        total += delta
        total_sq += delta * delta
    mean = total / R
    se = np.sqrt(np.clip(total_sq / R - mean**2, 0, None) / (R - 1))
    f32 = lambda a: np.asarray(a, dtype=np.float32).tolist()
    return {
        "p0": f32(p0.numpy()), "W": f32(W), "delta_mean": f32(mean), "delta_se": f32(se),
        "predicted_drift": f32(drift.numpy()), "own_term": f32(own),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs")
    parser.add_argument("--index", type=int)
    parser.add_argument("--config")
    parser.add_argument("--out", default="results/raw")
    args = parser.parse_args()

    if args.config:
        raw = json.loads(args.config)
    else:
        with open(args.jobs) as f:
            for i, line in enumerate(f):
                if i == args.index:
                    raw = json.loads(line)
                    break
            else:
                raise IndexError(f"no job {args.index} in {args.jobs}")

    cfg = dict(DEFAULTS, **raw)
    path = os.path.join(args.out, cfg["name"] + ".json.gz")
    if os.path.exists(path):
        return

    torch.set_num_threads(1)
    # Absorbed prompts reach p ~ 1e-320; subnormal floats are ~100x slower.
    torch.set_flush_denormal(True)
    t0 = time.time()
    result = {"train": run_train, "gradcheck": run_gradcheck, "onestep": run_onestep}[cfg["mode"]](cfg)
    result["config"] = cfg
    result["seconds"] = time.time() - t0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with gzip.open(tmp, "wt") as f:
        json.dump(result, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
