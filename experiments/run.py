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
               per-prompt success. "update" picks the step: "sampled" (GRPO
               with sampled groups), "expected" (the exact expected step on
               the sampled prompt batch, no response noise) or
               "expected_full" (the exact expected step on all prompts, a
               deterministic flow)
    gradcheck  H1: average many sampled gradients at fixed parameters and
               compare with the predicted mean -(1/B) sum_x W_G(p_x) grad p_x
    onestep    H2: average the change in every p_x over many independent
               one-step updates from the same parameters (full batch)
"""

import argparse
import copy
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
    grad_sq_norms,
    interference,
    predicted_drift,
    success_jacobian,
)
from synthetic_rl.policy import Policy  # noqa: E402
from synthetic_rl.synthetic_task import SyntheticTask, verify  # noqa: E402
from synthetic_rl.theory import advantage_gap  # noqa: E402
from synthetic_rl.trainer import Trainer, calculate_advantages  # noqa: E402
from synthetic_rl.warmstart import mixed_targets, set_tabular_success, supervised_warmstart  # noqa: E402

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
    "R": 2000, "probe": 32, "chunks": 1,
    "update": "sampled", "p_start": None, "drift_every": 0,
    "mix": 0.5, "sft_steps": 200,
    # v3 additions; the defaults reproduce earlier runs.
    "majority": None,            # T1: share of answer class 0 (others equal); adds a constant feature
    "used_classes": None,        # T22: number of answer classes that have prompts (None: all K); the remaining
                                 #      K - used_classes tokens are never correct ("junk" answers)
    "replace": True,             # T2: False draws each batch without replacement
    "track_every": 0,            # T0: also observe every prompt's p every k steps (0: log points only)
    "taus": [1e-3, 1e-2],        # T0: thresholds for ever- / persistent absorption
    "tail": 0.2,                 # T0: persistent = p < tau over the last 20% of steps
    "ppo_epochs": 1, "clip_eps": None, "grad_clip": None,   # T6
    "stop_at_half": False,       # T4: end the run at the first observed mean p >= 0.5
    "boot": 0,                   # T7: bootstrap draws of the mean z^2 null in gradcheck
    "branch_at": [],             # T11: fractions of training at which to measure A-only / B-only / mixed steps
    "branch_samples": 0,         # T11: sampled batches per branch point (0: exact expected branches only)
    "release_at": None,          # T18: fraction of training after which class A is no longer rewarded
    "release_beta": None, "release_kl": None,   # T18: KL strength and direction after the release (None: unchanged)
    "snapshot_at": [],           # T24: extra steps at which every prompt's success is logged and stored
    "adam_eps": 1e-8,            # H: Adam's epsilon (large values remove the per-coordinate normalisation)
    "anchor_at": None,           # G: fraction of training after which a KL anchor to the policy *at that step*
    "anchor_beta": None, "anchor_kl": "forward",   #    is switched on (all classes keep their reward)
    "differing_positions": None, # T26: with L > 1, positions 0..k-1 carry the answer class (the "procedure");
                                 #      the other positions follow one balanced rule shared by all prompts
    "seed": 0,
}

ABSORBED, SOLVED = 0.01, 0.99


def build(cfg):
    """Return (task, train_data, test_data, policy) with the base policy set."""
    seed = cfg["seed"]
    tabular = cfg["features"] == "onehot"
    d = cfg["n_train"] if tabular else cfg["d"]
    answers = "random" if tabular else cfg["answers"]
    shares = None
    if cfg["majority"] is not None:
        m = float(cfg["majority"])
        used = cfg["used_classes"] or cfg["K"]
        # A junk token gets a share too small for any prompt to draw it.
        shares = [m] + [(1 - m) / (used - 1)] * (used - 1) + [1e-12] * (cfg["K"] - used)
        d += 1  # the constant feature

    task = SyntheticTask(
        num_tokens=cfg["K"], response_length=cfg["L"], num_features=d,
        seed=cfg.get("task_seed", seed), features=cfg["features"],
        rho=cfg["rho"], answers=answers, class_shares=shares,
    )
    train = task.generate_dataset(cfg["n_train"])
    test = None if tabular else task.generate_dataset(cfg["n_test"])
    if cfg["differing_positions"] is not None:
        if cfg["majority"] is None or not 1 <= cfg["differing_positions"] <= cfg["L"]:
            raise ValueError("differing_positions needs a class-share task and 1 <= k <= L")
        # The shared positions use the same weights with an offset calibrated to equal shares (a twin task with
        # uniform class shares has identical weights; only its offset differs).
        balanced = SyntheticTask(
            num_tokens=cfg["K"], response_length=cfg["L"], num_features=d, seed=cfg.get("task_seed", seed),
            features=cfg["features"], rho=cfg["rho"], answers=answers, class_shares=[1.0 / cfg["K"]] * cfg["K"],
        )
        assert np.array_equal(balanced.weights, task.weights)
        train, test = (shared_skill_answers(task, data, cfg["differing_positions"], balanced.offset)
                       for data in (train, test))

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
    elif init == "sft_balanced":
        # T23: supervised warm start with equal total weight per answer class, until the mean of the class
        # means of success reaches p0. Every class then starts at about the same success, whatever its share.
        cls = train[1][:, 0]
        present = np.unique(cls)
        weights = 1.0 / (len(present) * np.bincount(cls, minlength=cfg["K"])[cls])
        steps, reached = supervised_warmstart(policy, train[0], train[1], float(cfg["p0"]), train[1],
                                              lr=cfg.get("sft_lr", 0.5), direction="up", weights=weights)
        cfg["_warmstart"] = {"steps": steps, "mean_success": reached}
    elif init == "sft_mixed":
        # Correct answers on a random share of prompts, decoys on the rest, for a
        # fixed number of steps: with shared parameters this spreads p widely.
        targets = mixed_targets(train[1], cfg["K"], cfg["mix"], rng)
        steps, reached = supervised_warmstart(
            policy, train[0], targets, 2.0, train[1], lr=cfg.get("sft_lr", 0.5),
            max_steps=cfg["sft_steps"], direction="up",
        )
        cfg["_warmstart"] = {"steps": steps, "mean_success": reached}
    elif init != "uniform":
        raise ValueError(f"unknown init {init}")

    return task, train, test, policy


def shared_skill_answers(task, data, k, offset):
    """T26: answer class (position 0, with the class shares) at positions 0..k-1; at positions k..L-1 a
    balanced rule (the task's own weights with `offset`, calibrated to equal shares), the same for every
    prompt."""
    if data is None:
        return None
    prompts, correct = data
    correct = correct.copy()
    scores = np.einsum("nd,lkd->nlk", prompts, task.weights) + offset
    for l in range(correct.shape[1]):
        correct[:, l] = correct[:, 0] if l < k else scores[:, l].argmax(axis=-1)
    return prompts, correct


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
        return torch.optim.Adam(policy.parameters(), lr=cfg["eta"], eps=cfg["adam_eps"])
    raise ValueError(f"unknown optimizer {cfg['optimizer']}")


def expected_step(policy, optimizer, prompts, correct, cfg, reference=None, weight=None):
    """One step on the exact expected GRPO update, -mean_x W_G(p_x) grad p_x.

    Same normalization as the sampled loss -(A * log pi).mean() over (B, G),
    whose expected gradient is exactly this (H1). With beta > 0 the exact
    KL term the sampled trainer uses is added (T8), so the two modes differ
    only in sampling noise. `weight` (one value per prompt) scales each prompt's
    reward term and leaves the denominator and the KL term alone (T18).
    """
    p = policy.success_probability(prompts, correct)
    W = torch.as_tensor(advantage_gap(p.detach().numpy(), cfg["G"], cfg["rule"], passk=cfg["passk"]), dtype=p.dtype)
    loss = -(W * p).mean() if weight is None else -(W * p * weight).mean()
    if cfg["beta"] > 0:
        loss = loss + cfg["beta"] * policy.kl_divergence(prompts, reference, direction=cfg["kl"]).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        after = policy.success_probability(prompts, correct)
    return int((after < p.detach()).sum().item())


def class_gradients(policy, prompts, correct, in_a, cfg):
    """Ascent gradients of the exact expected update split by class, and the minority's success gradient.

    g_a + g_b is the full expected ascent direction -grad of expected_step's loss (same W, same 1/n), so
    each part already carries its class's share of the prompts. h = grad of mean success over class B.
    """
    p = policy.success_probability(prompts, correct)
    W = torch.as_tensor(advantage_gap(p.detach().numpy(), cfg["G"], cfg["rule"], passk=cfg["passk"]), dtype=p.dtype)
    params = list(policy.parameters())
    weighted = W * p / len(p)
    g_a = torch.autograd.grad(weighted[in_a].sum(), params, retain_graph=True)
    g_b = torch.autograd.grad(weighted[~in_a].sum(), params, retain_graph=True)
    h = torch.autograd.grad(p[~in_a].mean(), params)
    return g_a, g_b, h


def flat(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def project_out(g_a, h):
    """g_a without its component along h when that component is negative (g_a . h < 0); else g_a."""
    dot, norm = flat(g_a) @ flat(h), flat(h) @ flat(h)
    if dot >= 0 or norm == 0:
        return list(g_a)
    return [a - hh * (dot / norm) for a, hh in zip(g_a, h)]


def step_with(policy, optimizer, ascent):
    """One optimizer step along a given ascent direction (SGD: theta += eta * ascent)."""
    for param, g in zip(policy.parameters(), ascent):
        param.grad = -g.clone()
    optimizer.step()


def class_step(policy, optimizer, prompts, correct, in_a, cfg):
    """T13: the expected step with the majority part projected, or shortened to the projected norm."""
    g_a, g_b, h = class_gradients(policy, prompts, correct, in_a, cfg)
    proj = project_out(g_a, h)
    if cfg["update"] == "shortened":
        scale = flat(proj).norm() / flat(g_a).norm().clamp(min=1e-300)
        proj = [scale * g for g in g_a]
    step_with(policy, optimizer, [a + b for a, b in zip(proj, g_b)])


def measure_branches(policy, optimizer, prompts, correct, in_a, cfg, rng):
    """T11: from copies of the current state, take one step along several directions and record the exact
    change in mean success of class A (answer class 0) and class B (the rest)."""
    g_a, g_b, h = class_gradients(policy, prompts, correct, in_a, cfg)
    fa, fb, fh = flat(g_a), flat(g_b), flat(h)
    proj = project_out(g_a, h)
    scale = flat(proj).norm() / fa.norm().clamp(min=1e-300)
    add = lambda x, y: [u + v for u, v in zip(x, y)]  # noqa: E731
    branches = {
        "zero": [torch.zeros_like(g) for g in g_a], "a": list(g_a), "b": list(g_b), "mixed": add(g_a, g_b),
        "a_proj": proj, "a_short": [scale * g for g in g_a],
        "mixed_proj": add(proj, g_b), "mixed_short": add([scale * g for g in g_a], g_b),
    }
    with torch.no_grad():
        p0 = policy.success_probability(prompts, correct)
    cos = lambda x, y: float(x @ y / (x.norm() * y.norm()).clamp(min=1e-300))  # noqa: E731
    out = {"n_a": int(in_a.sum()), "n_b": int((~in_a).sum()), "p_a": float(p0[in_a].mean()),
           "p_b": float(p0[~in_a].mean()), "norm_a": float(fa.norm()), "norm_b": float(fb.norm()),
           "dot_ab": float(fa @ fb), "cos_ab": cos(fa, fb), "dot_a_h": float(fa @ fh), "cos_a_h": cos(fa, fh)}

    def change(pol):
        with torch.no_grad():
            d = pol.success_probability(prompts, correct) - p0
        return float(d[in_a].mean()), float(d[~in_a].mean())

    for name, ascent in branches.items():
        pol, opt = copy.deepcopy((policy, optimizer))
        step_with(pol, opt, ascent)
        out["d_a_" + name], out["d_b_" + name] = change(pol)
    if cfg["beta"] == 0:
        # The mixed branch must be the step expected_step itself takes.
        pol, opt = copy.deepcopy((policy, optimizer))
        expected_step(pol, opt, prompts, correct, cfg)
        d_a, d_b = change(pol)
        out["mixed_check"] = max(abs(d_a - out["d_a_mixed"]), abs(d_b - out["d_b_mixed"]))

    if cfg["branch_samples"]:
        # Sampled GRPO steps: the same prompts, responses and rewards for every branch of a draw; a class's
        # contribution is masked out and the full batch denominator kept.
        state = torch.get_rng_state()
        draws = {k: [] for k in ("a", "b", "mixed")}
        missing = 0
        for _ in range(cfg["branch_samples"]):
            idx = torch.as_tensor(draw(rng, len(prompts), cfg))
            x, c, a_rows = prompts[idx], correct[idx], in_a[idx]
            if a_rows.all() or not a_rows.any():
                missing += 1
            responses = policy.sample(x, cfg["G"])
            rewards = torch.as_tensor(verify(responses.numpy(), c.numpy()), dtype=x.dtype)
            adv = calculate_advantages(rewards, kind=cfg["rule"], passk=cfg["passk"])
            for name, rows in (("a", a_rows), ("b", ~a_rows), ("mixed", torch.ones_like(a_rows))):
                pol, opt = copy.deepcopy((policy, optimizer))
                weighted = adv * pol.log_prob(x, responses) * rows[:, None]
                ascent = torch.autograd.grad(weighted.sum() / adv.numel(), list(pol.parameters()), allow_unused=True)
                step_with(pol, opt, [torch.zeros_like(q) if g is None else g for g, q in zip(ascent, pol.parameters())])
                draws[name].append(change(pol))
        torch.set_rng_state(state)
        for name, values in draws.items():
            values = np.asarray(values)
            out["s_d_a_" + name], out["s_d_b_" + name] = float(values[:, 0].mean()), float(values[:, 1].mean())
            out["s_sd_b_" + name] = float(values[:, 1].std())
        out["s_batches_missing_a_class"] = missing
    return out


def draw(rng, n, cfg):
    """Batch indices; with replacement keeps the earlier random stream."""
    if cfg["replace"]:
        return rng.integers(n, size=cfg["B"])
    return rng.choice(n, size=cfg["B"], replace=False)


def supervised_step(policy, optimizer, prompts, correct):
    """T1 control: one step of cross-entropy on the correct answers (full batch)."""
    loss = -policy.log_prob(prompts, correct[:, None, :]).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return 0


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
        reward_noise=cfg["reward_noise"], ppo_epochs=cfg["ppo_epochs"],
        clip_eps=cfg["clip_eps"], grad_clip=cfg["grad_clip"],
    )
    if cfg["p_start"] is not None:
        # The trainer has frozen the reference at p_ref; start the policy elsewhere.
        set_tabular_success(policy, train[1], float(cfg["p_start"]))
        p_init = success(policy, train)
        out["p_init"] = p_init.astype(np.float32).tolist()
    if cfg["update"] != "sampled" and (cfg["reward_noise"] > 0 or cfg["ppo_epochs"] > 1):
        raise ValueError("expected updates need an exact verifier and one update per batch")

    n = len(train[0])
    # Answer class 0 is "A" (the majority when cfg["majority"] > 1/K); the other classes are "B".
    in_a = torch.as_tensor(train[1][:, 0] == 0) if cfg["majority"] is not None else None
    if in_a is not None:
        # Where each class's probability sits at the start: success, and the mass on the other side's answers.
        with torch.no_grad():
            on_a = policy.probabilities(prompts_t)[:, 0, 0].numpy()
        a = in_a.numpy()
        out["start"] = {"p_a": float(p_init[a].mean()), "p_b": float(p_init[~a].mean()),
                        "b_on_a_answer": float(on_a[~a].mean()), "a_off_a_answer": float(1 - on_a[a].mean())}
    branch_steps = {int(round(f * cfg["steps"])) for f in cfg["branch_at"]}
    snap_at = {int(s) for s in cfg["snapshot_at"]}
    branch_rng = np.random.default_rng(cfg["seed"] + 20_000)
    branches = {}
    if (branch_steps or cfg["update"] in ("projected", "shortened")) and in_a is None:
        raise ValueError("branch_at and the projected/shortened updates need a class-share task (majority)")
    # T18: after the release step, class A's reward term is dropped (its prompts keep the KL term), and the KL
    # strength and direction can change. Everything before the release is the plain run.
    release_step = None
    if cfg["release_at"] is not None:
        if in_a is None or cfg["update"] != "expected_full":
            raise ValueError("release_at needs a class-share task (majority) and update='expected_full'")
        release_step = int(round(cfg["release_at"] * cfg["steps"]))
        released_cfg = dict(cfg, beta=cfg["beta"] if cfg["release_beta"] is None else cfg["release_beta"],
                            kl=cfg["release_kl"] or cfg["kl"])
        only_b = (~in_a).to(prompts_t.dtype)
        out["release_step"] = release_step
    checkpoints = {}
    absorbed_at = np.full(n, -1, dtype=np.int64)
    min_p = p_init.copy()
    history, snapshots = [], {}
    informative, expected = [], []
    # T0: last observed step with p >= tau (persistent absorption starts right
    # after it), and the largest p over the last `tail` share of training.
    last_above = {tau: np.full(n, -1, dtype=np.int64) for tau in cfg["taus"]}
    tail_start = int((1 - cfg["tail"]) * cfg["steps"])
    tail_max = np.zeros(n)
    half = {"step": None}  # first observed step with mean p >= 0.5 (T4)
    # T6 KL: largest decline from a running maximum, split at p_init (= p_ref when the
    # reference is the initial policy): the part above p_init and the part below it.
    run_max = p_init.copy()
    drop_above = np.zeros(n)

    def observe(step, p):
        if half["step"] is None and p.mean() >= 0.5:
            half["step"] = step
        np.minimum(min_p, p, out=min_p)
        np.maximum(run_max, p, out=run_max)
        np.maximum(drop_above, run_max - np.maximum(p, p_init), out=drop_above)
        newly = (absorbed_at < 0) & (p < ABSORBED) & (p_init >= ABSORBED)
        absorbed_at[newly] = step
        for tau, last in last_above.items():
            last[p >= tau] = step
        if step >= tail_start:
            np.maximum(tail_max, p, out=tail_max)

    def log(step):
        p = success(policy, train)
        observe(step, p)
        record = {"step": step, "train": summarize(p, p_init), "decreases": trainer.decreases}
        if in_a is not None:
            record["class_mean"] = [float(p[in_a.numpy()].mean()), float(p[~in_a.numpy()].mean())]
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
        if step % cfg["snapshot_every"] == 0 or step == cfg["steps"] or step == release_step or step in snap_at:
            snapshots[str(step)] = p.astype(np.float64).tolist() if snap_at else p.astype(np.float32).tolist()
        if cfg["drift_every"] and step % cfg["drift_every"] == 0 and step < cfg["steps"]:
            d = predicted_drift(policy, prompts_t, correct_t, cfg["rule"], cfg["G"], cfg["eta"], passk=cfg["passk"])
            W = advantage_gap(p, cfg["G"], cfg["rule"], passk=cfg["passk"])
            own = cfg["eta"] / n * W * grad_sq_norms(policy, prompts_t, correct_t)
            checkpoints[str(step)] = {
                "drift": d.numpy().astype(np.float32).tolist(),
                "own": own.astype(np.float32).tolist(),
            }

    log(0)
    # G: from anchor_step on, a KL term pulls toward the policy as it was at anchor_step (e.g. already collapsed).
    anchor_step = None
    if cfg["anchor_at"] is not None:
        if cfg["update"] != "expected_full" or cfg["release_at"] is not None or not cfg["anchor_beta"]:
            raise ValueError("anchor_at needs update='expected_full', no release, and anchor_beta > 0")
        anchor_step = int(round(cfg["anchor_at"] * cfg["steps"]))
        anchored_cfg = dict(cfg, beta=cfg["anchor_beta"], kl=cfg["anchor_kl"])
        out["anchor_step"] = anchor_step
    for step in range(1, cfg["steps"] + 1):
        if anchor_step is not None and step == anchor_step + 1:
            trainer.reference = policy.frozen_reference()
        if step - 1 in branch_steps:
            branches[str(step - 1)] = measure_branches(policy, trainer.optimizer, prompts_t, correct_t, in_a, cfg,
                                                       branch_rng)
        if cfg["update"] == "sampled":
            idx = draw(trainer.rng, n, cfg)
            metrics = trainer.train_step(train[0][idx], train[1][idx])
            informative.append(metrics["informative_before"])
            expected.append(metrics["expected_informative_before"])
        else:
            if cfg["update"] == "expected":
                idx = draw(trainer.rng, n, cfg)
                batch = (prompts_t[idx], correct_t[idx])
            elif cfg["update"] in ("expected_full", "supervised", "projected", "shortened"):
                batch = (prompts_t, correct_t)
            else:
                raise ValueError(f"unknown update {cfg['update']}")
            if cfg["update"] in ("projected", "shortened"):
                class_step(policy, trainer.optimizer, *batch, in_a, cfg)
            elif cfg["update"] == "supervised":
                supervised_step(policy, trainer.optimizer, *batch)
            elif release_step is not None and step > release_step:
                trainer.decreases += expected_step(policy, trainer.optimizer, *batch, released_cfg,
                                                   reference=trainer.reference, weight=only_b)
            else:
                now = anchored_cfg if anchor_step is not None and step > anchor_step else cfg
                trainer.decreases += expected_step(policy, trainer.optimizer, *batch, now, reference=trainer.reference)
        if step % cfg["log_every"] == 0 or step == cfg["steps"] or step in snap_at:
            log(step)
        elif cfg["track_every"] and step % cfg["track_every"] == 0:
            observe(step, success(policy, train))
        if cfg["stop_at_half"] and half["step"] is not None:
            if step % cfg["log_every"]:
                log(step)
            break

    out.update({
        "history": history,
        "snapshots": snapshots,
        "absorbed_at": absorbed_at.tolist(),
        "min_p": min_p.astype(np.float32).tolist(),
        "decreases_total": trainer.decreases,
        "last_above": {str(t): v.tolist() for t, v in last_above.items()},
        "tail_max": tail_max.astype(np.float32).tolist(),
        "step_to_half": half["step"],
        "drop_above": drop_above.astype(np.float32).tolist(),
    })
    if checkpoints:
        out["checkpoints"] = checkpoints
    if branches:
        out["branches"] = branches
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
    R, C = cfg["R"], cfg["chunks"]
    if R % C:
        raise ValueError("R must be a multiple of chunks")
    chunk_sums = torch.zeros(C, predicted.numel(), dtype=predicted.dtype)
    # Projection of each sample on the predicted direction: an exactly 1-D statistic.
    proj = torch.zeros(R, dtype=predicted.dtype)
    kept = torch.zeros(R, predicted.numel(), dtype=predicted.dtype) if cfg["boot"] else None
    for r in range(R):
        responses = policy.sample(prompts, cfg["G"])
        rewards = torch.as_tensor(verify(responses.numpy(), correct.numpy()), dtype=prompts.dtype)
        A = calculate_advantages(rewards, kind=cfg["rule"], passk=cfg["passk"])
        loss = -(A * policy.log_prob(prompts, responses)).mean()
        g = flat_grad(loss, policy)
        total += g
        total_sq += g * g
        chunk_sums[r * C // R] += g
        proj[r] = g @ predicted
        if kept is not None:
            kept[r] = g

    mean = total / R
    se = ((total_sq / R - mean**2).clamp(min=0) / (R - 1)).sqrt()
    norm = predicted.norm().item()
    mask = se > 0
    z = (mean - predicted)[mask] / se[mask]

    # Relative bias along the prediction: b = (m - pred).pred / |pred|^2.
    pp = predicted @ predicted
    extra = {}
    if C > 1:
        extra["bias_par"] = ((proj.mean() - pp) / pp).item()
        extra["bias_par_se"] = (proj.std() / np.sqrt(R) / pp).item()
        # Null for mean z^2 with the same coordinate correlation: half of the
        # chunks against the other half, each half-mean has twice the variance.
        chunk_means = chunk_sums / (R // C)
        rng = np.random.default_rng(cfg["seed"] + 2)
        null = []
        for _ in range(50):
            half = rng.permutation(C)[: C // 2]
            other = np.setdiff1d(np.arange(C), half)
            diff = chunk_means[half].mean(dim=0) - chunk_means[other].mean(dim=0)
            null.append(((diff[mask] / (2 * se[mask])) ** 2).mean().item())
        extra["mean_z2_null"] = null
    if kept is not None:
        # T7 null: resample the centred per-draw gradients, so the bootstrap mean is
        # centred on the truth by construction; mean z^2 of each draw against its own se.
        resid = kept - mean
        rng = np.random.default_rng(cfg["seed"] + 3)
        boot = []
        for _ in range(cfg["boot"]):
            draw = resid[torch.as_tensor(rng.integers(R, size=R))]
            m_b = draw.mean(dim=0)
            se_b = ((draw.pow(2).mean(dim=0) - m_b**2).clamp(min=0) / (R - 1)).sqrt()
            ok = mask & (se_b > 0)
            boot.append(((m_b[ok] / se_b[ok]) ** 2).mean().item())
        extra["mean_z2_boot"] = boot
    return {
        **extra,
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
    if cfg["mode"] == "enum":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extra", "experiments"))
        from enum_env import run_enum          # needs the third-party code (extra/experiments/third_party/fetch_wsw.sh)
        result = run_enum(cfg)
    else:
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
