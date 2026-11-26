"""
A correct minority class in the enumerable RLVR environment of Li, Pandey and Chen (2026),
"Wrong the Same Way" (MIT licence; fetched by experiments/third_party/fetch_wsw.sh at a pinned commit).

Their environment: 64 prompts (a, b), a 2-layer transformer shared by all prompts, and a response space of
100 two-digit strings, small enough that pi(.|x) is computed exactly. Their verifier accepts each prompt's
product plus designated wrong strings. Here the verifier is replaced: the prompts are split into K answer
classes, class 0 holding n_major of them, and a prompt's only accepted response is its class's string.
So the minority's answer is correct, and the question is the one of our T1/T11/T15: does training push the
minority classes' success down when class 0 is a majority?

Used from experiments/run.py with mode "enum". Reused from their code: the prompts, the policies
(shared transformer, tabular), the SFT routine, and the objectives (exact policy gradient with the
standardised advantage, sampled GRPO). The loop and the measurements are ours.

Config keys: K, n_major, partition ("random": classes unrelated to the prompt; "parity": class 0 = a*b even),
enum_policy ("shared", "tabular", "copies"), algo ("pg": exact expected update; "grpo": sampled, G per prompt),
optimizer, eta, steps, log_every, seed, G, floor, sft_steps, branch_eta.
Prevention levers (T21; all off by default): enum_beta and enum_kl ("reverse": KL(pi || base), "forward":
KL(base || pi)) add a KL anchor to the base model; enum_passk = k weights each prompt's loss by k (1 - p)^(k-1),
with p the prompt's exact success.
"""

import copy
import os
import sys

import numpy as np
import torch

WSW = os.environ.get("WSW_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "wsw", "src"))
ENUM_DEFAULTS = {"partition": "random", "enum_policy": "shared", "algo": "pg", "floor": 0.1, "sft_steps": 600,
                 "branch_eta": 0.001, "enum_beta": 0.0, "enum_kl": "reverse", "enum_passk": None}


def load():
    if not os.path.isdir(WSW):
        raise FileNotFoundError(f"{WSW} is missing; run experiments/third_party/fetch_wsw.sh")
    if WSW not in sys.path:
        sys.path.insert(0, WSW)
    import base
    import env
    import policies
    import rlvr
    return env, policies, base, rlvr


class ClassVerifier:
    """Each prompt accepts exactly one response: the string of its answer class. Class 0 is the majority."""

    def __init__(self, prompts, n_major, K, seed, partition="random"):
        n = len(prompts)
        rng = np.random.default_rng([seed, n_major, K])
        self.strings = rng.choice(100, size=K, replace=False)
        cls = np.empty(n, dtype=np.int64)
        if partition == "parity":
            if K != 2:
                raise ValueError("the parity partition has two classes")
            cls[:] = [(a * b) % 2 for a, b in prompts]          # class 0: even product
        elif partition == "random":
            order = rng.permutation(n)
            cls[order[:n_major]] = 0
            cls[order[n_major:]] = 1 + np.arange(n - n_major) % (K - 1)
        else:
            raise ValueError(f"unknown partition {partition}")
        self.n, self.cls, self.gold = n, cls, self.strings[cls]
        self.accept = np.zeros((n, 100))
        self.accept[np.arange(n), self.gold] = 1.0
        self.is_gold, self.is_hack = self.accept.copy(), np.zeros((n, 100))
        self.hack = np.zeros((n, 0), dtype=np.int64)


def class_target(ver, floor):
    """SFT target of the base model, the same for every prompt: `floor` spread over all 100 strings and the
    rest shared equally by the K class strings. The base then solves every prompt with about (1 - floor)/K."""
    q = np.full((ver.n, 100), floor / 100)
    q[:, ver.strings] += (1 - floor) / len(ver.strings)
    return torch.as_tensor(q / q.sum(-1, keepdims=True), dtype=torch.float32)


def per_prompt_loss(rlvr, lp, r):
    """Each prompt's exact policy-gradient loss with the standardised advantage; its mean is exact_pg_loss."""
    p = lp.exp()
    return -((p * rlvr.exact_advantage(p, r, "std")).detach() * lp).sum(-1)


def sampled_per_prompt_loss(lp, r, G, gen):
    """Each prompt's sampled GRPO loss; its mean is their grpo_loss with norm="std" (same sampling call)."""
    with torch.no_grad():
        idx = torch.multinomial(lp.exp(), G, replacement=True, generator=gen)
        rew = torch.gather(r, 1, idx)
        adv = rew - rew.mean(dim=1, keepdim=True)
        adv = adv / (rew.std(dim=1, unbiased=False, keepdim=True) + 1e-4)
    return -(adv * torch.gather(lp, 1, idx)).mean(dim=1)


def gold_mass(lp, accept):
    return (lp.exp() * accept).sum(-1)


def branches(rlvr, policy, accept, in_a, eta):
    """One plain gradient step of size eta on class A's part, class B's part, or all of the exact loss, from
    copies of the same state: the exact change in the mean success of class A and of class B."""
    with torch.no_grad():
        p0 = gold_mass(policy.full_logprobs(), accept)
    out = {"p_a": float(p0[in_a].mean()), "p_b": float(p0[~in_a].mean())}
    for name, mask in (("a", in_a), ("b", ~in_a), ("mixed", torch.ones_like(in_a))):
        pol = copy.deepcopy(policy)
        opt = torch.optim.SGD(pol.parameters(), lr=eta)
        loss = (per_prompt_loss(rlvr, pol.full_logprobs(), accept) * mask).sum() / len(mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            d = gold_mass(pol.full_logprobs(), accept) - p0
        out["d_a_" + name], out["d_b_" + name] = float(d[in_a].mean()), float(d[~in_a].mean())
    return out


def run_enum(cfg):
    cfg.update({k: cfg.get(k, v) for k, v in ENUM_DEFAULTS.items()})
    env, policies, base, rlvr = load()
    seed, steps, n_tail = cfg["seed"], cfg["steps"], int((1 - cfg["tail"]) * cfg["steps"])
    prompts = env.build_prompts(8)
    ver = ClassVerifier(prompts, cfg["n_major"], cfg["K"], seed, cfg["partition"])
    n, in_a = ver.n, torch.as_tensor(ver.cls == 0)
    accept = torch.as_tensor(ver.accept, dtype=torch.float32)

    shared = policies.TransformerPolicy(prompts, seed=seed)
    base.sft(shared, class_target(ver, cfg["floor"]), steps=cfg["sft_steps"], lr=3e-3, seed=seed)
    with torch.no_grad():
        lp0 = shared.full_logprobs().detach().clone()
    p_init = gold_mass(lp0, accept).numpy().astype(np.float64)
    out = {"p_init": p_init.tolist(), "cls": ver.cls.tolist(), "strings": ver.strings.tolist(),
           "share_a": float(in_a.float().mean()), "torch": torch.__version__,
           "wsw_commit": open(os.path.join(WSW, "..", "COMMIT")).read().strip()}

    kind = cfg["enum_policy"]
    if kind == "shared":
        out["branches"] = branches(rlvr, shared, accept, in_a, cfg["branch_eta"])
        models = [shared]
    elif kind == "tabular":
        models = [base.matched_tabular(lp0, n)]
    elif kind == "copies":
        models = [copy.deepcopy(shared) for _ in range(n)]       # copy i is trained on prompt i alone
    else:
        raise ValueError(f"unknown enum_policy {kind}")
    make = torch.optim.SGD if cfg["optimizer"] == "sgd" else torch.optim.Adam
    opts = [make(m.parameters(), lr=cfg["eta"]) for m in models]
    gen = torch.Generator().manual_seed(seed + 1)
    torch.manual_seed(seed + 1)

    def logprobs():
        if kind == "copies":
            return torch.cat([m.full_logprobs(prompt_idx=[i]) for i, m in enumerate(models)])
        return models[0].full_logprobs()

    min_p, tail_max, history = p_init.copy(), np.zeros(n), []
    for t in range(steps + 1):
        lp = logprobs()
        p = gold_mass(lp.detach(), accept).numpy().astype(np.float64)
        np.minimum(min_p, p, out=min_p)
        if t >= n_tail:
            np.maximum(tail_max, p, out=tail_max)
        if t % cfg["log_every"] == 0 or t == steps:
            history.append({"step": t, "class_mean": [float(p[ver.cls == 0].mean()), float(p[ver.cls != 0].mean())],
                            "mean": float(p.mean())})
        if t == steps:
            break
        if kind == "copies":
            # Each copy sees its own prompt only; the sum keeps every copy's own gradient, with no 1/n.
            loss = per_prompt_loss(rlvr, lp, accept).sum()
        elif cfg["algo"] not in ("pg", "grpo"):
            raise ValueError(f"unknown algo {cfg['algo']}")
        elif cfg["enum_passk"]:
            k = cfg["enum_passk"]
            weight = k * (1 - torch.as_tensor(p, dtype=lp.dtype)) ** (k - 1)
            per = (per_prompt_loss(rlvr, lp, accept) if cfg["algo"] == "pg"
                   else sampled_per_prompt_loss(lp, accept, cfg["G"], gen))
            loss = (weight * per).mean()
        elif cfg["algo"] == "pg":
            loss = rlvr.exact_pg_loss(lp, accept, norm="std")
        else:
            loss = rlvr.grpo_loss(lp, accept, G=cfg["G"], norm="std", gen=gen)
        if cfg["enum_beta"] > 0:
            if kind == "copies":
                raise ValueError("the KL anchor is implemented for one policy")
            if cfg["enum_kl"] == "reverse":
                kl = (lp.exp() * (lp - lp0)).sum(-1).mean()
            elif cfg["enum_kl"] == "forward":
                kl = (lp0.exp() * (lp0 - lp)).sum(-1).mean()
            else:
                raise ValueError(f"unknown enum_kl {cfg['enum_kl']}")
            loss = loss + cfg["enum_beta"] * kl
        for opt in opts:
            opt.zero_grad()
        loss.backward()
        for opt in opts:
            opt.step()
    out.update({"history": history, "min_p": min_p.tolist(), "tail_max": tail_max.tolist(), "p_final": p.tolist()})
    return out
