"""
Set the base policy's success probabilities before RL training.

Run these before creating the Trainer, which freezes the reference policy.
"""

import numpy as np
import torch


@torch.no_grad()
def set_tabular_success(policy, correct, p0):
    """Give prompt i success probability p0[i] in a tabular policy.

    The policy must be arch="linear" on one-hot prompts, so column i of the
    output weight holds prompt i's logits. Each position puts probability
    q = p0^(1/L) on its correct token and splits the rest evenly.
    """
    if policy.arch != "linear":
        raise ValueError("set_tabular_success needs arch='linear'")

    correct = np.asarray(correct)
    num_prompts, L = correct.shape
    K = policy.num_tokens
    p0 = np.broadcast_to(np.asarray(p0, dtype=float), (num_prompts,))
    if np.any((p0 <= 0) | (p0 >= 1)):
        raise ValueError("p0 must be in (0, 1)")

    q = p0 ** (1.0 / L)
    logits = np.repeat(np.log((1 - q) / (K - 1))[:, None, None], L, axis=1)
    logits = np.repeat(logits, K, axis=2)                     # (N, L, K)
    rows = np.arange(num_prompts)[:, None]
    logits[rows, np.arange(L)[None, :], correct] = np.log(q)[:, None]

    weight = policy.output.weight                             # (L*K, d)
    weight.zero_()
    weight[:, :num_prompts] = torch.as_tensor(
        logits.reshape(num_prompts, L * K).T,
        dtype=weight.dtype,
    )


def supervised_warmstart(policy, prompts, targets, target_success, correct,
                         lr=0.5, max_steps=20000, direction="up"):
    """Full-batch SGD on the cross-entropy of target tokens.

    direction "up": targets are the correct answers; stop once the mean
    success reaches target_success.
    direction "down": targets are decoy (wrong) answers; stop once the mean
    success falls to target_success. This makes low-p bases for L = 1.

    Returns (steps taken, mean success reached).
    """
    prompts = torch.as_tensor(prompts, dtype=next(policy.parameters()).dtype)
    targets = torch.as_tensor(targets, dtype=torch.long)
    correct = torch.as_tensor(correct, dtype=torch.long)
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)

    def mean_success():
        with torch.no_grad():
            return policy.success_probability(prompts, correct).mean().item()

    success = mean_success()
    for step in range(max_steps):
        if (direction == "up" and success >= target_success) or (
            direction == "down" and success <= target_success
        ):
            return step, success
        log_probs = policy.distribution(prompts).log_prob(targets)  # (N, L)
        loss = -log_probs.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        success = mean_success()

    return max_steps, success
