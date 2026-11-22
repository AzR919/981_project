"""
Exact per-prompt diagnostics: gradients of p, interference between prompts,
predicted drift and the effective step.

All gradients are of the exact success probability p_x(theta), so no
sampling is involved.
"""

import numpy as np
import torch
from torch.func import functional_call, jacrev, jvp

from .theory import advantage_gap


def _success_fn(policy, prompts, correct):
    names = [name for name, _ in policy.named_parameters()]

    def success(*params):
        logits = functional_call(policy, dict(zip(names, params)), (prompts,))
        log_probs = logits.log_softmax(dim=-1)
        picked = log_probs.gather(-1, correct.unsqueeze(-1)).squeeze(-1)
        return picked.sum(dim=-1).exp()

    params = tuple(p.detach() for _, p in policy.named_parameters())
    return success, params


def success_jacobian(policy, prompts, correct):
    """Return (p, J) with J[x] = d p_x / d theta flattened, shape (N, P)."""
    success, params = _success_fn(policy, prompts, correct)
    jac = jacrev(success, argnums=tuple(range(len(params))))(*params)
    J = torch.cat([j.reshape(j.shape[0], -1) for j in jac], dim=1)
    with torch.no_grad():
        p = success(*params)
    return p, J


def grad_sq_norms(policy, prompts, correct, chunk=128):
    """||grad p_x||^2 for every prompt, in chunks to bound the Jacobian's memory."""
    out = [
        (success_jacobian(policy, prompts[i:i + chunk], correct[i:i + chunk])[1] ** 2).sum(dim=1)
        for i in range(0, prompts.shape[0], chunk)
    ]
    return torch.cat(out).numpy()


def interference(policy, prompts, correct, subset=128, seed=0):
    """Interference measures from the gradient Gram matrix on a prompt subset.

    mean_cos      mean signed cosine between different prompts' gradients
    mean_abs_cos  mean absolute cosine (strength, regardless of sign)
    mean_sq_grad  mean ||grad p_x||^2
    mean_sq_grad_logit  mean ||grad logit p_x||^2 = ||grad p||^2 / (p(1-p))^2
    """
    n = prompts.shape[0]
    idx = np.random.default_rng(seed).choice(n, size=min(subset, n), replace=False)
    p, J = success_jacobian(policy, prompts[idx], correct[idx])
    sq = (J * J).sum(dim=1)
    norms = sq.sqrt().clamp(min=1e-300)
    cos = (J @ J.T) / (norms[:, None] * norms[None, :])
    off = ~torch.eye(len(idx), dtype=torch.bool)
    return {
        "mean_cos": cos[off].mean().item(),
        "mean_abs_cos": cos[off].abs().mean().item(),
        "mean_sq_grad": sq.mean().item(),
        "mean_sq_grad_logit": (sq / (p * (1 - p)) ** 2).mean().item(),
    }


def predicted_drift(policy, prompts, correct, rule, group_size, eta, passk=None):
    """First-order expected change of every p_x in one step, shape (N,).

    With prompts drawn uniformly, the expected update is
    eta * mean_j W_G(p_j) grad p_j, so d_x = eta * grad p_x . that vector.
    The own-prompt part alone is eta / N * W_G(p_x) ||grad p_x||^2.
    """
    success, params = _success_fn(policy, prompts, correct)
    params = tuple(p.clone().requires_grad_(True) for p in params)
    p = success(*params)
    W = torch.as_tensor(
        advantage_gap(p.detach().numpy(), group_size, rule, passk=passk),
        dtype=p.dtype,
    )
    direction = torch.autograd.grad((W * p).mean(), params)
    _, d = jvp(success, tuple(q.detach() for q in params), direction)
    return eta * d.detach()


def effective_step(p, group_size, rule, eta, batch_size, passk=None):
    """s_bar = (eta / B) * mean_x W_G(p_x) p_x (1 - p_x)."""
    p = np.asarray(p, dtype=float)
    W = advantage_gap(p, group_size, rule, passk=passk)
    return float(eta / batch_size * np.mean(W * p * (1 - p)))
