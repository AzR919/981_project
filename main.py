"""
GRPO playground: a one-question bandit with a binary verifier.

There are K possible answers and one is correct. The policy is a softmax
over K logits that starts uniform, so the initial success rate is 1/K.
Each step samples a group of G answers, scores them with the verifier,
computes advantages within the group, and takes one SGD step. Because the
policy is a plain softmax, the success probability p is known exactly.

Example usage:
    python main.py                          # GRPO, K=8, G=8
    python main.py -K 1000 -G 16            # sparse reward: p0 = 0.001
    python main.py --advantage mean         # Dr. GRPO (no std division)
    python main.py --beta 2 -n 2000         # KL penalty: settles below p = 1
    python main.py --advantage passk -k 4   # pass@k advantage shaping
"""

import math

import numpy as np

from args import parse_args

EPS = 1e-5  # std stabilizer: A = (r - mean) / sqrt(mean (1 - mean) + EPS)


def comb(n, k):
    """Binomial coefficient that returns 0 instead of raising for n < k."""
    return math.comb(n, k) if 0 <= k <= n else 0


def advantages(r, kind, k=4):
    """Advantages for one group of 0/1 rewards r (length G)."""
    G = len(r)
    m = r.mean()
    if kind == "mean":
        return r - m
    if kind == "rloo":
        return r - (r.sum() - r) / (G - 1)
    A = (r - m) / math.sqrt(m * (1 - m) + EPS)
    if kind == "grpo" or m in (0.0, 1.0):
        return A
    # passk: scale the GRPO advantage by the probability that the other
    # k-1 answers (drawn from the rest of the group) are all wrong
    # (Thrampoulidis et al., combinatorial form of their Eq. 14).
    n_wrong = int(G - r.sum())
    f_pos = comb(n_wrong, k - 1) / comb(G - 1, k - 1)
    f_neg = comb(n_wrong - 1, k - 1) / comb(G - 1, k - 1)
    return np.where(r == 1, f_pos * A, f_neg * A)


def omega(p, G, kind):
    """Finite-G weight Omega_G(p) from Proposition 1 of the proposal."""
    if kind == "mean":
        return (G - 1) / G
    if kind == "rloo":
        return 1.0
    total = 0.0
    for j in range(G):  # j = number of *other* correct answers
        w = math.comb(G - 1, j) * p**j * (1 - p) ** (G - 1 - j)
        cp, cm = (j + 1) / G, j / G
        total += w * ((1 - cp) / math.sqrt(cp * (1 - cp) + EPS)
                      + cm / math.sqrt(cm * (1 - cm) + EPS))
    return total


def predicted_fixed_point(p_ref, G, beta, kind):
    """Smallest p above p_ref with logit p = logit p_ref + Omega_G(p) / beta.

    This is where small-step SGD should settle when beta > 0.
    """
    z_ref = math.log(p_ref / (1 - p_ref))
    z = z_ref
    while z < 20:
        p = 1 / (1 + math.exp(-z))
        if z_ref + omega(p, G, kind) / beta - z <= 0:
            return p
        z += 1e-3
    return 1.0


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    K, G = args.num_answers, args.group_size
    correct = 0
    logits = np.zeros(K)
    log_ref = np.full(K, -math.log(K))  # uniform reference policy

    print(f"K={K} (p0={1 / K:.4g})  G={G}  advantage={args.advantage}  "
          f"lr={args.lr}  beta={args.beta}  steps={args.steps}")
    if args.beta > 0 and args.advantage != "passk":
        p_star = predicted_fixed_point(1 / K, G, args.beta, args.advantage)
        print(f"predicted fixed point p* = {p_star:.4f} (Proposition 1)")
    print(f"{'step':>6} {'p(correct)':>11} {'informative':>12} {'entropy':>8}")

    informative = 0
    for step in range(args.steps + 1):
        pi = np.exp(logits - logits.max())
        pi /= pi.sum()

        if step % args.print_every == 0 or step == args.steps:
            frac = informative / max(step, 1)
            entropy = -(pi * np.log(pi + 1e-300)).sum()
            print(f"{step:>6} {pi[correct]:>11.4f} {frac:>12.3f} {entropy:>8.3f}")
        if step == args.steps:
            break

        y = rng.choice(K, size=G, p=pi)      # sample a group of G answers
        r = (y == correct).astype(float)     # the verifier
        informative += 0.0 < r.mean() < 1.0  # all-equal groups give no signal
        A = advantages(r, args.advantage, args.passk)

        # d log pi(y) / d logits = onehot(y) - pi, so the policy gradient is
        # mean_i A_i (onehot(y_i) - pi).
        grad = (np.bincount(y, weights=A, minlength=K) - A.sum() * pi) / G
        if args.beta > 0:
            # Exact gradient of KL(pi || pi_ref): pi * (log(pi/pi_ref) - KL)
            log_ratio = np.log(np.maximum(pi, 1e-300)) - log_ref
            grad -= args.beta * pi * (log_ratio - (pi * log_ratio).sum())
        logits += args.lr * grad


if __name__ == "__main__":
    main()
