"""Unbiased pass@k from per-problem counts (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""

import numpy as np


def pass_at_k(c, n, k):
    """Probability that at least one of k samples (drawn without replacement from n, c correct) is correct."""
    c, n = np.asarray(c, dtype=float), np.asarray(n, dtype=float)
    out = np.ones_like(c)
    fail = n - c >= k
    # prod_{i=n-c+1}^{n} (1 - k / i), computed in logs for stability
    for j in np.nonzero(fail)[0]:
        i = np.arange(n[j] - c[j] + 1, n[j] + 1)
        out[j] = 1 - np.exp(np.sum(np.log1p(-k / i)))
    return out
