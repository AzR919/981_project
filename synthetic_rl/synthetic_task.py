"""
Synthetic task for GRPO experiments.

Each prompt is a vector of random features. A fixed hidden rule assigns
one correct token to each position in the response.

Current setup generates:
- 512 training prompts for training/learning
- 256 test prompts for eval

A response earns reward 1 only when every token is correct.

Run a quick check:
    python synthetic_task.py
"""

import numpy as np


FEATURES = ("gaussian", "onehot", "correlated")


class SyntheticTask:
    """Generate prompts and correct responses using a fixed hidden rule.

    features:
        gaussian    independent standard normal features (default)
        onehot      prompt i is the i-th basis vector, so a linear policy
                    without bias is tabular; num_features is the number of
                    distinct prompts and every dataset reuses them
        correlated  sqrt(rho) * c + sqrt(1 - rho) * g with one shared random
                    vector c, so every pair of prompts has feature cosine
                    about rho

    answers:
        rule        argmax of a fixed linear score (default; generalizes)
        random      uniform per prompt and position, so similar features do
                    not imply similar answers (controlled conflict)

    class_shares:
        None, or K target frequencies for the answer classes (rule answers).
        Answers become argmax(W x + b) with an offset b calibrated on a
        separate random stream, so the prompts and the rule W are identical
        for every choice of shares and only the class frequencies change.
        A constant feature is appended to the prompts (and num_features
        counts it), so a linear policy without bias can represent b.
    """

    def __init__(
        self,
        num_tokens=4,
        response_length=2,
        num_features=16,
        seed=0,
        features="gaussian",
        rho=0.0,
        answers="rule",
        class_shares=None,
    ):
        if features not in FEATURES:
            raise ValueError(f"features must be one of {FEATURES}")
        if answers not in ("rule", "random"):
            raise ValueError("answers must be 'rule' or 'random'")
        if not 0 <= rho < 1:
            raise ValueError("rho must be in [0, 1)")
        if num_tokens < 2:
            raise ValueError("num_tokens must be >= 2")
        if response_length < 1:
            raise ValueError("response_length must be >= 1")
        if num_features < 1:
            raise ValueError("num_features must be >= 1")

        self.num_tokens = num_tokens          # K
        self.response_length = response_length  # L
        self.num_features = num_features     # d
        self.features = features
        self.rho = rho
        self.answers = answers
        self.rng = np.random.default_rng(seed)

        # One fixed rule per response position.
        # Shape: (L, K, d).
        self.weights = self.rng.standard_normal(
            (response_length, num_tokens, num_features)
        )

        # Drawn after the rule so the gaussian datasets match earlier runs.
        if features == "correlated":
            self.shared = self.rng.standard_normal(num_features)

        self.class_shares = None
        self.offset = np.zeros((response_length, num_tokens))
        if class_shares is not None:
            shares = np.asarray(class_shares, dtype=float)
            if answers != "rule" or features == "onehot" or shares.shape != (num_tokens,):
                raise ValueError("class_shares needs rule answers, non-onehot features and K shares")
            self.class_shares = shares / shares.sum()
            self.offset = self._calibrate_offset(seed)

    def _features(self, rng, num_prompts):
        """Raw feature vectors (without the constant column)."""
        d = self.num_features - (1 if self.class_shares is not None else 0)
        x = rng.standard_normal((num_prompts, d))
        if self.features == "correlated":
            x = np.sqrt(self.rho) * self.shared[:d] + np.sqrt(1 - self.rho) * x
        return x

    def _calibrate_offset(self, seed, samples=50_000, iters=3000, lr=0.5):
        """Offset b with class frequencies of argmax(W x + b) equal to the shares."""
        x = self._features(np.random.default_rng(seed + 7_919), samples)
        x = np.concatenate([x, np.ones((samples, 1))], axis=1)
        scores = np.einsum("nd,lkd->nlk", x, self.weights)
        b = np.zeros((self.response_length, self.num_tokens))
        target = np.log(self.class_shares)
        for _ in range(iters):
            pick = (scores + b).argmax(axis=-1)
            freq = np.stack([np.bincount(pick[:, l], minlength=self.num_tokens) for l in range(self.response_length)])
            b += lr * (target - np.log((freq + 1) / (samples + self.num_tokens)))
        return b

    def generate_dataset(self, num_prompts):
        """Return prompt features and their correct token sequences.

        prompts: shape (num_prompts, d)
        correct: shape (num_prompts, L)
        """
        if num_prompts < 1:
            raise ValueError("num_prompts must be >= 1")

        if self.features == "onehot":
            if num_prompts > self.num_features:
                raise ValueError("onehot needs num_prompts <= num_features")
            prompts = np.eye(self.num_features)[:num_prompts]
        elif self.class_shares is not None:
            prompts = np.concatenate([self._features(self.rng, num_prompts), np.ones((num_prompts, 1))], axis=1)
        else:
            prompts = self.rng.standard_normal(
                (num_prompts, self.num_features)
            )
            if self.features == "correlated":
                prompts = (
                    np.sqrt(self.rho) * self.shared
                    + np.sqrt(1 - self.rho) * prompts
                )

        if self.answers == "random":
            correct = self.rng.integers(
                self.num_tokens,
                size=(num_prompts, self.response_length),
            )
            return prompts, correct

        # Score each possible token at each response position.
        # Shape: (num_prompts, L, K).
        scores = np.einsum("nd,lkd->nlk", prompts, self.weights) + self.offset

        # The highest-scoring token is correct at each position.
        correct = scores.argmax(axis=-1)

        return prompts, correct


def class_counts(n, majority, num_classes):
    """Deterministic class sizes summing to n: class 0 gets round(majority * n), the rest is split as evenly
    as possible over classes 1..K-1 (lower classes take the remainder)."""
    n_a = int(round(majority * n))
    q, r = divmod(n - n_a, num_classes - 1)
    counts = [n_a] + [q + (1 if c < r else 0) for c in range(num_classes - 1)]
    if min(counts) < 1:
        raise ValueError(f"class sizes {counts} leave a class empty; raise n or move majority away from 0 and 1")
    return counts


def block_dataset(counts, block_dim, shared_scale, rng, response_length=1):
    """Block features: a prompt of class c has |g|, g ~ N(0, I_block_dim), in block c, zeros in every other
    block, and a last coordinate equal to shared_scale. Its correct answer is c at every position.

    Prompts are ordered by class (class 0 first). The feature dimension is K * block_dim + 1, so different
    classes have inner product exactly shared_scale^2 and prompts of one class a non-negative one.
    """
    K, n = len(counts), int(sum(counts))
    cls = np.repeat(np.arange(K), counts)
    prompts = np.zeros((n, K * block_dim + 1))
    g = np.abs(rng.standard_normal((n, block_dim)))
    for c in range(K):
        rows = cls == c
        prompts[rows, c * block_dim:(c + 1) * block_dim] = g[rows]
    prompts[:, -1] = shared_scale
    return prompts, np.repeat(cls[:, None], response_length, axis=1)


def verify(responses, correct):
    """Score sampled responses using exact sequence matching.

    responses: shape (B, G, L)
    correct:   shape (B, L)
    rewards:   shape (B, G)

    B = number of prompts
    G = responses sampled per prompt
    L = response length
    """
    responses = np.asarray(responses)
    correct = np.asarray(correct)

    if responses.ndim != 3 or correct.ndim != 2:
        raise ValueError(
            "Expected responses shaped (B, G, L) and correct shaped (B, L)"
        )

    if (responses.shape[0], responses.shape[2]) != correct.shape:
        raise ValueError(
            "Responses and correct answers must have matching "
            "prompt counts and response lengths"
        )

    # All token positions must match to earn reward 1.
    return np.all(
        responses == correct[:, None, :],
        axis=-1,
    ).astype(float)


def main():
    task = SyntheticTask(
        num_tokens=4,
        response_length=2,
        num_features=16,
        seed=0,
    )

    # Both datasets use the same fixed hidden rule.
    x_train, correct_train = task.generate_dataset(512)
    x_test, correct_test = task.generate_dataset(256)

    print(f"Training prompts: {x_train.shape}")
    print(f"Training answers: {correct_train.shape}")
    print(f"Test prompts:     {x_test.shape}")
    print(f"Test answers:     {correct_test.shape}")

    # Check the verifier with two responses for the first prompt.
    responses = np.repeat(correct_train[:1, None, :], 2, axis=1)

    # Keep the first response correct; change one token in the second.
    responses[0, 1, 0] = (
        responses[0, 1, 0] + 1
    ) % task.num_tokens

    rewards = verify(responses, correct_train[:1])

    print(f"\nCorrect answer:   {correct_train[0]}")
    print(f"Sample responses:\n{responses[0]}")
    print(f"Rewards:          {rewards[0]}")

    assert np.array_equal(rewards, [[1.0, 0.0]])
    print("\nVerifier check passed.")


if __name__ == "__main__":
    main()