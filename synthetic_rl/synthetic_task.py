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
        scores = np.einsum("nd,lkd->nlk", prompts, self.weights)

        # The highest-scoring token is correct at each position.
        correct = scores.argmax(axis=-1)

        return prompts, correct


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