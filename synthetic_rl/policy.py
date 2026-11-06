"""
Shared neural-network (MLP) policy for the synthetic GRPO task.

Shapes:
    prompts:       (B, d)
    logits:        (B, L, K)
    probabilities: (B, L, K)
    responses:     (B, G, L)
    log_probs:     (B, G)

B = number of prompts
d = number of prompt features
L = response length
K = number of possible tokens
G = responses sampled per prompt
"""

from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Categorical


class Policy(nn.Module):
    """Predict a token distribution for each response position."""

    def __init__(
        self,
        num_features=16,
        num_tokens=4,
        response_length=2,
        hidden_size=128,
    ):
        super().__init__()

        if num_features < 1:
            raise ValueError("num_features must be >= 1")
        if num_tokens < 2:
            raise ValueError("num_tokens must be >= 2")
        if response_length < 1:
            raise ValueError("response_length must be >= 1")
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")

        self.num_features = num_features
        self.num_tokens = num_tokens
        self.response_length = response_length

        self.features = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Predict K token scores for each of the L positions.
        self.output = nn.Linear(
            hidden_size,
            response_length * num_tokens,
        )

        # Equal initial logits give uniform token probabilities.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, prompts):
        """Return unnormalized token scores shaped (B, L, K).

        prompts must be a floating-point tensor shaped (B, d).
        """
        if prompts.ndim != 2:
            raise ValueError("prompts must have shape (B, d)")
        if prompts.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features per prompt"
            )

        hidden = self.features(prompts)
        logits = self.output(hidden)

        return logits.reshape(
            prompts.shape[0],
            self.response_length,
            self.num_tokens,
        )

    def distribution(self, prompts):
        """Return one categorical distribution per prompt and position."""
        return Categorical(logits=self(prompts))

    def probabilities(self, prompts):
        """Return token probabilities shaped (B, L, K)."""
        return self.distribution(prompts).probs

    @torch.no_grad()
    def sample(self, prompts, group_size):
        """Sample responses shaped (B, G, L).

        Sampling does not build a gradient graph. The trainer calls
        log_prob() afterward to calculate differentiable log-probabilities.
        """
        if group_size < 1:
            raise ValueError("group_size must be >= 1")

        distribution = self.distribution(prompts)

        # PyTorch puts the sample dimension first: (G, B, L).
        responses = distribution.sample((group_size,))

        # The verifier expects the prompt dimension first: (B, G, L).
        return responses.permute(1, 0, 2).contiguous()

    def log_prob(self, prompts, responses):
        """Return sequence log-probabilities shaped (B, G).

        responses must be an integer tensor shaped (B, G, L).
        Token log-probabilities are summed across response positions.
        """
        if responses.ndim != 3:
            raise ValueError("responses must have shape (B, G, L)")
        if responses.shape[0] != prompts.shape[0]:
            raise ValueError("Prompt and response batch sizes must match")
        if responses.shape[2] != self.response_length:
            raise ValueError("Response length does not match the policy")

        distribution = self.distribution(prompts)

        # Match PyTorch's sample-first convention: (G, B, L).
        tokens = responses.permute(1, 0, 2)
        token_log_probs = distribution.log_prob(tokens)

        # Sum over L, then return to prompt-first order.
        return token_log_probs.sum(dim=-1).transpose(0, 1)

    def success_probability(self, prompts, correct):
        """Return exact success probabilities shaped (B,).

        correct must be an integer tensor shaped (B, L).
        This calculation assumes one correct sequence per prompt.
        """
        expected_shape = (
            prompts.shape[0],
            self.response_length,
        )
        if correct.shape != expected_shape:
            raise ValueError("correct must have shape (B, L)")

        distribution = self.distribution(prompts)
        token_log_probs = distribution.log_prob(correct)

        # Product of correct-token probabilities, calculated in log space.
        return token_log_probs.sum(dim=-1).exp()

    def entropy(self, prompts):
        """Return sequence entropy shaped (B,).

        Independent token positions have additive entropies.
        """
        return self.distribution(prompts).entropy().sum(dim=-1)

    def frozen_reference(self):
        """Copy this policy for use as the fixed KL reference.

        Call after any supervised warm-start and before RL training.
        The copy has independent parameters and does not track gradients.
        """
        reference = deepcopy(self)
        reference.requires_grad_(False)
        reference.eval()
        return reference

    def kl_divergence(self, prompts, reference, direction="reverse"):
        """Return exact sequence KL per prompt, shaped (B,).

        reverse: KL(self || reference), the default training penalty.
        forward: KL(reference || self), for the E2 comparison.
        Sum over tokens and positions; the trainer averages over prompts.
        Gradients flow only through this policy, never the reference.
        """
        if direction not in ("reverse", "forward"):
            raise ValueError("direction must be 'reverse' or 'forward'")
        if (
            self.num_features,
            self.response_length,
            self.num_tokens,
        ) != (
            reference.num_features,
            reference.response_length,
            reference.num_tokens,
        ):
            raise ValueError("Policy and reference task dimensions must match")

        log_probs = self(prompts).log_softmax(dim=-1)
        with torch.no_grad():
            reference_log_probs = reference(prompts).log_softmax(dim=-1)

        if direction == "reverse":
            token_kl = log_probs.exp() * (log_probs - reference_log_probs)
        else:
            token_kl = reference_log_probs.exp() * (reference_log_probs - log_probs)
        return token_kl.sum(dim=(-1, -2))


# Optional demonstration: generate a small example dataset to check that the
# policy, sampling shapes, and verifier work together. This does not train the
# policy
def main():
    """Check that the policy connects to the existing synthetic task."""
    from .synthetic_task import SyntheticTask, verify

    torch.manual_seed(0)

    task = SyntheticTask(
        num_features=16,
        num_tokens=4,
        response_length=2,
        seed=0,
    )

    prompts, correct = task.generate_dataset(8)

    # The task uses NumPy; the neural network uses PyTorch.
    prompts = torch.as_tensor(prompts, dtype=torch.float32)
    correct = torch.as_tensor(correct, dtype=torch.long)

    policy = Policy(
        num_features=task.num_features,
        num_tokens=task.num_tokens,
        response_length=task.response_length,
    )

    responses = policy.sample(prompts, group_size=4)
    log_probs = policy.log_prob(prompts, responses)

    # The existing verifier accepts NumPy arrays.
    rewards = verify(
        responses.cpu().numpy(),
        correct.cpu().numpy(),
    )

    with torch.no_grad():
        probabilities = policy.probabilities(prompts)
        success = policy.success_probability(prompts, correct)

    print("Probabilities:", tuple(probabilities.shape))
    print("Responses:    ", tuple(responses.shape))
    print("Rewards:      ", rewards.shape)
    print("Log-probs:    ", tuple(log_probs.shape))
    print("Mean success: ", success.mean().item())

    # Uniform policy: each correct token has probability 1/K.
    expected_success = task.num_tokens ** (-task.response_length)
    assert torch.allclose(
        success,
        torch.full_like(success, expected_success),
    )

    print("\nPolicy check passed.")


if __name__ == "__main__":
    main()
