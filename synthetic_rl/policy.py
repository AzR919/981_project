"""Factorized token policy for the synthetic GRPO task.

Shapes:
    prompts:       (B, d)
    logits:        (B, L, K)
    probabilities: (B, L, K)
    responses:     (B, G, L)
    log_probs:     (B, G)

B is the number of prompts, d is the number of prompt features,
L is the response length, K is the number of possible tokens,
and G is the number of responses sampled per prompt.
"""

from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Categorical


class Policy(nn.Module):
    """Predict a categorical distribution for each response position.

    The linear architecture acts as a tabular policy on one-hot prompts.
    Each prompt has its own logits because the layer has no shared bias.
    """

    def __init__(
        self,
        num_features=16,
        num_tokens=4,
        response_length=2,
        hidden_size=128,
        architecture="mlp",
        arch=None,
    ):
        """architecture "mlp": two ReLU layers of width hidden_size.
        architecture "linear": logits = W x with no bias, so one-hot prompts give a
        tabular policy and no parameter is shared between them.
        arch is an alias for architecture (the simulator in experiments/ uses it).
        """
        super().__init__()

        if arch is not None:
            architecture = arch

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

        self.architecture = self.arch = architecture
        num_logits = response_length * num_tokens

        if architecture == "mlp":
            self.features = nn.Sequential(
                nn.Linear(num_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.output = nn.Linear(hidden_size, num_logits)
        elif architecture == "linear":
            self.features = nn.Identity()
            self.output = nn.Linear(num_features, num_logits, bias=False)
        else:
            raise ValueError("architecture must be 'mlp' or 'linear'")

        # Equal initial logits give uniform token probabilities.
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, prompts):
        """Return unnormalized token scores shaped (B, L, K)."""
        if prompts.ndim != 2:
            raise ValueError("prompts must have shape (B, d)")
        if prompts.shape[1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features per prompt")

        features = self.features(prompts)
        logits = self.output(features)
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
        """Sample responses shaped (B, G, L)."""
        if group_size < 1:
            raise ValueError("group_size must be >= 1")

        distribution = self.distribution(prompts)
        responses = distribution.sample((group_size,))
        return responses.permute(1, 0, 2).contiguous()

    def log_prob(self, prompts, responses):
        """Return sequence log-probabilities shaped (B, G)."""
        if responses.ndim != 3:
            raise ValueError("responses must have shape (B, G, L)")
        if responses.shape[0] != prompts.shape[0]:
            raise ValueError("Prompt and response batch sizes must match")
        if responses.shape[2] != self.response_length:
            raise ValueError("Response length does not match the policy")

        distribution = self.distribution(prompts)

        # Categorical expects the sample dimension first.
        response_samples = responses.permute(1, 0, 2)
        token_log_probs = distribution.log_prob(response_samples)
        sequence_log_probs = token_log_probs.sum(dim=-1)
        return sequence_log_probs.transpose(0, 1)

    def success_probability(self, prompts, correct):
        """Return the exact success probability for each prompt."""
        if correct.shape != (prompts.shape[0], self.response_length):
            raise ValueError("correct must have shape (B, L)")

        distribution = self.distribution(prompts)
        token_log_probs = distribution.log_prob(correct)
        return token_log_probs.sum(dim=-1).exp()

    def entropy(self, prompts):
        """Return sequence entropy shaped (B,)."""
        return self.distribution(prompts).entropy().sum(dim=-1)

    def frozen_reference(self):
        """Copy the current policy for use as the fixed KL reference."""
        reference = deepcopy(self)
        reference.requires_grad_(False)
        reference.eval()
        return reference

    def kl_divergence(self, prompts, reference, direction="reverse"):
        """Return exact sequence KL for each prompt.

        Reverse KL is KL(policy || reference). Forward KL swaps the policies.
        Gradients flow only through the current policy.
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


def main():
    """Check both policies on the same synthetic prompts and answers."""
    from .synthetic_task import SyntheticTask

    torch.manual_seed(0)
    num_prompts = 32
    task = SyntheticTask(
        num_features=16,
        num_tokens=4,
        response_length=1,
        seed=0,
    )

    features, answers = task.generate_dataset(num_prompts)
    shared_prompts = torch.as_tensor(features, dtype=torch.float32)
    tabular_prompts = torch.eye(num_prompts, dtype=torch.float32)
    correct = torch.as_tensor(answers, dtype=torch.long)

    shared_policy = Policy(
        num_features=task.num_features,
        num_tokens=task.num_tokens,
        response_length=task.response_length,
    )
    tabular_policy = Policy(
        num_features=num_prompts,
        num_tokens=task.num_tokens,
        response_length=task.response_length,
        architecture="linear",
    )

    expected_p = task.num_tokens ** (-task.response_length)
    group_size = 8
    policy_examples = (
        ("MLP", shared_policy, shared_prompts),
        ("Tabular", tabular_policy, tabular_prompts),
    )

    for name, policy, prompts in policy_examples:
        success = policy.success_probability(prompts, correct)
        torch.testing.assert_close(success, torch.full_like(success, expected_p))

        responses = policy.sample(prompts, group_size=group_size)
        log_probs = policy.log_prob(prompts, responses)
        assert responses.shape == (num_prompts, group_size, task.response_length)
        assert log_probs.shape == (num_prompts, group_size)

        print(f"{name} check passed. Initial success = {expected_p:.2f}")

    check_tabular_prompt_isolation(tabular_policy, tabular_prompts, correct)
    print("Tabular prompt isolation check passed.")


def check_tabular_prompt_isolation(policy, prompts, correct):
    """Check that updating the first prompt leaves the other logits unchanged."""
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    logits_before = policy(prompts).detach()

    first_prompt_success = policy.success_probability(prompts[:1], correct[:1])
    loss = -first_prompt_success.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logits_after = policy(prompts).detach()
    assert not torch.equal(logits_before[0], logits_after[0]), (
        "The updated prompt's logits did not change"
    )
    torch.testing.assert_close(
        logits_after[1:],
        logits_before[1:],
        rtol=0,
        atol=0,
    )


if __name__ == "__main__":
    main()