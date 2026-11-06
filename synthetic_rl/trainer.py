"""
Training loop for the synthetic GRPO experiments.

Each training step:
1. Sample responses from the current policy.
2. Score responses using the task's verifier.
3. Calculate advantages within each prompt's group.
4. Calculate the policy loss and optional KL penalty.
5. Update the policy once.

Loss convention:
- Sum log-probabilities over tokens.
- Average the policy loss over prompts and responses.
- Average sequence KL over prompts.

Run the example from the project root:
    python3.12 -m synthetic_rl.trainer
"""

import numpy as np
import torch

from .policy import Policy
from .synthetic_task import SyntheticTask, verify


def calculate_advantages(rewards, kind="grpo", epsilon=1e-5):
    """Return advantages shaped (B, G) from binary rewards."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape (B, G), with G >= 2")

    group_mean = rewards.mean(dim=1, keepdim=True)
    centered = rewards - group_mean

    if kind == "mean":
        return centered

    if kind == "rloo":
        group_size = rewards.shape[1]
        other_mean = (
            rewards.sum(dim=1, keepdim=True) - rewards
        ) / (group_size - 1)

        return rewards - other_mean

    if kind == "grpo":
        # Population variance for binary rewards; no Bessel correction.
        variance = group_mean * (1 - group_mean)
        return centered / torch.sqrt(variance + epsilon)

    raise ValueError("kind must be 'grpo', 'mean', or 'rloo'")


class Trainer:
    """Train a policy using fresh on-policy samples at every step."""

    def __init__(
        self,
        policy,
        optimizer,
        group_size=8,
        advantage="grpo",
        beta=0.0,
        kl_direction="reverse",
        seed=0,
    ):
        if group_size < 2:
            raise ValueError("group_size must be >= 2")
        if advantage not in ("grpo", "mean", "rloo"):
            raise ValueError("Unknown advantage type")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and >= 0")
        if kl_direction not in ("reverse", "forward"):
            raise ValueError("Unknown KL direction")

        self.policy = policy
        self.optimizer = optimizer
        self.group_size = group_size
        self.advantage = advantage
        self.beta = beta
        self.kl_direction = kl_direction

        self.device = next(policy.parameters()).device

        # Freeze the policy as it exists when the trainer is created.
        # If using a warm-start, complete it before creating the trainer.
        self.reference = policy.frozen_reference()

        # Separate prompt-batch randomness from PyTorch response sampling.
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.steps = 0
        self.responses_sampled = 0

    def to_tensors(self, prompts, correct):
        """Convert task arrays to tensors on the policy's device."""
        prompts = torch.as_tensor(
            prompts,
            dtype=torch.float32,
            device=self.device,
        )
        correct = torch.as_tensor(
            correct,
            dtype=torch.long,
            device=self.device,
        )

        return prompts, correct

    def train_step(self, prompts, correct):
        """Update the policy once using a batch of prompts.

        Returned batch metrics describe the policy before this update.
        """
        self.policy.train()
        prompts, correct = self.to_tensors(prompts, correct)

        responses = self.policy.sample(
            prompts,
            group_size=self.group_size,
        )

        # The existing verifier works with NumPy arrays.
        reward_array = verify(
            responses.cpu().numpy(),
            correct.cpu().numpy(),
        )
        rewards = torch.as_tensor(
            reward_array,
            dtype=torch.float32,
            device=self.device,
        )

        advantages = calculate_advantages(
            rewards,
            kind=self.advantage,
        ).detach()

        # Shape: (B, G). Token log-probabilities are already summed.
        log_probs = self.policy.log_prob(prompts, responses)
        policy_loss = -(advantages * log_probs).mean()

        kl = self.policy.kl_divergence(
            prompts,
            self.reference,
            direction=self.kl_direction,
        ).mean()

        loss = policy_loss + self.beta * kl

        if not torch.isfinite(loss).item():
            raise FloatingPointError("Training loss is not finite")

        # Calculate diagnostics before changing the policy.
        with torch.no_grad():
            success = self.policy.success_probability(prompts, correct)
            expected_informative = (
                1
                - success.pow(self.group_size)
                - (1 - success).pow(self.group_size)
            )

            group_reward = rewards.mean(dim=1)
            informative = (group_reward > 0) & (group_reward < 1)

            metrics = {
                "loss": loss.item(),
                "batch_reward_before": rewards.mean().item(),
                "batch_kl_before": kl.item(),
                "informative_before": informative.float().mean().item(),
                "expected_informative_before": (
                    expected_informative.mean().item()
                ),
            }

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        self.responses_sampled += prompts.shape[0] * self.group_size

        return metrics

    @torch.no_grad()
    def evaluate(self, prompts, correct):
        """Calculate exact metrics without sampling or updating."""
        self.policy.eval()
        prompts, correct = self.to_tensors(prompts, correct)

        success = self.policy.success_probability(prompts, correct)
        entropy = self.policy.entropy(prompts)
        kl = self.policy.kl_divergence(
            prompts,
            self.reference,
            direction=self.kl_direction,
        )

        return {
            "success": success.mean().item(),
            "entropy": entropy.mean().item(),
            "kl": kl.mean().item(),
        }

    def record_metrics(self, train_data, test_data, batch_metrics=None):
        """Combine batch diagnostics and current dataset evaluations."""
        train_metrics = self.evaluate(*train_data)
        test_metrics = self.evaluate(*test_data)

        record = {
            "step": self.steps,
            "responses_sampled": self.responses_sampled,
            "tokens_sampled": (
                self.responses_sampled * self.policy.response_length
            ),
        }

        # No training batch exists at step zero.
        record.update({
            name: None
            for name in (
                "loss",
                "batch_reward_before",
                "batch_kl_before",
                "informative_before",
                "expected_informative_before",
            )
        })

        if batch_metrics is not None:
            record.update(batch_metrics)

        record.update({
            f"train_{name}": value
            for name, value in train_metrics.items()
        })
        record.update({
            f"test_{name}": value
            for name, value in test_metrics.items()
        })

        return record

    def train(
        self,
        train_data,
        test_data,
        steps=300,
        batch_size=32,
        log_every=25,
    ):
        """Train and return a list of metric dictionaries.

        Each dataset is the (prompts, correct) pair returned by
        SyntheticTask.generate_dataset().

        Prompt batches are sampled with replacement.
        """
        if steps < 1 or batch_size < 1 or log_every < 1:
            raise ValueError("steps, batch_size, and log_every must be >= 1")

        train_prompts, train_correct = train_data
        if len(train_prompts) == 0:
            raise ValueError("Training data must not be empty")
        if len(train_prompts) != len(train_correct):
            raise ValueError("Prompt and answer counts must match")

        history = []

        record = self.record_metrics(train_data, test_data)
        history.append(record)
        print_progress(record)

        for iteration in range(1, steps + 1):
            indices = self.rng.integers(
                len(train_prompts),
                size=batch_size,
            )

            batch_metrics = self.train_step(
                train_prompts[indices],
                train_correct[indices],
            )

            if iteration % log_every == 0 or iteration == steps:
                record = self.record_metrics(
                    train_data,
                    test_data,
                    batch_metrics,
                )
                history.append(record)
                print_progress(record)

        return history


def print_progress(record):
    """Print the main learning metrics."""
    print(
        f"step={record['step']:5d}  "
        f"train={record['train_success']:.4f}  "
        f"test={record['test_success']:.4f}  "
        f"entropy={record['test_entropy']:.4f}  "
        f"responses={record['responses_sampled']}"
    )


# Optional example: creates a small training run to check that the task,
# policy, and trainer work together. Runs only when this module is executed.
def main():
    seed = 0
    torch.manual_seed(seed)

    task = SyntheticTask(
        num_features=16,
        num_tokens=4,
        response_length=2,
        seed=seed,
    )

    # Both datasets use the same hidden rule.
    train_data = task.generate_dataset(512)
    test_data = task.generate_dataset(256)

    policy = Policy(
        num_features=task.num_features,
        num_tokens=task.num_tokens,
        response_length=task.response_length,
    )

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=1e-3,
    )

    trainer = Trainer(
        policy=policy,
        optimizer=optimizer,
        group_size=8,
        advantage="grpo",
        beta=0.0,
        seed=seed,
    )

    history = trainer.train(
        train_data=train_data,
        test_data=test_data,
        steps=300,
        batch_size=32,
        log_every=25,
    )

    print(f"\nRecorded {len(history)} evaluation points.")


if __name__ == "__main__":
    main()