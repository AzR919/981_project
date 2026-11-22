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

import math

import numpy as np
import torch

from .policy import Policy
from .synthetic_task import SyntheticTask, verify


ADVANTAGES = ("grpo", "mean", "rloo", "passk", "rloo_k", "maxrl", "sign", "heckel", "virtual", "pm_alpha", "nsr")


def log_comb(n, k):
    """log C(n, k) elementwise, -inf where k > n."""
    n = n.to(torch.float64)
    value = (
        torch.lgamma(n + 1)
        - torch.lgamma(torch.tensor(k + 1.0, dtype=torch.float64))
        - torch.lgamma((n - k + 1).clamp(min=1))
    )
    return torch.where(n >= k, value, torch.full_like(value, -torch.inf))


def calculate_advantages(rewards, kind="grpo", epsilon=1e-5, passk=None, alpha=0.1, lam=0.1):
    """Return advantages shaped (B, G) from binary rewards.

    grpo:  (r - mean) / sqrt(mean (1 - mean) + epsilon)
    mean:  r - mean (Dr. GRPO)
    rloo:  r - mean of the other G - 1 rewards
    passk: GRPO advantage reweighted for pass@k (Thrampoulidis et al.,
           combinatorial form of their Eq. 14; their GRPO_K, biased); needs passk = k <= G
    rloo_k: the same leave-one-out Fail@(k-1) weights on the RLOO advantage
           (their RLOO_K, Eq. 17): an unbiased estimate of grad pass@k
    maxrl: G r / K - 1 with K correct samples, 0 if K = 0 (MaxRL; equal to
           reward budgeting for binary rewards)
    sign:  2 r - 1, a fixed baseline (Nie et al.); not zero-sum

    T9, rules that give all-wrong groups a gradient:
    heckel:   Linear-R of Heckel et al. (2026): (r - mean) / mean, i.e. -1 for
              every wrong sample (also when mean = 0), 1/mean - 1 if correct
    virtual:  AVSPO with one virtual sample (He et al. 2026): a uniform group
              gets one virtual opposite reward in its mean and std, so all-wrong
              samples get -1/sqrt(G) and all-correct +1/sqrt(G); GRPO otherwise
    pm_alpha: GRPO, plus -alpha on all-wrong and +alpha on all-correct groups
              (RL-ZVP of Le et al. 2025 without its token-entropy factor)
    nsr:      W-REINFORCE (Zhu et al. 2025): lam if correct, -1 if wrong
    """
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape (B, G), with G >= 2")

    group_size = rewards.shape[1]
    group_mean = rewards.mean(dim=1, keepdim=True)
    centered = rewards - group_mean

    if kind == "mean":
        return centered

    if kind == "rloo":
        other_mean = (
            rewards.sum(dim=1, keepdim=True) - rewards
        ) / (group_size - 1)

        return rewards - other_mean

    # Population variance for binary rewards; no Bessel correction.
    grpo = centered / torch.sqrt(group_mean * (1 - group_mean) + epsilon)

    if kind == "grpo":
        return grpo

    if kind in ("passk", "rloo_k"):
        if passk is None or not 1 <= passk <= group_size:
            raise ValueError("passk needs 1 <= k <= G")
        n_wrong = group_size - rewards.sum(dim=1, keepdim=True)
        log_total = math.lgamma(group_size) - math.lgamma(passk) - math.lgamma(group_size - passk + 1)
        # Probability that the other k - 1 samples are all wrong.
        f_pos = torch.exp(log_comb(n_wrong, passk - 1) - log_total)
        f_neg = torch.exp(log_comb(n_wrong - 1, passk - 1) - log_total)
        base = grpo if kind == "passk" else rewards - (rewards.sum(dim=1, keepdim=True) - rewards) / (group_size - 1)
        return torch.where(rewards == 1, f_pos, f_neg).to(rewards.dtype) * base

    if kind == "maxrl":
        num_correct = rewards.sum(dim=1, keepdim=True)
        scaled = group_size * rewards / num_correct.clamp(min=1) - 1
        return torch.where(num_correct > 0, scaled, torch.zeros_like(scaled))

    if kind == "sign":
        return 2 * rewards - 1

    uniform = (group_mean == 0) | (group_mean == 1)

    if kind == "heckel":
        scaled = centered / group_mean.clamp(min=1.0 / group_size)
        return torch.where(group_mean > 0, scaled, -torch.ones_like(scaled))

    if kind == "virtual":
        # The virtual reward is 1 - r on a uniform group; mean and std over G + 1 values.
        mean_v = (group_size * group_mean + (1 - group_mean)) / (group_size + 1)
        virtual = (rewards - mean_v) / torch.sqrt(mean_v * (1 - mean_v) + epsilon)
        return torch.where(uniform, virtual, grpo)

    if kind == "pm_alpha":
        return torch.where(uniform, alpha * (2 * rewards - 1), grpo)

    if kind == "nsr":
        return torch.where(rewards == 1, lam * torch.ones_like(rewards), -torch.ones_like(rewards))

    raise ValueError(f"kind must be one of {ADVANTAGES}")


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
        passk=None,
        reward_noise=0.0,
        ppo_epochs=1,
        clip_eps=None,
        grad_clip=None,
    ):
        if group_size < 2:
            raise ValueError("group_size must be >= 2")
        if advantage not in ADVANTAGES:
            raise ValueError("Unknown advantage type")
        if advantage in ("passk", "rloo_k") and not (passk and 1 <= passk <= group_size):
            raise ValueError("passk needs 1 <= k <= group_size")
        if not 0 <= reward_noise < 0.5:
            raise ValueError("reward_noise must be in [0, 0.5)")
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
        self.passk = passk
        # Probability that the verifier flips a reward (0 = exact verifier).
        self.reward_noise = reward_noise
        # Realistic-training options (T6): several updates per sampled batch
        # with the PPO clipped ratio, and gradient-norm clipping. The
        # defaults give the plain on-policy update.
        if ppo_epochs < 1:
            raise ValueError("ppo_epochs must be >= 1")
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps
        self.grad_clip = grad_clip

        self.device = next(policy.parameters()).device

        # Freeze the policy as it exists when the trainer is created.
        # If using a warm-start, complete it before creating the trainer.
        self.reference = policy.frozen_reference()

        # Separate prompt-batch randomness from PyTorch response sampling.
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # Per-prompt decreases of p within sampled batches, summed over steps.
        self.decreases = 0

        self.steps = 0
        self.responses_sampled = 0

    def to_tensors(self, prompts, correct):
        """Convert task arrays to tensors on the policy's device."""
        prompts = torch.as_tensor(
            prompts,
            dtype=next(self.policy.parameters()).dtype,
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
            dtype=prompts.dtype,
            device=self.device,
        )
        if self.reward_noise > 0:
            flips = self.rng.random(reward_array.shape) < self.reward_noise
            rewards = torch.where(
                torch.as_tensor(flips, device=self.device),
                1 - rewards,
                rewards,
            )

        advantages = calculate_advantages(
            rewards,
            kind=self.advantage,
            passk=self.passk,
        ).detach()

        # Shape: (B, G). Token log-probabilities are already summed.
        log_probs = self.policy.log_prob(prompts, responses)
        policy_loss = -(advantages * log_probs).mean()

        if self.beta > 0:
            kl = self.policy.kl_divergence(
                prompts,
                self.reference,
                direction=self.kl_direction,
            ).mean()
        else:
            # Skip the reference forward pass when it has no effect.
            kl = torch.zeros((), dtype=prompts.dtype, device=self.device)

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
        self._clip_and_step()

        if self.ppo_epochs > 1:
            old_log_probs = log_probs.detach()
            for _ in range(self.ppo_epochs - 1):
                log_probs = self.policy.log_prob(prompts, responses)
                ratio = torch.exp(log_probs - old_log_probs)
                if self.clip_eps is None:
                    surrogate = ratio * advantages
                else:
                    clipped = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
                    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
                loss = -surrogate.mean()
                if self.beta > 0:
                    loss = loss + self.beta * self.policy.kl_divergence(
                        prompts, self.reference, direction=self.kl_direction
                    ).mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._clip_and_step()

        with torch.no_grad():
            success_after = self.policy.success_probability(prompts, correct)
            decreases = int((success_after < success).sum().item())
        self.decreases += decreases
        metrics["decreases"] = decreases

        self.steps += 1
        self.responses_sampled += prompts.shape[0] * self.group_size

        return metrics

    def _clip_and_step(self):
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

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