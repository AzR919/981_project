"""Generate one H1 gradient-norm comparison per advantage rule."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

from main import omega
from synthetic_rl.policy import Policy
from synthetic_rl.trainer import Trainer


GROUP_SIZE = 4
GROUPS_PER_STEP = 512
NUM_STEPS = 250
NUM_SEEDS = 10
LEARNING_RATE = 0.4
RULE_LABELS = {"mean": "Mean-only", "rloo": "RLOO", "grpo": "GRPO"}


def gradient_norms(rule):
    """Return the mean squared gradient norms across seeds for one rule."""
    prompt = torch.ones(1, 1)
    correct = torch.zeros(1, 1, dtype=torch.long)

    # Each row samples one independent group for the same prompt.
    batch_prompts = prompt.expand(GROUPS_PER_STEP, -1)
    batch_correct = correct.expand(GROUPS_PER_STEP, -1)
    predicted_norms = torch.zeros(NUM_STEPS, dtype=torch.float64)
    sampled_norms = torch.zeros_like(predicted_norms)

    for seed in range(NUM_SEEDS):
        policy = Policy(
            num_features=1,
            num_tokens=4,
            response_length=1,
            architecture="linear",
        )
        parameters = tuple(policy.parameters())
        trainer = Trainer(
            policy=policy,
            optimizer=torch.optim.SGD(parameters, lr=LEARNING_RATE),
            group_size=GROUP_SIZE,
            advantage=rule,
            seed=seed,
        )

        for step in range(NUM_STEPS):
            success = policy.success_probability(prompt, correct).squeeze()
            success_gradient = torch.autograd.grad(success, parameters)
            weight = omega(success.item(), GROUP_SIZE, rule)
            success_norm_sq = sum(
                gradient.square().sum().item() for gradient in success_gradient
            )
            predicted_norms[step] += weight**2 * success_norm_sq

            trainer.train_step(batch_prompts, batch_correct)

            # SGD leaves the pre-update loss gradients in .grad.
            sampled_norms[step] += sum(
                parameter.grad.square().sum().item() for parameter in parameters
            )

    return predicted_norms / NUM_SEEDS, sampled_norms / NUM_SEEDS


def save_gradient_norm_plot(path, label, evaluations, predicted, sampled, y_limits):
    """Save the predicted and sampled curves for one rule."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    ax.semilogy(
        evaluations,
        predicted,
        color="#244785",
        linestyle="--",
        linewidth=1.65,
        label="Advantage gap",
        zorder=3,
    )
    ax.semilogy(evaluations, sampled, color="#C75420", linewidth=2, label=label)
    ax.set_title(f"Advantage gap vs {label}", fontsize=15, pad=10)
    ax.set_ylabel(r"$\|g\|^2$")
    ax.set_xlabel("Number of gradient evaluations")
    ax.set_xlim(0, evaluations[-1])
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_locator(MultipleLocator(100_000))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_axisbelow(True)
    ax.grid(which="major", color="#DCE2E8", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper right", frameon=True)

    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_gradient_norms(output):
    """Measure each rule and save its figure using shared axis limits."""
    curves = {label: gradient_norms(rule) for rule, label in RULE_LABELS.items()}
    all_norms = torch.stack([norms for pair in curves.values() for norms in pair])
    y_limits = (
        all_norms[all_norms > 0].min().item() / 1.25,
        all_norms.max().item() * 1.25,
    )
    responses_per_step = GROUPS_PER_STEP * GROUP_SIZE
    evaluations = [step * responses_per_step for step in range(1, NUM_STEPS + 1)]

    output.mkdir(parents=True, exist_ok=True)
    for label, (predicted, sampled) in curves.items():
        filename = label.lower().replace("-", "_")
        path = output / f"advantage_gap_vs_{filename}.png"
        save_gradient_norm_plot(path, label, evaluations, predicted, sampled, y_limits)
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    plot_gradient_norms(args.output)


if __name__ == "__main__":
    main()
