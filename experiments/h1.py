"""Generate one H1 gradient-norm comparison per advantage rule.

python -m experiments.h1 --policy mlp --output results/h1_mlp_plots
python -m experiments.h1 --policy tabular --output results/h1_tabular_plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

from main import omega
from synthetic_rl.policy import Policy
from synthetic_rl.trainer import Trainer, calculate_advantages


GROUP_SIZE = 4
GROUPS_PER_STEP = 512
NUM_ANSWERS = 4
NUM_STEPS = 250
NUM_SEEDS = 10
LEARNING_RATE = 0.4
DIAGNOSTIC_GROUPS = 500_000 # for plotting mlp cz small sample make the plot really noisy
CHECKPOINT_INTERVAL = 10
RULE_LABELS = {"mean": "Mean-only", "rloo": "RLOO", "grpo": "GRPO"}


def estimate_checkpoint_gradient(log_probs, groups, advantages, rng, parameters):
    """Average a large sample's gradients using counts of each possible group."""
    # For one prompt with four answers, a joint draw over groups has 4**G outcomes.
    # Sampling their counts is equivalent to drawing and tallying individual groups.
    answer_probs = log_probs.detach().double().softmax(dim=0)
    group_probs = answer_probs[groups].prod(dim=1)
    group_probs /= group_probs.sum()
    counts = rng.multinomial(DIAGNOSTIC_GROUPS, group_probs.numpy())
    frequencies = torch.as_tensor(counts, dtype=log_probs.dtype) / DIAGNOSTIC_GROUPS
    group_scores = (advantages * log_probs[groups]).mean(dim=1)
    return torch.autograd.grad(frequencies @ group_scores, parameters)


def gradient_norms(rule, architecture, checkpoints):
    """Return the mean squared gradient norms across seeds for one rule."""
    prompt = torch.ones(1, 1)
    correct = torch.zeros(1, 1, dtype=torch.long)

    # Each row samples one independent group for the same prompt.
    batch_prompts = prompt.expand(GROUPS_PER_STEP, -1)
    batch_correct = correct.expand(GROUPS_PER_STEP, -1)
    predicted_norms = torch.zeros(len(checkpoints), dtype=torch.float64)
    sampled_norms = torch.zeros_like(predicted_norms)
    checkpoint_indices = {step: index for index, step in enumerate(checkpoints)}
    diagnose = architecture == "mlp"

    if diagnose:
        answers = torch.arange(NUM_ANSWERS)
        responses = answers.view(1, NUM_ANSWERS, 1)
        groups = torch.cartesian_prod(*[answers] * GROUP_SIZE)
        rewards = (groups == correct.item()).to(prompt.dtype)
        advantages = calculate_advantages(rewards, kind=rule)

    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        policy = Policy(
            num_features=1,
            num_tokens=NUM_ANSWERS,
            response_length=1,
            architecture=architecture,
        )
        parameters = tuple(policy.parameters())
        trainer = Trainer(
            policy=policy,
            optimizer=torch.optim.SGD(parameters, lr=LEARNING_RATE),
            group_size=GROUP_SIZE,
            advantage=rule,
            seed=seed,
        )
        if diagnose:
            diagnostic_rng = np.random.default_rng(seed)

        for step in range(NUM_STEPS + int(diagnose)):
            checkpoint = checkpoint_indices.get(step)
            if checkpoint is not None:
                success = policy.success_probability(prompt, correct).squeeze()
                success_gradient = torch.autograd.grad(success, parameters)
                weight = omega(success.item(), GROUP_SIZE, rule)
                predicted_norms[checkpoint] += weight**2 * sum(
                    gradient.square().sum().item() for gradient in success_gradient
                )

                if diagnose:
                    log_probs = policy.log_prob(prompt, responses).squeeze(0)
                    sampled_gradient = estimate_checkpoint_gradient(
                        log_probs, groups, advantages, diagnostic_rng, parameters
                    )
                    sampled_norms[checkpoint] += sum(
                        gradient.square().sum().item() for gradient in sampled_gradient
                    )

            if step == NUM_STEPS:
                break
            trainer.train_step(batch_prompts, batch_correct)

            if checkpoint is not None and not diagnose:
                # SGD leaves the pre-update loss gradients in .grad.
                sampled_norms[checkpoint] += sum(
                    parameter.grad.square().sum().item() for parameter in parameters
                )

    return predicted_norms / NUM_SEEDS, sampled_norms / NUM_SEEDS


def save_gradient_norm_plot(
    path, label, evaluations, predicted, sampled, y_limits, *, diagnostic=False
):
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
    ax.semilogy(
        evaluations,
        sampled,
        color="#C75420",
        linewidth=2,
        label=f"{label} checkpoint estimate" if diagnostic else label,
        nonpositive="mask",
    )
    ax.set_title(f"Advantage gap vs {label}", fontsize=15, pad=10)
    ax.set_ylabel(r"$\|g\|^2$")
    evaluations_name = "training gradient" if diagnostic else "gradient"
    ax.set_xlabel(f"Number of {evaluations_name} evaluations")
    ax.set_xlim(0, evaluations[-1])
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_locator(MultipleLocator(100_000))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_axisbelow(True)
    ax.grid(which="major", color="#DCE2E8", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper right", frameon=True)

    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_gradient_norms(output, architecture):
    """Measure each rule and save its figure using shared axis limits."""
    diagnostic = architecture == "mlp"
    if diagnostic:
        checkpoints = [*range(0, NUM_STEPS, CHECKPOINT_INTERVAL), NUM_STEPS]
    else:
        checkpoints = range(NUM_STEPS)
    curves = {
        label: gradient_norms(rule, architecture, checkpoints)
        for rule, label in RULE_LABELS.items()
    }
    all_norms = torch.stack([norms for pair in curves.values() for norms in pair])
    y_limits = (
        all_norms[all_norms > 0].min().item() / 1.25,
        all_norms.max().item() * 1.25,
    )
    responses_per_step = GROUPS_PER_STEP * GROUP_SIZE
    evaluations = [
        (step + (0 if diagnostic else 1)) * responses_per_step for step in checkpoints
    ]

    output.mkdir(parents=True, exist_ok=True)
    for label, (predicted, sampled) in curves.items():
        filename = label.lower().replace("-", "_")
        path = output / f"advantage_gap_vs_{filename}.png"
        plot_label = f"{label} (MLP)" if architecture == "mlp" else label
        save_gradient_norm_plot(
            path, plot_label, evaluations, predicted, sampled, y_limits,
            diagnostic=diagnostic,
        )
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("tabular", "mlp"), default="tabular")
    args = parser.parse_args()

    torch.set_num_threads(1)
    architecture = "linear" if args.policy == "tabular" else "mlp"
    plot_gradient_norms(args.output, architecture)


if __name__ == "__main__":
    main()
