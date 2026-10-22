"""
Command-line argument definitions.

All CLI arguments live here so main.py stays clean and everyone on the team
knows where to add new options.

Run `python main.py --help` to see the generated help text.
"""

import argparse


def build_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="GRPO on a one-question bandit: K answers, one correct.",
        # Shows default values in --help output automatically
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task
    parser.add_argument(
        "-K", "--num-answers",
        type=int,
        default=8,
        help="Number of possible answers. Uniform start gives p0 = 1/K.",
    )

    # Algorithm
    parser.add_argument(
        "-G", "--group-size",
        type=int,
        default=8,
        help="Answers sampled per step (the GRPO group).",
    )
    parser.add_argument(
        "--advantage",
        choices=["grpo", "mean", "rloo", "passk"],
        default="grpo",
        help="grpo: (r - mean)/std; mean: r - mean (Dr. GRPO); "
             "rloo: leave-one-out mean; passk: pass@k shaping.",
    )
    parser.add_argument(
        "-k", "--passk",
        type=int,
        default=4,
        help="k for --advantage passk (needs k <= G).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="KL penalty to the uniform starting policy (0 = off).",
    )

    # Optimization
    parser.add_argument(
        "--lr",
        type=float,
        default=0.5,
        help="Plain SGD step size on the logits.",
    )
    parser.add_argument(
        "-n", "--steps",
        type=int,
        default=300,
        help="Number of training steps.",
    )

    # Bookkeeping
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="Print a progress line every this many steps.",
    )

    return parser


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv: Optional list of strings to parse instead of sys.argv.
              Handy for testing, e.g. parse_args(["-K", "100", "-G", "16"]).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Custom validation that argparse can't express on its own
    if args.num_answers < 2:
        parser.error("--num-answers must be >= 2")
    if args.group_size < 2:
        parser.error("--group-size must be >= 2 (one sample has no baseline)")
    if args.advantage == "passk" and not 1 <= args.passk <= args.group_size:
        parser.error("--passk must satisfy 1 <= k <= G")
    if args.steps < 1 or args.print_every < 1:
        parser.error("--steps and --print-every must be >= 1")

    return args
