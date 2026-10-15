"""
Command-line argument definitions.

All CLI arguments live here so main.py stays clean and everyone on the team
knows where to add new options. Each argument below demonstrates a different
argparse feature; replace or extend them as the project takes shape.

Run `python main.py --help` to see the generated help text.
"""

import argparse


def build_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="981 project - template entry point.",
        # Shows default values in --help output automatically
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional argument: required, identified by position (no dashes)
    parser.add_argument(
        "name",
        type=str,
        help="A required positional argument (e.g. your name).",
    )

    # Optional argument with a short and long flag, a type, and a default
    parser.add_argument(
        "-n", "--num-iters",
        type=int,
        default=3,
        help="Number of iterations to run.",
    )

    # Restrict the value to a fixed set of choices
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "test"],
        default="train",
        help="Run mode.",
    )

    # store_true: a flag that is False unless provided
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )

    return parser


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv: Optional list of strings to parse instead of sys.argv.
              Handy for testing, e.g. parse_args(["john", "--debug"]).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Custom validation that argparse can't express on its own
    if args.num_iters < 1:
        parser.error("--num-iters must be >= 1")

    return args
