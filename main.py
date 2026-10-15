"""
Main entry point for the project.

Example usage:
    python main.py --help
    python main.py john
    python main.py john -n 5 --mode eval --debug
"""

from args import parse_args


def main():
    args = parse_args()

    print(f"Hello, {args.name}!")
    print()
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key:<10} = {value!r}")
    print()

    if args.debug:
        print("[debug] debug mode is on")

    for i in range(args.num_iters):
        print(f"[{args.mode}] iteration {i + 1}/{args.num_iters}")

    # TODO: replace the demo above with real project logic


if __name__ == "__main__":
    main()
