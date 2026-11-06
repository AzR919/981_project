import itertools
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def main():
    configurations = list(itertools.product(
        [0.01, 0.1, 0.5],  # Initial success probabilities.
        [2, 8, 32],        # Group sizes.
        [0, 1, 2],         # Random seeds.
    ))

    for index, (p0, group_size, seed) in enumerate(configurations, start=1):
        print(
            f"\nRun {index}/{len(configurations)}: "
            f"p0={p0}, G={group_size}, seed={seed}",
            flush=True,
        )

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "main.py"),
                "-K", "8",
                "--p0", str(p0),
                "-G", str(group_size),
                "--seed", str(seed),
                "--advantage", "grpo",
                "--beta", "0",
                "--lr", "0.1",
                "--steps", "2000",
                "--print-every", "200",
            ],
            cwd=PROJECT_DIR,
            check=True,
        )


if __name__ == "__main__":
    main()