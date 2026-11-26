"""
Write the job list for the LLM branch test C10 (PREREG_C10.md).

usage (from 981_project/):  python extra/llm/jobs/make_v3.py

  c10.txt   18 short runs from the base model: mixed, majority-only and minority-only steps of the
            76 + 4 composition (denominator 80), both role assignments, seeds 0-2; the 600 held-out
            problems are evaluated at N = 128 after 50 and 100 steps.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_v2 import P, c2_run, write  # noqa: E402

ARMS = {"mixed": (76, 4), "majority_only": (76, 0), "minority_only": (0, 4)}


def main():
    balanced = f"{P}/mix_m0.5_each_s0"     # 500 + 500 training problems; arms differ only in the strata
    lines = []
    for seed in (0, 1, 2):
        for major, minor in (("each", "total"), ("total", "each")):
            for n_major, n_minor in ARMS.values():
                lines.append(c2_run("/tmp/asa715/w_c10", f"--problems {balanced}_train_ids.json --seed {seed} "
                                    f"--strata procedure:{major}={n_major},{minor}={n_minor} --denominator 80",
                                    f"{P}/heldout_ids.json", steps=100, save_at=(50,), mid_n=128, final_n=128))
    write("c10.txt", lines)


if __name__ == "__main__":
    main()
