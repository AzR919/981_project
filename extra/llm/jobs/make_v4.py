"""
Write the job list for the C2 seed replication (PREREG_C2_amendment2.md).

usage (from 981_project/):  python extra/llm/jobs/make_v4.py

  c2_seeds.txt   10 runs: the problem draw of the one broadly degraded C2 run (mix seed 1, majority "each")
                 at m = 0.95 and m = 0.5, with five new training seeds each (3-7); no hint evaluations.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_v2 import P, c2_run, write  # noqa: E402


def main():
    lines = []
    for seed in (3, 4, 5, 6, 7):
        for m in ("0.95", "0.5"):
            mix = f"{P}/mix_m{m}_each_s1"
            lines.append(c2_run("/tmp/asa715/w_c2r", f"--problems {mix}_train_ids.json --seed {seed}", f"{mix}_eval_ids.json"))
    write("c2_seeds.txt", lines)


if __name__ == "__main__":
    main()
