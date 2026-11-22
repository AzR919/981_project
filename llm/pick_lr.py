"""
Apply the pre-registered SmolLM2 learning-rate rule (PREREG_C1.md) to the C0 pilots.

usage:
    python llm/pick_lr.py --base base_s0.json.gz --ids llm/pools/smol_c1_train.json 1e-5=dir 3e-5=dir ...

Rule: the smallest lr whose mean p-hat on the training problems rises by >= 0.05
over the base evaluation within 150 steps, with <= 2 BH-significant declines
(N = 32, q = 0.1). Prints the chosen lr, or NONE (the decision gate fires).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.declines import compare, load  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("pilots", nargs="+", help="lr=dir with after.json.gz")
    args = parser.parse_args()
    ids = set(json.load(open(args.ids)))
    base = {k: v for k, v in load(args.base).items() if k in ids}
    chosen = None
    for item in sorted(args.pilots, key=lambda s: float(s.split("=")[0])):
        lr, path = item.split("=", 1)
        df = compare(base, load(os.path.join(path, "after.json.gz")), 0.1)
        rise, declines = df.p_after.mean() - df.p_before.mean(), int(df.decline.sum())
        ok = rise >= 0.05 and declines <= 2
        print(f"lr {lr}: rise {rise:+.3f}, declines {declines}, {'meets' if ok else 'fails'} the rule", file=sys.stderr)
        if ok and chosen is None:
            chosen = lr
    print(chosen or "NONE")


if __name__ == "__main__":
    main()
