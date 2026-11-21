"""
Build the C2 problem sets from base-evaluated candidates (gen_procedures.py + evaluate.py).

usage:
    python llm/make_c2_sets.py --range ml --seed 1 --per-procedure 3000 \\
        --base c2_candidates_base.json.gz --out llm/pools/c2

1. Keep candidates with base p-hat (N = 32) in [0.15, 0.85].
2. Stratify: in each of 7 equal-width p-hat bins keep the same number of
   problems from both procedures, so the two procedures match in difficulty.
3. Per procedure: 1,000 problems form the training universe and 300 are held out.
4. For each m in {0.5, 0.8, 0.95}, majority procedure in {each, total} and seed
   in {0, 1, 2}: a 1,000-problem training mix drawn from the universe
   (gen_procedures.make_mix), written as a problem file; its evaluation ids are
   the mix plus all 600 held-out problems.
"""

import argparse
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.gen_procedures import PROCEDURES, generate, make_mix  # noqa: E402

LOW, HIGH, BINS = 0.15, 0.85, 7
UNIVERSE, HELD = 1000, 300
SHARES = (0.5, 0.8, 0.95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--range", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--per-procedure", type=int, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    counts = json.load(gzip.open(args.base, "rt"))["counts"]
    problems = [q for proc in PROCEDURES for q in generate(proc, args.per_procedure, args.range, args.seed)]
    for q in problems:
        c = counts[q["id"]]
        q["base_p"] = c[0] / c[2]
    rng = np.random.default_rng(args.seed)
    edges = np.linspace(LOW, HIGH, BINS + 1)
    keep = {p: [] for p in PROCEDURES}
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi == edges[-1]
        inside = {p: [q for q in problems if q["procedure"] == p and lo <= q["base_p"] and
                      (q["base_p"] <= hi if last else q["base_p"] < hi)] for p in PROCEDURES}
        k = min(len(v) for v in inside.values())
        for p, v in inside.items():
            keep[p] += [v[i] for i in rng.choice(len(v), k, replace=False)]
    sizes = {p: len(v) for p, v in keep.items()}
    if min(sizes.values()) < UNIVERSE + HELD:
        raise SystemExit(f"only {sizes} matched problems; need {UNIVERSE + HELD} per procedure")

    os.makedirs(args.out, exist_ok=True)
    universe, held = [], []
    for p in PROCEDURES:
        order = rng.permutation(len(keep[p]))
        universe += [keep[p][i] for i in order[:UNIVERSE]]
        held += [keep[p][i] for i in order[UNIVERSE: UNIVERSE + HELD]]
    json.dump(universe + held, open(os.path.join(args.out, "all.json"), "w"))
    json.dump([q["id"] for q in held], open(os.path.join(args.out, "heldout_ids.json"), "w"))
    summary = {p: dict(matched=sizes[p], universe_mean_p=float(np.mean([q["base_p"] for q in universe
                                                                         if q["procedure"] == p])))
               for p in PROCEDURES}
    for m in SHARES:
        for major in PROCEDURES:
            for seed in range(3):
                mix = make_mix(universe, major, m, UNIVERSE, seed)
                stem = os.path.join(args.out, f"mix_m{m}_{major}_s{seed}")
                json.dump(mix + held, open(stem + "_problems.json", "w"))
                json.dump([q["id"] for q in mix], open(stem + "_train_ids.json", "w"))
                json.dump([q["id"] for q in mix + held], open(stem + "_eval_ids.json", "w"))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
