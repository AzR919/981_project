"""
Draw the C1 problem sets from a model's pool (extra/llm/analyze_c0.py --pool).

usage:
    python extra/llm/make_sets.py extra/llm/pools/qwen.json --train 1000 --heldout 300 --seed 0

Writes <pool>_c1_train.json (problems trained on), <pool>_c1_heldout.json (pool
problems never trained on) and <pool>_c1_eval.json (both, for evaluate.py --ids).
"""

import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pool")
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--heldout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    pool = json.load(open(args.pool))
    pick = np.random.default_rng(args.seed).permutation(len(pool))
    train = sorted((pool[i] for i in pick[: args.train]), key=lambda i: int(i.split("-")[1]))
    held = sorted((pool[i] for i in pick[args.train: args.train + args.heldout]), key=lambda i: int(i.split("-")[1]))
    stem = args.pool[: -len(".json")]
    for name, ids in (("train", train), ("heldout", held), ("eval", train + held)):
        with open(f"{stem}_c1_{name}.json", "w") as f:
            json.dump(ids, f)
    print(f"{len(train)} train, {len(held)} held-out from {len(pool)} in {args.pool}")


if __name__ == "__main__":
    main()
