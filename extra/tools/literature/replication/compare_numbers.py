"""
Compare two numbers.json files written by figures.py of "Wrong the Same Way" (Li, Pandey and Chen).

usage: python compare_numbers.py published.json regenerated.json [--out table.md]

Every entry is a string as typeset in the paper. Numbers inside each string are extracted and compared:
exact (identical string), close (all numbers within 2% relative or 0.01 absolute), or different.
"""

import argparse
import json
import re

NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numbers(text):
    return [float(x) for x in NUM.findall(str(text).replace("{,}", "").replace(",", ""))]


def verdict(a, b):
    if str(a) == str(b):
        return "exact"
    na, nb = numbers(a), numbers(b)
    if na and len(na) == len(nb) and all(abs(x - y) <= max(0.02 * abs(x), 0.01) for x, y in zip(na, nb)):
        return "close"
    return "different"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("published")
    ap.add_argument("regenerated")
    ap.add_argument("--out")
    a = ap.parse_args()
    pub, reg = json.load(open(a.published)), json.load(open(a.regenerated))
    keys = sorted(set(pub) | set(reg))
    rows = [(k, pub.get(k, "(missing)"), reg.get(k, "(missing)"), verdict(pub.get(k), reg.get(k))) for k in keys]
    counts = {v: sum(r[3] == v for r in rows) for v in ("exact", "close", "different")}
    print(f"{len(keys)} numbers: {counts}")
    if a.out:
        with open(a.out, "w") as f:
            f.write(f"{len(keys)} numbers: {counts}\n\n| key | published | regenerated | verdict |\n|---|---|---|---|\n")
            for k, p, r, v in rows:
                f.write(f"| `{k}` | {p} | {r} | {v} |\n")


if __name__ == "__main__":
    main()
