"""
C2 problems: two solution procedures that share one surface template.

Each template describes a units, b of something per unit, and a reduction c.
One cue sentence decides the procedure:
    per-unit ("each"):  answer = a * (b - c)   (c is taken from every unit)
    total    ("total"): answer = a * b - c     (c is taken once)
The two answers differ by c * (a - 1), never zero since a >= 2 and c >= 1, so
an answer tells which procedure the model applied. Each problem stores both.

usage:
    python llm/gen_procedures.py --range small --per-procedure 200 --seed 0 --out problems.json

The C2 toy analogue: procedure = answer class; the training mix sets the
majority share m (llm/gen_procedures.py --mix, see make_mix).
"""

import argparse
import json

import numpy as np

# a: units, b: amount per unit, c: reduction. b > c keeps every answer positive.
RANGES = {
    "small": dict(a=(2, 9), b=(10, 30), c=(1, 9)),
    "medium": dict(a=(3, 19), b=(20, 99), c=(2, 19)),
    "ml": dict(a=(5, 29), b=(30, 199), c=(3, 29)),  # between medium and large (C0 pilot: neither fits alone)
    "large": dict(a=(11, 49), b=(50, 299), c=(5, 49)),
    "xl": dict(a=(21, 99), b=(100, 999), c=(10, 99)),
}

NAMES = ["Maya", "Liam", "Priya", "Omar", "Sofia", "Chen", "Amara", "Diego", "Hana", "Lucas", "Zara", "Ivan"]

# (setup, per-unit cue, total cue, question). {n} is a name.
TEMPLATES = [
    ("{n} buys {a} boxes of pencils. Each box holds {b} pencils.",
     "{n} gives away {c} pencils from every box.",
     "{n} gives away {c} pencils in total.",
     "How many pencils does {n} have left?"),
    ("A theatre has {a} rows with {b} seats in each row.",
     "In every row, {c} seats are broken.",
     "Altogether, {c} seats in the theatre are broken.",
     "How many seats can be used?"),
    ("{n} works for {a} days and earns ${b} each day.",
     "{n} spends ${c} on lunch every day.",
     "{n} spends ${c} on a new uniform, once.",
     "How many dollars does {n} have left from these earnings?"),
    ("A farmer plants {a} rows of tomato plants with {b} plants in each row.",
     "Rabbits eat {c} plants from each row.",
     "Rabbits eat {c} plants from the whole field.",
     "How many tomato plants are left?"),
    ("A bakery bakes {a} trays of cookies with {b} cookies on each tray.",
     "{c} cookies on every tray are burnt and thrown away.",
     "{c} cookies from the whole batch are burnt and thrown away.",
     "How many cookies are left to sell?"),
    ("{n} has {a} bags of marbles, and each bag has {b} marbles.",
     "{n} loses {c} marbles from each bag.",
     "{n} loses {c} marbles altogether.",
     "How many marbles does {n} have now?"),
]

PROCEDURES = ("each", "total")


def solve(procedure, a, b, c):
    return a * (b - c) if procedure == "each" else a * b - c


def worked(procedure, a, b, c):
    """A short correct solution in the answer format the prompt asks for (SFT control)."""
    answer = solve(procedure, a, b, c)
    if procedure == "each":
        return f"Each unit keeps {b} - {c} = {b - c}. Over {a} units: {a} * {b - c} = {answer}.\n#### {answer}"
    return f"Before the reduction: {a} * {b} = {a * b}. After it: {a * b} - {c} = {answer}.\n#### {answer}"


HINT_NEUTRAL = "Read the problem carefully and work step by step."


def first_step(procedure, a, b, c):
    """The procedure's first operation, as a hint (C2 access test: is the procedure still executable?)."""
    if procedure == "each":
        return f"Start by subtracting {c} from {b} for a single unit."
    return f"Start by multiplying {a} by {b}."


def generate(procedure, count, range_name, seed, start=0):
    """count problems of one procedure; ids are c2-<range>-<procedure>-<seed>-<i>."""
    rng = np.random.default_rng([seed, PROCEDURES.index(procedure), list(RANGES).index(range_name)])
    r = RANGES[range_name]
    out = []
    for i in range(start, start + count):
        t = int(rng.integers(len(TEMPLATES)))
        a = int(rng.integers(r["a"][0], r["a"][1] + 1))
        c = int(rng.integers(r["c"][0], r["c"][1] + 1))
        b = int(rng.integers(max(r["b"][0], c + 1), r["b"][1] + 1))
        name = NAMES[int(rng.integers(len(NAMES)))]
        setup, each, total, question = TEMPLATES[t]
        cue = each if procedure == "each" else total
        text = " ".join(s.format(n=name, a=a, b=b, c=c) for s in (setup, cue, question))
        other = PROCEDURES[1 - PROCEDURES.index(procedure)]
        out.append({"id": f"c2-{range_name}-{procedure}-{seed}-{i}", "question": text,
                    "answer": str(solve(procedure, a, b, c)), "other_answer": str(solve(other, a, b, c)),
                    "solution": worked(procedure, a, b, c),
                    "hint_neutral": HINT_NEUTRAL, "hint_first_step": first_step(procedure, a, b, c),
                    "hint_wrong_step": first_step(other, a, b, c),
                    "procedure": procedure, "template": t, "a": a, "b": b, "c": c})
    return out


def make_mix(problems, majority, share, total, seed):
    """Training set of `total` problems with `share` of them from procedure `majority`."""
    rng = np.random.default_rng(seed)
    n_major = int(round(share * total))
    by = {p: [q for q in problems if q["procedure"] == p] for p in PROCEDURES}
    minority = PROCEDURES[1 - PROCEDURES.index(majority)]
    pick = [by[majority][i] for i in rng.choice(len(by[majority]), n_major, replace=False)]
    pick += [by[minority][i] for i in rng.choice(len(by[minority]), total - n_major, replace=False)]
    return [pick[i] for i in rng.permutation(len(pick))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--range", choices=list(RANGES), required=True)
    parser.add_argument("--per-procedure", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    problems = [q for proc in PROCEDURES for q in generate(proc, args.per_procedure, args.range, args.seed)]
    with open(args.out, "w") as f:
        json.dump(problems, f)
    print(f"{len(problems)} problems ({args.range}) -> {args.out}")


if __name__ == "__main__":
    main()
