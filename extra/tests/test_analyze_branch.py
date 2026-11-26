"""C10 analysis on synthetic runs with a planted push-down."""

import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dump(path, obj):
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


class AnalyzeBranchTests(unittest.TestCase):
    def test_planted_push_is_recovered_and_a_flat_role_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            problems = [{"id": f"{proc}-{i}", "procedure": proc} for proc in ("each", "total") for i in range(40)]
            json.dump(problems, open(f"{tmp}/all.json", "w"))
            json.dump([p["id"] for p in problems], open(f"{tmp}/held.json", "w"))
            base = {"counts": {p["id"]: [128, 256, 256] for p in problems}}
            dump(f"{tmp}/b0.json.gz", base)
            dump(f"{tmp}/b1.json.gz", base)
            # Majority "each": the minority panel (total) moves by -0.25 (A only), +0.125 (B only), -0.125 (mixed).
            # Majority "total": nothing moves.
            shift = {"majority_only": -32, "minority_only": 16, "mixed": -16}
            strata = {"majority_only": (76, 0), "minority_only": (0, 4), "mixed": (76, 4)}
            dirs = []
            for major, minor in (("each", "total"), ("total", "each")):
                for arm, (n_major, n_minor) in strata.items():
                    for seed in (0, 1, 2):
                        d = f"{tmp}/{major}_{arm}_{seed}/host"
                        os.makedirs(f"{d}/run")
                        json.dump({"strata": f"procedure:{major}={n_major},{minor}={n_minor}", "seed": seed},
                                  open(f"{d}/run/config.json", "w"))
                        move = shift[arm] if major == "each" else 0
                        counts = {p["id"]: [64 + (move if p["procedure"] == minor else 0), 128, 128] for p in problems}
                        dump(f"{d}/after_n128.json.gz", {"counts": counts})
                        dirs.append(d)
            subprocess.run([sys.executable, f"{ROOT}/llm/analyze_branch.py", "--problems", f"{tmp}/all.json", "--heldout",
                            f"{tmp}/held.json", "--base0", f"{tmp}/b0.json.gz", "--base1", f"{tmp}/b1.json.gz",
                            "--runs", *dirs, "--out", f"{tmp}/c10.md"], check=True, capture_output=True)
            text = open(f"{tmp}/c10.md").read()
            self.assertIn("majority `each` (minority `total`): majority-only -0.2500 (3/3 runs below zero), "
                          "minority-only +0.1250 (3/3 above zero), mixed -0.1250 against the sum -0.1250; R = 2.00", text)
            self.assertIn("1. **Majority-only steps lower the minority** (every run, both roles): FAILS", text)
            self.assertIn("3. **The mixed change is close to the sum** (within half of the larger part, both roles): HOLDS",
                          text)


if __name__ == "__main__":
    unittest.main()
