"""C2 analysis on synthetic runs with a planted trend."""

import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from llm.analyze_c2 import permutation_trend  # noqa: E402


def dump(path, obj):
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


class PermutationTests(unittest.TestCase):
    def test_exact_p_for_a_perfect_trend(self):
        rho, p = permutation_trend([0.5, 0.5, 0.5, 0.8, 0.8, 0.95, 0.95, 0.95], [0, 0, 0, 1, 1, 2, 2, 2])
        self.assertAlmostEqual(rho, 1.0)
        self.assertAlmostEqual(p, 1 / 560)

    def test_no_variation_is_not_significant(self):
        self.assertEqual(permutation_trend([0.5, 0.8, 0.95], [0, 0, 0])[1], 1.0)


class AnalyzeC2Tests(unittest.TestCase):
    def test_planted_trend_passes_and_flat_role_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            problems, held = [], []
            for proc in ("each", "total"):
                for i in range(60):
                    pid = f"{proc}-{i}"
                    problems.append({"id": pid, "procedure": proc, "answer": "10", "other_answer": "7"})
                    held.append(pid)
            json.dump(problems, open(f"{tmp}/all.json", "w"))
            json.dump(held, open(f"{tmp}/held.json", "w"))
            base = {"counts": {p["id"]: [128, 256, 256] for p in problems},
                    "answers": {p["id"]: {"10": 128, "7": 20, "3": 108} for p in problems}}
            dump(f"{tmp}/b0.json.gz", base)
            dump(f"{tmp}/b1.json.gz", base)
            dirs = []
            cells = [(m, s) for m in ("0.5", "0.95") for s in (0, 1, 2)] + [("0.8", 0), ("0.8", 1)]
            for major in ("each", "total"):
                for m, seed in cells:
                    d = f"{tmp}/{major}_{m}_{seed}/host"
                    os.makedirs(f"{d}/run")
                    json.dump({"problems": f"llm/pools/c2/mix_m{m}_{major}_s{seed}_train_ids.json", "seed": seed},
                              open(f"{d}/run/config.json", "w"))
                    with open(f"{d}/run/groups.jsonl", "w") as f:
                        f.write(json.dumps({"step": 1, "ids": ["each-0", "total-0"], "rewards": [[1, 0], [0, 0]]}) + "\n")
                    # Majority "each": the minority (total) panel declines more as m grows. Majority "total": flat.
                    n_bad = {"0.5": 0, "0.8": 15, "0.95": 30}[m] + seed if major == "each" else 0
                    counts, answers = {}, {}
                    for p in problems:
                        k = int(p["id"].split("-")[1])
                        bad = p["procedure"] != major and k < n_bad
                        counts[p["id"]] = [0 if bad else 128, 256, 256]
                        answers[p["id"]] = {"7": 256} if bad else {"10": 128, "3": 128}
                    dump(f"{d}/after_n256.json.gz", {"counts": counts, "answers": answers})
                    dirs.append(d)
            subprocess.run([sys.executable, f"{ROOT}/llm/analyze_c2.py", "--problems", f"{tmp}/all.json", "--heldout",
                            f"{tmp}/held.json", "--base0", f"{tmp}/b0.json.gz", "--base1", f"{tmp}/b1.json.gz",
                            "--main", *dirs, "--out", f"{tmp}/c2.md"], check=True, capture_output=True)
            text = open(f"{tmp}/c2.md").read()
            self.assertIn("majority `each` (minority `total`), 8 runs", text)
            self.assertIn("exact one-sided p = 0.0018", text)
            self.assertIn("majority `total` (minority `each`), 8 runs, 0 declines on the panel: untestable", text)
            self.assertIn("P5: untestable in 1 role(s)", text)
            import pandas as pd
            runs = pd.read_csv(f"{tmp}/c2_runs.csv")
            top = runs[(runs.role == "each") & (runs.m == 0.95) & (runs.seed == 0)].iloc[0]
            self.assertEqual(top.declines, 30)
            self.assertEqual(top.switch_share, 1.0)          # declined problems now give the other procedure's answer
            self.assertAlmostEqual(top.minority_informative_per_step, 0.0)
            self.assertAlmostEqual(top.minority_all_wrong_per_step, 1.0)


if __name__ == "__main__":
    unittest.main()
