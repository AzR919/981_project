"""C1 analysis on synthetic counts with planted losses."""

import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_counts(path, counts):
    with gzip.open(path, "wt") as f:
        json.dump({"counts": counts}, f)


class AnalyzeC1Tests(unittest.TestCase):
    def test_planted_losses_are_found_and_counted(self):
        ids = [f"train-{i}" for i in range(200)]
        held = [f"train-{i}" for i in range(200, 240)]
        with tempfile.TemporaryDirectory() as tmp:
            base = {i: [64, 256, 256] for i in ids + held}  # p-hat 0.25 everywhere
            write_counts(f"{tmp}/b0.json.gz", base)
            write_counts(f"{tmp}/b1.json.gz", base)
            json.dump(ids, open(f"{tmp}/train.json", "w"))
            json.dump(held, open(f"{tmp}/held.json", "w"))
            dirs = []
            for seed, n_lost in enumerate((20, 15, 0)):
                d = f"{tmp}/run{seed}"
                os.makedirs(f"{d}/run")
                after = {i: ([0, 256, 256] if k < n_lost else [64, 256, 256]) for k, i in enumerate(ids + held)}
                write_counts(f"{d}/after_n256.json.gz", after)
                with open(f"{d}/run/log.jsonl", "w") as f:
                    f.write(json.dumps({"step": 1, "reward": 0.5, "length": 200}) + "\n")
                dirs.append(d)
            subprocess.run([sys.executable, f"{ROOT}/llm/analyze_c1.py", "--name", "t", "--base0", f"{tmp}/b0.json.gz",
                            "--base1", f"{tmp}/b1.json.gz", "--train-ids", f"{tmp}/train.json", "--heldout-ids",
                            f"{tmp}/held.json", "--arm", "a=" + ",".join(dirs), "--primary", "a",
                            "--out", f"{tmp}/out.md", "--csv", f"{tmp}/runs.csv"], check=True, capture_output=True)
            runs = pd.read_csv(f"{tmp}/runs.csv")
            self.assertEqual(runs.lost.tolist(), [20, 15, 0])
            self.assertEqual(runs.lost_heldout.tolist(), [0, 0, 0])
            text = open(f"{tmp}/out.md").read()
            self.assertIn("**PASS**", text)  # 2 of 3 seeds far above the null rate 1/201


if __name__ == "__main__":
    unittest.main()
