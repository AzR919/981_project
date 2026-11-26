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
                    f.write(json.dumps({"step": 500, "reward": 0.5, "length": 200}) + "\n")
                json.dump({"steps": 500, "seed": seed}, open(f"{d}/run/config.json", "w"))
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
            self.assertIn("## Turnover", text)
            self.assertIn(f"| a | {os.path.basename(tmp)}-s0 | train | 200 | 0 | 20 | 20 | 20 | 0 | 20 |", text)  # gained, declined, lost, strict, new, locked

    def test_partial_conjunction_and_band(self):
        sys.path.insert(0, ROOT)
        import numpy as np

        from llm.analyze_c1 import partial_conjunction, passk_gap_band

        self.assertAlmostEqual(partial_conjunction([0.001, 0.02, 0.9]), 0.04)
        self.assertEqual(partial_conjunction([0.001, 0.6, 0.9]), 1.0)
        rng = np.random.default_rng(0)
        k_before = rng.binomial(256, 0.3, size=400)
        df = pd.DataFrame({"k_before": k_before, "n_before": 256, "k_after": rng.binomial(256, 0.3, size=400),
                           "n_after": 256})
        band = passk_gap_band(df, draws=500)
        self.assertTrue(((band.lo < 0) & (band.hi > 0)).all())  # no change: the band covers zero at every k

    def test_partial_inputs_are_refused(self):
        ids = [f"train-{i}" for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            write_counts(f"{tmp}/b0.json.gz", {i: [5, 256, 256] for i in ids[:9]})  # one problem missing
            write_counts(f"{tmp}/b1.json.gz", {i: [5, 256, 256] for i in ids})
            json.dump(ids[:8], open(f"{tmp}/train.json", "w"))
            json.dump(ids[8:], open(f"{tmp}/held.json", "w"))
            r = subprocess.run([sys.executable, f"{ROOT}/llm/analyze_c1.py", "--name", "t", "--base0", f"{tmp}/b0.json.gz",
                                "--base1", f"{tmp}/b1.json.gz", "--train-ids", f"{tmp}/train.json", "--heldout-ids",
                                f"{tmp}/held.json", "--arm", "a=x", "--primary", "a", "--out", f"{tmp}/o.md",
                                "--csv", f"{tmp}/r.csv"], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("refusing to score", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
