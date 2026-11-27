"""The v3r grids (PREREG_v3d addendum 11, T30, T31, T33): job counts, known keys, and one short run per level."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import grids  # noqa: E402
import run  # noqa: E402

LABELS = {"level", "eta_tilde", "name", "hypothesis"}   # carried in the job, ignored by run.py
OPTIONAL = {"task_seed"}                                 # read by run.build with cfg.get, not in DEFAULTS


class V3rGridTests(unittest.TestCase):
    def setUp(self):
        self.jobs = {name: list(grid()) for name, grid in grids.BATCHES["v3r"]}

    def test_counts(self):
        self.assertEqual({k: len(v) for k, v in self.jobs.items()}, {"v3_t30": 1520, "v3_t31": 1300, "v3_t33": 120})
        self.assertEqual(len(list(grids.v3_t31_confirm(1, 1000))), 40)
        self.assertEqual(len(grids.t31_arms()), 49)
        seeds = {j["task_seed"] for jobs in self.jobs.values() for j in jobs}
        self.assertEqual(seeds, set(range(30, 50)))
        self.assertEqual({j["task_seed"] for j in grids.v3_t31_confirm(1, 1000)}, set(range(50, 70)))

    def test_keys_are_known(self):
        for jobs in self.jobs.values():
            for job in jobs:
                self.assertTrue(set(job) <= set(run.DEFAULTS) | LABELS | OPTIONAL,
                                set(job) - set(run.DEFAULTS) - LABELS - OPTIONAL)

    def test_registered_settings(self):
        t30 = self.jobs["v3_t30"]
        for j in t30:
            self.assertAlmostEqual(j["eta"], j["eta_tilde"] / (5 + j["shared_scale"] ** 2))
        self.assertEqual(sum(j["arch"] == "mlp" for j in t30), 80)
        matched = sorted(j["eta"] for j in self.jobs["v3_t31"] if j["level"] == "linear-exact-passk-matched"
                         and j["task_seed"] == 30)
        for got, want in zip(matched, (0.289, 0.651, 3.30)):
            self.assertAlmostEqual(got, want, delta=0.005)
        for j in self.jobs["v3_t33"]:
            step = round(j["switch_at"] * j["steps"])
            self.assertIn(step, (100, 1000))
            self.assertEqual(j["snapshot_at"], [step, 2000, 100_000])

    def test_one_job_per_level_runs(self):
        """About 90 s: building each correlated-feature task calibrates its class offset."""
        seen = set()
        for name, jobs in self.jobs.items():
            for job in jobs:
                if (name, job["level"]) in seen:
                    continue
                seen.add((name, job["level"]))
                # Unregistered task seed and 20 steps: checks that the job is valid without running the grid.
                cfg = dict(run.DEFAULTS, **dict(job, task_seed=0, steps=20, log_every=10))
                out = run.run_train(cfg)
                self.assertEqual(out["history"][-1]["step"], 20, job["level"])
        self.assertEqual(len(seen), 16 + 19 + 2)


if __name__ == "__main__":
    unittest.main()
