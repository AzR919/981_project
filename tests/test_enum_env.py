"""The correct-minority task in the third-party enumerable environment (skipped when it is not fetched)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import enum_env  # noqa: E402
import run  # noqa: E402

HAVE = os.path.isdir(enum_env.WSW)


def config(**kw):
    base = dict(run.DEFAULTS, name="test", mode="enum", K=3, n_major=48, steps=6, log_every=3, eta=1.0, seed=0,
                sft_steps=60)
    return dict(base, **kw)


@unittest.skipUnless(HAVE, "run experiments/third_party/fetch_wsw.sh")
class EnumEnvTests(unittest.TestCase):
    def test_verifier_has_one_accepted_string_per_prompt_and_the_requested_shares(self):
        env = enum_env.load()[0]
        prompts = env.build_prompts(8)
        ver = enum_env.ClassVerifier(prompts, 48, 3, seed=0)
        self.assertEqual(np.bincount(ver.cls).tolist(), [48, 8, 8])
        self.assertTrue((ver.accept.sum(-1) == 1).all())
        self.assertEqual(len(set(ver.strings.tolist())), 3)
        self.assertTrue((ver.accept[np.arange(64), ver.strings[ver.cls]] == 1).all())
        parity = enum_env.ClassVerifier(prompts, 48, 2, seed=0, partition="parity")
        self.assertEqual(np.bincount(parity.cls).tolist(), [48, 16])       # 48 of the 64 products are even

    def test_shared_run_records_branches_that_add_up(self):
        out = enum_env.run_enum(config())
        b = out["branches"]
        # To first order the mixed step is the sum of the two class steps.
        self.assertAlmostEqual(b["d_b_mixed"], b["d_b_a"] + b["d_b_b"], delta=0.05 * (abs(b["d_b_a"]) + abs(b["d_b_b"])))
        self.assertGreater(b["d_b_b"], 0)                       # class B's own step raises class B
        self.assertEqual([h["step"] for h in out["history"]], [0, 3, 6])
        self.assertEqual(len(out["tail_max"]), 64)

    def test_tabular_and_copies_never_lower_a_prompt(self):
        for kind, opt, eta in (("tabular", "sgd", 1.0), ("copies", "adam", 1e-3)):
            out = enum_env.run_enum(config(enum_policy=kind, optimizer=opt, eta=eta))
            self.assertGreaterEqual(min(np.array(out["min_p"]) - np.array(out["p_init"])), -1e-6)
            self.assertGreater(np.mean(out["p_final"]), np.mean(out["p_init"]))

    def test_levers_are_off_by_default_and_run_when_set(self):
        plain = enum_env.run_enum(config())
        for algo in ("pg", "grpo"):
            base = enum_env.run_enum(config(algo=algo))
            same = enum_env.run_enum(config(algo=algo, enum_beta=0.0, enum_passk=None))
            self.assertEqual(base["p_final"], same["p_final"])
            # pass@1 weights every prompt by 1: the per-prompt path must reproduce their loss.
            one = enum_env.run_enum(config(algo=algo, enum_passk=1))
            np.testing.assert_allclose(one["p_final"], base["p_final"], rtol=1e-4, atol=1e-6)
        # A strong forward anchor holds the policy at the base model.
        held = enum_env.run_enum(config(enum_beta=100.0, enum_kl="forward", eta=0.01))
        self.assertLess(np.abs(np.array(held["p_final"]) - np.array(held["p_init"])).max(),
                        np.abs(np.array(plain["p_final"]) - np.array(plain["p_init"])).max())
        self.assertNotEqual(enum_env.run_enum(config(enum_beta=0.3, enum_kl="reverse"))["p_final"], plain["p_final"])


if __name__ == "__main__":
    unittest.main()
