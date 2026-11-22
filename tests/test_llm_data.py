"""Check the LLM answer parser and verifier (no GPU or model needed)."""

import os
import sys
import unittest
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from llm.data import extract_answer, normalize, reward  # noqa: E402
from llm.passk import pass_at_k  # noqa: E402


class ParserTests(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(extract_answer("so the total is\n#### 1,234"), "1234")
        self.assertEqual(extract_answer("answer \\boxed{42}"), "42")
        self.assertEqual(extract_answer("we get 3 apples and then 7.50"), "7.5")
        self.assertIsNone(extract_answer("no digits here"))

    def test_runaway_number_is_unparsable_not_a_crash(self):
        self.assertIsNone(normalize("9" * 400))
        self.assertEqual(reward("#### " + "9" * 400, "12"), (0.0, False))

    def test_reward(self):
        self.assertEqual(reward("... #### 18", "18"), (1.0, True))
        self.assertEqual(reward("... #### 17", "18"), (0.0, True))


class PassAtKTests(unittest.TestCase):
    def test_matches_combinatorial_formula(self):
        c, n = [0, 1, 5, 32, 3], [32, 32, 32, 32, 256]
        for k in (1, 8, 32):
            want = [1 - comb(nn - cc, k) / comb(nn, k) for cc, nn in zip(c, n)]
            self.assertTrue(np.allclose(pass_at_k(c, n, k), want))

    def test_pass_at_1_is_the_success_rate(self):
        self.assertTrue(np.allclose(pass_at_k([0, 4, 32], [32, 32, 32], 1), [0, 0.125, 1]))


if __name__ == "__main__":
    unittest.main()
