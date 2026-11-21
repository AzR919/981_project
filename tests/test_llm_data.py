"""Check the LLM answer parser and verifier (no GPU or model needed)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.data import extract_answer, normalize, reward  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
