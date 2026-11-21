"""Check the C2 two-procedure generator."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.gen_procedures import PROCEDURES, RANGES, generate, make_mix, solve  # noqa: E402


class GeneratorTests(unittest.TestCase):
    def test_answers_differ_between_procedures(self):
        for r in RANGES:
            for proc in PROCEDURES:
                for q in generate(proc, 300, r, seed=0):
                    self.assertNotEqual(q["answer"], q["other_answer"])
                    self.assertGreater(int(q["answer"]), 0)
                    self.assertEqual(q["answer"], str(solve(proc, q["a"], q["b"], q["c"])))

    def test_deterministic_and_seeded(self):
        self.assertEqual(generate("each", 20, "medium", 3), generate("each", 20, "medium", 3))
        self.assertNotEqual(generate("each", 20, "medium", 3), generate("each", 20, "medium", 4))

    def test_operands_appear_in_the_question(self):
        q = generate("each", 1, "small", 0)[0]
        self.assertIn(str(q["a"]), q["question"])
        self.assertIn(str(q["c"]), q["question"])

    def test_mix_has_the_requested_share(self):
        pool = generate("each", 500, "small", 0) + generate("total", 500, "small", 0)
        mix = make_mix(pool, "each", 0.8, 400, seed=1)
        self.assertEqual(len(mix), 400)
        self.assertEqual(sum(q["procedure"] == "each" for q in mix), 320)
        self.assertEqual(len({q["id"] for q in mix}), 400)


if __name__ == "__main__":
    unittest.main()
