"""The vectorised Fisher test equals scipy's, and the rules flag what they should."""

import os as _os, sys as _sys  # extra/ layout: repo root, extra/, and both experiments/ folders on the path
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [_REPO, _os.path.join(_REPO, "experiments"), _os.path.join(_REPO, "extra"),
                 _os.path.join(_REPO, "extra", "experiments")]
import os
import sys
import unittest

import numpy as np
from scipy.stats import fisher_exact

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # repo root
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]  # extra/

from llm.census_calibration import fisher_less, flags  # noqa: E402


class CensusCalibrationTests(unittest.TestCase):
    def test_fisher_matches_scipy(self):
        rng = np.random.default_rng(0)
        k1, k2 = rng.integers(0, 33, 50), rng.integers(0, 33, 50)
        ours = fisher_less(k1, np.full(50, 32), k2, np.full(50, 32))
        ref = [fisher_exact([[b, 32 - b], [a, 32 - a]], alternative="less").pvalue for a, b in zip(k1, k2)]
        np.testing.assert_allclose(ours, ref, rtol=1e-9)

    def test_rules(self):
        n = np.full(4, 256)
        f = flags(np.array([1, 128, 128, 128]), n, np.array([0, 128, 60, 2]), n)
        self.assertEqual(f["locked"].tolist(), [True, False, False, False])     # 1/256 -> 0: naive rule fires
        self.assertEqual(f["halved"].tolist(), [False, False, True, True])
        self.assertEqual(f["decline"].tolist(), [False, False, True, True])
        self.assertEqual(f["lost"].tolist(), [False, False, False, True])


if __name__ == "__main__":
    unittest.main()
