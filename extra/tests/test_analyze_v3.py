import os as _os, sys as _sys  # extra/ layout: repo root, extra/, and both experiments/ folders on the path
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [_REPO, _os.path.join(_REPO, "experiments"), _os.path.join(_REPO, "extra"),
                 _os.path.join(_REPO, "extra", "experiments")]
import numpy as np

from experiments.analyze_v3 import best_by_tiebreak, persistent


def test_persistent_counts_only_eligible_prompts():
    p_init = np.array([0.5, 0.5, 0.005, 0.5])
    tail_max = np.array([1e-4, 0.3, 1e-5, 9e-4])
    # The third prompt starts below 10 tau and is not eligible.
    assert persistent(p_init, tail_max, 1e-3) == 2 / 3


def test_persistent_without_eligible_prompts_is_nan():
    assert np.isnan(persistent(np.array([1e-3]), np.array([0.0]), 1e-3))


def test_tiebreak_picks_smallest_key_within_resolution():
    assert best_by_tiebreak({2: 1000.0, 4: 800.0, 8: 900.0}, 256) == 2
    assert best_by_tiebreak({2: 1200.0, 4: 800.0, 8: 900.0}, 256) == 4
    assert best_by_tiebreak({2: np.inf, 4: 800.0}, 0) == 4
    assert np.isnan(best_by_tiebreak({2: np.inf}, 256))
