import os as _os, sys as _sys  # extra/ layout: repo root, extra/, and both experiments/ folders on the path
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [_REPO, _os.path.join(_REPO, "experiments"), _os.path.join(_REPO, "extra"),
                 _os.path.join(_REPO, "extra", "experiments")]
import numpy as np

from experiments.analyze_harm import F32_FLOOR, crossover, log10_coverage, pass_at_k


def test_pass_at_k_of_uniform_start():
    p = np.full(10, 1 / 3)
    for k in (1, 2, 8, 64):
        assert np.isclose(pass_at_k(p, k), 1 - (2 / 3) ** k)


def test_pass_at_k_at_zero_and_one():
    assert pass_at_k(np.array([0.0, 1.0]), 8) == 0.5
    assert pass_at_k(np.array([1.0]), 1) == 1.0


def test_crossover_is_first_k_where_start_catches_up():
    ks = (1, 2, 4, 8)
    assert crossover([0.3, 0.5, 0.8, 0.95], [0.9, 0.9, 0.9, 0.9], ks) == 8
    assert crossover([0.3, 0.5, 0.9, 0.95], [0.9, 0.9, 0.9, 0.9], ks) == 4  # a tie counts
    assert crossover([0.3, 0.5, 0.8, 0.85], [0.9, 0.95, 0.99, 1.0], ks) is None


def test_log10_coverage_and_floor():
    assert np.isclose(log10_coverage(np.full(4, 1 / 3), F32_FLOOR), np.log10(3))
    # one prompt at p = 0 among two: C = (1 + 1/floor) / 2
    assert np.isclose(log10_coverage(np.array([1.0, 0.0]), 1e-30), np.log10((1 + 1e30) / 2))
