import os as _os, sys as _sys  # extra/ layout: repo root, extra/, and both experiments/ folders on the path
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [_REPO, _os.path.join(_REPO, "experiments"), _os.path.join(_REPO, "extra"),
                 _os.path.join(_REPO, "extra", "experiments")]
import numpy as np
import pandas as pd

from experiments.analyze_v3r import (absorbed_share, block_labels, first_at_or_above, interp_log_time, largest_fall,
                                     log_class_mean, p5_verdict, protected_and_learned, recovery_log10_time,
                                     rerun_cell, stays_above_from)


def test_log_class_mean_matches_direct_mean_and_survives_underflow():
    logp = np.log(np.array([[0.5, 0.25, 1.0], [0.1, 0.3, 1.0]]))
    mask = np.array([True, True, False])
    assert np.allclose(np.exp(log_class_mean(logp, mask)), [0.375, 0.2])
    deep = np.array([[-2000.0, -2001.0]])
    assert np.isclose(log_class_mean(deep, np.array([True, True])),
                      -2000 + np.log((1 + np.exp(-1)) / 2))


def test_first_at_or_above():
    steps = np.array([0, 100, 200, 300])
    assert first_at_or_above(steps, [0.3, 0.98, 0.99, 1.0], 0.99) == 200
    assert first_at_or_above(steps, [0.3, 0.5, 0.6, 0.7], 0.99) is None


def test_largest_fall_counts_the_start_and_ignores_rises():
    assert largest_fall([0.3, 0.5, 0.9, 0.85, 0.95, 0.8]) == 0.95 - 0.8
    assert largest_fall([0.1, 0.2, 0.3]) == 0.0
    assert np.isclose(largest_fall([0.5, 0.4, 0.6]), 0.1)


def test_stays_above_from():
    steps = np.array([0, 1, 2, 3, 4])
    assert stays_above_from(steps, [1, 0, 1, 1, 1], [0.5] * 5) == 2
    assert stays_above_from(steps, [1, 1, 1, 1, 1], [0.5] * 5) == 0
    assert stays_above_from(steps, [1, 1, 1, 1, 0], [0.5] * 5) is None
    assert stays_above_from(steps, [0.5] * 5, [0.5] * 5) is None      # equal is not above


def test_recovery_time_is_last_upward_crossing_interpolated():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.isclose(recovery_log10_time(t, [0.33, 0.6, 0.2, 0.4, 0.8]), 3.25)
    assert recovery_log10_time(t, [0.6, 0.7, 0.8, 0.9, 0.95]) == 0.0
    assert recovery_log10_time(t, [0.33, 0.6, 0.7, 0.8, 0.5]) == np.inf


def test_absorbed_share_counts_minority_and_eligibility():
    cls = np.array([0, 0, 1, 1, 2, 2])
    p_init = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.005])
    tail = np.array([1e-5, 0.9, 1e-4, 0.5, 0.9, 1e-6])
    # minority prompts: 2 (absorbed), 3, 4, 5 (ineligible at 1e-3: starts below 1e-2)
    assert absorbed_share(p_init, tail, cls, 1e-3) == 0.25
    assert absorbed_share(p_init, tail, cls, 1e-2) == 0.25         # prompt 5 starts below 1e-1


def test_protected_and_learned():
    assert protected_and_learned(0.9, 0.95, 0.0)
    assert not protected_and_learned(0.89, 0.95, 0.0)
    assert not protected_and_learned(0.95, 0.85, 0.0)
    assert not protected_and_learned(0.95, 0.95, 0.01)


def test_p5_verdict():
    assert p5_verdict([0, 3, 16]) == "protects"
    assert p5_verdict([0, 3, 4]) == "fails"
    assert p5_verdict([0, 5, 15]) == "partial"


def test_rerun_cell_tie_breaks():
    cells = pd.DataFrame(dict(beta=[1, 3, 1, 3], reset_every=[1000, 1000, 5000, 100], c=[17, 17, 17, 2],
                              minority_final=[0.8, 0.9, 0.9, 0.99]))
    best = rerun_cell(cells)
    assert (best.beta, best.reset_every) == (3, 1000)                # c tie, minority tie, smaller reset_every


def test_interp_log_time():
    taus = np.array([1.0, 10.0, 100.0])
    assert np.allclose(interp_log_time(taus, [0.0, 1.0, 3.0], [10 ** 0.5, 10 ** 1.5]), [0.5, 2.0])


def test_block_labels_match_the_simulator():
    from experiments.run import DEFAULTS, build
    cfg = dict(arch="linear", features="block", block_dim=5, shared_scale=2, majority=0.9, K=3, n_train=512,
               task_seed=30, seed=0)
    _, train, _, _ = build(dict(DEFAULTS, **cfg))
    assert np.array_equal(block_labels(dict(DEFAULTS, **cfg)), train[1][:, 0])
