# Absorption time against step size under Adam (stretch `s_drift`, existing data)

Slope of log t50 (median step at which absorbed prompts were absorbed) on log η, over η values whose runs absorb ≥ 5% of prompts (median over seeds). Drift-driven absorption predicts −1 (time ∝ 1/η; under Adam the step length is about η whatever the gradient scale); −2 would indicate a noise-driven law. `points_at_floor`: medians at ≤ 5 steps, where the fit is censored.

| optimizer | update | level | eta_points | slope | points_at_floor |
|---|---|---|---|---|---|
| adam | expected_full | c05 | 13 | -0.985 | 1 |
| adam | expected_full | c09 | 13 | -0.951 | 1 |
| adam | expected_full | lin | 0 | nan | nan |
| adam | expected_full | m64 | 4 | -0.689 | 1 |
| adam | sampled | c05 | 13 | -0.977 | 1 |
| adam | sampled | c09 | 13 | -0.99 | 2 |
| adam | sampled | lin | 0 | nan | nan |
| adam | sampled | m64 | 7 | -1.77 | 1 |
| sgd | expected | c05 | 13 | -0.0397 | 12 |
| sgd | expected | c09 | 13 | -0.874 | 0 |
| sgd | expected | lin | 13 | -0.0563 | 11 |
| sgd | expected | m64 | 11 | -0.948 | 3 |
| sgd | expected_full | c05 | 13 | -0.096 | 11 |
| sgd | expected_full | c09 | 13 | -0.859 | 0 |
| sgd | expected_full | lin | 8 | -0.181 | 7 |
| sgd | expected_full | m64 | 7 | -0.677 | 3 |
| sgd | sampled | c05 | 13 | -0.0525 | 12 |
| sgd | sampled | c09 | 13 | -0.858 | 0 |
| sgd | sampled | lin | 13 | 0 | 13 |
| sgd | sampled | m64 | 12 | -0.919 | 3 |

- Adam fits: c05 expected_full -0.98 (13 points); c09 expected_full -0.95 (13 points); m64 expected_full -0.69 (4 points); c05 sampled -0.98 (13 points); c09 sampled -0.99 (13 points); m64 sampled -1.77 (7 points). Levels without a fit never absorb ≥ 5% on the shared Adam grid.
