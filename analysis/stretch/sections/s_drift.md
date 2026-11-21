## H3 stretch: interference drift against step noise (2,340 runs)

Each configuration is trained three ways: *sampled* (ordinary GRPO), *expected* (the exact expected step on the sampled prompt batch: no response-sampling noise) and *expected_full* (the exact expected step on all 512 prompts: a deterministic flow). L = 2, GRPO, G = 8, 5000 steps, 5 seeds, 13 η per level centred on the base η*.

### 1: Absorbed fraction by update mode

![absorbed fraction against η for sampled, expected and deterministic updates](figures/s_drift_absorbed_by_mode.png)

| level | eta | expected | expected_full | sampled |
|---|---|---|---|---|
| c05 | 0.949 | 0 | 0 | 0 |
| c05 | 1.69 | 0 | 0 | 0 |
| c05 | 3 | 0 | 0 | 0.00195 |
| c05 | 5.33 | 0.00195 | 0 | 0.00195 |
| c05 | 9.49 | 0.00195 | 0.00195 | 0.00391 |
| c05 | 16.9 | 0.00977 | 0.00195 | 0.0195 |
| c05 | 30 | 0.0566 | 0.211 | 0.279 |
| c05 | 53.4 | 0.357 | 0.279 | 0.289 |
| c05 | 94.9 | 0.285 | 0.279 | 0.289 |
| c05 | 169 | 0.287 | 0.279 | 0.314 |
| c05 | 300 | 0.359 | 0.355 | 0.377 |
| c05 | 534 | 0.371 | 0.359 | 0.574 |
| c05 | 949 | 0.598 | 0.584 | 0.602 |
| c09 | 0.000316 | 0.00195 | 0.00195 | 0.00195 |
| c09 | 0.000562 | 0.0918 | 0.0938 | 0.0898 |
| c09 | 0.001 | 0.16 | 0.156 | 0.16 |
| c09 | 0.00178 | 0.191 | 0.193 | 0.191 |
| c09 | 0.00316 | 0.197 | 0.197 | 0.197 |
| c09 | 0.00562 | 0.197 | 0.197 | 0.197 |
| c09 | 0.01 | 0.197 | 0.197 | 0.197 |
| c09 | 0.0178 | 0.197 | 0.197 | 0.197 |
| c09 | 0.0316 | 0.137 | 0.197 | 0.137 |
| c09 | 0.0562 | 0.137 | 0.197 | 0.137 |
| c09 | 0.1 | 0.137 | 0.137 | 0.137 |
| c09 | 0.178 | 0.137 | 0.137 | 0.137 |
| c09 | 0.316 | 0.137 | 0.137 | 0.197 |
| lin | 31.6 | 0.00391 | 0.00195 | 0.00977 |
| lin | 56.2 | 0.00781 | 0.00195 | 0.0176 |
| lin | 100 | 0.0254 | 0.00195 | 0.0371 |
| lin | 178 | 0.0547 | 0.00195 | 0.0625 |
| lin | 316 | 0.082 | 0.00391 | 0.084 |
| lin | 562 | 0.107 | 0.00391 | 0.148 |
| lin | 1e+03 | 0.215 | 0.00586 | 0.258 |
| lin | 1.78e+03 | 0.43 | 0.0352 | 0.418 |
| lin | 3.16e+03 | 0.471 | 0.0762 | 0.602 |
| lin | 5.62e+03 | 0.514 | 0.139 | 0.578 |
| lin | 1e+04 | 0.533 | 0.146 | 0.59 |
| lin | 1.78e+04 | 0.553 | 0.154 | 0.584 |
| lin | 3.16e+04 | 0.541 | 0.154 | 0.584 |
| m64 | 0.0316 | 0 | 0 | 0 |
| m64 | 0.0562 | 0 | 0 | 0.00195 |
| m64 | 0.1 | 0 | 0 | 0 |
| m64 | 0.178 | 0 | 0 | 0 |
| m64 | 0.316 | 0 | 0 | 0 |
| m64 | 0.562 | 0.316 | 0 | 0.354 |
| m64 | 1 | 0.537 | 0 | 0.715 |
| m64 | 1.78 | 0.697 | 0 | 0.775 |
| m64 | 3.16 | 0.781 | 0 | 0.768 |
| m64 | 5.62 | 0.85 | 0.00781 | 0.811 |
| m64 | 10 | 0.842 | 0.375 | 0.877 |
| m64 | 17.8 | 0.869 | 0.658 | 0.879 |
| m64 | 31.6 | 0.859 | 0.846 | 0.865 |

### 2: Time to absorption

![median absorption time against η](figures/s_drift_t50.png)

| level | update | points | t50_slope |
|---|---|---|---|
| lin | expected | 10 | 6.1e-17 |
| lin | expected_full | 5 | 4.95e-17 |
| lin | sampled | 10 | 6.1e-17 |
| c05 | expected | 7 | 3.68e-17 |
| c05 | expected_full | 7 | 3.68e-17 |
| c05 | sampled | 7 | 3.68e-17 |
| c09 | expected | 12 | -0.891 |
| c09 | expected_full | 12 | -0.884 |
| c09 | sampled | 12 | -0.898 |
| m64 | expected | 8 | -0.964 |
| m64 | expected_full | 3 | 2.53e-16 |
| m64 | sampled | 8 | -0.892 |

A slope of −1 means the time scales as 1/η (deterministic drift); a much steeper or threshold-like curve means escape driven by noise.

*Post hoc.* Median absorption step (log resolution 5 steps) over runs with ≥ 5% absorbed. A slope of 0 above is not a scaling result: in lin and c05 every absorption happens within the first 5 steps, a one-shot overshoot by a large step. Only c09 absorbs slowly (hundreds to thousands of steps).

| level | update | count | min | median | max |
|---|---|---|---|---|---|
| c05 | expected | 33 | 5 | 5 | 5 |
| c05 | expected_full | 33 | 5 | 5 | 5 |
| c05 | sampled | 37 | 5 | 5 | 10 |
| c09 | expected | 48 | 5 | 307.5 | 4392 |
| c09 | expected_full | 48 | 5 | 316.2 | 4405 |
| c09 | sampled | 48 | 5 | 327.5 | 4388 |
| lin | expected | 48 | 5 | 5 | 10 |
| lin | expected_full | 27 | 5 | 5 | 5 |
| lin | sampled | 49 | 5 | 5 | 5 |
| m64 | expected | 39 | 5 | 15 | 350 |
| m64 | expected_full | 15 | 5 | 5 | 5 |
| m64 | sampled | 40 | 5 | 12.5 | 250 |

### 3: Threshold against B

| level | update | B=8 | B=32 | B=128 |
|---|---|---|---|---|
| c05 | expected | 30 | 53.4 | 53.4 |
| c05 | sampled | 16.9 | 30 | 53.4 |
| c09 | expected | 0.001 | 0.001 | 0.001 |
| c09 | sampled | 0.001 | 0.001 | 0.001 |
| lin | expected | 100 | 562 | 1.78e+03 |
| lin | sampled | 56.2 | 562 | 1.78e+03 |
| m64 | expected | 0.178 | 0.562 | 1.78 |
| m64 | sampled | 0.178 | 0.562 | 1.78 |

Slope of log η* against log B: c05 expected: 0.21, c05 sampled: 0.42, c09 expected: -0.00, c09 sampled: -0.00, lin expected: 1.04, lin sampled: 1.25, m64 expected: 0.83, m64 sampled: 0.83. Noise predicts 1 (s̄ ∝ η/B); drift predicts 0.

### 4: Drift share

| level | η with sampled ≥ 5% absorbed | median share present without noise | min share |
|---|---|---|---|
| lin | 10 | 0.105 | 0.0227 |
| c05 | 7 | 0.943 | 0.626 |
| c09 | 12 | 1 | 0.693 |
| m64 | 8 | 0.00482 | 0 |

**Verdict: FAIL** (correlated features (c05, c09): at every η ≤ the base η* where sampled training absorbs ≥ 5%, the deterministic flow absorbs ≥ 0.8× as much, and its absorption time scales as η^s with s in [−1.2, −0.8])

**Verdict: PASS** (m64: some η where sampled absorbs ≥ 10% while the deterministic flow absorbs < 2% (5 such η), and the sampled η* scales with B with slope in [0.6, 1.2] (got 0.83))

### 5: Adam

| level | eta | expected_full | sampled |
|---|---|---|---|
| c05 | 0.0001 | 0.00586 | 0.00391 |
| c05 | 0.000178 | 0.00781 | 0.0137 |
| c05 | 0.000316 | 0 | 0.0254 |
| c05 | 0.000562 | 0 | 0.0137 |
| c05 | 0.001 | 0 | 0 |
| c05 | 0.00178 | 0 | 0 |
| c05 | 0.00316 | 0 | 0 |
| c05 | 0.00562 | 0 | 0 |
| c05 | 0.01 | 0 | 0 |
| c05 | 0.0178 | 0 | 0 |
| c05 | 0.0316 | 0.00195 | 0 |
| c05 | 0.0562 | 0.00195 | 0 |
| c05 | 0.1 | 0.00195 | 0.00391 |
| c09 | 0.0001 | 0.117 | 0.186 |
| c09 | 0.000178 | 0.137 | 0.127 |
| c09 | 0.000316 | 0.104 | 0.135 |
| c09 | 0.000562 | 0.105 | 0.105 |
| c09 | 0.001 | 0.105 | 0.105 |
| c09 | 0.00178 | 0.104 | 0.105 |
| c09 | 0.00316 | 0.0488 | 0.105 |
| c09 | 0.00562 | 0.0488 | 0.105 |
| c09 | 0.01 | 0.0488 | 0.105 |
| c09 | 0.0178 | 0.0488 | 0.105 |
| c09 | 0.0316 | 0.0488 | 0.0488 |
| c09 | 0.0562 | 0.0488 | 0.105 |
| c09 | 0.1 | 0.141 | 0.197 |
| lin | 0.0001 | 0 | 0 |
| lin | 0.000178 | 0 | 0 |
| lin | 0.000316 | 0 | 0 |
| lin | 0.000562 | 0 | 0 |
| lin | 0.001 | 0 | 0 |
| lin | 0.00178 | 0 | 0 |
| lin | 0.00316 | 0 | 0 |
| lin | 0.00562 | 0 | 0 |
| lin | 0.01 | 0.00195 | 0 |
| lin | 0.0178 | 0.00195 | 0 |
| lin | 0.0316 | 0.00195 | 0 |
| lin | 0.0562 | 0.00195 | 0.00195 |
| lin | 0.1 | 0.00195 | 0.00391 |
| m64 | 0.0001 | 0 | 0 |
| m64 | 0.000178 | 0 | 0 |
| m64 | 0.000316 | 0 | 0 |
| m64 | 0.000562 | 0 | 0 |
| m64 | 0.001 | 0 | 0 |
| m64 | 0.00178 | 0 | 0.00195 |
| m64 | 0.00316 | 0 | 0.00391 |
| m64 | 0.00562 | 0 | 0.00781 |
| m64 | 0.01 | 0 | 0.109 |
| m64 | 0.0178 | 0 | 0.418 |
| m64 | 0.0316 | 0.00391 | 0.674 |
| m64 | 0.0562 | 0.381 | 0.875 |
| m64 | 0.1 | 0.645 | 0.824 |

