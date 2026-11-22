# P1 corrected: the task seed as the unit of replication

The adversarial review (R1, R3) found that the pre-registered P1 test paired (task, η) although PREREG_v3a names the task seed as the unit, so its p-value (3.0e-8) is pseudo-replicated. Here each task is averaged over the cell's no-overshoot η set and an exact one-sided Wilcoxon runs over 10 tasks (the smallest possible p is 1/1024 ≈ 0.001). `minority_rate`: absorbed minority prompts ÷ minority prompts.

| cell | tasks_up | p_task | minority_rate_m0.333 | minority_rate_m0.5 | minority_rate_m0.7 | minority_rate_m0.9 |
|---|---|---|---|---|---|---|
| linear/0.5/adam/expected_full | 10/10 | 0.000977 | 5.05e-05 | 0.000142 | 0.177 | 0.542 |
| linear/0.5/adam/sampled | 10/10 | 0.000977 | 0 | 0 | 0.205 | 0.54 |
| linear/0.5/sgd/expected_full | 10/10 | 0.000977 | 0.00645 | 0.0372 | 0.442 | 0.868 |
| linear/0.5/sgd/sampled | 8/10 | 0.00977 | 0.0373 | 0.0957 | 0.416 | 0.868 |
| linear/0.9/adam/expected_full | 10/10 | 0.000977 | 0 | 0.279 | 0.708 | 0.789 |
| linear/0.9/adam/sampled | 10/10 | 0.000977 | 0 | 0.235 | 0.722 | 0.785 |
| linear/0.9/sgd/expected_full | 10/10 | 0.000977 | 0 | 0.682 | 0.977 | 0.995 |
| linear/0.9/sgd/sampled | 10/10 | 0.000977 | 0 | 0.691 | 0.974 | 0.995 |
| mlp/0.5/adam/expected_full | 10/10 | 0.000977 | 0.00994 | 0.0603 | 0.498 | 0.843 |
| mlp/0.5/adam/sampled | 10/10 | 0.000977 | 0.000516 | 0.0708 | 0.587 | 0.938 |
| mlp/0.5/sgd/expected_full | 8/10 | 0.00488 | 0.0167 | 0.0128 | 0.83 | 0.964 |
| mlp/0.5/sgd/sampled | 10/10 | 0.000977 | 0.000173 | 0.0318 | 0.783 | 0.981 |
| mlp/0.9/adam/expected_full | 10/10 | 0.000977 | 0.000824 | 0.576 | 0.875 | 1 |
| mlp/0.9/adam/sampled | 10/10 | 0.000977 | 0 | 0.681 | 0.999 | 1 |
| mlp/0.9/sgd/expected_full | 10/10 | 0.000977 | 0 | 0.977 | 1 | 1 |
| mlp/0.9/sgd/sampled | 10/10 | 0.000977 | 0 | 0.978 | 1 | 1 |

- P1 still holds in all 16 cells; the largest task-level p is 0.0098 (Bonferroni × 5 = 0.049). This replaces 3.0e-8 as the p-value passed to Holm.
- Per minority prompt, absorption rises monotonically with m in 15 of 16 cells (≈ 0 when balanced). The absorbed *fraction* falls from m = 0.7 to 0.9 only because the minority set shrinks.
