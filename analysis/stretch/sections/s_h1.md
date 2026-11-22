## H1 re-run: loose, medium and tight tests (414 runs, R = 20,000)

Same 414 configurations as the base H1, with 5× the samples, split into 20 chunks. Two new statistics: the projected relative bias b∥ = (m − pred)·pred/‖pred‖² with an exact 1-D standard error, and for every run a null distribution of mean z² from 50 half-against-half splits of its own chunks (same coordinate correlation, no model involved).

### Loose: per-rule medians (the base criterion)

| rule | runs | median_ratio | median_z2 | frac_z_gt3 |
|---|---|---|---|---|
| grpo | 72 | 0.963 | 0.922 | 0.00199 |
| maxrl | 72 | 0.974 | 0.931 | 0.00225 |
| mean | 72 | 0.964 | 0.937 | 0.00216 |
| passk2 | 72 | 0.952 | 0.935 | 0.00213 |
| passk4 | 54 | 1.02 | 1.03 | 0.00345 |
| rloo | 72 | 0.964 | 0.937 | 0.00216 |

The base per-run band (ratio ≤ 1.2, z² in [0.8, 1.25]) is missed by 135 of 414 runs at R = 20,000 (base run, R = 4,000: 174).

**Verdict: PASS** (per-rule median ratio ≤ 1.2 and median z² in [0.8, 1.25]; |z| > 3 share < 1%)

### Medium: each run against its own calibrated null

Runs whose mean z² falls outside the central 95% of their own null: **14 of 414 (0.034)**. Expected under no bias: about 0.05 (a little more, since the 2.5/97.5% points come from 50 draws).

| split | chi2_p | miss_rate_by_group |
|---|---|---|
| rule | 0.941 | grpo: 0.03, maxrl: 0.03, mean: 0.03, passk2: 0.04, passk4: 0.06, rloo: 0.03 |
| level | 0.0416 | lin: 0.01, m64: 0.02, tab: 0.07 |
| G | 0.925 | 2: 0.02, 4: 0.04, 8: 0.04, 32: 0.04 |
| L | 0.00363 | 1: 0.07, 2: 0.00, 4: 0.03 |
| init | 0.863 | sft_up: 0.03, tabular: 0.03, uniform: 0.04 |

![calibrated null ranks and projected-bias z-scores](figures/s_h1_null_rank_and_bias.png)

**Verdict: FAIL** (miss rate ≤ 0.08 (got 0.034) and no split with χ² p ≤ 0.01 (min p 0.004))

*Post hoc.* Bias can only make the error larger than its null, i.e. a *high* miss. High misses: **7 of 414 (0.017)**, below the 0.025 expected by chance. The split that fails is L: the L = 1 tabular runs have 7 *low* misses (error smaller than the null) and L = 2 has none at all, so the split-half null is somewhat too wide or too narrow depending on the configuration. That is a calibration limit of the null, not a sign of bias.

| L | level | high | inside | low |
|---|---|---|---|---|
| 1 | lin | 2 | 44 | 0 |
| 1 | m64 | 0 | 46 | 0 |
| 1 | tab | 1 | 38 | 7 |
| 2 | lin | 0 | 46 | 0 |
| 2 | m64 | 0 | 46 | 0 |
| 2 | tab | 0 | 46 | 0 |
| 4 | lin | 0 | 46 | 0 |
| 4 | m64 | 3 | 43 | 0 |
| 4 | tab | 1 | 45 | 0 |

### Tight: bound on the bias along the predicted direction

Runs whose 95% CI for b∥ leaves ±2%: **36**. Runs with |b∥/SE| > 3.9 (Bonferroni over 414 at 5%): **0**. Median CI half-width: 0.0025.

| rule | pooled_bias | pooled_se | max_abs_z |
|---|---|---|---|
| grpo | 0.0000 | 0.0000 | 2.0515 |
| maxrl | 0.0001 | 0.0001 | 2.0178 |
| mean | 0.0000 | 0.0001 | 2.0642 |
| passk2 | 0.0001 | 0.0000 | 2.0354 |
| passk4 | 0.0001 | 0.0001 | 2.1919 |
| rloo | 0.0000 | 0.0001 | 2.0642 |

| name | rule | G | L | level | init | bias_par | bias_par_se |
|---|---|---|---|---|---|---|---|
| s_h1/00276 | mean | 2 | 4 | tab | uniform | 0.00423 | 0.00834 |
| s_h1/00277 | rloo | 2 | 4 | tab | uniform | 0.00423 | 0.00834 |
| s_h1/00278 | grpo | 2 | 4 | tab | uniform | 0.00423 | 0.00834 |
| s_h1/00279 | maxrl | 2 | 4 | tab | uniform | 0.00423 | 0.00834 |
| s_h1/00280 | passk2 | 2 | 4 | tab | uniform | 0.00632 | 0.00796 |
| s_h1/00281 | mean | 4 | 4 | tab | uniform | -0.00981 | 0.00556 |
| s_h1/00282 | rloo | 4 | 4 | tab | uniform | -0.00981 | 0.00556 |
| s_h1/00283 | grpo | 4 | 4 | tab | uniform | -0.00983 | 0.00556 |
| s_h1/00284 | maxrl | 4 | 4 | tab | uniform | -0.00983 | 0.00556 |
| s_h1/00285 | passk2 | 4 | 4 | tab | uniform | -0.00963 | 0.0055 |
| s_h1/00286 | passk4 | 4 | 4 | tab | uniform | -0.00929 | 0.0055 |
| s_h1/00322 | mean | 2 | 4 | lin | uniform | 0.00661 | 0.00919 |
| s_h1/00323 | rloo | 2 | 4 | lin | uniform | 0.00661 | 0.00919 |
| s_h1/00324 | grpo | 2 | 4 | lin | uniform | 0.00661 | 0.00919 |
| s_h1/00325 | maxrl | 2 | 4 | lin | uniform | 0.00661 | 0.00919 |
| s_h1/00327 | mean | 4 | 4 | lin | uniform | 0.00988 | 0.00616 |
| s_h1/00328 | rloo | 4 | 4 | lin | uniform | 0.00988 | 0.00616 |
| s_h1/00329 | grpo | 4 | 4 | lin | uniform | 0.00978 | 0.00616 |
| s_h1/00330 | maxrl | 4 | 4 | lin | uniform | 0.00952 | 0.00616 |
| s_h1/00331 | passk2 | 4 | 4 | lin | uniform | 0.00883 | 0.00605 |
| s_h1/00368 | mean | 2 | 4 | m64 | uniform | -0.00935 | 0.0155 |
| s_h1/00369 | rloo | 2 | 4 | m64 | uniform | -0.00935 | 0.0155 |
| s_h1/00370 | grpo | 2 | 4 | m64 | uniform | -0.00935 | 0.0155 |
| s_h1/00371 | maxrl | 2 | 4 | m64 | uniform | -0.00935 | 0.0155 |
| s_h1/00372 | passk2 | 2 | 4 | m64 | uniform | -0.0138 | 0.013 |
| s_h1/00373 | mean | 4 | 4 | m64 | uniform | 0.00251 | 0.00984 |
| s_h1/00374 | rloo | 4 | 4 | m64 | uniform | 0.00251 | 0.00984 |
| s_h1/00375 | grpo | 4 | 4 | m64 | uniform | 0.00248 | 0.00982 |
| s_h1/00376 | maxrl | 4 | 4 | m64 | uniform | 0.00236 | 0.00983 |
| s_h1/00377 | passk2 | 4 | 4 | m64 | uniform | 0.00242 | 0.00945 |
| s_h1/00379 | mean | 8 | 4 | m64 | uniform | 0.00857 | 0.00675 |
| s_h1/00380 | rloo | 8 | 4 | m64 | uniform | 0.00857 | 0.00675 |
| s_h1/00381 | grpo | 8 | 4 | m64 | uniform | 0.00871 | 0.00669 |
| s_h1/00382 | maxrl | 8 | 4 | m64 | uniform | 0.00887 | 0.00668 |
| s_h1/00383 | passk2 | 8 | 4 | m64 | uniform | 0.00843 | 0.00663 |
| s_h1/00384 | passk4 | 8 | 4 | m64 | uniform | 0.00784 | 0.00655 |

**Verdict: FAIL** (every 95% CI inside ±2%, no |b∥/SE| > 3.9, pooled bias per rule within ±0.5%)

*Post hoc.* The ±2% bound fails for lack of precision, not because of bias: every run outside it is at L = 4 with uniform starts (p ≈ 0.012, so few samples carry signal), L 4 uniform: 36; their |b∥| is at most 0.014 and their |z| at most 1.77. Across all 414 runs the largest |b∥/SE| is 2.19 and the pooled bias per rule is within ±0.0001. The rules share samples within a configuration, so these runs are not independent.

