# Theory of the fall: results, part 1 (in-sample)

**Read this as in-sample.** The model (`analysis/theory/fall.md`, code `experiments/fall_ode.py` at `afef9a4`) was
written by an agent who had read the summaries of v3e, v3g, v3j, v3k and v3l. The comparison below is therefore a fit
check, not evidence, whatever the result (`PREREG_fall_heldout.md`, `review` `5275c11`). The test that counts is
the held-out one on v3o (part 2, below, once run).

Order of events:
- Predictions for all 6,980 runs were computed on 11-24 from each run's configuration and starting success alone,
  and committed on 11-25 (`fb7c0d2`), before any comparison.
- The model has no fitted parameter.
- Tables: `fall_compare.md`, `fall_runs.csv`, `fall_thresholds.csv` (`experiments/check_fall.py compare`).

## The registered tests (primary model: class-symmetric; linear policy, exact update)

| Test | Pass if | v3e | v3g | v3j | v3k | v3l |
|---|---|---|---|---|---|---|
| P1 depth ranks across runs | Spearman ≥ 0.8 | 0.993 | 0.982 | 0.997 | 0.988 | **0.799 (fails narrowly)** |
| P2 depth level | median error within ±0.5 decades | 0.00 | 0.00 | 0.09 | 0.20 | 0.26 |
| P3 ranks within a cell | median ≥ 0.5 | 0.96 | 0.95 | 0.96 | 0.82 | 0.94 |
| P4 absorbed or not | accuracy ≥ 0.85 | 0.993 | 0.984 | 1.000 | 0.988 | 0.858 |
| P6 per-prompt order (test particles) | median ≥ 0.6 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |

**P5 (v3e threshold in m):** the predicted threshold equals the measured one in all 6 linear cells, including the
two cells where no grid share collapses.

- Every test passes in every batch except P1 in v3l, which misses by 0.001.
- v3l is the long-horizon batch (flow time 10⁴–10⁵), where the late phase dominates. That is where the class-level
  closure was expected to be weakest (`fall.md`, failure mode (a)): it predicts absorption too often (15 false
  positives against 2 false negatives).
- **Sampled updates** (reported separately, not registered): the same picture. Spearman 0.976–0.995, accuracy
  0.975–0.995.
- **MLP** (out of scope, nothing registered): see `fall_compare.md`.

## Reading

In these batches the minority's fall is close to an exact consequence of the class-level equations: a handful of
numbers per task (block kernel means, shares, the GRPO weight, the start), with nothing fitted. The depth is
predicted to within hundredths of a decade in the short-horizon batches.

This is not yet evidence, because the model's design came after its author had seen these batches. The held-out
test decides.

## Deviations

- **A bug in the comparison code, not the model.** In the first `compare` run, `runs.update` (pandas'
  `DataFrame.update` method) was compared instead of the `update` column, so every per-scope table was empty and only
  the v3e threshold table appeared. Fixed to `runs["update"]` and rerun on 11-25. The predictions were not touched.
- `check_fall.py` gained `--batches`, `--tag` and `--linear-sgd` for the held-out run. `fall_ode.py` is unchanged
  since `afef9a4`.

## Part 2: held-out test on v3o (the test that counts)

Order of events:
- Predictions for the 1,360 in-scope v3o runs were made from configurations and starting success only. Scope:
  linear policy, exact update, SGD, β = 0.
- They were committed (`9421c44`) before any v3o outcome was read, with the model unchanged since `afef9a4`.
- Tables: `fall_compare_heldout.md`, `fall_runs_heldout.csv`.

| Test (pass if) | v3_a1, P1 grid (1,280 runs) | v3_a2, β = 0 arm (40) | v3_a5, warm start (40) |
|---|---|---|---|
| P1 depth rank (≥ 0.8) | **0.755: fails** | 0.992 | 0.972 |
| P2 level (within ±0.5 decades) | 0.10 | 0.18 | 0.15 |
| P3 within cell (≥ 0.5) | 0.94 | 0.97 | 0.89 |
| P4 absorbed or not (≥ 0.85) | 0.879 | 1.000 | 1.000 |
| P6 per prompt (≥ 0.6) | 0.99 (879 runs) | 1.00 | 1.00 |
| P5 analogue: threshold in m, within one grid step (≥ 5 of 6 cells; here 16 cells) | 15 of 16 | — | — |

**Verdict as registered:** the model passes every rule on two of the three grids. On the P1 grid it passes P2–P6,
and the P5 analogue, but **fails P1**.

**Why P1 fails: step size.** This breakdown is reported in addition; it does not change the registered verdict. The
P1 grid is the only one with large SGD steps (η up to 30):

| η | ≤ 1 (800 runs) | 3 | 10 | 30 |
|---|---|---|---|---|
| Depth Spearman | 0.988 | 0.908 | 0.905 | 0.866 |
| Median error (decades) | +0.03 to +0.07 | +1.07 | +8.25 | +31.8 |
| Absorbed accuracy | 0.93–0.96 | 0.93 | 0.73 | 0.61 |

The model is the small-step (gradient-flow) limit, with no step size in it. With η ≥ 3, discrete SGD overshoots and
falls far deeper than the flow, and the model predicts falls that are too shallow. The registered scope did not
exclude large steps. That was an oversight: the model's own derivation assumes the flow limit.

## Reading

- **Within its own regime, the small-step flow, the class-level model predicts the minority's fall on unseen tasks.**
  Depth rank is 0.99, the level is right to within a tenth of a decade, which runs collapse is right 95% of the
  time, prompt order within a run is 0.99, and the collapse threshold in the majority share is right in 15 of 16
  cells. It has no fitted parameter.
- **Outside that regime (η ≥ 3) it fails, in the predicted direction:** large discrete steps add a fall the flow
  does not have.
- **For the report:** "the fall theory is supported out of sample in the flow regime (exploratory; the registered
  rank test failed on the full grid because of large-step runs)." Not "the fall is fully explained".

## Deviations (part 2)

- **Scope narrowed:** the held-out scope (`PREREG_fall_heldout.md`) named the linear exact-update runs of `v3_a1`,
  `v3_a2` (β = 0) and `v3_a5`. We also restricted to SGD (excluding Adam, about which the model says nothing) and to
  β = 0 in `v3_a1`. This was decided before reading outcomes, by `--linear-sgd`.
- **P5 adapted:** P5 was defined on v3e. Its held-out analogue is the same threshold rule per (ρ, η) cell of
  `v3_a1`, with 16 cells instead of 6.
- **The step-size breakdown** is post hoc and reported only as an explanation.
