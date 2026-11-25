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

## Part 2: held-out test on v3o

To be added after the held-out predictions are committed and compared (`fall_*_heldout.*`).
