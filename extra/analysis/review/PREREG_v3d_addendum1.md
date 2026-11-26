# PREREG_v3d, addendum 1 (2026-11-22): threshold in class share (T15) and two analyses of existing runs

Committed before batch `v3e` launches. Exploratory; nothing here enters the Holm family.

## What has been seen

- **The v3d results** (`analysis/v3d/SUMMARY.md`, `analysis-v3` `79d2161`).
  - **T11:** at step 0 the ratio R = −Δ_B(A)/Δ_B(B) is the majority's push on the minority over the
    minority's own gain. At K = 3 it is 0.78–0.93 for m = 1/3, 3.6–4.8 for m = 0.7 and 10.7–17.1 for
    m = 0.9.
  - **T12:** some collapsed runs recover within 10× the horizon, at ρ = 0.5 only.
- **The reviews** of Schaul et al. (1904.11455) and Li et al. (2611.13218). The reviewer's reading of
  Li et al.'s Prop. 2 is that suppression of a minority label starts at m = 1/K from a uniform start.
- **Local smoke runs** of the T15 grid (200 steps, one task per K). Only that they run and the class
  counts were looked at.

## T15: where does the mixed step start to lower the minority?

- **Grid:** K ∈ {2, 3, 5}; m = 1/K + {−0.05, 0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.2, 0.35}; linear and
  MLP; ρ ∈ {0.5, 0.9}; SGD at η = 0.01, exact expected update, 10,000 steps; 10 task seeds; branch
  measurement at step 0.
- **Step-0 threshold m\*:** per (K, arch, ρ), the m at which the median over tasks of Δ_B(mixed)
  crosses zero, by linear interpolation in logit m between grid points.
- **Absorption threshold:** the smallest grid m at which the median minority absorption rate
  exceeds 0.5.

**Predictions:**
1. **m\* is strictly above 1/K.** At m = 1/K, Δ_B(mixed) > 0 in ≥ 9 of 10 tasks in every cell. The
   classes are exchangeable there, so all gain. This contradicts "suppression starts exactly at
   m = 1/K" as a statement about decline.
2. **Odds scaling.** R grows in proportion to the odds m/(1 − m): the slope of log R on logit m is 1
   (within 0.8–1.2) in every cell.
3. **Location for K = 3.** From the T11 ratios at m = 1/3 and odds scaling, m\* ≈ 0.39 (linear,
   ρ = 0.5) and ≈ 0.35 (linear at ρ = 0.9, and MLP). Predicted intervals: [0.37, 0.42] and
   [0.34, 0.37].
4. **Order.** m\* − 1/K is smaller at ρ = 0.9 than at ρ = 0.5, for every K and architecture.
5. **Absorption follows the step-0 sign.** No cell has median minority absorption above 0.5 at an m
   below m\*. The absorption threshold is at or above m\*.

## M2 (analysis of the T12 runs): is the minority still sinking at 10× the horizon?

For each T12 run, over the last decade of training (from T/10 to T):

g = Δ[−log p̄_B] / Δ[−log(1 − p̄_A)]

This is the growth of the minority's wrong-side margin relative to the majority's margin, from
class-mean success.

**Reading:**
- g ≥ 1: the minority sinks at least as fast as the majority improves, with no sign of recovery;
- 0 < g < 1: it sinks more slowly;
- g ≤ 0: it is recovering.

No prediction: this was written after T12 was seen, so it is descriptive, per cell.

## M4 (analysis of existing runs): is it imbalance or margin?

From the T13 `expected_full` runs (the T1 configuration): the absorption rate of minority prompts by
tercile of score margin (terciles over all prompts of the task), and of majority prompts in the same
terciles.

**Prediction:** minority prompts are absorbed at similar rates in every margin tercile (within 0.15
of each other, where the class is absorbed), and majority prompts in the lowest tercile are absorbed
at a rate below 0.05. That is, class membership decides, not margin.

## Deviations

Listed in the write-up.
