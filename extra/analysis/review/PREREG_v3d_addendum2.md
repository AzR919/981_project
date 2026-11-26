# PREREG_v3d, addendum 2 (2026-11-22): pass@k rules at matched progress (T16)

Committed before batch `v3f` launches. Exploratory.

## What has been seen

The v3c results (`analysis/v3c/SUMMARY.md`), on the minority task (linear, ρ = 0.9, m = 0.9) at
5,000 steps:
- grpo absorbs the whole minority set and ends at overall success 0.897;
- pass@2 rules absorb 2–5% and reach 0.92–0.94 at η = 1;
- for k = 4 and 8 no minority prompt is absorbed, but overall success ends at 0.49–0.87. That is
  below grpo's, so protection cannot be told apart from slow learning.

## Design

The same task and rules (grpo, mean, passk and rloo_k at k ∈ {2, 4, 8}), G ∈ {8, 32} at B·G = 256,
η ∈ {0.03, 0.3, 1}, 10 task seeds, `sampled`, **20,000 steps** (4× v3c), with class means logged
every 100 steps.

## Measures

- **Matched point:** the first logged step at which the majority class mean reaches 0.80.
- **At the matched point:** the minority class mean.
- **At the end:** the minority class mean, the majority class mean, and the share of minority prompts
  absorbed.

## Predictions

1. **Protection at matched progress.** For passk and rloo_k at every k, in runs that reach the
   matched point, the minority class mean there is above 0.2 in ≥ 9 of 10 tasks per (rule, G, η) cell.
   Under grpo it is below 0.01 in ≥ 9 of 10.
2. **Not just slow.** For k = 4 and k = 8 at η = 1, the majority class mean reaches 0.80 within
   20,000 steps in ≥ 9 of 10 tasks, and no minority prompt is absorbed at the end.
3. **Cost.** The matched point comes later for larger k: the median step to reach it increases with
   k for both rules.

## Deviations

Listed in the write-up.
