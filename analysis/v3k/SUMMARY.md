# v3k: does the collapse need a class-agnostic start? (T23, class-balanced warm start)

All 320 runs returned and none failed. Launched from a clean commit (`toy-v3` `a717aab`; job digest
`04c06c4b14a9`). Pre-registered in `PREREG_v3d_addendum5.md` (`review` `2a24e93`); exploratory.
`experiments/analyze_v3k.py` regenerates `FINDINGS.md`, `runs.csv` and `cells.csv`. (In
`FINDINGS.md`, a value of −1 in the first-step table marks the sampled arm, where the first step was
not measured.)

Setup: the T1 minority task (K = 3, ρ = 0.9), m ∈ {0.7, 0.9}, SGD η = 0.1, 10,000 steps, exact and
sampled updates, 10 tasks. Starts: uniform, or a supervised warm start with equal weight per class
until the mean of the class means reaches p0 ∈ {0.5, 0.7, 0.9}.

## Start values (medians)

| Start | Majority success | Minority success | Minority's mass on the majority's answer |
|---|---|---|---|
| uniform | 0.33 | 0.33 | 0.33 |
| p0 = 0.5 | 0.36–0.45 | 0.53–0.57 | 0.16–0.25 |
| p0 = 0.7 | 0.66–0.74 | 0.70–0.72 | 0.15–0.23 |
| p0 = 0.9 | 0.87–0.90 | 0.90 | 0.05–0.06 |

For comparison, the C2 base model answers minority problems correctly about 0.52–0.55 of the time
and gives the other procedure's answer on 0.006–0.054 of samples.

## Result

Share of minority prompts absorbed (mean over 10 tasks; exact and sampled updates agree within 0.08):

| | uniform | p0 = 0.5 | p0 = 0.7 | p0 = 0.9 |
|---|---|---|---|---|
| m = 0.7, linear | 1.00 | 0.07–0.14 | 0 | 0 |
| m = 0.7, MLP | 1.00 | 0.00 | 0.00 | 0.05–0.06 |
| m = 0.9, linear | 1.00 | **0.99** | 0.13 | 0 |
| m = 0.9, MLP | 1.00 | **0.69–0.70** | 0.04–0.11 | 0.20–0.21 |

Minority success at the end (median): with a warm start of 0.5 at m = 0.7, 0.78 (linear) and 1.00
(MLP); at m = 0.9 and p0 = 0.5, below 1e-4 in both. With p0 ≥ 0.7 it ends at 0.59–0.86 (linear) and
0.98–1.00 (MLP).

| Prediction | Result |
|---|---|
| 1. The first step still lowers the minority at m = 0.9 for every warm start | **6 of 6 cells** (10 of 10 tasks each) |
| 2. Absorption does not increase with p0 | **4 of 8 cells.** Holds for the linear policy; fails for the MLP, where p0 = 0.9 absorbs more (0.20) than p0 = 0.7 (0.04–0.11) |
| 3. Nothing absorbed at p0 = 0.9 | **4 of 8 cells.** Holds for the linear policy; fails for the MLP (5–21% of minority prompts absorbed) |
| 4. p0 = 0.5 against the uniform start (reported) | At m = 0.7: **gone or much reduced** in all 4 cells. At m = 0.9: **as at a uniform start** in the linear cells (0.99); between the two for the MLP (0.69–0.70) |

## Reading

- **A head start protects the minority, and the larger the majority's share, the larger the head
  start it needs.** At m = 0.7 a modest one (both classes near 0.5) prevents the collapse. At
  m = 0.9 that is not enough: the minority is still absorbed, completely for the linear policy. It
  takes a start near 0.7 to keep most minority prompts.
- **The first step does not tell you which.** It lowers the minority at every start (prediction 1),
  including starts that end with the minority at 0.98. This is the second batch, after T22, where
  the first-step sign and the outcome disagree.
- **For the LLM result.** The C2 base model starts near the toy's p0 = 0.5 on success. At the C2
  share of 0.95 the toy at that start still collapses (at 0.9 it does). So the base model's head
  start does not by itself explain why the LLM shows no collapse; the remaining candidate is the one
  C10 measured directly: majority-only training raises the minority procedure. One difference is
  not matched: the toy's minority puts 0.16–0.25 of its mass on the majority's answer at p0 = 0.5,
  the LLM's 0.006–0.054. T22 found that mass placed on answers outside both classes does not
  protect, but a start that is both accurate and almost free of the majority's answer was not
  tested.
- **The MLP at p0 = 0.9 absorbs a fifth of the minority prompts** where p0 = 0.7 absorbs less. Not
  explained here; it is the cell where the minority's starting mass on the majority's answer is
  lowest, so it runs against the simple reading above and is reported as is.

## Deviations

None. The first-step table in `FINDINGS.md` shows −1 for the sampled arm, which has no first-step
measurement by design.
