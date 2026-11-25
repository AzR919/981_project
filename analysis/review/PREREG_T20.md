# Pre-registration: T20, a correct minority class in Li, Pandey and Chen's environment (2026-11-23)

Committed before batch `v3h` launches. Code: branch `enum-env` ("T20" commit),
`experiments/enum_env.py`. Exploratory.

## Why

Our toy uses feature-vector prompts and a linear or MLP policy. The closest prior work (Li, Pandey
and Chen, "Wrong the Same Way", NeurIPS 2026 workshop) uses text-like prompts, a 2-layer
transformer and a response space small enough to enumerate, and studies *wrong* strings that a
leaky verifier accepts. T20 asks our question in their environment: when one answer class is a
majority, does training push a *correct* minority class down?

## What has been seen

- Our toy results (T1–T19) and the odds law (`analysis/theory/odds_law.md`).
- Their paper and code. Two cells of their headline sweep were re-run and matched their published
  values within 0.016 (on torch 2.2.2; they pin 2.15, which the lab machines have).
- Three local smoke runs of 100 steps (one per policy kind) checked that the code runs and how long
  it takes. Their outcomes were not inspected.

## Design

- **Reused unchanged** (pinned commit `05ef005`): the 64 prompts (a, b); the 100 two-digit
  responses; the shared transformer and the tabular policy; their SFT routine; their exact
  policy-gradient loss with the standardised advantage and their sampled GRPO loss (G = 8).
- **Changed:** the verifier. The prompts are split into K classes, class 0 holding n_major prompts
  and the others equal shares. Each class has one random 2-digit string, the only accepted response
  for its prompts.
  - `random` partition: class membership is unrelated to the prompt.
  - `parity` partition (K = 2): class 0 is "a·b is even" (48 of 64 prompts).
- **Base model:** SFT to the same target for every prompt: 0.9 shared equally by the K class
  strings and 0.1 spread over all 100. Every prompt starts near success 0.9/K.
- **Runs** (2,000 steps; 10 seeds; SGD at lr 0.3, 1, 3 and Adam at 1e-3, 3e-3, their grids):
  - K = 3, n_major ∈ {22, 32, 40, 48, 56, 60}; K = 2, n_major ∈ {32, 40, 48, 56, 60}; parity.
  - shared transformer with the exact update and with sampled GRPO; tabular with the exact update;
  - per-prompt copies of the transformer (Adam 1e-3, 3 seeds, three tasks). 1,809 runs.
- **Measures:** class means over time; per prompt, the minimum over time and the maximum over the
  last 20% of steps. Absorbed: that maximum is below 1e-3. At step 0 (shared policy): the exact
  change of the minority's mean success under a majority-only, a minority-only and a mixed gradient
  step of size 0.001, and R = −(majority-only change)/(minority-only change).

## Predictions

A cell is one task × optimizer × lr; "holds" means in ≥ 8 of 10 seeds unless stated.

1. **Collapse at high share, none at balance** (shared, exact update, SGD):
   - at n_major ≥ 56 the mean share of absorbed minority prompts is ≥ 0.5 at lr 1 and lr 3;
   - at the balanced tasks (K = 3, n_major = 22; K = 2, n_major = 32) no minority prompt is absorbed
     in ≥ 9 of 10 seeds at every lr.
2. **No sharing, no decline:** in every tabular run and every copies run, no prompt's success ever
   falls more than 1e-6 below its start.
3. **Odds law:** for K = 2 and K = 3 (random partition), the slope of log R on logit m, from the
   seed medians at the shares where R > 0, is within 0.8–1.2.
4. **Threshold near 1/K:** R = 1 is crossed within 0.1 of 1/K (random partition, both K). The
   reason: class membership is unrelated to the prompt, so same-class and cross-class prompts are
   about equally similar and c ≈ K − 1.
5. **Not predicted, reported:** the Adam cells; sampled GRPO against the exact update; the parity
   task (where the class is a function of the prompt, so c < 1 and a higher threshold are expected);
   whether absorbed prompts recover within 2,000 steps at the smallest lr.

## Readings, fixed in advance

- **1–4 hold:** the collapse of a correct minority, its absence without sharing and the odds law
  carry over to a transformer on token prompts. The toy result is not an artefact of feature-vector
  prompts.
- **1 fails with 2–4 holding:** the first-step push is there but the transformer recovers or never
  absorbs within 2,000 steps; the collapse is then a property of low-capacity shared policies, and
  the report says so.
- **2 fails:** the measurement or the control is wrong; investigate before any claim.

## Deviations

Listed in the T20 write-up.
