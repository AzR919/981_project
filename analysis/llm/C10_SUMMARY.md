# C10: who pushes whom in the LLM

All 18 runs finished (queue `r2`, tags `r2_41`–`r2_58`, 0 failed; code commit `b9747d1`).
Pre-registered in `PREREG_C10.md` (`review` `75e0436`); exploratory. `llm/analyze_branch.py`
regenerates `c10.md` and `c10_runs.csv`.

Setup: Qwen2.5-0.5B from the base weights, GRPO, lr 1e-6, 100 steps. Three arms with the same loss
denominator (80 prompts): mixed (76 majority + 4 minority prompts per step), majority-only (76),
minority-only (4). Both role assignments, 3 seeds. Measured: the change in mean success on the 300
held-out problems of the minority procedure (128 samples each) against the base model. Null (base
seed 1 against seed 0): +0.001 and −0.002.

## Result at step 100

| Majority procedure | Majority-only | Minority-only | Mixed | Sum of the two parts | R |
|---|---|---|---|---|---|
| `each` (minority `total`) | **+0.26** (seeds: +0.11, +0.34, +0.34) | +0.33 | +0.37 | +0.60 | −0.79 |
| `total` (minority `each`) | **+0.25** (seeds: +0.20, +0.31, +0.24) | +0.31 | +0.42 | +0.56 | −0.79 |

Every interval excludes zero on the positive side, in all 18 runs and at both steps.

| Prediction | Result |
|---|---|
| 1. Majority-only training lowers the minority (all 3 seeds, both roles) | **Fails in all 6 runs.** It raises the minority, by +0.11 to +0.34 |
| 2. Minority-only training raises it | **Holds** in all 6 runs (+0.30 to +0.35) |
| 3. The mixed change is close to the sum of the two parts | **Fails**: the mixed change is 62% and 75% of the sum |

## Reading

- **This is the "positive transfer" outcome named in the pre-registration.** Training only on the
  majority procedure improves the minority procedure, by about 0.8 of what the minority's own
  training gives. In the toy the same ratio is +4 to +17 the other way at this imbalance: the
  majority's step undoes the minority's.
- **It works in both directions.** Four minority prompts per step also raise the majority panel, by
  +0.15 and +0.23.
- **Why additivity fails.** The two parts teach largely the same thing (reading the problem,
  carrying out the arithmetic), so their effects overlap; and success is bounded: from a base near
  0.52–0.55, a gain of +0.6 is not possible.
- **In the terms of the odds law,** R = c·m/(1 − m) and c carries how the two classes' gradients
  relate. In the toy and in the transformer environment the classes compete for the same output and
  c is positive. Here c has the other sign: the classes' updates are aligned.
- **With C2, this settles the LLM part of the report:** in this setting GRPO does not push a
  minority procedure down. The few percent of held-out problems that are lost in every run
  (`C2_SUMMARY.md`) are not caused by the majority's pressure; the rejection-sampling controls test
  what does cause them.
- **C11 (a fix in the LLM) is not run.** Its gate was a P5 pass or a push-down here; neither
  happened, and there is no collapse to fix.

## Limits

- 100 steps from the base model, one model, one pair of procedures. A pair of tasks whose correct
  behaviours conflict could behave differently; that is the case the toy and T20 cover.
- A short-horizon branch, not a single step: the three arms drift apart over 100 steps.

## Deviations

The composition is 76 + 4 at denominator 80, not the plan's 61 + 3; this is stated in the
pre-registration.
