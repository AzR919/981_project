# PREREG_v3d, addendum 5 (2026-11-23): does the collapse need a class-agnostic start? (T23, warm start)

Committed before batch `v3k` launches. Code: `toy-v3`, "Toy v3k" commit (`a717aab`). Exploratory.

## Why

Review finding 5 (`ADVERSARIAL_v5.md`): every toy run starts from a policy that does not tell the
answer classes apart, while the base LLM in C2 already answers minority problems correctly about
half the time and almost never gives the other procedure's answer. T22 tested one version of this
("the wrong mass is on junk answers") and it failed: junk delays the collapse and does not prevent
it (`analysis/v3j/SUMMARY.md`). T23 tests the other version directly: start the toy from a policy
that already answers every class.

## What has been seen

All earlier results including T22. Unit tests of the new warm start. One local check of the start
values only (task seed 0, m = 0.9): both classes start within 0.05 of p0 for the linear policy at
all three p0 and for the MLP at 0.5 and 0.9 (the MLP at 0.7 starts at 0.84 and 0.65); the
minority's mass on the majority's answer is 0.23, 0.15 and 0.05 (linear) against 1/3 at a uniform
start. No training outcome of a T23 run has been seen.

## Runs

The T1 minority task (K = 3, ρ = 0.9), m ∈ {0.7, 0.9}, linear and MLP, SGD at η = 0.1, 10,000
steps, exact and sampled updates, 10 task seeds. Starts: uniform (the control), and a
class-balanced supervised warm start to p0 ∈ {0.5, 0.7, 0.9} (equal total weight per answer class;
stop when the mean of the class means reaches p0). 320 runs.

Measures: the start values (each class's success; the minority's mass on the majority's answer);
the exact one-step changes at step 0 (exact arm); the share of minority prompts persistently
absorbed at 1e-3 and 1e-2; the minority's lowest and final class mean.

## Predictions

A cell is one architecture × update × m.

1. **The first step still pushes the minority down.** At m = 0.9, the mixed step lowers the
   minority's mean success at step 0 for every p0, in ≥ 9 of 10 tasks (exact arm). Reason: when a
   prompt's wrong mass is split evenly over the other classes, the cross-class factor and the
   same-class factor shrink together (−3q² and 6q² for K = 3, with q the mass on each wrong class),
   so their ratio does not change with p0.
2. **Absorption does not increase with p0:** per cell, the share of minority prompts absorbed is
   non-increasing along uniform, 0.5, 0.7, 0.9 (slack 0.05).
3. **At p0 = 0.9 nothing is absorbed** (no minority prompt, in ≥ 9 of 10 tasks, every cell). Reason:
   the majority has little left to learn, so the total push it can exert is small.
4. **p0 = 0.5 is reported, not predicted.** It is the start closest to the LLM's (about 0.5 on the
   correct answer for both classes), and I do not know which way it goes.

## Reading, fixed in advance

- **Absorption at p0 = 0.5 is as at a uniform start** (within 0.1): the class-agnostic start is not
  what makes the toy collapse. The LLM result then cannot be put down to its start, and aligned
  updates between the procedures remain the candidate.
- **Absorption at p0 = 0.5 is gone or much reduced** (below half of the uniform value): the toy's
  collapse depends on starting before the classes are told apart. The LLM result is then what the
  toy predicts for a model that starts where the LLM does, and the report's claim narrows to
  "collapse when learning the classes from scratch under imbalance".
- **In between:** reported as partial dependence on the start.
- After T22, a first-step prediction (1) is not taken as a prediction of the outcome.

## Deviations

Listed in the v3k write-up.
