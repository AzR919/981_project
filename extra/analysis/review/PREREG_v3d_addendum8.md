# PREREG_v3d, addendum 8 (2026-11-24): shared skills (T26)

Committed before batch `v3n` launches. Code: `toy-v3` `c991964` (`differing_positions`, grid `v3_t26`).
Exploratory.

## Why

In the LLM, training only on the majority procedure *raises* the minority procedure (C10), and no
collapse appears (C2). The leading explanation is that the two procedures share most of their steps.
T26 tests that explanation in the toy, where the classes so far share nothing.

## Design

Four-token answers; success requires all four tokens. Positions 0..k−1 carry the answer class (the
"procedure", with the majority share m); positions k..3 follow one balanced rule that is the same for
every prompt (the "shared skill"; its offset is calibrated to equal shares). k ∈ {1, 2, 3, 4} (k = 4: no
shared skill). Linear and MLP; ρ ∈ {0.5, 0.9}; m ∈ {0.7, 0.9}; exact expected update; SGD, η = 0.1;
10,000 steps; 10 tasks; branches (majority-only, minority-only, mixed) at 0, 10% and 50% of training.
320 runs. One local smoke run of 50 steps (linear, k = 1) checked that the code runs; its outcome was not
inspected beyond the step-0 branch values.

## What theory says (fixed now)

At a uniform start, two prompts' success directions overlap by k(x, y)·p_x·p_y·Σ_l (δ_l − 1/K), where δ_l is
1 if their correct tokens at position l agree. Procedure positions contribute −1/K for every
majority–minority pair; a shared position contributes 1 − 1/K for pairs that agree there and −1/K for those
that do not. The push ratio R = −Δ_B(A)/Δ_B(B) is computed per task from these sums (no fitted constant).

## Predictions

1. **The identity holds with several positions:** the measured step-0 R is within 5% of the computed
   value, in ≥ 95% of tasks.
2. **Shared skills can reverse the first step:** in every task where the computed majority–minority sum
   is positive, the majority-only step raises the minority at step 0.
3. **Shared skills prevent the collapse (the hypothesis under test):** the share of minority prompts
   absorbed (below 1e-3 over the last 20% of steps) decreases as k decreases (more sharing), in every
   (architecture, ρ, m) cell; and at k = 1 (three shared positions) at most 10% of minority prompts are
   absorbed in ≥ 9 of 10 tasks per cell.
4. **Reported, not predicted:** the sign of the majority-only effect at 10% and 50% of training, i.e.
   whether the help persists once the shared skill is learned.

## Reading

- **3 holds:** sharing protects the minority in the toy, which supports the shared-skills
  explanation of the LLM result.
- **2 holds, 3 fails:** sharing helps only while the shared skill is being learned; once it is learned,
  the procedure positions compete as before. Then shared skills alone do not explain the LLM result.

## Deviations

Listed in the v3n write-up.
