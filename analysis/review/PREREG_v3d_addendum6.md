# PREREG_v3d, addendum 6 (2026-11-23): the late phase, prompt by prompt (T24)

Committed before batch `v3l` launches. Code: `toy-v3` `5487480` (`snapshot_at`, grid `v3_t24`).
Exploratory, but with a quantitative prediction rule fixed here.

## What has been seen

- T12 (`analysis/v3d`): a run's smallest γ_y = û·φ_y (û: the majority's ℓ2 max-margin direction)
  ranks the late margin growth (Spearman 0.89; within cells, median 0.78). A per-prompt rule
  "γ_y < 1 recovers" failed (0 of 147 such prompts recovered under SGD). A depth comparison on the same
  data was partly circular (a prompt that never recovered is deepest at the end by construction) and
  is not used as evidence. (`analysis/theory/late_time.md`.)
- S1: the ℓ∞ max-margin geometry does not predict Adam's late sinking better than ℓ2 does.
- SymPy (`tests/test_theory_symbolic.py`): the scalar late model dL/dτ = γ/τ − c e^{−L} has an exact
  solution. No T24 run has been made.

## Runs

Linear policy, SGD, exact expected update, K = 3, ρ ∈ {0.5, 0.7, 0.9}, m ∈ {0.7, 0.9},
η ∈ {0.1, 1}, 100,000 steps, 10 tasks; every prompt's success stored at steps 10, 20, 50, 100, 200,
500, 1,000, 2,000, 5,000, 10,000, 20,000, 50,000 and 100,000. 120 runs.

## The prediction rule (fixed now)

Flow time τ = η·step; W0 = W_G(0) for GRPO with G = 8.

1. **Start of the late phase, τ0:** the earliest stored step at which the majority's class mean is at
   least 0.99. Runs that never reach it are excluded and counted.
2. **State at τ0:** for each minority prompt, L_y = log((1 − p_y)/p_y) from the stored success.
3. **The majority as a clock:** M(τ) = log(2 p̄_A / (1 − p̄_A)) from the logged majority class mean,
   interpolated linearly in τ. The majority itself is not modelled.
4. **Minority model:** for every minority prompt y,
   dL_y/dτ = γ_y·dM/dτ − (1/n) Σ_{y' minority} k(y, y')·W_G(p_y')·p_y'·s(y, y')·(1 − p_y'),
   with γ_y = û·φ_y (û scaled so the majority's closest prompts have margin 1), k the linear kernel,
   p = 1/(1 + e^{L}), s = 2 for a classmate (including y itself) and 1 for a prompt of the other
   minority class. Integrated from τ0 to each later stored step.
5. **Baselines** (same start): B1 persistence (L unchanged); B2 push only (no restoring term); B3
   restoring from y itself only (no classmates); B4 one γ for all prompts (the run's mean γ_y).

## Predictions (at the last step; a run counts if it has at least 20 minority prompts below 0.5 at τ0)

1. **The model tracks each prompt:** median over runs of the Spearman correlation between predicted
   and simulated log p_y is at least 0.8.
2. **It beats the baselines:** its mean absolute error in log10 p_y is smaller than B1's, B2's and
   B4's in at least 80% of runs.
3. **Recovery is collective:** in runs with at least 10 recovering prompts (simulated p_y > 0.1 at the
   last step), the model beats B3 in at least 70% of them.
4. **Recovery counts:** the predicted number of minority prompts above 0.1 at the last step is within
   25% of the simulated number in at least 70% of runs with at least 10 recoveries.
5. **Replication of the probe:** within each (ρ, m, η) cell, the smallest γ_y ranks the late margin
   growth with Spearman at least 0.6 (median over cells).

## Reading, fixed in advance

- **1–4 hold:** the late phase is explained by the majority's implicit-bias geometry plus the
  minority's collective, vanishing restoring signal. This becomes the second phase of the theory.
- **1 holds but 3 fails:** the late phase is per-prompt after all; report.
- **1 fails:** the reduced model is wrong; the theory stays at the first-step identity plus the
  run-level association.

## Deviations

Listed in the v3l write-up.
