# Pre-registration: v3 batch 2 (`v3b`): P3 (T5), and the exploratory T7, T9 and batch-1 follow-ups

This file is written and committed before batch `v3b` launches. The code is on branch `toy-v3`
(commits `ef29059` and `e61626f`), and the analysis goes on `analysis-v3`. The T0 definitions in
`PREREG_v3a.md` apply unchanged:
- eligible prompts are those with p_init ≥ 10τ;
- persistently absorbed means max p over the last 20% of steps is < τ;
- τ = 1e-3;
- task seeds are the unit of replication;
- bootstrap CIs use 10,000 draws over task seeds.

**Disclosed before writing:**
- **T5 warm-starts.** They were tuned before launch so that p_init has sd ≥ 0.25 on task seeds 0–2.
  The sd values were: lin 0.32–0.36, c05 0.29–0.38, c09 0.25–0.42, m64 0.36.
  - The tuning looked only at p_init, never at outcomes.
- **T9 analytic predictions.** The values below were computed from `theory.advantage_gap` before any
  T9 run.
- **Local smoke runs.** One run of each job kind, of 300 steps or fewer, checked only that the code
  runs. Their outcomes were not inspected.

## P3 (T5, primary): initial drift predicts absorption on balanced tasks

- **Runs.**
  - Balanced classes (m = 1/3), K = 3, L = 1, GRPO, G = 8, B = 32, `sampled`, 5,000 steps;
  - levels lin, c05, c09, m64 × 5 η values × 10 task seeds;
  - warm-start `sft_mixed`.
- **Label:** persistently absorbed at τ = 1e-3, among eligible prompts.
- **Runs used:** those with ≥ 20 absorbed and ≥ 20 non-absorbed eligible prompts. If fewer than 10
  runs qualify, P3 is reported as untestable, not failed.
- **Predictors, at step 0:**
  - drift: the predicted Δp from the full JVP, `checkpoints["0"]["drift"]`;
  - rarity: the share of the prompt's 10 nearest training prompts (Euclidean, in feature space) with
    a different answer class;
  - logit(p_init).
- **Models,** logistic, with a run fixed effect:
  - base = rarity + logit(p_init);
  - full = base + drift.
- **Fit.** Predictions for each task seed come from models fitted on the other 9 (leave-one-task-out).
- **Statistic:** ΔAUC = AUC(full) − AUC(base), pooled over the qualifying runs.
- **Test.** The 95% CI resamples task seeds 10,000 times.
  - **P3 holds** if ΔAUC ≥ 0.03 and the CI's lower bound is > 0.
  - The one-sided p-value passed to Holm (P1–P5) is the share of bootstrap draws with ΔAUC ≤ 0,
    floored at 1e-4.
- **Secondary:**
  - the same test per level;
  - AUC of each single predictor;
  - windowed prediction: drift at checkpoint t predicting absorption entry in (t, t + 500], with
    windows of ≥ 20 events only.

## T7 (exploratory): calibrated H1 medium tier

- **Statistic.** For every gradcheck run, the rank of the observed mean z² within its 200 bootstrap
  draws.
- **Null.** The identity E[ĝ] = W_G ∇p is exact, so the ranks should be Uniform(0, 1) in every level.
- **Calibration check, tabular.** KS test of the tab ranks against uniform.
  - If p ≥ 0.05, the bootstrap is calibrated, and the medium tier is applied to lin and m64: KS per
    level, and the share of runs with rank > 0.95, expected 5%.
  - If tab fails (p < 0.05), the medium tier is dropped, as v3 planned.

## T9 (exploratory): all-wrong-group fixes

**Rules:**
- `heckel`: Linear-R, (r − p̂)/p̂, so −1 on every wrong sample;
- `virtual`: AVSPO with one virtual sample, ±1/√G on uniform groups;
- `pm_alpha`: ±0.1 on uniform groups;
- `nsr`: W-REINFORCE with λ = 0.1;
- controls: grpo and mean.

**Analytic predictions** (G = 8), from `advantage_gap`:

| Rule | W₈(0.01) | W₈(0.9) | Ratio of low-p to high-p restoring weight |
|---|---|---|---|
| grpo | 2.61 | 2.33 | 1.12 |
| mean | 0.875 | 0.875 | 1.00 |
| heckel | 7.73 | 1.11 | **6.96** |
| virtual | 2.94 | 2.50 | 1.18 |
| pm_alpha | 2.70 | 2.38 | 1.14 |
| nsr | 1.10 | 1.10 | 1.00 |

1. **t1-minority** (linear, ρ = 0.9, m = 0.9, SGD, `expected_full` and `sampled`, η ∈ {0.1, 1}).
   - Minority collapse is driven by the majority's mean update, so the rule's ratio of low-p to high-p
     weight sets how much protection a minority prompt gets.
   - **Prediction:**
     - heckel has lower persistent absorption than grpo (paired one-sided Wilcoxon over tasks);
     - virtual and pm_alpha do not differ from grpo by more than 0.02;
     - nsr and mean are not lower than grpo.
2. **t2-kick** (lin, L = 2, B = 8, η = 300). This is noise-driven, and there is no analytic prediction.
   heckel's large advantages (up to 1/p̂) may raise the kick size. Descriptive only.
3. **t6-flip20** (tabular, 20% verifier flips, η ∈ {30, 300}). Descriptive: material drops and
   absorption per rule.

## Follow-ups to batch 1 (exploratory)

- **(a) T1 MLP, init seed 1.** The P1 test as pre-registered, applied to the 8 MLP cells. Prediction:
  it passes in each cell, as with seed 0.
- **(b) T2 at B = 8, with G ∈ {4, 16, 64}** (G = 8 is in batch 1).
  - Prediction: the response-noise excess, `sampled` − `expected`, shrinks as G grows. Spearman(G,
    excess) < 0, pooled over levels.
- **(c) T6 KL with `drop_above`.**
  - Prediction: kl01 and kl1 have no material drop below p_init at η ≤ 3. Their declines from a
    running maximum, `drop_above`, may be > 0 (relaxation toward the reference).
  - Control has `drop_above` = 0 at every η.
  - Drops below p_init happen only at η ≥ 30, as in batch 1.
- **(d) T4 edge cells, η ∈ {3000, 10000}, stopping at mean p = 0.5.**
  - Report whether each batch-1 edge cell's best η becomes interior, and whether the tie-broken best
    G changes.
  - No direction is predicted.

## Deviations

Any departure from this file is listed in the v3b write-up under "Deviations", with its reason.
