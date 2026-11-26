# Adversarial review D (code and raw caches, branch `experiments`)

All numbers were recomputed from `results/{base,stretch}/cache` or from small local runs in a scratch directory. No repo file other than this one was changed.

## Findings, ranked

### HIGH 1. c09 (and mostly c05) "interference drift" is majority-answer collapse. A label-only baseline predicts it as well as drift does.
- **Evidence.** I rebuilt each c09 task (K=3, L=2, ρ=0.9, `task_seed = seed`) and compared it with `p_final` in `prompts_s_drift.npz`:
  - Across all 5 seeds, every η and both `sampled` and `expected_full`, **not one absorbed prompt has the majority answer sequence.**
  - In seeds 0, 2 and 3 the absorbed set is exactly the full minority set at η ≥ 0.003: 178/178, 101/101 and 54/54 prompts. The surviving prompts reach a mean p of 0.99–1.0.
  - The policy collapses to one constant answer.
- **The headline numbers are this artifact.** The "≈20% absorbed plateau" (0.197) is 101/512, the minority share of seed 2, which happens to be the seed median. The minority share differs a lot between tasks (seeds 0–4: 0.35, 0.65, 0.20, 0.10, 0.46), so a seed median taken across different tasks hides most of the variation.
- **Label-only baseline.** Score each prompt by its answer's rarity, −Σ_l log freq(answer_l), with no model involved. Median within-run AUC on the S_PRED runs:

  | level | rarity | step-0 drift |
  |---|---|---|
  | c09, sft_mixed | 1.000 | 0.999 |
  | c09, uniform | 1.000 | 1.000 |
  | c05, sft_mixed | 0.913 | 0.897 |
  | c05, uniform | 0.880 | 0.870 |
  | lin, sft_mixed | 0.51 | 0.73 |
  | m64, sft_mixed | 0.53 | 0.63 |

  My drift AUCs match FINDINGS, so the task reconstruction is right.
- **Threatens:** stretch "c09 is slow interference drift", "initial drift predicts absorbed prompts better than p and ‖∇p‖" (p and ‖∇p‖ are straw baselines), and base 4b "absorption set by the geometry".
- **What it is not.** It is not a capacity effect (see the dismissal of lead 2). It is RL's p-weighting (∇p = p∇log p) together with ill-conditioning: with ρ=0.9 the shared feature direction acts like a bias term and has about 145× the variance of the discriminative directions.
- **Fix.** Report the rarity baseline in S_PRED. Rename the mechanism "majority-answer collapse via the shared feature direction". Report absorbed counts per task, not seed medians. Add a balanced-label c09 control: centre the features, or draw answers so the classes are balanced.

### HIGH 2. H5's "best G" table is chosen by an arbitrary tie-break, at an η stuck on the grid edge.
- **Evidence.** In `h5_best_eta.csv` the best η is **10, the top of the grid, in 28/28 tabular cells** and in 45/56 linear cells. "η tuned per configuration" never actually brackets the optimum.
- **Resolution.** `step_to_half` is resolved to 25 steps, so many G tie exactly. The tied G ranges for tabular GRPO are [32–128], [16–128], [4–128] and [2–128] for p₀ = 0.01 to 0.3. The p₀ = 0.3 column is the same 6,400 samples for every G.
- **The tie-break.** `analyze_h5` picks the winner with `sort_values("samples").groupby(...).first()`, which uses an unstable quicksort. So the published values (128, 128, 4, 2) are arbitrary picks inside each tie. Another example: lin GRPO at p₀ = 0.3 reports 128 from a tie spanning 2–128.
- **At fixed η, larger G is never worse in tabular** (h5 raw `step_to_half`). "Best G" is therefore a saturation point, not an optimum.
- **The mean-rule check.** Its max/min ≤ 2 passes at exactly 2.00, which is 12,800 / 6,400: two logging intervals against one.
- **Threatens:** H5. With a principled tie-break (smallest G in the tie) the trend survives: 32, 16, 4, 2. But the verdict as computed does not support it.
- **Fix.** Extend the η grid above 10, log every step until p crosses the target (or interpolate), and define best G as the smallest G within the noise of the minimum.

### MEDIUM 3. In lin and c05 most "absorbed" prompts are not absorbed.
- **Evidence.** Share of prompts that ever drop below 0.01 but end above it:
  - H3: lin 0.83 / 0.80 / 0.70 (L = 1/2/4); c05 0.56 / 0.51 / 0.30.
  - S_DRIFT sampled: lin 0.56, c05 0.65.
- **Effect on the end-of-run measure.** The final absorbed fraction understates the peak. Median `max_absorbed/absorbed` is 1.6–1.9 for c05.
- **Effect on timing.** `t50` and `absorbed_at` use the first crossing, so they are dominated by prompts that later recover. Restricted to prompts that end absorbed, the c05 median and 90th-percentile times are both still 5 steps, so the overshoot claim survives. lin's 90th percentile rises to 30–45 steps.
- **Threatens:** the η* thresholds (an end-of-run snapshot of a population still changing), the "absorption" label for lin and c05, and the stretch split "lin = batch noise, c05 = one-shot". lin also has t50 = 5, so it is just as one-shot as c05.
- **Fix.** Report both "ever below 0.01" and "below 0.01 for the last 20% of training". Base thresholds on the latter.

### MEDIUM 4. c09's η* is set by the training horizon, not by stability. The "η^−0.89" exponent comes from censoring.
- **Evidence.** For `expected_full`, η·t50 is 3.70, 3.70, 3.73, 3.79, 3.80 for η from 0.0056 to 0.056, a slope of −1.0. Below that, η·t50 falls to 2.0 because the absorbed share is still rising at step 5000 (0.096 → 0.27): only the fast prompts get counted.
- So absorption happens at a fixed flow time (η·t ≈ 3.7), and the c09 η* ≈ 3.7/5000. The base claim "absorbing at the smallest η, not a step-size effect" holds only for this horizon: at η = 3e−4 it absorbs 0.2%.
- **Threatens:** the c09 time exponent; the base collapse test (H3 §3), where c09 enters with an η* of a different kind, which explains its +4.7-decade shift; and the "doesn't depend on B" claim, which holds trivially for any deterministic flow.
- **Fix.** Fit only uncensored η values, quote the time as flow time η·t, and drop c09 from the stability-threshold collapse, or rerun it with more steps.

### MEDIUM 5. Momentum and Adam do not break the tabular proposition. Their "decreases" are almost certainly round-off.
- **The argument.** With a zero-sum advantage, each tabular logit gradient is sign-consistent: the correct-token coordinate is ≥ 0 (Σ_{i: y_il = c} A_i ≥ 0) and every wrong-token coordinate is ≤ 0. Momentum (a positive sum of such vectors) and Adam (per-coordinate positive scaling of m_t) keep these signs, so p cannot fall in exact arithmetic.
- **The data agree.** `s_abl` shows `max_drop` = 0 and absorbed = 0 for both. The decreases are tiny: momentum median 1.5 per run (at most 47 out of 160k checks), against 10³–10⁵ for the real breaks.
- **Threatens:** S_ABL "every broken condition lowers p". The verdict passes on `decreases > 0`, which is an ulp-level counter.
- **Fix.** Count a decrease only when `before − after > 1e-9`, and record the largest decrease. Report momentum and Adam as preserving monotonicity, which is a cleaner result anyway.

### MEDIUM 6. H2 "shared runs don't sit on the W_G line" holds only for the MLP.
- **Evidence.** lin meets the tabular PASS criterion itself: slope 0.965 and R² 0.962 (0.979 with the worst 5% of bins dropped), with a median absolute deviation of 0.031, despite a cross_share of 0.93. m64 fails robustly (R² 0.42 after trimming).
- "RMSE increasing with interference share" is a monotonicity check on 3 points.
- **Threatens:** the H2 narrative.
- **Fix.** State that the linear model follows W_G and only the MLP departs from it. Add c05 and c09 to H2 to get a real interference axis.

### LOW 7. Post-hoc rescues and statistics
- **H1 PASS is a relaxed criterion.** 174/414 runs miss the pre-registered per-run band, and the stretch medium and tight tests FAIL as pre-registered. The post-hoc reading is reasonable, but the abstract should not say "PASS" without that qualifier.
- **Rules are not independent tests.** Rules share samples, and rloo is mean scaled by G/(G−1), so its statistics match exactly.
- **Thresholds rest on few seeds.** Every η* is a median of 3 seeds, i.e. the middle seed, and on correlated levels each seed is a different task (finding 1). There are about 25 verdicts with no multiplicity control.
- **What I checked and did not hold against them.** The stretch criteria did not change between commit `ab9c270` ("pre-set pass criteria") and the final analysis `4642444`.
- **Grid centre mismatch.** `BASE_ETA_STAR[("c09", 2)] = 0.01` and the s_drift `centre` use 0.01, but the base table gives 0.003. This is harmless here, since the grid covers both.

### LOW 8. H4(a) agreement comes mostly from interior fixed points
- The median |error| of 1e-4 is robust for p* < 0.9 (900 runs).
- For p* > 0.99 (21 runs), the error relative to 1−p* has a median of **14×**, so the theory does not predict the approach to 1.
- The 64 replicate prompts share batch draws. That matters only for H4(b) variances, not for the mean.

## Leads dismissed
- **Lead 1, flushing denormals.** Runs with and without `set_flush_denormal` are **bit-identical**. I compared final p, `absorbed_at`, the decrease counts and the absorbed fraction on three configurations:
  - c09, L=2, η=0.1, sampled, 1500 steps;
  - lin, L=2, η=3000, 1000 steps, where min p reaches 0;
  - m64, L=2, η=10, 600 steps, where min p reaches 0 and the final mean is exactly 80/512: a deterministic one-hot policy.

  Logits stay finite, so only the reported p underflows. Absorption is permanent because the softmax is saturated (all-wrong groups → zero advantage), not because of flushing. Script: `<scratch>/script`.
- **Lead 2, realizability.** Rule answers are the argmax of a linear score and the policy is W x with no bias, so W = s·(rule weights) solves every prompt:
  - At s = 20, every prompt has p > 0.5 on c09 seeds 1 and 2.
  - Supervised cross-entropy (`supervised_warmstart`, lr 0.5, 20k steps) reaches a mean p of 0.95 and 0.89, with **min p 0.40 and 0.27, and no prompt below 0.01**.

  So it is not a capacity effect. Supervised learning keeps the minority prompts that RL kills, which is the real story behind finding 1.
- **Lead 5, normalization.** `expected_step` uses −mean_x W(p_x) p_x over the batch, which matches −mean_{B,G}(A log π) in expectation (W is detached). The SGD comparisons are fair. Under Adam the sampled noise inflates v and shrinks steps, so the Adam block compares different effective step sizes. That block is not a headline.
- **Lead 4, uniform initialization.** This is a real issue for the p baselines, but it is moot: on correlated levels the rarity baseline beats drift regardless of initialization (finding 1).
- **Lead 8, "within 5 steps".** The claim is consistent with the 5-step resolution, and it survives restricting to prompts that end absorbed. Only the c09 exponent is affected, by censoring rather than resolution (finding 4).

## The single most important fix
Re-examine the correlated-feature "interference drift" claim against answer imbalance (finding 1):
1. add the label-rarity baseline to S_PRED;
2. report absorbed sets per task against the minority sets;
3. run a balanced-label or centred-feature c09 control.

If absorption disappears with balanced labels, the stretch batch's main mechanism is majority-answer collapse, and the "three mechanisms" story and the predictor result both need rewording. The H5 tie-break (finding 2) is the cheapest correctness fix: one line in `analyze_h5` plus a wider η grid.
