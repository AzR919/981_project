# v3r: a stylized ladder, a one-knob prevention table, pass@k (T30–T33)

Pre-registered in `extra/analysis/review/PREREG_v3d_addendum11.md` (2026-11-26), with Amendment 1 written before any `v3r`
run or flow. The batch answers the instructor's feedback of 2026-11-26. T30 has one confirmatory prediction (its
prediction 1 is a theorem); everything else is exploratory. Task seeds 30–49, never run before, unless stated.

`extra/experiments/analyze_v3r.py` regenerates `FINDINGS.md`, `results.csv` (every verdict) and the tables
(`t30_*.csv`, `t31_*.csv`, `t32_*.csv`, `t33a_*.csv`, `t33b_*.csv`). Its `baseline` step wrote `t32_baseline.md`
before any pass@k flow was opened (see T32).

Abbreviations: GRPO is group relative policy optimisation (here with G = 8 samples per prompt); KL is the
Kullback–Leibler divergence; SGD is stochastic gradient descent; MLP is a multilayer perceptron; pass@k is the chance
that at least one of k samples is correct, and pass@1 is the mean success; `rloo_k` is the unbiased (leave-one-out)
pass@k reward rule; LSODA, Radau and BDF (backward differentiation formula) are ODE solvers. Class 0 is the majority
answer class, with share m; the minority class mean is the mean success over the prompts of the other classes pooled.
"Absorbed" means a minority prompt's success stays below 1e-3 over the last 20% of training.

## Jobs

| Grid | Commit | Runs | Failed |
|---|---|---|---|
| `v3_t30` (T30, block features) | `toy-v3` `943e886`, clean, job digest `6b62fe691730` | 1,520 | 0 |
| `v3_t31` (T31, 49 arms × 20 tasks, then 2 × 8 robustness arms × 20 tasks) | same | 1,300 | 0 |
| `v3_t33` (T33(b), switch runs) | same | 120 | 0 |
| T30 exact flows (s × m × 20 tasks, to flow time 1e8) | same code, run locally | 360 | 0 |
| T32 exact flows (100 T25 tasks × GRPO, pass@2, pass@4, to 1e14) | same code, run locally | 300 | 0 |

- The 2,940 runs ran on six lab machines and were fetched with `extra/cluster/fetch.sh`: 2,940 results, 0 failed jobs, six
  machines done. The job list (`~/rlvr-data/v3r/jobs.jsonl`) has the digest `6b62fe691730`, and `grids.py v3r`
  reproduces it on both `toy-v3` `943e886` and `analysis-v3`.
- All 660 flows finished with LSODA; no flow needed the Radau or BDF fallback.
- T33(a) also reads 840 existing sampled runs: T16 (`~/rlvr-data/v3f`, 480 runs, task seeds 0–9) and A3
  (`~/rlvr-data/v3o`, grid `v3_a3`, 360 runs, task seeds 10–29), all from a uniform start.

## T30: non-negative interactions at full size (block features)

K = 3, 512 prompts, block features with a shared coordinate s; linear policy, GRPO. At s = 0 the condition holds
exactly; at s > 0 every cross-class pair violates it.

| Prediction | Pass rule | Result | Verdict |
|---|---|---|---|
| 1 (confirmatory). s = 0, exact flow: no prompt's success falls more than 1e-6 below its running maximum | 60 of 60 tasks | 60 of 60; the largest fall over all prompts and 221 flow times to 1e8 is exactly 0 | **holds** |
| 2. s = 0, discrete exact runs: no minority prompt absorbed and every class mean ends above 0.99 | ≥ 95% of runs | 118 of 120 (98.3%). No run absorbs a prompt; the two misses are task 48 at m = 0.9, whose class 2 ends at 0.985 and 0.990 | **holds** |
| 3. m = 0.9, exact, η̃ = 1.5: absorbed share 0 at s = 0; at s = 8 above it, median ≥ 0.5 | 20 of 20 at 0; ≥ 18 of 20 above; median ≥ 0.5 | 20 of 20 at 0; 20 of 20 above; median 1.00 | **holds** |
| 4. m = 1/3: median absorbed share ≤ 0.05 at every s | every (update, η̃, s) cell | No prompt is absorbed in any m = 1/3 run (24 of 24 cells at 0) | **holds** |
| 5. Sampled GRPO, m = 0.9: median absorbed share ≤ 0.02 at s = 0 and ≥ 0.5 at s = 8 | both, at each η̃ | 0 and 1.00 at η̃ = 0.5 and at η̃ = 1.5 | **holds** |
| 6. Reported: the first-step sign against R > 1 | an identity | The minority class mean falls at the flow's first time point exactly when R > 1, in 360 of 360 tasks | reported |

**Dose–response (reported).** The absorbed share is 0 or 1 in almost every task, so the curve is a step. In every
(update, η̃) cell the median share is 0 at s ≤ 2 and 1 at s ≥ 4 when m = 0.9, 0 at s ≤ 4 and 1 at s = 8 when m = 0.7,
and 0 at every s when m = 1/3; exact and sampled updates and both step sizes give the same curve. The smallest s with
a median of at least 0.05 at m = 0.9 (exact, η̃ = 1.5) is s = 4.

**R(s) and γ_min(s) per task at m = 0.9 (reported).** R is the odds-law ratio of the first step
(`check_odds_law.predict`); γ_min is T25's smallest projection of a minority prompt on the majority's max-margin
direction. Both apply unchanged to block features and are computed from each task's features. Medians over 20 tasks:

| s | 0 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| R | 0 | 0.70 | 2.5 | 7.1 | 13.0 | 16.5 |
| γ_min | 0 | 0.50 | 0.80 | 0.94 | 0.986 | 0.996 |
| tasks with γ_min ≥ 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| absorbed share, exact runs (η̃ = 1.5) | 0 | 0 | 0 | 0 | 1 | 1 |
| minority class mean in the flow at 1e8 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.7e-11 |

- R > 1 from s = 1 on, but nothing is absorbed until s = 4. At s = 1 and 2 the minority falls and comes back within
  the run.
- The runs stop at flow time η̃ × 20,000/(5 + s²), which is 435 at s = 8 and η̃ = 1.5, against 6,000 at s = 0. At
  s = 4 every run ends absorbed, but the flows are back at 1.00 by 1e8. At s = 8 the flows are still absorbed at 1e8
  in 20 of 20 tasks, although γ_min < 1 in 19 of them.
- The exploratory MLP (s ∈ {0, 8}, m = 0.9, η̃ = 0.5) gives the same picture: nothing is absorbed at s = 0, and
  everything is absorbed at s = 8, under both updates.

## T31: prevention with one knob changed

The standard cell: linear, ρ = 0.9, m = 0.9, exact update, SGD η = 0.1, 100,000 steps (flow time 1e4). "Protected and
learned" means a final minority mean ≥ 0.9, a final majority mean ≥ 0.9 and no minority prompt absorbed at 1e-2.

**Main table, end of training (step 100,000).** Medians over 20 tasks, ranges in brackets. The full table, all 49
arms and step 2,000, is in `FINDINGS.md` (`t31_arms_end.csv`, `t31_arms_2000.csv`). "Best" uses the rerun rule of
prediction 5: the largest c, then the higher median final minority mean.

| Arm | Minority mean | Lowest minority | Majority mean | Step majority ≥ 0.99 (tasks) | pass@1 | pass@4 | pass@16 | log10 median 1/p, minority | Absorbed 1e-3 | Absorbed 1e-2 | Protected and learned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Start policy | 0.333 | | 0.333 | | 0.333 | 0.802 | 0.998 | 0.48 | 0 | 0 | |
| Plain GRPO | 1.3e-6 (6.7e-7–5.7e-6) | 1.3e-6 | 1.000 | 100 (20) | 0.901 (0.88–0.94) | 0.901 | 0.901 | 6.19 | 1 | 1 | 0 |
| Best fixed reverse KL (β = 0.3) | 0.80 (0.50–0.91) | 5.8e-4 | 0.987 (0.979–0.992) | 100 (20) | 0.968 (0.93–0.98) | 0.994 | 0.998 | 0.05 | 0 (0–0.09) | 0 (0–0.26) | 1 |
| Best fixed forward KL (β = 0.03) | 0.65 (0.010–0.74) | 4.9e-3 | 0.976 (0.967–0.991) | 100 (20) | 0.940 (0.92–0.95) | 0.991 | 0.998 | 0.17 | 0 | 0 (0–0.57) | 0 |
| Best reverse reset (β = 3, N = 1,000) | 0.83 (0.72–0.93) | 0.27 | 0.993 (0.987–0.996) | 51,250 (16) | 0.977 (0.97–0.99) | 0.992 | 0.994 | 0.03 | 0 (0–0.07) | 0.03 (0–0.10) | 3 |
| Reverse reset (β = 1, N = 5,000), tied on c | 0.83 (0.73–0.93) | 0.17 | 0.993 (0.987–0.996) | 52,700 (16) | 0.977 (0.97–0.99) | 0.993 | 0.995 | 0.03 | 0 (0–0.07) | 0.008 (0–0.09) | 3 |
| pass@2 | 0.68 (0.43–0.79) | 0.030 | 0.966 (0.944–0.983) | never (0) | 0.933 (0.91–0.95) | 0.995 | 1.000 | 0.16 | 0 (0–0.41) | 0 (0–0.45) | 0 |
| pass@4 | 0.61 (0.56–0.66) | 0.17 | 0.894 (0.86–0.91) | never (0) | 0.866 (0.83–0.89) | 0.993 | 1.000 | 0.22 | 0 | 0 | 0 |
| pass@8 | 0.46 (0.43–0.48) | 0.25 | 0.696 (0.66–0.72) | never (0) | 0.672 (0.64–0.69) | 0.975 | 1.000 | 0.35 | 0 | 0 | 0 |
| Matched pass@2 (η = 0.289) | 0.80 (0.47–0.86) | 0.031 | 0.975 (0.966–0.984) | never (0) | 0.956 (0.93–0.97) | 0.998 | 1.000 | 0.09 | 0 (0–0.40) | 0 (0–0.40) | 0 |
| Matched pass@4 (η = 0.651) | 0.73 (0.68–0.78) | 0.20 | 0.948 (0.927–0.964) | never (0) | 0.927 (0.91–0.95) | 0.997 | 1.000 | 0.15 | 0 | 0 | 0 |
| Matched pass@8 (η = 3.30) | 0.64 (0.59–0.69) | 0.29 | 0.892 (0.85–0.92) | never (0) | 0.866 (0.83–0.89) | 0.994 | 1.000 | 0.21 | 0 | 0 | 0 |

**At step 2,000 (flow time 200)**, no arm is protected and learned in any task. The minority mean is 6.1e-5 under
plain GRPO, 7.0e-4 under fixed reverse KL at β = 0.3, 6.2e-3 under fixed forward KL at β = 0.03, 0.25 and 0.35 under
the reverse resets at β = 1 (N = 1,000 and 5,000), 0.32 at β = 3 and N = 1,000, and 0.06, 0.25 and 0.28 under
pass@2, 4 and 8. The majority mean there is 1.00 under plain GRPO and 0.52–0.94 under the resets and pass@k.

| Prediction | Pass rule | Result | Verdict |
|---|---|---|---|
| 1 (identity). `reset_every` = 1 reproduces plain GRPO | class means within 1e-6 at every logged step | 60 of 60 runs; largest difference 3.3e-16 (mean rule: 2.2e-16) | **holds** |
| 2. Plain GRPO: the minority mean ends below 1e-3 | ≥ 19 of 20 tasks | 20 of 20; median 1.3e-6 | **holds** |
| 3. Fixed reverse KL: no β is protected and learned in ≥ 16 of 20 tasks | every β has c < 16 | c = 0, 0, 1, 0, 0 at β = 0.03, 0.1, 0.3, 1, 3. Small β collapses (minority 1.6e-6 and 2.7e-6 at β = 0.03, 0.1); large β caps the majority (0.78 at β = 1, 0.46 at β = 3) | **holds** |
| 4. Fixed forward KL: no minority prompt absorbed at 1e-3 in ≥ 18 of 20 tasks at every β, and protected and learned in ≤ 4 | both, at every β | none absorbed in 20 of 20 tasks at every β; c = 0 at every β | **holds** |
| 5. Reverse-KL resets, two-sided: "protects" if some cell has c ≥ 16, "fails" if every cell has c ≤ 4, otherwise "partial" | as stated | largest c = 3, in three cells (β = 1 at N = 1,000 and 5,000; β = 3 at N = 1,000); all 15 cells have c ≤ 4 | **fails** |
| 6. β = 1 reverse resets: final minority mean at N = 1,000 above that at N = 10 | ≥ 16 of 20 tasks | 20 of 20; medians 0.79 and 1.4e-6 | **holds** |
| 7. pass@k (unscaled): final minority mean ≥ 0.5 for k = 2 and 4; the majority reaches 0.99 later than under plain GRPO for every k | ≥ 18 of 20 (k = 2, 4); 20 of 20 (every k) | minority ≥ 0.5 in 15 of 20 (k = 2) and 20 of 20 (k = 4). The majority reaches 0.99 later in 20 of 20 for every k (plain GRPO at step 100; pass@k never within 100,000 steps) | **fails** (k = 2 part) |
| 8. Reported: forward resets, mean-rule resets, robustness rows | | below | reported |

**Prediction 5 grid, c(β, N)** (tasks of 20 protected and learned):

| β | N = 1 | N = 10 | N = 100 | N = 1,000 | N = 5,000 |
|---|---|---|---|---|---|
| 0.3 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 3 | 3 |
| 3 | 0 | 0 | 1 | 3 | 0 |

- **Verdict: fails.** No cell is rerun on seeds 50–69, because the rerun is required only under "protects". By the
  tie-break rule the best cell would be β = 3, N = 1,000.
- The registered expectation ("protects" at β ≥ 1 and N ≥ 1,000) was wrong. Its second half held: cells with N ≤ 10,
  and also N = 100 at every β, end like plain GRPO.
- The cells with N ≥ 1,000 at β ≥ 1 fail on the minority threshold. The majority is ≥ 0.9 in 20 of 20 tasks, but the
  minority reaches 0.9 in at most 3, with a median of 0.79–0.83. No minority prompt is absorbed at 1e-2 in 7–13 of 20
  tasks.

**Reported (prediction 8).**
- **Forward resets:** c ≤ 3 in every cell (3 at β = 0.3, N = 5,000 and at β = 1, N = 1,000). The minority ends
  near 0.8 (median 0.79–0.82) only at β = 0.3, N = 5,000 and at β = 1, N ≥ 1,000; every other cell collapses,
  including β = 0.1 at every N.
- **Mean-rule reverse resets (β = 1):** c = 0 at every N. N ≤ 100 collapses as the plain mean rule does (final
  minority about 4e-6). The final minority mean is 0.66 at N = 1,000 and 0.72 at N = 5,000.
- **Robustness, MLP with the exact update:** protected and learned in 20 of 20 tasks under fixed forward KL
  (β = 0.1), and in 8 of 20 under fixed reverse KL (β = 0.3) and under reverse resets (β = 1, N = 1,000). It is 2 of
  20 under pass@4 and 0 under plain GRPO, fixed reverse KL at β = 1, and resets at N = 10 and 100.
- **Robustness, linear with sampled updates (B = 32):** at most 2 of 20 in every arm (resets at β = 1, N = 1,000: 2;
  fixed reverse β = 0.3: 1; every other arm 0). The medians are close to the exact-update rows.

## T32: pass@k over long times (exact flow to 1e14)

The 100 tasks of T25 under GRPO (recomputed), pass@2 and pass@4 (`rloo_k`). All 300 flows finished with LSODA.

**Step 1, done first (Amendment 1).** The script's `baseline` step compared the recomputed GRPO flows with the saved
T25 flows and wrote `t32_baseline.md`, committed as `0e6e3d3`, before any `rloo_k2` or `rloo_k4` file was opened.
`collect` keeps that order.
- **Per prompt**, over 100 tasks × 80 flow times × 512 prompts, the difference in log10 success has a largest value
  of 0.094 decades (task ρ = 0.9, m = 0.7, seed 2), a median of 0 and a 99th percentile of 3.1e-5.
- **Minority class mean:** the largest difference is 0.025 decades, and the median is 0.
- **Task outcomes:** all 100 agree; 54 tasks recover in both.
- **The two stiff tasks** that T25 finished with Radau and BDF differ by at most 0.0014 decades.

| Prediction | Pass rule | Result | Verdict |
|---|---|---|---|
| 1. Tasks recovered at 1e14 (minority class mean > 0.5) | ≥ 85 of 100 (pass@2); ≥ 95 of 100 (pass@4) | 100 and 100 (GRPO: 54) | **holds** |
| 2. Among the 30 tasks with γ_min ≥ 1: recovered under pass@2 | ≥ 24 | 30 of 30 (GRPO: 3 of 30) | **holds** |
| 3. Lowest minority class mean higher than under GRPO | ≥ 95 of 100, each k | 100 and 100; median lowest 10^−0.99 (pass@2) and 10^−0.64 (pass@4), against 10^−7.67 | **holds** |
| 4. Tasks recovering under both pass@2 and GRPO: pass@2 recovers earlier | ≥ 90% | 49 of 54 (90.7%); median recovery time 10^2.1 against 10^3.8 | **holds** (narrowly; see Deviations 5) |

**Reported.**

| | recovered | not recovered |
|---|---|---|
| GRPO, γ_min < 1 | 51 | 19 |
| GRPO, γ_min ≥ 1 | 3 | 27 |
| pass@2, γ_min < 2 | 100 | 0 |
| pass@4, γ_min < 4 | 100 | 0 |

Every task has γ_min < 2, so the γ_min ≥ k rows are empty for k = 2 and 4.

Majority class mean at flow times 1e2, 1e3 and 1e4 (the cost; medians over 100 tasks):

| Rule | 1e2 | 1e3 | 1e4 |
|---|---|---|---|
| GRPO | 0.999 | 1.000 | 1.000 |
| pass@2 | 0.867 (0.75–0.96) | 0.920 | 0.967 (0.89–0.99) |
| pass@4 | 0.680 (0.55–0.75) | 0.799 | 0.907 (0.77–0.95) |

## T33: pass@k training measured on pass@1

**(a) Existing sampled runs (T16, A3).** The 600 pass@k runs are k = 2, 4 and 8, in both forms (biased `passk`
and unbiased `rloo_k`), with G ∈ {8, 32} and η ∈ {0.03, 0.3, 1}. Each is paired with the GRPO run of the same task,
G and η.

| Prediction | Pass rule | Result | Verdict |
|---|---|---|---|
| 1. The majority class mean never falls more than 0.05 below its running maximum (sampled); 0.02 in T31's exact pass@k arms | ≥ 90% of the sampled pass@k runs; ≥ 18 of 20 tasks per k (exact) | sampled: 301 of 600 (50.2%), largest fall 0.41. Exact: k = 2: 0 of 20 (largest 0.040); k = 4: 20 of 20; k = 8: 20 of 20 | **fails** (both parts) |
| 2. At GRPO's first logged step with majority ≥ 0.99, mean pass@1 under pass@k is below GRPO's | ≥ 90% of pairs, k = 4 and 8 | k = 4: 360 of 360; k = 8: 120 of 120 (k = 2, reported: 120 of 120). Every GRPO run reaches 0.99 | **holds** |
| 3. Reported: the first step from which mean pass@1 under pass@k stays above GRPO's | | k = 2: from step 1,400–9,200 (median per cell) at η ≥ 0.3, and almost never at η = 0.03. k = 4: at η = 1 in 21 and 24 of 30 tasks for the biased form (median step about 12,000) and in 15 and 14 of 30 for the unbiased form; at η ≤ 0.3 in at most 3 of 30. k = 8: never within 20,000 steps | reported |

- **Prediction 1, sampled part.** The falls grow with the step size and the group size. The share within 0.05 is 97
  and 87 of 100 runs at η = 0.03 (G = 8 and 32), and 29 and 6 of 100 at η = 1. In the median run the final majority
  mean is 0.006–0.017 below its largest logged value. The GRPO runs never fall by more than 1.5e-5.
- **Prediction 1, exact part.** Under exact pass@2 the majority rises to a median of 0.945 by about step 650. It then
  falls by 0.028–0.040 (median trough 0.91, at about step 4,500) while the minority climbs from 0.03 to 0.15, and it
  rises again later. Under k = 4 and 8 it never falls by more than 1.5e-5.

**(b) Switching to pass@k after a collapse** (the T31 cell; plain GRPO up to step 100 or 1,000, then `rloo_k`). The
control is T31's plain arm, whose majority never falls after step 100 and whose minority ends at a median of 1.3e-6
(0 of 20 tasks above 0.5). At the switch at step 100 the majority mean is 0.998 and the minority mean is 1.2e-3
(medians).

| Prediction | Pass rule | Result | Verdict |
|---|---|---|---|
| 4. k = 8, switch at 100: the majority falls ≥ 0.05 below its value at the switch | ≥ 16 of 20 tasks | 20 of 20; median largest fall 0.48 | **holds** |
| 5. Median largest fall of the majority ordered k = 2 < 4 < 8 | strict order | 0.089 < 0.31 < 0.48 (switch at 1,000: 0.090 < 0.31 < 0.47) | **holds** |
| 6. k = 8, switch at 100: mean pass@1 below its switch value while mean pass@8 is above its switch value, at some logged step | ≥ 16 of 20 tasks | 20 of 20 (at steps 2,000 and 100,000) | **holds** |
| 7. Rescue: k = 4, switch at 100, minority class mean above 0.5 at the end | ≥ 16 of 20 tasks | 20 of 20; median 0.60 | **holds** |

Reported for prediction 7:
- **Switch at 100:** the minority ends above 0.5 in 14 of 20 tasks under k = 2 (median 0.67) and in 0 of 20 under
  k = 8 (median 0.45).
- **Switch at 1,000:** 9 of 20 under k = 2 (median 0.48), 16 of 20 under k = 4 (0.56) and 0 of 20 under k = 8 (0.43).
- **Cost:** the final majority mean is 0.97, 0.89 and 0.70 under k = 2, 4 and 8 (switch at 100). No switch run is
  protected and learned.

## What this adds

- **T30.** At full size (512 prompts) the non-negative-interaction condition holds exactly in the flow: no prompt's
  success ever falls, in 60 of 60 tasks. Breaking it with a shared coordinate gives a step, not a ramp. At m = 0.9
  nothing is absorbed in the 20,000-step runs up to s = 2 (R up to 7) and everything is absorbed from s = 4. At
  m = 1/3 nothing is absorbed at any s.
- **T31, prediction 5 fails.** With one knob changed, no arm is protected and learned in more than 3 of 20 tasks.
  Reverse-KL resets at β ≥ 1 and N ≥ 1,000 keep the minority near 0.8 with the majority at 0.99, which is close to a
  fixed reverse anchor at β = 0.3. N ≤ 100 collapses like plain GRPO. Every such arm is still rising at the end
  (post hoc context below).
- **T31, prediction 7 fails** (its k = 2 part: 15 of 20 tasks). Unscaled pass@k absorbs no prompt at k = 4 and 8, but
  the majority never reaches 0.99 within 100,000 steps under any k.
- **T32.** Over long times pass@2 and pass@4 recover all 100 T25 tasks, including the 30 with γ_min ≥ 1, of which
  GRPO recovers 3. The cost is a slower majority: 0.87 (pass@2) and 0.68 (pass@4) at flow time 1e2, against 0.999.
- **T33(a), prediction 1 fails.** pass@k from a uniform start does reverse the majority by more than 0.05 in half of
  the sampled runs, mostly at larger steps, and by up to 0.04 under exact pass@2. Prediction 2 holds: at GRPO's 0.99
  point, mean pass@1 under pass@k is lower in every pair.
- **T33(b).** Switching to pass@k after a collapse lowers the majority and mean pass@1 while pass@8 rises, which is
  the trade Barakat et al. predict, ordered in k. k = 4 brings the minority back above 0.5 in 20 of 20 tasks; k = 8
  does not in any.

## Post hoc context for T31 (not registered)

Three descriptive checks made after the verdicts, on these runs and on earlier ones. They change no verdict.

- **Every protecting arm is still rising at step 100,000.** In each arm below the minority class mean rose by more
  than 0.005 between steps 80,000 and 100,000 in 20 of 20 tasks. Medians over the 20 tasks:

  | Arm | Step 20,000 | Step 60,000 | Step 100,000 |
  |---|---|---|---|
  | Reverse reset, β = 3, N = 1,000 | 0.60 | 0.78 | 0.83 |
  | Reverse reset, β = 1, N = 5,000 | 0.60 | 0.78 | 0.83 |
  | Fixed reverse KL, β = 0.3 | 0.002 | 0.71 | 0.80 |
  | Matched pass@2 | 0.59 | 0.74 | 0.80 |
  | pass@2 | 0.36 | 0.61 | 0.68 |
  | pass@4 | 0.47 | 0.57 | 0.61 |

  So "fails" in prediction 5 means that no cell met the bar within 100,000 steps. Where these arms end is not known.
- **Shield or floor.** The resets and pass@4 and pass@8 never let the minority fall far: the lowest logged minority
  mean is 0.17–0.27 (medians). The fixed anchors let it fall and bring it back: the lowest value is 5.8e-4 under
  reverse KL at β = 0.3 and 4.9e-3 under forward KL at β = 0.03, and at step 20,000 the fixed reverse anchor still has
  the minority at 0.002.
- **The 0.9 bar against supervised training.** T31 has no supervised arm, so its table has no ceiling row. The nearest
  reference is A1's supervised runs on the same kind of task (linear, ρ = 0.9, m = 0.9; task seeds 10–29; 2,000 steps
  at η = 1, flow time 2,000): the majority ends at 0.974 (median) and the minority at 0.754 (range 0.59–0.81), and no
  task reaches 0.9. The tasks and the horizon differ from T31's, so this is a rough reference only. It suggests that
  the linear policy learns these minority prompts slowly even without interference, and that the 0.9 bar was strict
  for this cell.

## Deviations and choices

1. **Flows ran locally, and all finished with LSODA.** The 360 T30 flows and 300 T32 flows ran on the laptop with
   `extra/experiments/flow_batch.py`, not on the lab machines. None needed the registered Radau and BDF fallbacks, so no
   task counts as "not recovered" for a solver failure.
2. **The two-prompt example is not part of this batch.** It is a derivation with a numerical check
   (`extra/analysis/theory/minimal_example.md`), as the addendum states.
3. **T32's baseline was compared first.** The script and the commit order (`0e6e3d3`, then `cf30d00`) record that the
   recomputed GRPO flows were compared with the saved T25 flows before any pass@k flow was opened. The paired
   comparisons use the recomputed baseline.
4. **T31 prediction 5 has no rerun.** The rerun on seeds 50–69 is required only if the verdict is "protects". The
   verdict is "fails", so `grids.v3_t31_confirm` is not run.
5. **T32 prediction 4 is close to its threshold and depends on a choice.** Recovery time is the flow time from which
   the minority class mean stays above 0.5, interpolated linearly in log10 time between T25's grid points (80 points
   over 15 decades). This rule was in the script committed before any pass@k flow was read (`0e6e3d3`). With it,
   pass@2 is earlier in 49 of 54 tasks (90.7%). Read on the grid points alone, it is earlier in 48, tied in 4 and
   later in 2 (88.9%), which would fail.
6. **T33(b) prediction 6** can be checked only where per-prompt success is stored after the switch: steps 2,000 and
   100,000. Mean pass@1 is logged every 100 steps, but pass@8 is not.
7. **T30's runs end at different flow times.** The step size is η̃/(5 + s²), as registered, so the runs at large s
   stop earlier in flow time. The dose–response is a statement about the runs as registered; the flows show where the
   same tasks go by 1e8.
8. **T30 γ_min accuracy.** γ_min uses T25's max-margin solver. Its smallest majority margin is 0.9967 instead of 1,
   so γ_min is accurate to about 0.003.
9. **T31 has no supervised arm.** The pre-registration named none, so the one-knob table has no ceiling row; the
   post hoc section above gives the nearest reference.

Choices not fixed by the pre-registration (all in `FINDINGS.md`, "Choices not fixed by the pre-registration"):
- **The minority class mean** is the pooled mean over classes 1 and 2, as logged. The absorbed share is over
  minority prompts, with T0's eligibility rule.
- **T30 prediction 2's "every class mean"** is each class's mean in the final snapshot. **Prediction 4** is read over
  every (update, η̃) cell, and **prediction 5** at each η̃.
- **T31 at step 2,000:** the absorbed shares use the snapshot there, because no tail window exists. "Never reaches
  0.99" counts as later in prediction 7.
- **T33(a) prediction 1's 90%** is over all sampled pass@k runs pooled; the exact part uses the three unscaled arms.
  **Prediction 2** leaves out pairs whose GRPO run never reaches 0.99 (none did).
- **T33(b) prediction 5** is read at the switch at step 100; step 1,000 is reported.
