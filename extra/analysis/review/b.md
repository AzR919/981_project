# Review B: stretch goals, planned against executed

Scope: the original stretch list (`HANDOFF.md:185-191`), the executed stretch batch (`HANDOFF.md:99-155`,
`experiments/grids.py:98-190`), criteria in `experiments/analyze.py:1084-1475`, results in
`analysis/stretch/FINDINGS.md` (cited as F:line). Numbers marked *[recomputed]* come from
`results/stretch/cache` and `results/base/cache`; the scripts are at the end.

Process check: the batch launched 16:22 (`fa7f182`), the criteria were committed 16:25 (`ab9c270`), and
results finished around 16:40. Between `ab9c270` and `4642444` the only removed line in `analyze.py` is
the predictor's `pa, da = p[alive], drift[alive]` (it now clips p to [1e-6, 1-1e-6]). No verdict line
changed. So the pre-registration holds, apart from one small bug fix in the scoring code.

## 1. Goal-by-goal table

| Goal | Planned | Executed (deviations) | Pre-registered verdict | Post-hoc additions: diagnosis or rescue | Remains |
|---|---|---|---|---|---|
| H1 tiers (new in executed plan) | — | s_h1, R = 20,000, 20 chunks, split-half null, b∥ (`grids.py:114-117`) | Loose PASS; Medium FAIL (χ² on L p = 0.004); Tight FAIL (36 runs outside ±2%) (F:9-11) | **Medium: leans rescue (med).** F:58 switches to a one-sided count ("only high misses count"). The pre-registered test was two-sided, and the failing split (L = 1 tab, 7 low misses) shows the null itself is miscalibrated, so this tier can't certify anything. **Tight: justified diagnosis**, but the conclusion is limited: the 36 runs are about 7 configurations (rules share samples, F:126), and at L = 4 uniform the CI half-width reaches about 0.03 (SE 0.0155, F:107). `HANDOFF.md:127` ("no bias at any tier that has the precision") is stronger than what the data show. | Report "bias ≤ 2% except L = 4 uniform (unresolved, ≤ ~3%)". |
| H3-1 proposition + ablations | noisy 0.05/0.2, momentum, Adam, β>0, non-zero-sum | All run, at L 1/2/4, G 4/8, η 0.3–300 (`grids.py:147-167`). `sign` stands in for "non-zero-sum" (documented, `HANDOFF.md:57`) | PASS (F:17) | None. **The criterion is weak (med):** "every broken condition lowers p in ≥ 1 run" (`analyze.py:1405`) counts momentum and Adam (max_drop **0**, max_absorbed 0, median decreases 1.5 and 4; F:378-379). It also counts KL, which lowers p above the fixed point by design. The PASS label hides the real result: momentum, Adam and β = 0.1 do **not** break monotonicity in any material way. | Rephrase the verdict around max_drop and absorption. |
| H3-2 predictor | AUC vs p_init and ‖∇p‖ | s_pred: sft_mixed + uniform starts, checkpoints every 500 steps (`grids.py:137-144`) | PASS on c05/c09; "MLP drift does not beat baseline" FAIL, which is a positive surprise; "checkpoint > step-0 on MLPs" FAIL (F:14-16) | `HANDOFF.md:143` reports the MLP surprise honestly. **Missing caveat (med):** "checkpoint drift predicts at chance" (`HANDOFF.md:145`) rests on few events. Only 1,070/33,885 (c05), 4,150/54,659 (lin) and 5,128/54,105 (m64) absorptions happen after step 500 *[recomputed]*. The first-500-step window is never scored. c05 ck_grad = 0.23 (F:342), a strong inverse signal, is not discussed. | Score a 0–500 window at finer checkpoints. Discuss the AUC levels (MLP drift is only 0.63, F:329). |
| H3-3 drift vs noise | fine η grid; t ∝ 1/η vs 1/η²; SGD **and** Adam | 13 η × 5 seeds × {sampled, expected, expected_full}, a B block, an Adam block (`grids.py:120-134`). Adam has no `expected` mode, and its η grid isn't centred per level | c05/c09 FAIL; m64 PASS (F:12-13) | **Overshoot: justified diagnosis (low).** F:213 is labelled post hoc, keeps the FAIL, and the data back it (c05 t50 = 5 steps in all modes, F:217-219). But it moves c05 out of the "interference drift" class that base FINDINGS assigned it (`base/FINDINGS.md:186`) without saying so. **"Three mechanisms" (`HANDOFF.md:134`): partly rescue (high), see §3.** | Adam fit (§2). An experiment that separates noise from in-batch step size (§3). |
| H3-4 lost problems | absorbed prompts with p_init > 0.05; pass@64 before/after | **Not run on stretch data.** Base did it with uniform starts, where it is degenerate (`base/FINDINGS.md:257`) | none | — | §2: 5 minutes from the cache. |
| H4 equilibria | start on both sides of the unstable root; switching vs η; optional ref resets | s_eq: 3 (p_ref, β) settings × 11 offsets × 5 η, plus a mean-rule control (`grids.py:170-190`). No resets | Basin PASS, roots PASS, control PASS; crossings-rise-with-η FAIL (F:18-21) | **Rescue (med).** F:449-457 changes the metric from all crossings to second-half crossings away from offset 0. That is a reasonable fix for a badly specified metric, but the new metric is non-zero in only one of three settings, at two η (6 and 538, F:454). "As a barrier-crossing picture predicts" is untested (§3). | A deterministic control. Reference resets. |
| H5 all-wrong-group fixes | virtual correct sample, ±α on uniform groups, NSR; rerun failing H3 configs | **Not run.** No rule or grid exists (grep finds nothing in `synthetic_rl/`, `experiments/`) | none | — | §2 |
| H2 Barakat | shared models: non-constant-W rules lower mean pass@1 by interference | **Not run** | none | — | §2 (a preview from existing data contradicts the claim) |

## 2. Goals never run or partly run, ranked by value to the final report

1. **Lost problems on stretch data (value high, cost about 5 min, analysis only).** sft_mixed runs have a
   real spread of p_init. In the 247 sft_mixed runs with > 10% absorbed *[recomputed]*, 79–95% of
   absorbed prompts started above p = 0.05 and 18–59% above 0.3. On linear levels mean pass@1 *rises*
   while pass@64 *falls*: lin 0.29→0.64 vs 0.99→0.64, c05 0.32→0.72 vs 0.99→0.73. That is the RLVR
   coverage-narrowing signature, and it directly supports the H3 headline. One caveat: these runs are
   selected by absorption, and many use η = 3–10× η*.
2. **H5 all-wrong-group fixes (value high, cost about 0.5–1 day of code plus about 300–600 runs, a few CPU-hours).**
   This is the one intervention that tests whether absorption is *caused* by all-wrong groups.
   Rerun it on the s_drift configurations that absorb (c09 η = 0.01, m64 η = 1, lin η = 1,000). The
   milestone already promises it (`latex/milestone.tex:323-325`).
3. **1/η vs 1/η² under Adam (value medium, cost about 10 min from the cache; a re-centred grid
   needs about 260 runs).** The data exist but were never fitted. My fit *[recomputed]*: c09 slope
   −0.91 (expected_full) and −0.98 (sampled); m64 sampled −1.6 over 5 points, floored at 5 steps. lin
   and c05 never reach 5% absorbed on the shared Adam grid (F:262-300), so no fit is possible there.
   Under Adam the step length is about η whatever the gradient scale, so 1/η is the expected drift law.
4. **Barakat test (value medium, cost about 1 h from base H2, or 1 small grid).** Preview from the
   `prompts_h2.npz` one-step data *[recomputed]*: 0 of 138 shared-model runs lower mean pass@1, for
   every rule. Interference is large and **positive** for the constant-W mean rule too (median
   (Δp̄ − own)/own ≈ 14 for lin/mean and 13.6 for m64/mean). At this warm start (p ≈ 0.4, η = 0.5,
   one step) the data contradict both halves of the Barakat claim. That is worth one sentence in
   the report, and it should be stated as a single-θ result.
5. **Reference resets (value low, cost about 0.5 day).** They would extend only the tabular KL picture.
   Drop them, or mention them as future work.

## 3. Where the design cannot support the claim

- **"Batch-composition noise, not response noise" (high).** `expected` resamples B prompts each step
  (`run.py:250-252`), so expected vs expected_full differs in two ways. One is randomness. The other is
  that each in-batch prompt gets a step about n/B = 16× larger, while all prompts move together in
  expected_full. The data fit the step-size reading. m64's expected_full threshold (between η = 5.6,
  absorbed 0.008, and η = 10, absorbed 0.375) is 10–18× the B = 32 `expected` threshold (0.562),
  close to n/B = 16 *[recomputed, F:183-188]*. η* ∝ B^0.83 (F:243) is predicted equally by
  "noise ∝ η/B" and "per-prompt step ∝ η/B". Absorption is also fast: m64 median t50 is 12.5–15
  steps (F:226-228), which looks like overshoot. Two other problems:
  - The decomposition is not factorial. There is no full-batch-with-sampled-responses cell, so
    "response noise doesn't matter" is only shown *given* batch noise.
  - It doesn't hold for c05 or small-B lin, where sampled η* is half of expected's (16.9 vs 30, and
    56 vs 100, F:234-238). At c05 η = 30 the deterministic flow absorbs **more** than `expected`
    (0.211 vs 0.057, F:145).

  Fix: run deterministic cyclic batches (a fixed partition, no randomness), or expected_full at η·n/B.
- **The t50 discriminator does not discriminate (high).** F:211 says slope −1 means drift. m64, the
  "noise" level, has slope −0.89 (sampled) and −0.96 (expected) (F:207-209), the same as c09's −0.89.
  With `log_every = 5` (`grids.py:122`), lin, c05 and m64 are floored at 5–15 steps. Only c09 is
  resolved, so the planned 1/η-vs-1/η² question is answered for c09 alone.
- **c09 "pure drift" (medium).** Supported for noise-independence. Sampled and expected_full absorb
  the *same* prompts (median Jaccard 1.0 over 61 (η, seed) pairs; lin 0.09, m64 0.003 *[recomputed]*).
  But:
  - the plateau is set by the task seed. At η = 0.01 the five seeds absorb 0.348, 0.543, 0.197,
    0.105 and 0.049. "0.197" (F:157-159) is just seed 2's value.
  - seed 1 is non-monotone in η (0.625 → 0.137), and its absorbed set changes (Jaccard 0.23).
  - t50 medians drop runs below 5% absorbed (seed 4).
  - A seed-fixed set of victims is exactly what a capacity or realizability limit would produce.
    The stretch never checks that the linear model can solve all c09 prompts.

  Claim "noise-free and η-independent", not yet "dynamical drift".
- **Bistability switching as barrier crossing (medium).** Expected updates refuse β > 0
  (`run.py:204-205`), so s_eq has no noise-free control. At p_ref 0.03, η = 10, starts at −3 logit end
  in the upper state 47% of the time *[recomputed from `s_eq_basin.csv`]*. The low state is
  effectively gone, which a deterministic large-step instability would also produce. Two non-zero
  points can't establish "grows steeply with η". Minor: F:435-436 prints p_ref 0.01 and 0.03 as "0.0".
- **β = 1 "curvature limit" (low).** `HANDOFF.md:149` is an interpretation. Only η = 300 absorbs
  (F:407), and no test was run.

## 4. Overall verdict

Execution was strong and the process was honest. The executed batch covered H3-1, H3-2, H3-3 (SGD) and
H4 with good controls. Criteria were committed before the results and never edited. Every post-hoc
paragraph is labelled, and FAILs stay FAILs in `FINDINGS.md`. The weak points are elsewhere:
- The synthesis in `HANDOFF.md` §3c turns two FAILs (H1 medium/tight, S_DRIFT c05) and one rescued
  metric (S_EQ switching) into positive statements.
- The marquee new claim, "batch-composition noise, not response noise" and "three mechanisms", rests
  on a comparison that confounds randomness with the n/B step concentration. The planned t50
  discriminator turned out unable to separate the mechanisms.
- The ablation PASS is near-trivial as coded.
- Four planned goals were never run: lost problems, H5 fixes, Barakat, resets. The Adam fit was run
  but not analysed. Two of these (lost problems, the Adam fit) cost minutes from existing caches, and
  a Barakat preview from base data already contradicts the hypothesis.

Before the final report: do those three cheap analyses, add the cyclic-batch control (about 100 runs),
and run the H5 fix.

## Appendix: recomputation snippets

```python
# c09 per seed; prompt-set overlap between modes (results/stretch/cache)
r=pd.read_csv('runs_s_drift.csv'); z=np.load('prompts_s_drift.npz'); pos={int(v):i for i,v in enumerate(z['row_train'])}
ab=lambda i:(z['p_final'][pos[i]]<.01)&(z['p_init'][pos[i]]>=.01)   # Jaccard(sampled, expected_full) per (eta, seed)
# Adam t50 slope: polyfit(log10 eta, log10 median t50) over runs with absorbed >= .05, >= 3 runs per eta
# lost problems: s_pred sft_mixed, absorbed > 10%: mean(1-(1-p)^64) on p_init vs p_final
# Barakat preview: prompts_h2.npz, onestep rows: delta_mean.mean() < 0 ? ; (delta_mean.mean()-own_term.mean())/|own_term.mean()|
# late events: s_pred absorbed_at > 500 vs 0 < absorbed_at <= 500
```
