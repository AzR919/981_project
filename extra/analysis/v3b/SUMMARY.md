# v3 batch 2 (`v3b`): what the results say

All 5,280 runs returned and none failed. The tests follow `analysis/review/PREREG_v3b.md` (`review`,
`aa144dc`), and the code ran from `toy-v3` at `e61626f`. The numbers are in `FINDINGS.md`, which
`experiments/analyze_v3b.py all` regenerates.

| Test | Pre-registered verdict | In one line |
|---|---|---|
| **P3** (T5) | **FAIL** | Initial drift adds ΔAUC 0.009 (CI 0.004–0.014) to rarity + p_init. A real effect, but below the 0.03 bar |
| T7 | PASS | The bootstrap null is calibrated in tabular runs (KS p = 0.73); the medium tier of H1 holds for lin and m64 |
| T9 | FAIL (prediction too strong) | Heckel weighting cuts minority collapse from 0.10 to 0.06 at η = 0.1 (p = 0.002), but not at η = 1, and it doubles absorption in the big-step setting |
| (a) T1, seed 1 | PASS | P1 replicates in all 8 MLP cells with a second initialisation |
| (b) T2, G sweep | PASS | The response-noise excess at B = 8 shrinks as G grows (Spearman −0.33, p = 5e-10) |
| (c) T6 KL | PASS | KL runs drop below p_ref only at η ≥ 30; the control never falls from a peak |
| (d) T4, wider η | descriptive | The best G stays 2 (lin at p₀ = 0.01: 4); most former grid-edge cells now have an interior optimum |

## P3 in more detail

- **Single predictors.** As a single predictor, the initial drift ranks absorbed prompts better (AUC
  0.639) than rarity (0.543) or p_init (0.455).
- **Why the pre-registered model gains little.** It pools one drift coefficient across runs whose η
  spans 100×, and drift scales with η.
- **Post-hoc sensitivity** (not pre-registered; the verdict stays FAIL): with predictors standardised
  within each run, drift adds ΔAUC 0.079, CI 0.036–0.120.
- **Reading.** On balanced tasks the drift does carry information beyond rarity and p_init. The
  pre-registered specification lost it to scaling.
- **Implementation clarifications** (reported as such):
  - AUC is computed within runs, because run fixed effects cannot be predicted for held-out tasks;
  - the bootstrap resamples tasks over held-out scores without refitting.

## T9 in more detail

| Setting | Effect of the rules |
|---|---|
| Minority collapse, η = 0.1 | **heckel lowers absorption 0.102 → 0.059** (expected_full) and → 0.065 (sampled), p = 0.002 in both. virtual and pm_alpha are unchanged, and nsr and mean barely move, as the analytic W ratios predicted |
| Minority collapse, η = 1 | Nothing helps. All rules sit at 0.103, the whole minority set |
| Big kicks (lin, B = 8, η = 300) | heckel **raises** absorption (expected 0.21 → 0.42); mean (0.07) and nsr (0.16) lower it. Large advantages on rare successes make the kicks bigger |
| Noisy verifier (flip 20%, η = 300) | heckel is the worst (0.049 against 0.010 for grpo) |

- **What gets absorbed.** Every absorbed prompt in the minority task is a minority-class prompt,
  under every rule.
- **Lesson.** Fixing all-wrong groups helps only where collapse is driven by the mean drift and the
  step is small. It hurts wherever noise or large steps drive absorption. No rule removes minority
  collapse at η = 1.

## Deviations and clarifications

1. **P3:** within-run AUC and a bootstrap without refitting (see above).
2. **T9:** the cells mix task seeds (minority, kick) with seeds (tabular); the paired unit is
   whichever applies.
