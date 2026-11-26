# Final results (2026-11-25)

The single reference for the report: every claim we can make, its status, its numbers and the file it comes from.
All experiments have finished and are analysed.

**Scope labels** (promised vs extra) are given per section; definitions in `docs/EXPERIMENTS.md`, "Promised vs extra".

**Status labels:**
- **Confirmatory:** pre-registered as a primary test with error control, and passed.
- **Replicated:** an exploratory finding that held again on fresh tasks or in a second environment, without being a
  primary test.
- **Exploratory:** pre-registered, but not a primary test.
- **Refuted:** a pre-registered prediction that failed.
- **In-sample:** a fit to data the model's author had seen.
- **Known:** already in the literature; we reproduce it.

## 1. The milestone hypotheses

**Scope: Promised.**

| | Claim | Status | Evidence | Source |
|---|---|---|---|---|
| H1 | The expected GRPO update is W_G(p)·∇p | Known (Davis & Recht); reproduced | 414 configurations, vector test, cosine ≥ 0.989 | `analysis/base/FINDINGS.md` |
| H2 | G moves the objective from p toward 2 arcsin √p | Known; reproduced (tabular) | Mixing share λ = 0, 0.23, 0.63, 0.89 at G = 2, 4, 16, 64; with sharing, 93–97% of each prompt's change comes from other prompts | `analysis/base/FINDINGS.md`, `extra/analysis/review/a.md` |
| H3 | One effective step decides stability | **Refuted** | Best candidate misses by up to 4.7 decades; correlated tasks collapse at η = 0.003 and at a fixed flow time | `analysis/base/FINDINGS.md`, `extra/analysis/review/SUMMARY.md` |
| H4(a) | KL fixed points at finite G | Known (Vojnovic & Yun); reproduced | Median error about 1e-4 (960 runs) | `analysis/base/FINDINGS.md` |
| H4(b) | Fluctuations scale with η/β | Refuted | Scale with η, but not as 1/β (pooled slopes 0.40, 0.79) | same |
| H4 stretch | Two stable KL states at G ≥ 16 | Exploratory (extends Mroueh) | Basin boundary at the predicted unstable root | `analysis/stretch/FINDINGS.md` |
| H5 | The best G depends on difficulty | Weak | The direction survives with the tie-break fixed (32, 16, 4, 2), but many G values tie | `extra/analysis/review/a.md` |

## 2. Minority collapse (the main finding)

**Scope: Core follow-up** (it grew out of H3's failure; the report's main finding).

| Claim | Status | Evidence | Source |
|---|---|---|---|
| Class imbalance causes the minority to collapse under shared parameters (P1) | **Confirmatory, twice** | Original: 16/16 cells, Holm p 0.049. Fresh tasks (A1): 20/20 tasks in all 16 cells, Holm p 4.8e-6 | `extra/analysis/v3a/P1_TASK_LEVEL.md`, `extra/analysis/v3o/SUMMARY.md` |
| Without sharing (tabular), no prompt ever decreases | Proved (Lean) + 2,268 runs | 0 decreases | `extra/formal/`, `analysis/base/FINDINGS.md` |
| Collapse follows the answer class, not difficulty | Exploratory | Hard majority prompts are never absorbed (M4) | `extra/analysis/v3d/SUMMARY.md` |
| The majority actively pushes the minority down | Exploratory | A majority-only step lowers the minority in every run; push 4–17 times the minority's own gain | `extra/analysis/v3d/SUMMARY.md` |
| Removing the harmful component prevents collapse | Exploratory | Projection prevents it in 16/16 | `extra/analysis/v3d/SUMMARY.md` |
| P2: sampling noise is negligible | Refuted | The kick size matters | `extra/analysis/v3a/SUMMARY.md` |
| P3: early drift predicts the victims | Refuted | ΔAUC 0.009 against a bar of 0.03 | `extra/analysis/v3b/SUMMARY.md` |

## 3. Theory

**Scope: Core follow-up**, except the Lean proofs and the Adam ℓ∞ hint (**Extra**).

| Claim | Status | Evidence | Source |
|---|---|---|---|
| First-step law R = c·m/(1 − m), c from feature-kernel blocks | Proved (Lean, 6 theorems), plus numerical checks | Median error 0.14% (1,080 runs); out of sample in the transformer env (slope 1.00) and with junk answers; extends to 4-token answers (320/320 within 5%) | `extra/analysis/theory/odds_law.md`, `extra/analysis/v3h/SUMMARY.md`, `extra/analysis/v3j/SUMMARY.md`, `extra/analysis/v3n/SUMMARY.md` |
| The first step does not decide the outcome | Exploratory | Junk answers move the threshold as predicted, but collapse happens from m = 0.6; absorption needs more imbalance than the first decline | `extra/analysis/v3j/SUMMARY.md`, `extra/analysis/v3e/SUMMARY.md` |
| Late sinking rate set by the majority's ℓ2 max-margin direction (run level) | Exploratory, replicated | Rank 0.78 (T12), then 0.86 pre-registered (T24) | `extra/analysis/v3l/SUMMARY.md`, `extra/analysis/theory/late_time.md` |
| Per-prompt late model | Refuted | Median Spearman 0.28 against 0.8 | `extra/analysis/v3l/SUMMARY.md` |
| **Theory of the fall** (class-level ODE, no fitted parameter) | In-sample: passes almost everything. **Held-out: mixed** | Unseen runs: all rules pass on the KL and warm-start grids. On the P1 grid it fails depth rank (0.755) because of large-step runs; at η ≤ 1, rank 0.988, level within 0.03–0.07 decades, accuracy 0.93–0.96, threshold right in 15/16 cells | `extra/analysis/theory/fall_results.md` |
| Adam follows a different (ℓ∞) geometry | Hint only | Under Adam at small ε, ℓ∞ γ ranks the late rate better (0.89–0.92 against 0.77–0.79); 10 tasks per correlation | `extra/analysis/v3p/SUMMARY.md` |
| Permanent or plateau (exact flow to 1e14) | Per-prompt predictions refuted (T25: 1–3 fail); task-level finding post hoc | Mostly a plateau: recovery at 1e2–1e5 for ρ ≤ 0.5, 1e7.5–1e12 for ρ ≥ 0.9 (training ~1e3–1e4); nothing recovers at ρ = 0.95, m = 0.9. γ_min ≥ 1: 3 of 30 tasks recover; < 1: 51 of 70 (p 1e-11) | `extra/analysis/v3m/SUMMARY.md` |

## 4. Prevention levers

**Scope: Core follow-up** (KL comes from the promised H4, group size from H5), except Adam's normalisation (**Extra**).

| Claim | Status | Evidence | Source |
|---|---|---|---|
| Forward KL to the start policy prevents absorption (β 0.03–0.3) | **Confirmatory** (A2) | 20/20 fresh tasks; 480/480 runs with no minority prompt absorbed at 1e-3; partial at 1e-2 for β = 0.03 | `extra/analysis/v3o/SUMMARY.md`, `extra/analysis/v3g/SUMMARY.md` |
| … but only while the reference still solves the minority | Exploratory (G) | Anchors after the fall never restore it (180/180) | `extra/analysis/v3p/SUMMARY.md` |
| Reverse KL does not help at usual strengths | Exploratory | 100% absorbed at β ≤ 0.1 | `extra/analysis/v3g/SUMMARY.md` |
| pass@4 protects at matched progress, at a speed cost | **Confirmatory** (A3) | Minority 0.28–0.44 against GRPO's 5e-5 to 5e-3; majority slower | `extra/analysis/v3o/SUMMARY.md`, `extra/analysis/v3f/SUMMARY.md` |
| A class-balanced warm start at p0 = 0.5 protects at m = 0.7 | **Confirmatory** (A5) | 20/20; absorbed share 0.0003–0.022 | `extra/analysis/v3o/SUMMARY.md` |
| … but not at m = 0.9 | Exploratory | 0.99 linear, 0.69 MLP absorbed | `extra/analysis/v3k/SUMMARY.md` |
| Group size is not a lever | Exploratory | It cannot change the first step; no protection | `extra/analysis/v3g/SUMMARY.md` |
| No second resting point after release | Exploratory | The minority returns when a restoring force remains | `extra/analysis/v3g/SUMMARY.md` |
| Adam's per-coordinate normalisation drives its extra recovery | Registered prediction refuted (4/8); post hoc support | Recovery switches off between ε = 1e-3 and 1e-2 at fixed lr | `extra/analysis/v3p/SUMMARY.md` |

## 5. A second environment (small transformer, Li, Pandey & Chen)

**Scope: Extra** (a second setting beyond the promised toy task).

| Claim | Status | Evidence | Source |
|---|---|---|---|
| Their paper replicates | Replication | Their figure code: 494/494 numbers; our rerun: 130 identical, 193 within 2% (base models not bit-identical) | `extra/tools/literature/replication/RESULT.md` |
| Collapse and the first-step law carry over | Exploratory (law), **confirmatory** (share effect, A4) | Law slope 1.00; A4 20/20 fresh seeds under Adam | `extra/analysis/v3h/SUMMARY.md`, `extra/analysis/v3o/SUMMARY.md` |
| The share effect also holds under small-step SGD | Exploratory (J) | Class 0 absorbed 0/80, minority 80/80 at high share, 4/40 at balance | `extra/analysis/v3q/SUMMARY_J.md` |
| Forward KL prevents it there | Exploratory | All 8 cells at β = 0.3 | `extra/analysis/v3i/SUMMARY.md` |

## 6. Small LLMs (Qwen2.5-0.5B, Llama-3.2-1B; for the short transfer section)

**Scope: Extra** (the milestone and the instructor's feedback set a toy-only project).

| Claim | Status | Evidence | Source |
|---|---|---|---|
| P4: GRPO loses training problems | Refuted | 0 of 1,000 lost, both models, 3 seeds | `extra/analysis/llm/C1_C5_SUMMARY.md` |
| P5: the minority procedure declines with the share | Refuted | Improves +0.36 to +0.41; trend p 0.0536 | `extra/analysis/llm/C2_SUMMARY.md` |
| Majority-only training raises the minority procedure | Exploratory | +0.25 and +0.26 in all 6 runs | `extra/analysis/llm/C10_SUMMARY.md` |
| A few unseen problems decline | Exploratory | 8–12 of 300 decline, 1–5 lost | `extra/analysis/llm/C1_C5_SUMMARY.md` |
| KL removes those declines | Exploratory | 14–26 declines without KL, 0–4 with β ≥ 0.02; forward ≈ reverse; G = 16 settles above G = 4; per-problem fixed points add nothing beyond base success | `extra/analysis/llm/C7_C8_SUMMARY.md` |
| Shared skills explain the transfer (toy test) | Not tested fairly | Neutral shared positions dilute but don't prevent collapse; the test of positive transfer did not apply | `extra/analysis/v3n/SUMMARY.md` |

## 7. What not to claim

- **Not new:** the identity (Davis & Recht), the KL fixed points (Vojnovic & Yun), and interference through shared
  parameters in general (Schaul et al. 2019, "ray interference"; read before writing the related work).
- Not "the fall is explained": the fall model is supported out of sample only in the small-step regime.
- Not "collapse is permanent" in general: it is a plateau whose length grows steeply with feature similarity; effectively permanent (to 1e14) only where γ_min ≥ 1.
- Not "LLMs are immune": our two small models and GSM8K procedures share most steps; the toy test of that
  explanation was not fair.
- Not "forward KL fixes it": only while the reference still solves the minority (G).
- Every LLM claim is exploratory. The report's main body is the toy ladder, plus one short transfer section.
