# Final results (2026-11-25; updated 2026-11-26 with batch `v3r`)

The single reference for the report: every claim we can make, its status, its numbers and the file it comes from.
All experiments have finished and are analysed.

**Update of 2026-11-26.** After the instructor's feedback we added a two-prompt minimal example
(`analysis/theory/minimal_example.md`), harm measures on the confirmatory runs (`analysis/v3o/HARM.md`) and batch
`v3r` (T30–T33, `analysis/v3r/SUMMARY.md`, pre-registered in `analysis/review/PREREG_v3d_addendum11.md`). Of its 22
registered predictions, 19 hold and 3 fail; the failures are named below.

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
| H2 | G moves the objective from p toward 2 arcsin √p | Known; reproduced (tabular) | Mixing share λ = 0, 0.23, 0.63, 0.89 at G = 2, 4, 16, 64; with sharing, 93–97% of each prompt's change comes from other prompts | `analysis/base/FINDINGS.md`, `analysis/review/a.md` |
| H3 | One effective step decides stability | **Refuted** | Best candidate misses by up to 4.7 decades; correlated tasks collapse at η = 0.003 and at a fixed flow time | `analysis/base/FINDINGS.md`, `analysis/review/SUMMARY.md` |
| H4(a) | KL fixed points at finite G | Known (Vojnovic & Yun); reproduced | Median error about 1e-4 (960 runs) | `analysis/base/FINDINGS.md` |
| H4(b) | Fluctuations scale with η/β | Refuted | Scale with η, but not as 1/β (pooled slopes 0.40, 0.79) | same |
| H4 stretch | Two stable KL states at G ≥ 16 | Exploratory (extends Mroueh) | Basin boundary at the predicted unstable root | `analysis/stretch/FINDINGS.md` |
| H5 | The best G depends on difficulty | Weak | The direction survives with the tie-break fixed (32, 16, 4, 2), but many G values tie | `analysis/review/a.md` |

## 2. Minority collapse (the main finding)

**Scope: Core follow-up** (it grew out of H3's failure; the report's main finding).

| Claim | Status | Evidence | Source |
|---|---|---|---|
| Class imbalance causes the minority to collapse under shared parameters (P1) | **Confirmatory, twice** | Original: 16/16 cells, Holm p 0.049. Fresh tasks (A1): 20/20 tasks in all 16 cells, Holm p 4.8e-6 | `analysis/v3a/P1_TASK_LEVEL.md`, `analysis/v3o/SUMMARY.md` |
| Without sharing (tabular), no prompt ever decreases | Proved (Lean) + 2,268 runs | 0 decreases | `formal/`, `analysis/base/FINDINGS.md` |
| Collapse follows the answer class, not difficulty | Exploratory | Hard majority prompts are never absorbed (M4) | `analysis/v3d/SUMMARY.md` |
| The majority actively pushes the minority down | Exploratory | A majority-only step lowers the minority in every run; push 4–17 times the minority's own gain | `analysis/v3d/SUMMARY.md` |
| Removing the harmful component prevents collapse | Exploratory | Projection prevents it in 16/16 | `analysis/v3d/SUMMARY.md` |
| P2: sampling noise is negligible | Refuted | The kick size matters | `analysis/v3a/SUMMARY.md` |
| P3: early drift predicts the victims | Refuted | ΔAUC 0.009 against a bar of 0.03 | `analysis/v3b/SUMMARY.md` |
| With non-negative interactions between prompts' features, no prompt ever falls (softmax-linear policy, expected-update flow, any rule with non-negative weights) | Proved (three lines; the condition is the instructor's); **Confirmatory** at full size (T30, prediction 1) | 60 of 60 block-feature tasks at s = 0: the largest fall of any prompt's success is exactly 0 over flow times to 1e8 | `analysis/theory/minimal_example.md`, `analysis/v3r/SUMMARY.md` |
| Breaking that condition with one shared feature brings the collapse back | Exploratory (T30, predictions 3–5 hold) | Majority share 0.9: absorbed share 0 for shared scale s ≤ 2 and 1 for s ≥ 4 within the run; nothing absorbed at balance at any s; same under sampled GRPO | `analysis/v3r/SUMMARY.md` |
| The collapse survives in a two-prompt example (two answers, two weights, always separable) | Derivation with numerical checks, not registered | Three regimes by the overlap γ at share 0.9: never falls (γ ≤ 0), falls and recovers (lowest 0.17 at γ = 0.5, 5e-4 at γ = 0.9), stays down (1e-17 at γ = 1.2 at flow time 1e14). Same outcome under the mean rule and GRPO at G = 2, 8 and ∞; sampled GRPO: 20 of 20 runs below 1e-3 at γ = 1.2 | `analysis/theory/minimal_example.md` |
| Losing the minority is harmful by published measures | Descriptive, on the confirmatory runs | Majority share 0.9: the start policy overtakes the trained one at k = 8 in 16 of 16 cells and 20 of 20 tasks (pass@64 falls from 1.00 to 0.90–0.95); worst-class pass@1 below 1e-3 in 13 of 16 cells; coverage E[1/p] from 3 to 10^3.9–10^21. The start policy is uniform by construction | `analysis/v3o/HARM.md` |

## 3. Theory

**Scope: Core follow-up**, except the Lean proofs and the Adam ℓ∞ hint (**Extra**).

| Claim | Status | Evidence | Source |
|---|---|---|---|
| First-step law R = c·m/(1 − m), c from feature-kernel blocks | Proved (Lean, 6 theorems), plus numerical checks | Median error 0.14% (1,080 runs); out of sample in the transformer env (slope 1.00) and with junk answers; extends to 4-token answers (320/320 within 5%) | `analysis/theory/odds_law.md`, `analysis/v3h/SUMMARY.md`, `analysis/v3j/SUMMARY.md`, `analysis/v3n/SUMMARY.md` |
| The first step does not decide the outcome | Exploratory | Junk answers move the threshold as predicted, but collapse happens from m = 0.6; absorption needs more imbalance than the first decline | `analysis/v3j/SUMMARY.md`, `analysis/v3e/SUMMARY.md` |
| Late sinking rate set by the majority's ℓ2 max-margin direction (run level) | Exploratory, replicated | Rank 0.78 (T12), then 0.86 pre-registered (T24) | `analysis/v3l/SUMMARY.md`, `analysis/theory/late_time.md` |
| Per-prompt late model | Refuted | Median Spearman 0.28 against 0.8 | `analysis/v3l/SUMMARY.md` |
| **Theory of the fall** (class-level ODE, no fitted parameter) | In-sample: passes almost everything. **Held-out: mixed** | Unseen runs: all rules pass on the KL and warm-start grids. On the P1 grid it fails depth rank (0.755) because of large-step runs; at η ≤ 1, rank 0.988, level within 0.03–0.07 decades, accuracy 0.93–0.96, threshold right in 15/16 cells | `analysis/theory/fall_results.md` |
| Whether the minority stays down: a collapsed branch exists only when the overlap γ exceeds α/β, the ratio of how fast the majority's push and the minority's pull fade (1 for REINFORCE and GRPO at any G, k for pass@k, none for supervised learning), and the fall must also be deep enough | Proved for the tail system of the two-prompt example (SymPy); checked against the exact flow | Late decay of the minority's success matches time^(−γ/α) to three digits; the boundary is a straight line in the tail, and 20 of 20 flows started on either side go the predicted way. Where a fall from the uniform start lands is computed, not derived | `analysis/theory/minimal_example.md` |
| Adam follows a different (ℓ∞) geometry | Hint only | Under Adam at small ε, ℓ∞ γ ranks the late rate better (0.89–0.92 against 0.77–0.79); 10 tasks per correlation | `analysis/v3p/SUMMARY.md` |
| Permanent or plateau (exact flow to 1e14) | Per-prompt predictions refuted (T25: 1–3 fail); task-level finding post hoc | Mostly a plateau: recovery at 1e2–1e5 for ρ ≤ 0.5, 1e7.5–1e12 for ρ ≥ 0.9 (training ~1e3–1e4); nothing recovers at ρ = 0.95, m = 0.9. γ_min ≥ 1: 3 of 30 tasks recover; < 1: 51 of 70 (p 1e-11) | `analysis/v3m/SUMMARY.md` |

## 4. Prevention levers

**Scope: Core follow-up** (KL comes from the promised H4, group size from H5), except Adam's normalisation (**Extra**).

| Claim | Status | Evidence | Source |
|---|---|---|---|
| Forward KL to the start policy prevents absorption (β 0.03–0.3) | **Confirmatory** (A2) | 20/20 fresh tasks; 480/480 runs with no minority prompt absorbed at 1e-3; partial at 1e-2 for β = 0.03 | `analysis/v3o/SUMMARY.md`, `analysis/v3g/SUMMARY.md` |
| … but only while the reference still solves the minority | Exploratory (G) | Anchors after the fall never restore it (180/180) | `analysis/v3p/SUMMARY.md` |
| Reverse KL does not help at usual strengths | Exploratory | 100% absorbed at β ≤ 0.1 | `analysis/v3g/SUMMARY.md` |
| pass@4 protects at matched progress, at a speed cost | **Confirmatory** (A3) | Minority 0.28–0.44 against GRPO's 5e-5 to 5e-3; majority slower | `analysis/v3o/SUMMARY.md`, `analysis/v3f/SUMMARY.md` |
| A class-balanced warm start at p0 = 0.5 protects at m = 0.7 | **Confirmatory** (A5) | 20/20; absorbed share 0.0003–0.022 | `analysis/v3o/SUMMARY.md` |
| … but not at m = 0.9 | Exploratory | 0.99 linear, 0.69 MLP absorbed | `analysis/v3k/SUMMARY.md` |
| Group size is not a lever | Exploratory | It cannot change the first step; no protection | `analysis/v3g/SUMMARY.md` |
| No second resting point after release | Exploratory | The minority returns when a restoring force remains | `analysis/v3g/SUMMARY.md` |
| Adam's per-coordinate normalisation drives its extra recovery | Registered prediction refuted (4/8); post hoc support | Recovery switches off between ε = 1e-3 and 1e-2 at fixed lr | `analysis/v3p/SUMMARY.md` |
| With one knob changed and one yardstick, some lever fully protects (final minority ≥ 0.9, majority ≥ 0.9, none absorbed at 1e-2) | **Refuted as registered** (T31, prediction 5: "fails") | Standard linear cell, 100,000 steps, 20 fresh tasks: no arm meets the bar in more than 3 tasks. Post hoc: every protecting arm is still rising at the end, and supervised training on similar tasks ends the minority at 0.75 | `analysis/v3r/SUMMARY.md` |
| KL to the latest policy (reference resets) shields the minority, if each reset has enough steps | Exploratory (T31, predictions 1 and 6 hold) | Reverse KL, strength ≥ 1, ≥ 1,000 steps per reset: minority 0.79–0.83 at the end with a lowest logged value of 0.12–0.28 (medians), majority 0.99, mean pass@1 up to 0.977 (plain GRPO: minority 1.3e-6, majority 1.00, pass@1 0.901). ≤ 100 steps per reset collapses like plain GRPO; resetting every step is identical to it | `analysis/v3r/SUMMARY.md` |
| A fixed reverse KL either fails to protect or caps the majority | Exploratory (T31, prediction 3 holds) | Strength ≤ 0.1: minority about 2e-6. Strength 1 and 3: majority capped at 0.78 and 0.46. Best, 0.3: minority 0.80 after falling to 6e-4, majority 0.987 | `analysis/v3r/SUMMARY.md` |
| pass@k removes the long-time collapse on our tasks | Exploratory (T32, predictions 1–3 hold; 4 holds narrowly and depends on interpolating between grid points) | Exact flow to 1e14 on the 100 T25 tasks: pass@2 and pass@4 recover 100 of 100 (GRPO 54), including 30 of 30 with γ_min ≥ 1 (GRPO 3) | `analysis/v3r/SUMMARY.md` |
| … at a cost in majority speed and pass@1 | Exploratory (T33, prediction 2 holds; T31, prediction 7 fails in its k = 2 part) | The majority never reaches 0.99 within 100,000 steps under unscaled pass@k (0.97, 0.89, 0.70 for k = 2, 4, 8; 0.98, 0.95, 0.89 with a matched initial step). At GRPO's 0.99 point mean pass@1 under pass@k is lower in every pair. pass@2 keeps the final minority ≥ 0.5 in only 15 of 20 tasks | `analysis/v3r/SUMMARY.md` |
| From a uniform start pass@k slows the majority but never reverses it | **Refuted** (T33, prediction 1) | Exact pass@2: the majority falls 0.03–0.04 below its peak in 20 of 20 tasks; sampled runs: within 0.05 of the peak in only 301 of 600 (mostly fluctuations at larger steps) | `analysis/v3r/SUMMARY.md` |
| Switching to pass@k after a collapse lowers pass@1 while pass@k rises (the trade Barakat et al. predict), and pass@4 rescues the minority | Exploratory (T33, predictions 4–7 hold) | Median largest fall of the majority 0.09, 0.31, 0.48 for k = 2, 4, 8; under k = 8 mean pass@1 falls while pass@8 rises in 20 of 20 tasks; under k = 4 the minority is back above 0.5 in 20 of 20 | `analysis/v3r/SUMMARY.md` |
| Every lever has a constructed failure | Derivation with numerical checks (two-prompt example), not registered | pass@k: a collapsed state with γ > k stays down (pass@4 at γ = 5). KL to the start policy: a weak reverse KL, or a reference taken after the collapse for either direction. Resets: too few steps per reset | `analysis/theory/minimal_example.md` |

## 5. A second environment (small transformer, Li, Pandey & Chen)

**Scope: Extra** (a second setting beyond the promised toy task).

| Claim | Status | Evidence | Source |
|---|---|---|---|
| Their paper replicates | Replication | Their figure code: 494/494 numbers; our rerun: 130 identical, 193 within 2% (base models not bit-identical) | `tools/literature/replication/RESULT.md` |
| Collapse and the first-step law carry over | Exploratory (law), **confirmatory** (share effect, A4) | Law slope 1.00; A4 20/20 fresh seeds under Adam | `analysis/v3h/SUMMARY.md`, `analysis/v3o/SUMMARY.md` |
| The share effect also holds under small-step SGD | Exploratory (J) | Class 0 absorbed 0/80, minority 80/80 at high share, 4/40 at balance | `analysis/v3q/SUMMARY_J.md` |
| Forward KL prevents it there | Exploratory | All 8 cells at β = 0.3 | `analysis/v3i/SUMMARY.md` |

## 6. Small LLMs (Qwen2.5-0.5B, Llama-3.2-1B; for the short transfer section)

**Scope: Extra** (the milestone and the instructor's feedback set a toy-only project).

| Claim | Status | Evidence | Source |
|---|---|---|---|
| P4: GRPO loses training problems | Refuted | 0 of 1,000 lost, both models, 3 seeds | `analysis/llm/C1_C5_SUMMARY.md` |
| P5: the minority procedure declines with the share | Refuted | Improves +0.36 to +0.41; trend p 0.0536 | `analysis/llm/C2_SUMMARY.md` |
| Majority-only training raises the minority procedure | Exploratory | +0.25 and +0.26 in all 6 runs | `analysis/llm/C10_SUMMARY.md` |
| A few unseen problems decline | Exploratory | 8–12 of 300 decline, 1–5 lost | `analysis/llm/C1_C5_SUMMARY.md` |
| KL removes those declines | Exploratory | 14–26 declines without KL, 0–4 with β ≥ 0.02; forward ≈ reverse; G = 16 settles above G = 4; per-problem fixed points add nothing beyond base success | `analysis/llm/C7_C8_SUMMARY.md` |
| Shared skills explain the transfer (toy test) | Not tested fairly | Neutral shared positions dilute but don't prevent collapse; the test of positive transfer did not apply | `analysis/v3n/SUMMARY.md` |

## 7. What not to claim

- **Not new:** the identity (Davis & Recht), the KL fixed points (Vojnovic & Yun), and interference through shared
  parameters in general (Schaul et al. 2019, "ray interference"; read before writing the related work).
- Not "the fall is explained": the fall model is supported out of sample only in the small-step regime.
- Not "collapse is permanent" in general: it is a plateau whose length grows steeply with feature similarity; effectively permanent (to 1e14) only where γ_min ≥ 1.
- Not "LLMs are immune": our two small models and GSM8K procedures share most steps; the toy test of that
  explanation was not fair.
- Not "forward KL fixes it": only while the reference still solves the minority (G).
- Not "reference resets fix it": the registered verdict is "fails" at the 0.9 bar within 100,000 steps. What holds is
  narrower: with enough steps per reset they keep the minority near 0.8 and never let it fall far.
- Not "pass@k prevents collapse" in general: a collapsed state with γ > k stays down, pass@k slows the majority, and
  after a collapse pass@8 lowers pass@1. On our tasks every γ_min is below 2, which is why pass@2 and pass@4 recover.
- Not ours: the non-negative-interaction condition (the instructor's, unpublished), the kernel ⟨∇p_x, ∇p_x′⟩ and the
  pass@k-against-pass@1 conflict (Barakat et al. 2026), reference resets (Huang et al. 2026). Huang et al.'s coverage
  collapse is a different mechanism (one prompt, graded rewards, no sharing).
- Not "the two-prompt example is new": the instructor's submitted paper may contain a similar counterexample; ask
  before the report claims it.
- Not "the start policy beats the trained one at k = 8" as a general number: the start policy is uniform by
  construction, and the crossover k depends on it.
- Every LLM claim is exploratory. The report's main body is the toy ladder, plus one short transfer section.
