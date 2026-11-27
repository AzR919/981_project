# Presentation outline: GRPO unraveled

Draft, 2026-11-25. Toy results only (the LLM work is set aside for now). Each slide gives a heading,
what goes on it, and a suggested figure. Numbers come from `analysis/` and `docs/Azaan.md`. Figures: `figures/` (build with `python figures/make_figures.py`).
Results marked *pending* wait on the replication batch (A1–A5).

---

## Slide 1: GRPO unraveled: a small-scale empirical study of RLVR

- CMPT 981 project. Ankush Singh, Simon Doan, Azaan Rehman
- One line: *we test what GRPO theory predicts, in a toy where every quantity is exact*

---

## Slide 2: Why GRPO?

- RLVR: fine-tune a model on prompts whose answers a verifier can check (reward 0 or 1)
- GRPO (DeepSeek): sample G answers per prompt and score each against the group's average, with no critic
- Theory has recently explained parts of it, but each paper looks at **one prompt in isolation**
- **Our question:** does that per-prompt theory predict what training actually does?

---

## Slide 3: One equation behind three papers

- For binary rewards, the expected GRPO update on a prompt is

  **E[ĝ] = W_G(p) · ∇p**, where W_G = E[A | correct] − E[A | wrong] (the *advantage gap*)
- Each paper reads one property off this equation:

  | Paper | Question | What W_G sets |
  |---|---|---|
  | Thrampoulidis et al. | What is optimized? | the surrogate objective, F′ = W |
  | Mroueh | Where does training stop? | the fixed point with a KL penalty |
  | Suk & Duan | Is it stable? | the effective step size |
- Credit: Davis & Recht prove this identity, finite G included
- *Figure: `figures/out/s03_advantage_gap.png`*

---

## Slide 4: The instructor's feedback shaped the plan

- Proposal comment: "trying to do too much … fix the setting … relate everything"
- What we did with it:
  - **one synthetic task, no LLM**, where p and ∇p are exact;
  - every hypothesis is a reading of the same equation.

---

## Slide 5: The setting: one task, a ladder of sharing

- A prompt is a feature vector, and the answer is L tokens. Reward 1 if all of them are correct
- The same task at increasing amounts of parameter sharing:

  | Level | Policy | Sharing |
  |---|---|---|
  | tabular | each prompt has its own logits | none |
  | linear | logits = W·x | shared weights |
  | linear, correlated features | prompts look alike | more |
  | MLP (width 64 or 256) | shared nonlinear features | most |
- Plain SGD, exact or sampled updates, many seeds
- *Figure: `figures/out/s05_ladder.png`*

---

## Slide 6: Five hypotheses (as planned in the milestone)

| | Hypothesis | Source |
|---|---|---|
| H1 | The identity holds at finite G | all three papers |
| H2 | Group size changes what GRPO optimizes (from p toward 2 arcsin√p) | Thrampoulidis |
| **H3** | **One effective step, ηW_G·p(1−p)/B, plus interference, decides stability** (main study) | Suk & Duan |
| H4 | KL sets where training stops (finite-G p*) | Mroueh |
| H5 | Group size and escape from p ≈ 0 (separate study) | Bay & Yearick, Nie et al. |

---

## Slide 7: What we ran

- **Base batch:** about 20,000 runs covering H1–H5, on the planned grid
  - H3 alone: 13,608 runs (6 sharing levels × L × batch/group pairs × 3 rules × 12 step sizes × 3 seeds)
- **Follow-ups:** about 30,000 more runs to understand what H3 showed
- **Method:**
  - predictions committed before each batch;
  - failed predictions stay failed in the write-up;
  - each batch was independently reviewed.

---

## Slide 8: H1 — the identity holds

- 414 configurations, sampled updates averaged and compared with W_G·∇p: cosine ≥ 0.989, no detectable bias
- Holds for every rule, sharing level and response length
- *Figures: `figures/out/s08_h1_identity.png`; Simon's norm plots in `figures/out/s08_h1_norms_tabular/` and `s08_h1_norms_mlp/`*

---

## Slide 9: H2 — group size changes the objective, but only without sharing

- Tabular: each prompt moves exactly as W_G predicts (slope 1.000, R² 1.000)
- GRPO's weight moves toward the arcsin surrogate as G grows: λ = 0, 0.02, 0.22, 0.63 at G = 2, 4, 16, 64
- **With sharing, over 90% of a prompt's movement comes from other prompts**
  - Its own W_G no longer decides its progress, but the full shared drift still does (R² 0.999)
- *Figure: `figures/out/s09_h2_sharing.png`*

---

## Slide 10: H4 — KL fixes where training stops

- Measured p* matches the finite-G fixed point: median error 1e-4 over 960 runs, both KL directions
- Example: p_ref = 0.05, β = 2, G = 4 gives 0.109, as predicted (Mroueh's G → ∞ limit gives about 0.17)
- New: at large G there can be **two** stable points, and SGD keeps the side it starts on (60 of 60 runs)
- Failed: our scaling claim for the spread around p* (∝ η/β)
- *Figure: `figures/out/s10_h4_fixed_points.png`*

---

## Slide 11: H5 — group size is not a lever

- Our first "best G" table was an artefact of the step-size grid
- With a wide grid, the best G is 2 almost everywhere
- Group size does not move the collapse threshold
- Fixes for all-wrong groups (Heckel et al.) help only at small steps

---

## Slide 12: H3 as planned — it failed

- **Without sharing nothing ever collapses:** 2,268 tabular runs, success never drops
- **With sharing, no single effective step lines up the levels:** off by up to 4.7 orders of magnitude on held-out levels
- *Figure: `figures/out/s12_h3_thresholds.png`*

---

## Slide 13: Why it failed — two different routes to collapse

1. **Large kicks.** Each sampled prompt gets a big step in one lump, so prompts are knocked out. This is a step-size effect, close to Suk & Duan's picture
2. **Majority pressure.** With correlated features, prompts collapse even at the **smallest** step size and under the exact (noise-free) update
   - The prompts lost are exactly those whose correct answer is a **minority** class
- H3's effective step only sees a prompt's own update; route 2 is driven by the other prompts' updates

---

## Slide 14: H3 restated

- **Original:** one effective step per prompt decides stability. *Failed*
- **Restated:** with shared parameters, a prompt's progress is set by the drift from all prompts,

  Δp_y ≈ (η/B) Σ_x W_G(p_x) ⟨∇p_y, ∇p_x⟩

  When prompts form answer classes, **a majority class pushes a correct minority down and absorbs it**
- Both versions go in the report, kept apart
- The restated version: exploratory so far; its replication on 20 fresh tasks is *pending*

---

## Slide 15: The push, measured and derived

- **Measured (T11):** from one state, step along the majority's part of the update, the minority's part, or both
  - the majority's step lowers the minority; the minority's own step raises it;
  - the push is 4–5× the minority's gain at 70% majority and 11–17× at 90%.
- **Derived (the odds law):** at the first step, push ratio R = c · m/(1 − m)
  - m is the majority share, c comes from feature similarity;
  - verified against 1,080 runs with nothing fitted (median error 0.14%), and proved in Lean;
  - it describes the first step only, not the whole run.
- *Figure: `figures/out/s15_push.png`*

---

## Slide 16: Dose–response and why the collapse lasts

- **More imbalance, more collapse** (16 of 16 cells; holds under SGD and Adam)
  - 54–100% of minority prompts lost at 90% majority; about 0 with balanced classes
- Supervised training on the same task loses nothing
- **Why the minority stays down:**
  - once success is near 0, its own signal vanishes;
  - the majority keeps pushing;
  - remove the majority's reward and the minority comes back, slower the deeper it fell.
- *Figures: `figures/out/s16_dose_response.png`, `figures/out/s16_trajectory.png`*

---

## Slide 17: What protects the minority

| Lever | Effect | Cost |
|---|---|---|
| Forward KL to the start policy | no prompt absorbed from β = 0.1 (fully at both thresholds) | caps final success |
| Reverse KL | nothing at β ≤ 0.1 | works only near β = 0.3–1 |
| pass@k weighting (k = 4) | minority kept while the majority learns | majority 4–40× slower |
| Projecting out the harmful part | removes the collapse | needs class labels |
| Larger group size | no effect | — |

- Each paper's knob, read under sharing: the objective (pass@k), the KL anchor, the step size
- *Figure: `figures/out/s17_levers.png`*

---

## Slide 18: It carries over to a transformer

- Li, Pandey & Chen's environment: token prompts and a shared 2-layer transformer. We replicated their paper first
- With our correct-answer classes:
  - the same minority collapse appears, only with sharing;
  - the odds law holds there too, again with nothing fitted;
  - forward KL protects the minority there as well.

---

## Slide 19: What is known, and what is ours

- **Known:**
  - the identity and the finite-G surrogate (Davis & Recht);
  - fixed-point equations (Vojnovic & Yun);
  - interference and plateaus through shared parameters (Schaul et al., 2019);
  - the first-step decomposition (Li et al.).
- **Ours:**
  - exact tests of these predictions on one controlled task;
  - the majority share as a controlled driver of collapse of a *correct* minority under GRPO;
  - the kernel correction c in the odds law;
  - the levers compared at matched progress.

---

## Slide 20: Limitations and next steps

- **Limitations:**
  - a toy: answer tokens are independent given the prompt;
  - binary rewards only;
  - the collapse needs a start that does not yet separate the classes.
- **Pending:** confirmatory replication (A1–A5) on 20 fresh tasks, about 180 CPU-hours
- **Not tested:** whether this happens in real LLM training

---

## Slide 21: Takeaways

1. The per-prompt theory is **exact without sharing** (H1, H2, H4)
2. With sharing, **one effective step does not decide stability** (H3 as planned failed)
3. What decides it is **the drift from other prompts**: a majority answer class absorbs a correct minority, with a strength set by the odds of its share
4. **Forward KL and hard-prompt weighting** prevent it, at a cost; group size does not
