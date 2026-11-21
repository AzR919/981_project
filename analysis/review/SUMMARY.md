# Review summary: milestone, stretch goals, literature, adversarial (2026-11-20)

**How this was produced:**
- Four independent reviews, each by an AI reviewer agent:
  - [a.md](a.md): milestone promises against delivery;
  - [b.md](b.md): stretch goals, planned against executed;
  - [c.md](c.md): adjacent research and gaps;
  - [d.md](d.md): adversarial flaw hunt.
- The main session then re-checked the high- and medium-severity claims against the code and the
  caches. Each claim below is marked **verified**, **corrected** or **rejected**, with the main
  session's own evidence.

## Verdicts

| Question | Verdict |
|---|---|
| (a) Milestone delivery | About 60% delivered as stated. **H1 and H4 are delivered well.** **H3, the headline, was refuted as stated** (no single effective step), and the replacement story needs further corrections (see (d)). **H2 was run as a different question.** The promised p → 2 arcsin √p shift is testable from existing data and points the right way. **H5 passes only through a tie-break artifact.** **The Heckel and Huang side promises were never attempted.** |
| (b) Stretch goals | **Run honestly.** Criteria were committed at 16:25, before the results (`ab9c270`). Every FAIL stays FAIL, and post-hoc text is labelled. But two post-hoc readings went too far: "batch-composition noise" and "three mechanisms". The ablation PASS is over-credited. Never run: H5 all-wrong-group fixes, the Barakat test, reference resets, and the Adam timing fit. |
| (c) Literature | **Known:** the theory we started from (Davis & Recht; Vojnovic & Yun). **Extends prior work:** bistability (Mroueh already notes non-unique fixed points). **Contradicts:** Ghosh et al. Prop. 10 (the trajectory is G-invariant). **The real contribution** is an exact-gradient anatomy of absorption under binary rewards, small models and SGD. 16+ citations are missing from the milestone. |
| (d) Flaws | **Two high-severity, both verified:** c09/c05 "interference drift" is majority-answer collapse, and the H5 best-G table is an arbitrary tie-break. **One high-severity flaw from (b), verified:** the "batch noise" claim is confounded by per-prompt step size. **Dismissed:** denormal flushing (bit-identical runs) and realizability (supervised warm start solves c09). |

## Verified claims (main-session evidence)

| # | Claim (source) | Status | Main-session check |
|---|---|---|---|
| 1 | **c09/c05 absorption is minority-answer collapse** (D-H1) | **verified** | I rebuilt the 5 c09 and 5 c05 tasks with `run.build`. Across all 10 tasks, 0 absorbed prompts have the majority answer. In c09 tasks 0, 2 and 3, every minority prompt is absorbed (178/178, 101/101, 54/54). The 0.197 plateau is task 2's minority share. |
| 2 | **c09 has no step-size threshold:** absorption at a fixed flow time (D-M4) | **verified** | For η ≥ 0.0056, η·t50 = 3.70–4.00 in `expected_full`. Smaller η is cut off by the 5000-step budget, so the "η^−0.89" exponent is a censoring effect. |
| 3 | **The "batch-composition noise" claim is confounded:** `expected` gives each sampled prompt an n/B = 16× larger step than `expected_full` (B-H1) | **verified** | `run.py:159` averages over the batch. The `expected_full` threshold is 17.8× (m64) and 10× (lin) the B = 32 `expected` threshold, close to 16. η* ∝ B^0.83 fits per-prompt step size equally well. The claim is **withdrawn**: m64 and lin absorption is most simply a large per-prompt step (overshoot). |
| 4 | **The H5 best-G table is an arbitrary tie-break** (A-5, D-H2) | **verified** | Tabular GRPO ties at the minimum over G ∈ {32, 64, 128} (p₀ 0.01) and {16, …, 128} (p₀ 0.03); 128 is reported both times. The smallest G at the minimum gives 32, 16, 4, 2, so the trend survives but the table and verdict as computed don't. D also reports the best η at the top of the grid in 28/28 tabular cells (not re-checked). |
| 5 | **The arcsin promise (H2) is testable from existing data** (A-3) | **corrected** | The trend is verified. GRPO's measured weight moves from constant (λ = 0 at G = 2) toward 1/√(p(1−p)) as G grows. My free-scale fit gives λ = 0, 0.015, 0.215, 0.627 at G = 2, 4, 16, 64 (A reported 0, 0.229, 0.626, 0.886; the values depend on the parameterization). The measured W matches exact W_G to within 1% (not A's 0.01–0.03%). |
| 6 | **Bistability extends Mroueh** (C) | **verified** | Mroueh 2503.06639, p. 7: "the fixed points are not unique in general, and p* = 1 is almost always a fixed point". |
| 7 | **We contradict Ghosh et al. Prop. 10** (C) | **verified** | `papers/notes/2026_Ghosh_…md:32`: they claim p* and the deterministic trajectory are G-independent. Our H2 short-training success varies with G up to 1.3× (m64 GRPO), while RLOO is flat in G (W ≡ 1). |
| 8 | **c09 victim count is set by the task seed** (B) | **verified** | At η = 0.01 the five seeds absorb 0.35, 0.54, 0.20, 0.11 and 0.05, matching each task's minority share. |
| 9 | **Momentum and Adam "decreases" are round-off** (D-M5) | **verified (data)** | Ablation table: max_drop is 0 and max absorbed is 0 for both, with a median of 1.5 and 4 decreases against 10³–10⁵ for real breaks. D's exact-arithmetic sign argument was not re-derived. |
| 10 | **Denormal flushing is harmless** (D, dismissed) | accepted | D's runs with and without flushing were bit-identical on 3 configurations. Not re-run. |
| 11 | **Most lin/c05 "absorbed" prompts recover** (D-M3, A-1) | accepted | A's final/peak absorbed ratio: c05 0.58, lin 0.15; D's 50–83% recover. This is consistent with `max_absorbed/absorbed` in the caches. |

## What we can and can't claim now

**Safe:**
- The identity and finite-G surrogate replicate Davis & Recht. Cite them; this isn't new.
- The finite-G KL fixed points match Vojnovic & Yun's equations to about 1e-4 over 856 converged runs.
- GRPO with KL(π‖π_ref) at G ≥ 16 is bistable under SGD:
  - the basin boundary sits at the predicted unstable root;
  - noise switches between states only at a small barrier and large η.

  This is a finite-G, SGD extension of Mroueh.
- With a deterministic verifier, a zero-sum rule, SGD and no sharing, p never decreases (2,268 runs).
- A noisy verifier and the sign rule (Nie et al.'s fix) cause absorption.
- Sharing makes over 90% of each prompt's drift come from other prompts. Without sharing, rules follow W_G exactly.
- Short-training success depends on G (against Ghosh et al. Prop. 10), and RLOO doesn't.
- GRPO's weight moves toward the arcsin surrogate as G grows (H2 as promised; from existing data).

**Must change:**
- **H3: drop "three mechanisms" and "batch-composition noise".** The supported reading has two
  routes:
  - **Large per-prompt steps** knock prompts out within a few steps. The threshold scales with η/B;
    tested on MLP, linear and c05.
  - **Answer imbalance under shared features.** The gradient flow itself, with no noise, collapses
    minority-answer prompts onto the majority answer at a fixed flow time (c09, partly c05). There
    is no step-size threshold, and supervised learning keeps the same prompts.
- **Predictor:** say "predicts at the level of an answer-rarity baseline". On c09/c05 the rarity
  AUC is 1.000/0.913 against drift 0.999/0.897 (D).
- **H5:** recompute with a smallest-G tie-break. Report it as tabular-only, with η at the grid edge.
- **H4(b) and the H1 medium tier:** report as calibration limits, not passes.
- **Milestone wording:**
  - lines 258–260 and 322 (Davis & Recht are finite-G);
  - H3 at 309–311 and 316–319;
  - the threshold remark at 289–292 (GRPO's lower threshold appears only at large G);
  - Heckel and Huang at 325–331 (deliver them or drop them).
- **Citations:** add the 16 keys listed in [c.md](c.md) (all already in `papers/bib`), plus Song et al.
  2607.01083. Merge the two bib files. Check Schaul et al. 2019 "Ray interference" before calling
  any interference finding new; it may anticipate the majority-collapse result.

## Prioritised to-do

| # | Task | Why | Cost |
|---|---|---|---|
| 1 | **Balanced-answer c09/c05 control:** equal answer classes, or centred features. Plus the rarity baseline in `predictor_scores` and absorbed-set against minority-set tables. | Decides whether the main "interference" result is a general dynamics effect or an answer-imbalance effect. Highest value. | ~1 h of code plus about 300 runs (about 2 CPU-hours) |
| 2 | **Fix the H5 tie-break** (one line in `analyze_h5`); extend the η grid above 10; log every 5 steps for p₀ ∈ {0.1, 0.3}. | The current table is wrong. | analysis 5 min; rerun about 600 runs (~3.5 CPU-hours) |
| 3 | **H5 all-wrong-group fixes:** Heckel weights, a virtual correct sample, ±α advantage. Rerun the absorbing configurations. | The only causal test of the absorption mechanism, and promised in the milestone. | ~0.5 day of code plus about 500 runs |
| 4 | **Report the arcsin λ table** (H2 as promised) and **the Bay & Yearick fit** on the H5 caches. | Delivers milestone promises with no new runs. | ~1 h |
| 5 | **Lost problems from the stretch caches** (pass@1 up, pass@64 down; B's numbers). | The coverage-narrowing link (Yue et al.), from existing data. | ~15 min |
| 6 | **Response-noise isolation:** a full-batch `sampled` cell (B = 512), against `expected_full`, at matched per-prompt step. | Only needed if the noise question stays in the report. | ~200 runs |
| 7 | **Barakat test write-up** from the base H2 data (B: 0/138 one-step runs lower pass@1). | A negative result, cheap. | ~1 h |
| 8 | **Drop reference resets**, or list them as future work. | Low value under binary rewards (A, B). | 0 |
