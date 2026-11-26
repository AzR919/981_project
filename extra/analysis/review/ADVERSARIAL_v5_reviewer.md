# Adversarial review v5 (2026-11-23)

**State reviewed.** `analysis-v3` at `d7c5feb` and `review` at `f1f97f0`. Where raw results were on
this machine (`~/rlvr-data/v3g`, `v3h`, `llm/c1q`), the reviewer recomputed numbers with the
project's own readers (`experiments/analyze_v3g.py`, `llm/declines.py`).

**Terms.** *Absorbed*: a prompt whose success stays below τ = 1e-3 over the last 20% of steps (the
T0 definition). *Per-prompt rate*: the share of a class's prompts that are absorbed. *Class-level*:
at least half of a class's prompts absorbed. *Minority*: every class other than class 0.

## Findings ranked by severity

| # | Severity | Claim and where it is made | Problem | Cheapest fix |
|---|---|---|---|---|
| 1 | High | "Forward KL prevents absorption at every strength tried, including β = 0.01"; "with the MLP under forward KL every prompt, minority included, ends at the single-prompt resting point" (v3g, T17 Reading); ledger H4 | True only at τ = 1e-3. At the registered sensitivity threshold τ = 1e-2, β = 0.01 absorbs 34% of minority prompts | Re-analyse existing data |
| 2 | High | "The KL direction decides whether the minority survives" (ledger H4; v3g; v3i) | Shown only with a reference that gives the minority's answer 1/K (toy) or 0.3 (T20); this is Schaul et al.'s f(0) > 0 remedy | Reword; new experiment (about 120 CPU runs) |
| 3 | High | T20: "Balanced classes, Adam lr 1e-3: neither class absorbed in 18"; "the clean evidence … 118 of 120 runs"; "only when parameters are shared" (v3h) | Every balanced Adam run absorbs 17–20% of each class's prompts; SGD lr 0.3 fails the balanced control | Re-analyse existing data; reword |
| 4 | High | "A majority answer class drives the minority to near-zero success (P1; 16 of 16 cells)" (ledger H3; story) | P1 tests a rise with m, not a level; per-prompt rate at m = 0.9 is 0.54 in two cells. Regresses ADVERSARIAL_v4 item 3 | Reword |
| 5 | High | "The collapse needs classes whose updates oppose each other" (C2 Reading); "here c has the other sign" (C10); "in two small LLMs the same pressure does not appear" (story) | Toy and T20 start class-agnostic; the LLM base gives the majority's answer on minority problems 6–57 times less often. The start alone predicts a weak push | Reword; new experiment (160 CPU runs, or 8 GPU runs) |
| 6 | Medium | Novelty sentences (ledger H3; v3h "New: …"; v3g "Why") | Multi-step regression, f(0) = 0, 1/ε plateaus and the f(0) > 0 remedy are Schaul et al.; zero drift at m = 1/K and step-size sign changes are Li, Pandey and Chen | Reword |
| 7 | Medium | Odds law "parameter-free, checked on 1,080 runs"; "New here: the odds law as an identity" (ledger; `odds_law.md`; v3h; v3i) | A first-order identity verified numerically, not a law of training; in T20 c = K − 1 by construction | Reword |
| 8 | Medium | "Sampled updates block recovery that the exact update allows" (v3h, v3i table); "at least 10× longer when features are strongly correlated" (v3d) | Exact-update recovery appears only with Adam | Reword |
| 9 | Medium | C10: R = −0.79 against "+4 to +17 the other way at this imbalance"; additivity prediction | 8-bit AdamW makes the arms non-additive; toy values quoted are for m = 0.7–0.9, C10 is m = 0.95 | Reword |
| 10 | Medium | Story: "training loses no problem it trains on"; "a small, repeatable loss of unseen problems" | Holds at lr 1e-6 only; C2/C10 one family; held-out declines are mostly different problems per seed | Reword |
| 11 | Medium | P5: "The failure is not a near miss in substance either" (C2) | It is a near miss and both correlations are positive | Reword |
| 12 | Medium | "Absorption from" tables (v3e) and "smallest share with absorption" (v3g T19) | Two definitions give different thresholds for the same configuration | Re-analyse existing data |
| 13 | Medium | T18 "time to return rises with 1/p (Spearman 0.81)" (v3g) | Pooled over release steps; within a release step ρ = 0.38 and 0.47, p = 0.10 and 0.07 | Re-analyse existing data; reword |
| 14–19 | Low | See the table of low findings | Number slips, stale notes, an undisclosed Holm input | Reword |

## High findings

**1. Forward KL's floor sits just above τ.** PREREG_v3a registers τ = 1e-2 as the sensitivity
threshold; v3g reports only τ = 1e-3. Recomputed from the raw T17 results, forward KL absorbs 0.011
of minority prompts at τ = 1e-3 but **0.339 at τ = 1e-2** at β = 0.01, and 0.001 against **0.109**
at β = 0.03; at β ≥ 0.1, none. Worst cells at τ = 1e-2: linear, η = 0.1, β = 0.01 (0.885 exact,
0.852 sampled); MLP, sampled, η = 1, β = 0.03 (0.662). In that MLP cell (`t17_runs.csv`) the
minority ends at 0.005 in 8 of 10 tasks at β = 0.03 and 0.05 in 8 of 10 at β = 0.3, not at the
stated resting points 0.992–0.920. Overall, 58 of 320 forward-KL runs at β ≤ 0.3 end with the
minority below 0.05. The pre-registration's own heuristic predicts this floor: p ≈ β·p_ref/c ≈ 0.005
at β = 0.03. *Fix:* report τ = 1e-2 and the final minority mean; say "forward KL keeps the minority
above a floor of about β·p_ref/c".

**2. The KL result depends on the reference.** The reference is the uniform start in T17–T18 (p_ref
= 1/3) and the class-agnostic base in T21 (0.3 per class string, `class_target`). Forward KL's
gradient on the correct logit, β(π − π_ref), stays near β/3 as p → 0. That is Schaul et al.'s f(0) >
0, and their uniform-data remedy is the same pull. Li, Zhou et al. (2509.07430) and GX-Chen et al.
(2510.20817) (in `papers/notes`) are not cited in v3g. *Fix:* say "an anchor to a reference that gives
the minority's answer probability p_ref". New experiment: T17 at m = 0.9 with the reference set to
the plain-GRPO policy at step 100 (minority near 1e-3); 2 directions × 3 β × 2 architectures × 10
tasks, exact update: 120 CPU runs.

**3. T20's balanced control and "clean" cells.** In `v3h/cells.csv`, balanced tasks, exact update,
Adam lr 1e-3, "seeds with none absorbed" is **0/10 for both K**; per-prompt rates are 0.17–0.20 in
each class, and final class means are 0.78–0.80. "Neither class absorbed in 18" uses the class-level
criterion, which prediction 1b (no minority prompt absorbed) does not use. Runs with a class-level
loss at balance: 2 of 20 for Adam 1e-3, 11 for Adam 3e-3, **17 for SGD lr 0.3**, 20 for SGD lr 1 and
3. At high share, SGD lr 0.3 also absorbs class-0 prompts in 11 of 40 runs. By the authors' own
test, SGD lr 0.3 belongs with lr 1 and 3. Only Adam lr 1e-3 isolates the effect: the per-prompt rate
rises from about 0.2 at balance to 0.85–1.00 at share ≥ 0.875. The per-prompt-copies control ran
only there (9 runs). *Fix:* cite that cell alone, as a rise from about 0.2 to about 0.9; word
prediction 1a as "passed as registered, but the registered cells break the majority too; the
isolating cell was not named in advance".

**4. The headline sentence.** P1 tests that absorption at m = 0.9 exceeds absorption at m = 1/3. In
`P1_TASK_LEVEL.md` the per-prompt minority rate at m = 0.9 is 0.54 in both linear ρ = 0.5 Adam cells
(0.18–0.21 at m = 0.7). *Fix:* restore v4's "absorbed, more completely as m grows", from a start
where every prompt gives every class 1/K.

**5. The uniform start is the untested alternative for the LLM gap.** Every toy run with m > 1/3
starts from π = 1/K (no T1 or T11–T19 grid sets `init`), and T20's base is fitted to one target for
every prompt. By the decomposition the summaries use (Li, Pandey and Chen, Lemma 1 and A.4), the push on a
minority prompt scales with the probability it gives the majority's answer: 1/3 in the toy. The C2
base gives the other procedure's answer on **0.58%** (minority `total`) and **5.4%** (minority
`each`) of samples (`c2_runs.csv`, `other_answer_base`). So the start alone predicts the C2 null. No
LLM pair with opposing updates was tested, so "boundary" is a hypothesis. *Fix:* reword. New
experiment, either T1 at m = 0.9 from a warm start where minority prompts give the majority's answer
at most 0.05 (`init: sft_up`; 160 CPU runs), or C2 with a one-token cue (v4 item 14; 8 GPU runs of
about 80 minutes).

## Medium findings

**6. Novelty.** Per `1904.11455.md`, Schaul et al. give multi-step winner-take-all regression of
correct contexts, SL and tabular contrasts, f(0) = 0, O(1/ε) plateaus and f(0) > 0 remedies.
Overclaims: "the multi-step outcome for a correct class … [is] not" there (ledger H3); "New:
recovery depends on having a signal at low success" (v3h, which is f(0) = 0); "the forward-KL pull
does not shrink" (v3g, which is f(0) > 0); T18's 1/p return time (the O(1/ε) plateau). Li, Pandey
and Chen's Prop. 2 (zero leading-order drift at m = 1/K) and section 6.1 (sign depends on step size)
cover "New: balance is a knife edge". That reading is also a post-hoc rescue of failed prediction
1b, and the cleanest cell does not fit it (both classes lose a fifth). *What remains new:* class
share as a controlled driver of a correct minority's absorption under GRPO, with a dose–response and
two thresholds; the kernel correction c < K − 1; levers compared at matched progress; the enumerable
transformer with a correct verifier; the calibrated LLM census and positive transfer.

**7. The odds law.** In any first-order model, m/(1 − m) is just n_A/n_B (the two terms sum over n_A
and n_B prompts). The content is c, computed from the features the simulator differentiates. The
check compares a first-order expansion with a finite step (residual: the second-order term). It
verifies the algebra (the reviewer re-derived it), not a law of training; "1,080 runs" are
evaluations of one identity. In T20, c = K − 1 by construction, and parity (R = 2.93, as random)
does not exercise the correction. "First decline at 1/(1 + c)" needs c at the threshold: c at 1/K
gives 0.340 against 0.354 measured (K = 3, linear, ρ = 0.9).

**8. Adam, not sampling, decides recovery.** On parity, the exact update recovers 10 of 10 with
Adam, but with SGD lr 0.3 it absorbs 92.5% (1 of 10 recovers), like sampled SGD. In T12 at ρ = 0.9,
24 of 80 exact-update Adam runs recover, against 0 of 80 sampled-Adam and 0 of 240 SGD runs
(`t12_cells.csv`). *Fix:* "with Adam the exact update recovers and sampling blocks it; with SGD
neither recovers"; add "under SGD or with sampled updates" to v3d's "at least 10× longer".

**9. C10 magnitudes.** 8-bit AdamW normalises each step, so the 4-prompt arm is not "the minority's
part" of the mixed update: it moves its own panel +0.31 to +0.33, close to the +0.38 to +0.40 of the
76-prompt arm. Additivity was not expected, and R = −0.79 is not comparable with the toy's SGD
ratio, which is about 22–36 at m = 0.95 (T11's 10.7–17.1 at m = 0.9, times the odds ratio 19/9). The
sign result stands (18 of 18 intervals above zero, verified).

**10. LLM scope.** C2 and C10 use one model; at lr 3e-6, C1 loses 15, 12 and 1 training problems.
From the raw Qwen evaluations (reproducing 12, 8, 9 declines; 1, 2, 5 lost), held-out declines
overlap by 1, 2 and 2 problems between seed pairs (0.24–0.36 expected by chance): the rate repeats,
the problems mostly do not. *Fix:* scope the story's LLM sentences to these facts.

**11. P5.** Reproduced: 30 and 4 of 560 assignments (p = 0.0536, 0.0071). P5 fails as registered,
but both Spearman correlations are positive (0.64, 0.86); the counter-reading rests on the
exploratory matched-minority arm (6 runs per role). *Fix:* "fails narrowly".

**12. Two "absorption from" definitions.** v3e uses median > 0.5 and v3g mean ≥ 0.05, giving MLP ρ =
0.5: 0.68 against 0.53, and linear ρ = 0.9: 0.53 against 0.45. The T19 deviation is defensible:
under the v3e definition (`t19_thresholds.csv`) predictions 2 and 3 still fail. Use one definition.

**13. T18.** The pooled Spearman mainly separates release steps 100 and 1,000; report within-step
values.

## Low findings

| # | Claim and where | Evidence | Fix |
|---|---|---|---|
| 14 | v3a SUMMARY: P1 p = 3.0e-8 | No correction note pointing to `P1_TASK_LEVEL.md`. The correction redid part (a) only; part (b), the Spearman test, is still pseudo-replicated. Holm-adjusted 0.049 rests on one cell (8 of 10 tasks) | Reword; re-analyse (b) |
| 15 | Holm table: P2 enters with p = 1 | PREREG_v3a fixes P2's Holm input as the largest kick-size p (0.0027). Using 1 is conservative (P1 would be 0.039) but undisclosed | Reword |
| 16 | v3e "exactly 1/K for two classes"; C1 "1–3% end at ≤ 0.05" in both families; C2 "0.92–0.97"; v3g "group size: 100% at every G", "projection 0% in 16 of 16"; v3f "pass@4 … not at η = 0.03"; C1 "17–27%" | m\* = 0.498–0.529; Qwen 0.3–1.7%; lowest is 0.906 (`r2_27`); 0.91 at G = 2, m = 0.883; 0.00–0.01 at m = 0.9; biased pass@4 reaches 0.80 in 3 and 6 of 10 tasks; the third seed is 100% (one problem) | Reword |
| 17 | Ledger "Open" list; H1 "null is uniform (KS p = 0.73)" | C10 and T20 are finished; v4 asked for "not rejected" | Reword |
| 18 | "Without sharing, success never decreases (2,268 runs)"; T20's 600 tabular runs; T19 prediction 1 | Guaranteed: zero-sum advantages cannot lower a tabular correct logit; at a uniform start G is a common factor | Reword as sanity checks |
| 19 | T13 "prediction 1 holds where it can be tested" | It holds in 8 of 16 cells; the other 8 are ties at zero | Reword |

**Pre-registration moves.** T20 prediction 1a: defensible only as worded in finding 3, without SGD
lr 0.3. Prediction 1b re-cast as "New: knife edge": not defensible as a finding. T18: defensible;
the correction is in the committed addendum, came from a fixed-point calculation, and no T17–T19 run
had been inspected. T19 `absorption_from`: defensible (finding 12).

## Checked and found sound

- LLM: P5 p-values; C2 gains, decline range and the degraded run; switch share 0; Holm arithmetic
  (0.049, 0.21); C1 Qwen counts from raw data; P4; modal-wrong shares; census figures; C10 values.
- Transformer: T20 counts under the class-level definition, the R table and parity numbers; T21 ranges.
- Toy: T11, T15 and T16 tables; T17 medians; T18 return counts; the odds-law algebra (the output
  layer starts at zero, as assumed); M4 terciles are over all prompts, so they do control margin.

## Not checked

- P1 part (b) at task level and the v3a–v3c raw results (not on this machine).
- T21 at τ = 1e-2 (raw v3i not on this machine); Llama per-problem overlap.
- The model behind the unfiltered census table; the count of 2,268 runs.
- Running jobs (rejection-sampling controls, C2 replication, long and samples-per-step arms, C7–C9).
  The story's last sentence and the "boundary" reading depend on the rejection-sampling controls.
