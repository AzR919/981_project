# Critique of NEXT_PLAN.md (2026-11-20)

Checks against code and caches were done with small read-only scripts. Nothing in the repo was changed apart from this file.

## 1. Verdict

The plan is honest and well organised. Pre-registration, the definitions in T0, flow-time grids and supervised controls are the right instincts. But its three central controls do not isolate what they claim:
- **T1:** the "centred" arm removes the correlation itself, and balancing changes the margin.
- **T2:** the "matched-step" pair is mismatched by n/B in total displacement.
- **T3:** route 1 is already refuted by the base data.

The LLM block's definition of "lost" has a noise floor that is near zero by construction, so it validates nothing. Evaluation, not training, will eat the GPU budget. Five days cannot hold Phases A, B and C plus a second four-way review. Fix T1, T2, T3 and C0, then cut C4, C6, T10 and most of T3's ambition.

## 2. Critical issues (fix before running)

**C1. The T1 controls are confounded or self-contradictory (§3 T1).** [verified]
- **Problem 1: `centered=True` "keeps correlation" is impossible.** Subtracting the shared component removes the correlation. On the five c09 tasks, the mean pairwise feature cosine falls from 0.84–0.92 to −0.002 after centring. This arm is just the `gaussian` level.
- **Problem 2: balancing changes difficulty and geometry.**
  - **Margin.** The natural balanced rule projects the rule orthogonal to the shared vector c. That gives a median geometric margin of 0.08 for every prompt. The imbalanced tasks have 0.12–0.33 for majority prompts and 0.10–0.19 for minority prompts.
  - **Number of classes.** Imbalanced c09 tasks populate 2–5 of the 9 answer classes; balanced ones populate all 9.
  - **Rarity and margin are confounded inside a task.** Minority prompts sit closer to the decision boundary (task 3: 0.10 against 0.33). So "minority-answer collapse" may partly be "small-margin collapse".
  - **The decision rule is biased.** Balanced tasks are lower-margin, so "balanced still absorbs, so general interference is back" (§7) cannot be told apart from "harder tasks absorb".
- **Problem 3: the imbalance levels are undefined.** "Majority share m ∈ {0.4 … 0.95}" has no stated construction for 9 joint classes.
- **Fix:**
  - Fix ρ, the noise draws g and the orthogonal rule W⊥.
  - Create imbalance only through a scalar offset α·(W c) added to the rule scores, sweeping α to reach the target m. Then the discriminative geometry is constant across arms.
  - Report the margin distribution per arm. Fit per-prompt absorption ~ rarity + margin + p_init, with task random effects.
  - Drop the "centred" arm, or rename it what it is: a ρ = 0 control.

**C2. The T2 "matched-step" pair is not matched (§3 T2).** [verified, `run.py` `expected_step`]
- **Problem:** `expected` at η equals `expected_full` at η *in expectation*, which is already flow-matched. `expected_full` at η·n/B moves every prompt with weight η/B. That makes the total parameter displacement, and every cross-prompt term, n/B = 16× larger per step.
- **Why it matters:** your own H2 result says over 90% of a prompt's drift comes from other prompts. So the pair matches the minor term and mismatches the dominant one. It does not "isolate batch composition".
- **Further mismatches:**
  - W_G(p) and ‖∇z‖² evolve differently in the two arms, so s is matched only at step 0.
  - The "full-batch `sampled`" cell draws `rng.integers(n, size=B)` with replacement. At B = n only about 63% of prompts are distinct, so batch-composition noise remains.
- **Fix:**
  - Use a 2-D design: η grids for `expected` and `expected_full`, analysed at matched own-step and at matched flow time.
  - Add `expected` with without-replacement epochs. It has the same per-prompt step as `expected` and the same per-epoch flow as `expected_full`, and differs only in ordering noise.
  - Use a permutation for the B = n sampled cell.

**C3. T3 route 1 is pre-refuted, and the pass bar is weak (§3 T3).** [verified, base FINDINGS H3 §3]
- **Problem 1: route 1 has already failed.** Route 1's variable (η/B)·W_G·‖∇z‖² is candidate (ii) of the failed H3 collapse. Its level shifts are lin +1.37 and m64 −2.75 decades: about 4 decades apart, or 3.6 at L ≤ 2. Both are overshoot levels, so one constant s* cannot fit them. Passing would require a separate s* per architecture, which is the tuning the plan forbids.
- **Problem 2: the "held-out" configurations interpolate.** MLP-128 sits between m64 and m256, whose shifts are nearly identical. ρ = 0.7 sits between c05 and c09.
- **Problem 3: the pass bar is already met.** s̄ alone already reaches held-out AUC 0.78–0.80. A per-architecture logistic in log η would probably clear AUC ≥ 0.8.
- **Problem 4:** "a third route is needed for < 10%" has no operational definition.
- **Problem 5: s* cannot be fitted from the caches.** "Fitted on base data only" under the new definition (persistent, τ = 1e-3) cannot use the caches, which hold only p_final, min_p and absorbed_at. It needs the raw files, whose snapshots are every 500 steps: three points in the last 20%.
- **Problem 6: τ = 1e-2 drops L = 3 entirely.** With the start rule p ≥ 10τ, uniform-init L = 3 prompts (p = 1/27 ≈ 0.037) are excluded at τ = 1e-2, and L = 3 is in the held-out set.
- **Fix:**
  - Pre-register the baseline the model must beat: a logistic in log η, with architecture fixed effects.
  - Use extrapolative held-out configurations: width 16, ρ ∈ {0.3, 0.95}, K = 5, d = 32.
  - Define a third route operationally, for example "absorbed with s < s*/10 and a minority share above 0.5".
  - Or downgrade T3 to descriptive.

**C4. C0's noise floor cannot fail, and "lost" mixes absorption with ordinary decline (§4 C0).** [verified arithmetic]
- **Problem:** for a problem that truly stays at p ≥ 0.05, P(0/128) ≤ 0.0014; at p = 0.1 it is 1.4e-6. Two evaluations of the base model will therefore show about 0 false losses, whatever the real error sources are.
- **The real threats are different:**
  - **Genuine partial declines.** A drop from 0.1 to 0.02 gives 0/128 with probability 7.5%.
  - **Answer-format or extraction drift,** a well-known RLVR effect on Instruct models.
- **Fix:**
  - Classify every sample as parseable or not, and report losses only among parseable outputs.
  - Report the full before/after scatter, with "declined" (upper bound < p_before/2) kept separate from "lost".
  - Use about 300 samples for the after-evaluation of candidate problems (0/300 gives an upper bound near 0.01), sampled sequentially: stop at the first correct sample.
  - Use an untrained-on-these-problems model (the held-out split) as the null for training-induced change, not a re-evaluation of the base model.

**C5. The evaluation budget is unplanned (§4, §6).** [reasoned]
- **The arithmetic:**
  - One evaluation is 128 × 2,500 = 320k samples × about 250 tokens ≈ 80M tokens.
  - At a plausible 5–15k tokens/s for 0.5B in vLLM on a 4090, that is 1.5–4.5 GPU-hours per evaluation.
  - About 70 runs (C1, C2, C4 and C5, plus SFT controls) need 100–300 GPU-hours of evaluation alone.
  - A training run is about 154k samples, half an evaluation.
- **Available capacity:** the 03:00 wipe, the model reloads and shared machines leave far fewer usable GPU-hours than 12 × 24 × 3.
- **Fix:**
  - Evaluate after-training only on problems with p̂_before ∈ [0.05, 0.7], sequentially.
  - Evaluate before-training once per problem set, not once per run.
  - Budget from C0's measured throughput, with an explicit cut list.

## 3. Major issues (should fix)

**M1. C4's W_G ratio test contradicts your own data (§4 C4).** [verified, review D-M6 and SUMMARY]
- **Problem 1:** the toy showed that rules follow W_G per prompt only without sharing. m64 fails (R² 0.42), and sharing carries over 90% of drift. An LLM has far more sharing, so your own results predict that per-bin Δp̂ ratios will *not* match W_G ratios. The test is not identifiable.
- **Problem 2: G changes more than G.** At a fixed 512 samples, G = 2 uses 256 prompts per step against 32 at G = 16, so each problem is revisited a very different number of times.
- **Problem 3: Dr. GRPO differs in more than W.** It also changes length aggregation.
- **Problem 4: two seeds** is too few for bin-level Δp̂.
- **Problem 5: low payoff.** Ghosh et al. is a mean-field preprint (Assumption 1; `papers/notes/2026_Ghosh_…`). That GRPO depends on G follows directly from Davis & Recht's W_G, so refuting Ghosh is not worth 2 GPU-days.
- **Fix:** cut C4, or keep only GRPO against RLOO at G ∈ {4, 16}, with prompts per step fixed and a qualitative "shift toward low-p bins" outcome.

**M2. C2's toy-to-LLM mapping is unstated, and its signature needs a base rate (§4 C2).** [reasoned]
- **What the toy collapse is:** a constant answer *sequence*, driven by a shared feature direction acting as a bias.
- **What the LLM analogue would be:** in GSM8K-style problems every answer is a different number, so the analogue is the *procedure*, not the answer. Say so, and derive what the mechanism predicts under that mapping.
- **Difficulty:**
  - Add-or-subtract word problems are likely near ceiling for Qwen2.5-0.5B-Instruct, so there will be no mid-range p. Pilot the base p̂ first.
  - Stratifying on base p̂ selects different operand ranges per procedure, which is another confound.
- **The signature needs a base rate.** "Majority-procedure answer among the errors" must be measured on base-model errors first, because operation swaps are common base errors.
- **Swapped roles** cancel only additive difficulty differences.

**M3. The statistics plan is mis-targeted (T0, T5).** [reasoned]
- **Multiplicity:** Holm "across all verdicts in a batch" makes the family size depend on what happened to launch together. Name 4–6 primary confirmatory hypotheses (T1, T2, C1, C2 and the T9 headline), Holm over those, and label the rest descriptive.
- **Unit of replication:** the task, not the run or the prompt.
  - Prompts within a run are dependent (shared parameters), so Kaplan–Meier confidence intervals must come from a bootstrap over tasks.
  - "Run fixed effects" in T5 absorb nothing harmful within-run, but give no inference about tasks. Use task random effects.
- **Kaplan–Meier is the wrong tool here:**
  - "Ever-absorbed" is not an absorbing state in lin and c05 (50–83% recover).
  - First-passage Kaplan–Meier measures dips. Model the onset of persistent absorption, or use a multi-state or competing-risks model.
- **Power:** 10 tasks is enough for the large between-task effects (minority share ranges 0.05–0.65). It is thin for a logistic threshold confidence interval, and for ΔAUC = 0.03 in T5. Justify the 0.03 or simulate power.

**M4. T4's stopping rule may never terminate in the tabular case (§3 T4).** [reasoned from the H3 proposition and D-H2]
- **Problem:** with tabular softmax and a zero-sum rule, p never falls at any η. So samples-to-target should be non-increasing in η, and an interior best η need not exist. D already found η at the top of the grid in 28/28 cells.
- **Fix:** define best-η as saturation (within 5% of the asymptote), or fix η by a stability criterion. Do not loop "until interior".

**M5. T9's predictions are asserted, not derived (§3 T9).** [verified, `theory.py` docstring]
- **The fixes are reweightings:** every listed fix depends only on rewards, so by the identity each one is a reweighting W_G → W′. Under `expected` and `expected_full` its effect is exactly computable with zero runs.
- **Derive the flow time T*(W′) for imbalance collapse analytically, and use the runs as confirmation.** "Should not prevent imbalance collapse" is not obvious. The fixes increase low-p prompts' weight in the shared direction and could delay collapse substantially.
- **Negative-sample reinforcement** has W ≡ 1, the mean-rule weight, so it is not an "all-wrong-group fix".
- **"Matched effective step"** is ill-defined when W′ differs in shape over p.

**M6. Limitations the plan misses or leaves untested.** [reasoned unless marked]
- **Adam:** LLMs use Adam. The toy's Adam block already showed m64 absorbing under sampled Adam at η ≈ 0.01, while the exact flow does not (C §2). That noise × Adam interaction is untested outside T10. Add an Adam arm to T1 and T2.
- **TRL loss aggregation:** `GRPOTrainer` defaults (loss type, `scale_rewards`, token against sequence averaging, clipping) differ from the toy rules. "Rules identical to the toy" holds only in the custom loop. Pin them.
- **Qwen-specific effects:** Shao et al. 2506.10947 (in `papers/notes`) find that Qwen2.5 gains under spurious and random rewards. C5's prediction "verifier flips increase losses" may fail for Qwen-specific reasons. GSM8K contamination makes "lost" ambiguous between memorised and reasoned answers. The second model family should be mandatory for C1 and C5, not optional.
- **KL bistability:** it gets no transfer test (β = 0 everywhere). Either run one β > 0 arm, or state that its LLM relevance is untested.
- **Sampling:** temperature 1.0 for both training and evaluation is fine. Fix top-p and top-k at 1 and 0, and say so.
- **Scale against parameterisation:** 1.5B with LoRA against 0.5B fully fine-tuned confounds scale with parameterisation.

**M7. Novelty positioning is under-budgeted (W1).** [reasoned]
- **Close prior work:**
  - Schaul et al. 2019, "Ray interference": on-policy learning plus shared parameters gives winner-take-all across contexts. This is very close to minority collapse.
  - Diversity collapse, pass@k narrowing and entropy collapse: Yue et al., Wu et al., Yuan 2606.15455, Cui 2505.22617.
- **Schaul is not even in `papers/notes`.**
- **Fix:** budget half a day. State what the exact setting adds: identifying majority collapse as gradient-flow-driven, with no noise and no step threshold, plus the supervised contrast.

## 4. Minor issues

- **A1 is a check, not a test.** "λ rises monotonically with G" follows from the exact W_G, which the measured W already matches to within 1%. Label it a check, and fix the p-range of the fit, because λ depends on it.
- **The headline threshold needs raw files.** Persistent absorption at τ = 1e-3 on base data needs the raw files, not the caches [verified: the caches lack snapshots].
- **lr 1e-6 for 300 steps may barely move a 0.5B model** beyond format. Make C0 include a 100-step pilot that must move pass@1 by at least 3 points. Otherwise raise the lr to 2e-6–5e-6 before C1.
- **GSM8K difficulty is acceptable for 0.5B-Instruct** (greedy accuracy is around 50%, so plenty of mid-range p). The 1.5B model will have far fewer low-p problems; report the base p̂ histograms.
- **T10 needs a pre-registered success criterion,** or it cannot fail.
- **Day 5's second four-way review consumes the writing day.** Use one adversarial reviewer.
- **Retire the "c09 η*" label everywhere, not only in T1/T3.** It is a horizon artifact.

## 5. What is sound (keep)

- Pre-registration committed before launch, with conclusions stated for both outcomes.
- The T0 definitions:
  - three absorption forms;
  - start ≥ 10τ;
  - separate task and init seeds;
  - time-to-event in flow time;
  - the analysis unit tests and golden files.
- The flow-time grid (η·T fixed) in T1, which directly answers D-M4.
- The supervised controls in T1 and C2; swapped roles in C2.
- T7: exact-enumeration calibration, with drop-if-it-fails.
- T8: a noise-free KL control.
- A2, A3 and A6: honest re-scoring.
- Dropping Huang reference resets.
- C0 before anything else.
- Reporting C1 even if it is null.
- Using `calculate_advantages` in the LLM loop.
- The disk and wipe handling.

## 6. Cuts and reprioritisation (5 days)

**Cut first:** C4, C6, T10, A5, A7, the second model family's headline-only role (replace it with a mandatory second family on C1), and T3's predictive-law pass test (make T3 descriptive unless C3's fix is adopted).

**Schedule:**

| Day | Work |
|---|---|
| 1 | T0; A1–A4 and A6; GPU environment; C0, including the lr pilot, the format check and measured throughput. |
| 2 | Redesigned T1 (offset-controlled imbalance, plus an Adam arm); redesigned T2; T4 with the saturation rule; T6–T8. C1 on 0.5B, 3 seeds. |
| 3 | T9 (analytic predictions first), T5. C2 after the difficulty pilot. |
| 4 | C5, reduced to: verifier flip 20%, lr ×10, the sign rule. Analysis. |
| 5 | Writing: related work (Schaul, diversity collapse), a limitations section, figures. One adversarial review. |

A grader will expect a related-work section that positions the work against Schaul and the diversity-collapse literature, an explicit limitations section, and a short statement of what each LLM experiment could and could not show.
