# Plan: close the known limitations, then test what transfers to small LLMs (v2)

**Status:** v2 (2026-11-20), revised after the academic pressure test in
[plan_critique.md](plan_critique.md). §0 records which criticisms were accepted and how each one
changed the plan.

**Inputs:**
- the four-way review [SUMMARY.md](SUMMARY.md) and `a`–`d.md`;
- `analysis/{base,stretch}/FINDINGS.md`;
- `HANDOFF.md`.

**Branch:** `experiments`, local commits only. Every new experiment commits its prediction and
pass/fail rule before launch.

## 0. Changes from v1 after the pressure test

| Critique | Accepted? | Change |
|---|---|---|
| **C1:** T1's controls change the task. `centered=True` removes the correlation (feature cosine 0.84–0.92 → −0.002), and rule-based balancing changes the margins and the populated classes | **Yes** | T1 now creates imbalance with a **scalar class offset on one fixed rule**. Feature geometry and margins stay constant across arms; only the class frequencies change. The centred control is dropped. The sweep uses L = 1 (K = 3 classes), so "majority share" is well defined. Margin is reported as a covariate. |
| **C2:** the T2 "matched-step" pair is not matched. `expected` and `expected_full` at the same η already have the same mean update, and full-batch `sampled` with replacement covers only 63% of prompts | **Yes** (63% checked: 1−(1−1/512)^512 = 0.632) | T2 is redesigned around the right decomposition (§3). This also **corrects SUMMARY item 3**. The difference between `expected` and `expected_full` is not a step-size artifact: at equal mean drift it isolates *per-kick size*, meaning each prompt moves in lumps of η/B when it is sampled. Response noise is isolated by `sampled` against `expected`. The full-batch cell draws without replacement. |
| **C3:** T3 route 1 reuses the refuted collapse variable; the held-out set interpolates; the pass bar is already met by s̄ | **Yes** | T3's predictive-law pass test is **cut**. T3 becomes a descriptive decomposition: for each absorbed prompt, the share explained by the kick-size and imbalance routes. No universal-law claim. |
| **C4:** the C0 "lost" definition cannot fail. P(0/128 \| p = 0.05) = 0.0014, so a duplicate evaluation shows about 0 false losses regardless; real threats are partial declines and format drift | **Yes** (0.95^128 = 0.0014 checked) | C0 now has a **calibrated decline test**: a per-problem binomial likelihood ratio for a drop, with Benjamini–Hochberg FDR across problems, the full before/after transition matrix of p̂, and a separate **format-failure rate** (answers that won't parse) with a lenient parser. |
| **C5:** evaluation dominates the GPU budget (about 80M tokens per full evaluation) | **Yes** (argued; C0 measures it) | Two-stage evaluation: N = 32 samples on every problem, then N = 256 adaptively on candidates (high before, low after, ~10–15% of problems). Evaluation sets: 1,000 train + 300 held-out problems, selected for mid-range base p̂. |
| **Major:** C4 (W_G ratio) is not identifiable under sharing, confounds G with prompts per step, and isn't worth 2 GPU-days | **Yes** | **C4 cut** (moved to future work). The G-dependence claim against Ghosh rests on the toy, where it is exact. |
| **Major:** C2's toy-to-LLM mapping is unstated; add/subtract is at ceiling for 0.5B; no base rate for majority-procedure errors | **Yes** | Mapping stated: toy *answer class* ↔ LLM *solution procedure*. A difficulty pilot tunes the operand size until base p̂ is in 0.3–0.7 for both procedures. The base model's rate of majority-procedure errors is measured before training. |
| **Major:** Holm over "all verdicts" is the wrong family; bootstrap over prompts; Kaplan–Meier on first crossings measures dips | **Yes** | **5 named primary hypotheses** with Holm (§2). Everything else is secondary and labelled exploratory. Bootstrap over task seeds. Kaplan–Meier on entry into *persistent* absorption. |
| **Major:** T4's "extend η until interior" may never stop (tabular p can't fall) | **Yes** | Saturation rule: stop extending when samples-to-target changes by less than the logging resolution. Report "monotone in η" where it is. |
| **Major:** T9's fixes are all W_G reweightings, so their effect is derivable | **Yes** | T9 derives the predicted change exactly from `theory.py` first; the runs confirm it. Grid reduced. |
| **Major:** missed limitations: Adam, TRL loss aggregation, Qwen spurious-reward and contamination, no KL transfer | **Yes**, except KL | T1 gains an Adam arm. The LLM loop uses our own `calculate_advantages` and a stated loss aggregation (sequence-mean, as in the toy), not TRL defaults. **A second model family is mandatory on C1.** KL transfer is **not** added (no budget); it is listed as a limitation. |
| **Major:** novelty against Schaul et al. ("Ray interference") and diversity-collapse papers is under-budgeted | **Yes** | W1 gets half a day, and Schaul is read before any T1 or C2 claim is written. |
| **Cuts:** C4, C6, T10, A5, A7 | **Yes** | All moved to future work. |

## 1. Limitations register (v2)

**Addressed by an experiment or analysis:**

| ID | Limitation | Addressed in |
|---|---|---|
| L1 | Absorption is confounded with answer imbalance | T1 |
| L2 | Kick size against response noise; the earlier "withdrawn" claim needs restating | T2, SUMMARY correction |
| L3 | c09 threshold is a horizon artifact | T0 flow-time grid, T1 |
| L4 | H5 tie-break and edge-of-grid η | T4 |
| L5 | H2 as promised (arcsin) | A1 |
| L6 | H3 headline refuted | T3 (descriptive), report reframing |
| L7 | The absorption definition | T0 |
| L8 | Degenerate baselines under uniform init | T1 offsets, T5 |
| L9 | Predictor no better than rarity | T5 |
| L10 | Ablation pass over-credited | A2, T6 |
| L11, L19 | H4(b) sufficiency; H4(a) at the boundary | A3 |
| L12 | H1 medium tier miscalibrated | T7 |
| L13 | KL switching has no noise-free control | T8 |
| L14 | Statistics | T0, §2 |
| L15 | Never-run promises | T9, A2, A4, A6; resets dropped |
| L16 | External validity | the LLM block (C0–C3, C5); T6 adds clipping and μ > 1 |
| L17 | Novelty and related work | W1 |
| L18 | Silent analysis bugs | T0 tests |
| L20 | Milestone threshold wording | A1 |

**Not addressed, listed in the final report:**
- KL in LLMs;
- graded rewards;
- the autoregressive toy (T10);
- the LLM predictor (C6);
- the LLM group-size test (C4);
- reference resets.

## 2. Primary hypotheses

These five are the only confirmatory claims. Holm–Bonferroni applies across them; everything else
is exploratory.

1. **P1 (T1):** at fixed feature geometry, persistent absorption rises with majority share m, and
   is ≈ 0 for balanced classes. It holds under SGD and under Adam.
2. **P2 (T2):** at equal mean drift, absorption depends on the per-kick size η/B, and does not
   depend on response-sampling noise (`sampled` ≈ `expected`).
3. **P3 (T5):** on balanced tasks, initial drift predicts which prompts get absorbed beyond rarity
   + p_init (ΔAUC ≥ 0.03, CI above 0).
4. **P4 (C1):** small-LLM GRPO produces problem losses above the calibrated decline-test FDR in
   both model families.
5. **P5 (C2):** in the LLM, minority-procedure problems are lost more often as m grows, and
   swapping the roles of the two procedures doesn't change that.

## 3. Phase A: existing data, no new runs (~1 day)

- **A1. H2 as promised.** Fit W_meas = c·[(1−λ) + λ/√(p(1−p))] with c free, plus a scale-free
  ratio version. Report λ(G) with bootstrap CIs and the exact λ(G) from theory, and fix milestone
  lines 289–292.
- **A2. Re-score the ablations.**
  - Material drop: max_drop > 1e-3.
  - Absorption.
  - Momentum and Adam reported as "no material decrease", with the sign-preservation lemma.
  - KL split into "toward p_ref" and "below p_ref".
- **A3. H4(b) and H4(a).** Fixed-β slopes as primary; H4(a) errors in logit space.
- **A4. Lost problems from stretch `s_pred`.** pass@1 against exact pass@64, over all runs at each
  η (not only absorbing runs).
- **A6. Barakat write-up,** with its limits stated.
- **SUMMARY correction:** restate item 3 (see §0, C2).

## 4. Phase B: toy reruns on the CPUs (~1.5 days)

### T0. Fixes before any run

- **Absorption is reported three ways:**
  - **end-absorbed;**
  - **persistent:** p < τ over the last 20% of steps;
  - **ever-absorbed.**
- **Thresholds and eligibility:** τ ∈ {1e-3, 1e-2}; a prompt is eligible only if p_init ≥ 10τ.
  Headline: persistent at τ = 1e-3.
- **Time-to-event:** Kaplan–Meier on entry into persistent absorption, censored at the horizon, in
  flow time η·t.
- **Seeds:** `task_seed` separate from `init_seed`; ≥ 10 task seeds for P1–P3.
- **Thresholds and CIs:** threshold estimates from a logistic fit pooled over seeds; bootstrap over
  task seeds.
- **Unit tests:** threshold, the deterministic tie-break, Kaplan–Meier, `predictor_scores`, and
  golden files for the existing tables.
- **Logging:** `min_p` every step; unrounded float32 snapshots.

### T1 (P1). Imbalance at fixed geometry

**Task:**
- correlated features, ρ ∈ {0.5, 0.9};
- L = 1, K = 3;
- answers = argmax(w·x + b), where w is one fixed rule;
- the offset vector b is solved to give majority share m ∈ {1/3 (balanced), 0.5, 0.7, 0.9}.

Margins change only through b. The per-class margin distribution is reported per arm.

**Arms:**
- linear and MLP-64;
- `expected_full` and `sampled` (B = 32);
- SGD and Adam;
- a flow-time η grid;
- 10 task seeds × 2 init seeds.

**Supervised control:** cross-entropy on the same tasks.

**Analysis:**
- persistent absorption against m, with a logistic fit and a margin covariate;
- per-class absorption against class frequency;
- Jaccard overlap of the absorbed set with the minority classes.

**Analytic companion:** expected-flow drift for a linear softmax with a class offset, giving which
classes lose and when.

**Outcomes:**
- **P1 holds:** the headline is "minority-class collapse".
- **Balanced arms absorb:** general interference collapse.

Either outcome is reportable.

### T2 (P2). Kick size against response noise, at equal mean drift

**The decomposition.** At the same η, these three modes have identical expected updates:
- `expected_full` (every prompt, weight η/n);
- `expected` (B sampled prompts, weight η/B each);
- `sampled` (the same plus response noise).

So, at equal mean drift:
- `expected` against `expected_full` isolates **per-kick size**;
- `sampled` against `expected` isolates **response noise**.

**Grid:**
- `expected` at B ∈ {8, 32, 128, 512 without replacement} at fixed η (kick size η/B varies; mean
  drift fixed);
- `sampled` at the same cells;
- levels lin, c05, m64, plus the T1 balanced task;
- L = 2; 10 seeds.

**Pass (P2):** absorption is monotone in η/B at fixed η, and `sampled` − `expected` has a CI that
includes 0 (equivalence margin ±0.05 absorbed fraction).

### T3. Descriptive decomposition (no law)

For every absorbed prompt in T1 and T2, attribute it as:
- **kick-size:** absent under `expected_full` at the same η;
- **imbalance:** a minority class, and present under `expected_full`;
- **other.**

Report the shares. This replaces the predictive test.

### T4. H5 rerun

- **Setup:**
  - η extended upward with the saturation stop rule;
  - logging every step;
  - 10 seeds;
  - tabular, lin and m64;
  - p₀ ∈ {0.01, 0.03, 0.1, 0.3}.
- **Best G:** the deterministic tie-break, with a bootstrap CI.
- **Shape:** say whether "best G" is an optimum or a saturation point.

### T5 (P3). Predictor on balanced tasks

- **Tasks:** T1 balanced tasks and MLPs, with a `sft_mixed` spread ≥ 0.25 sd, checked.
- **Scores:** drift, logit-drift, rarity, p_init, ‖∇p‖, and a logistic combination of rarity and
  p_init.
- **Test:** ΔAUC of drift beyond rarity + p_init, with run fixed effects and a bootstrap over task
  seeds. Checkpoint windows contain ≥ 20 events.

### T6. Ablations under realistic training

- **Conditions:**
  - the existing conditions;
  - PPO clipping ε = 0.2;
  - μ ∈ {1, 4} updates per batch;
  - gradient-norm clipping.
- **Setup:** tabular, 10 seeds.
- **Outcomes:** material drop and absorption.

### T7. H1 medium tier

- A parametric-bootstrap null, checked on an exact-enumeration case (tabular, L = 1, small G).
- Drop the tier if it still miscalibrates.

### T8. KL switching without noise

- Exact expected update with the analytic KL gradient (tabular).
- Bistable configurations at η ∈ {3, 10}.

### T9. All-wrong-group fixes

- **Rules:** Heckel weights, a virtual correct sample, ±α on uniform groups, and negative-sample
  reinforcement, added to `calculate_advantages` and `theory.py`.
- **Method:** derive each rule's W_G and the predicted change in drift near p ≈ 0 first, then run
  the T1 imbalanced and T2 large-kick configurations to confirm.
- **Prediction:** they don't prevent imbalance collapse, which is a direction effect.
- **Resets:** dropped, with the reason given.

**CPU budget:** about 3,500 runs and about 35 CPU-hours across two batches.

## 5. Phase C: small LLMs on the RTX 4090s

**Hardware** (checked 2026-11-20): 12 lab machines, each with an RTX 4090 (24 GB), 62 GB RAM and
1.7 TB local `/tmp`. The home disk is 91% full.

**Practical setup:**
- weights cached in `/tmp` and re-fetched after the 03:00 wipe;
- checkpoints every ≤ 30 min, with resume;
- only results go home;
- only machines with no other user.

**Models:**
- Qwen2.5-0.5B-Instruct (full fine-tuning);
- a **mandatory second family:** Llama-3.2-1B-Instruct or SmolLM2-1.7B-Instruct, with LoRA if
  needed.

This guards against the known Qwen spurious-reward effect and GSM8K contamination.

**Loop:**
- vLLM sampling;
- PyTorch update;
- our own `calculate_advantages`;
- sequence-mean loss aggregation (stated);
- binary exact-match verifier with a lenient parser.

Every config is logged.

### C0. Measurement and feasibility pilot (day 1–2)

- **Throughput:** tokens per second for sampling and training on one 4090. This fixes the
  evaluation and run budget.
- **Learning-rate pilot:** the smallest lr at which mean p̂ moves measurably in 300 steps.
- **Problem selection:** pick GSM8K/MATH problems with base p̂ in [0.1, 0.9] for each model.
- **Decline test:**
  - per-problem binomial likelihood ratio, before against after, at N = 32;
  - N = 256 for candidates;
  - BH-FDR at 10%;
  - the transition matrix of p̂.
- **Format failures:** the unparseable-answer rate, before and after, reported separately.

### C1 (P4). Lost-problem census

- **Runs:** 2 families × 3 seeds.
- **Evaluation:** 1,000 train problems + 300 held-out.
- **Measures:**
  - FDR-controlled declines;
  - pass@1 against pass@64 (unbiased estimator);
  - train against held-out.
- **A null result is reported as such.**

### C2 (P5). Minority-procedure collapse

**Mapping:** toy answer class ↔ LLM solution procedure.

**Task:** a template generator with two procedures that share surface templates. A difficulty pilot
tunes the operand size so that base p̂ is in 0.3–0.7 for both procedures; instances are
stratified on base p̂.

**Arms:**
- m ∈ {0.5, 0.8, 0.95};
- both role assignments (swapped);
- 3 seeds;
- 0.5B.

**Controls:**
- the base model's rate of majority-procedure errors;
- SFT on the same mix.

**Measures:**
- the minority-problem decline rate against m;
- the share of lost minority problems whose new answer equals the majority procedure's output.

### C3. Natural-data rarity (no training)

- **Pre-specified rarity measures:**
  - answer format;
  - strategy cluster (k fixed by silhouette before looking at the losses);
  - operation sequence.
- **Test:** logistic regression of decline on rarity, controlling for p̂_before and length.

### C5. Loss drivers

- **Changes** from the C1 configuration: verifier flip 20%, the sign rule, lr ×10.
- **Setup:** 0.5B, 3 seeds.

**GPU budget:** set by C0. The planning estimate, with two-stage evaluation, is about 120–180
GPU-hours. That is 10–15 hours across 12 GPUs.

## 6. Writing (W1, half a day plus the final days)

- **Related work:** Schaul et al. 2019 "Ray interference", the RLVR diversity-collapse and pass@k
  narrowing papers (Yue et al.), and the 16 missing keys plus Song et al. 2607.01083. Merge the bib
  files.
- **Framing:** an exact, controlled study of how prompts get lost under binary rewards, followed by
  a transfer test on small LLMs.
- **Limitations section:** everything in §1 "Not addressed", plus SGD-only toy results outside T1
  and T6.
- **Milestone wording fixes:** lines 258–260, 289–292, 309–311, 316–319 and 322–331.

## 7. Schedule (adopted from the critique)

| Day | Work |
|---|---|
| 1 | T0; A1–A4, A6; SUMMARY correction; GPU environment; C0 (throughput, lr pilot, format check) |
| 2 | T1, T2, T4, T6–T8 (CPU batch 1); C1 on both families |
| 3 | T9 (analytic first), T5, T3 (analysis); C2 after its difficulty pilot; C3 |
| 4 | C5; analysis of everything; Holm on P1–P5 |
| 5 | Writing (related work, limitations, figures); one adversarial review |

## 8. Decision rules and risks

| If | Then |
|---|---|
| P1 balanced arms show no absorption | The headline is "minority-class collapse under shared features"; C2 is its transfer test |
| Balanced arms absorb | General interference collapse; P3 tests drift |
| P4 is null | Report it; C2 and C5 test whether the toy drivers can induce losses |
| vLLM fails on the lab image | Pure-PyTorch sampling, 0.5B only, C5 cut |
| The time budget slips | Cut in this order: C5, then C3, then T8, then T7 |

**Machines:** checkpoint and resume; skip any host another user is on.

## 9. Verification

- Unit and golden-file tests pass.
- Predictions and pass rules for P1–P5 are committed before the launches.
- Holm-adjusted table for P1–P5.
- The format-failure rate and FDR are reported with every LLM loss count.
- A final adversarial review before submission.
