# Plan v3 (final): close the known limitations, then test transfer to small LLMs

Supersedes v2 (commit `ac2edd7`). Changes from v2: the LLM KL stretch goal C7 is added, and the second model family is fixed as SmolLM2-1.7B-Instruct. The v2 table of changes made after [plan_critique.md](plan_critique.md) still applies.


## Context

Two CPU batches have run: 24,638 runs on the synthetic task. A four-way review
(`981_project/analysis/review/SUMMARY.md`) overturned part of the story:
- the "interference drift" is minority-answer collapse;
- "noise" is really per-kick size;
- the H5 table was a tie-break artifact;
- the one-step-size law (H3) was refuted.

A draft plan was pressure-tested (`analysis/review/plan_critique.md`) and revised to v2
(`analysis/review/NEXT_PLAN.md`, commit `ac2edd7`).

This v3 makes two changes to v2:
1. **New LLM stretch goal C7, KL in LLMs.** v2 had listed KL as an unaddressed limitation.
2. **The second model family is fixed:** SmolLM2-1.7B-Instruct.

On approval, `NEXT_PLAN.md` is updated to this v3, committed locally, and Day 1 starts. Nothing is
pushed.

## Second model family: SmolLM2-1.7B-Instruct (HuggingFaceTB)

- **Ungated** (Apache-2.0), so the lab machines need no Hugging Face token or licence step.
  Llama-3.2 and Gemma are gated.
- **A different lab and different training data** from Qwen. That guards against Qwen's known
  spurious-reward effect and against GSM8K contamination specific to Qwen.
- **Mid-range GSM8K accuracy** (reported around 45–50%). Absorption needs problems with p in
  0.1–0.9; Qwen2.5-0.5B is also mid-range, so the two families overlap in difficulty.
- **Fits a 24 GB RTX 4090:** LoRA (rank 64) on 1.7B, with vLLM sampling in the same process at
  about 40% of GPU memory. Qwen2.5-0.5B is trained with full fine-tuning.
- **Fallback:** Llama-3.2-1B-Instruct, if SmolLM2's base p̂ distribution on the selected problems
  turns out too extreme in the C0 pilot. That needs the user's Hugging Face access to the gated
  model.

## Primary hypotheses

These are the only confirmatory claims. Holm–Bonferroni applies across P1–P5; everything else is
exploratory, including C7.

1. **P1 (T1):** at fixed feature geometry, persistent absorption rises with the majority share m
   and is ≈ 0 when classes are balanced. It holds under SGD and under Adam.
2. **P2 (T2):** at equal mean drift, absorption depends on the per-kick size η/B, and response
   sampling doesn't matter (`sampled` ≈ `expected`, equivalence margin ±0.05).
3. **P3 (T5):** on balanced tasks, initial drift predicts absorbed prompts beyond rarity + p_init
   (ΔAUC ≥ 0.03, CI above 0).
4. **P4 (C1):** small-LLM GRPO produces problem losses beyond the calibrated decline-test FDR, in
   **both** families.
5. **P5 (C2):** LLM minority-procedure problems are lost more as m grows, robust to swapping the two
   procedures' roles.

## Phase A: existing data, no runs (Day 1)

- **A1. H2 as promised.** λ(G) toward the 2 arcsin √p objective:
  - fit c·[(1−λ) + λ/√(p(1−p))] with c free, plus a scale-free version;
  - bootstrap CIs;
  - the exact theoretical λ(G);
  - fix milestone lines 289–292.
- **A2. Re-score the ablations.**
  - A material drop is max_drop > 1e-3; absorption is counted separately.
  - Momentum and Adam reported as "no material decrease", with the sign-preservation lemma.
  - KL split into "toward p_ref" and "below p_ref".
- **A3. H4.** Fixed-β variance slopes as the primary result; H4(a) errors in logit space.
- **A4. Lost problems from stretch `s_pred`.** pass@1 against exact pass@64, over all runs at each
  η.
- **A6. Barakat write-up,** with its limits stated.

## Phase B: toy reruns on the CPUs (Days 1–3, ~3,500 runs, ~35 CPU-hours)

- **T0. Fixes before any run.**
  - **Absorption reported three ways:** end-absorbed; *persistent* (p < τ over the last 20% of
    steps); and ever-absorbed.
  - **Thresholds and eligibility:** τ ∈ {1e-3, 1e-2}; eligible only if p_init ≥ 10τ. Headline:
    persistent at τ = 1e-3.
  - **Time-to-event:** Kaplan–Meier on entry into persistent absorption, in flow time η·t.
  - **Seeds:** `task_seed` separate from `init_seed`; ≥ 10 task seeds for P1–P3; bootstrap over
    tasks.
  - **Tests:** threshold, the deterministic tie-break (smallest G within the logging resolution),
    Kaplan–Meier, `predictor_scores`, and golden files for the existing tables.
  - **Logging:** `min_p` every step; unrounded snapshots.
- **T1 (P1). Imbalance at fixed geometry.**
  - **Task:** correlated features, ρ ∈ {0.5, 0.9}; L = 1, K = 3; answers = argmax(w·x + b) for one
    fixed w; the offset b is solved for majority share m ∈ {1/3, 0.5, 0.7, 0.9}. Per-class margins
    are reported.
  - **Arms:** linear and MLP-64; `expected_full` and `sampled`; SGD and Adam; a flow-time η grid;
    10 task seeds × 2 init seeds.
  - **Supervised control:** cross-entropy on the same tasks.
  - **Analytic companion:** the expected flow with a class offset.
- **T2 (P2). Kick size against response noise, at equal mean drift.**
  - `expected` at B ∈ {8, 32, 128, 512 without replacement}, at fixed η.
  - `sampled` at the same cells.
  - Levels lin, c05, m64, plus the T1 balanced task; L = 2; 10 seeds.
- **T3. Descriptive attribution** of each absorbed prompt: kick-size, imbalance, or other. No
  predictive-law test.
- **T4. H5 rerun.**
  - η extended upward with a saturation stop rule;
  - logging every step; 10 seeds; tabular, lin and m64;
  - p₀ ∈ {0.01, 0.03, 0.1, 0.3};
  - best G by the deterministic tie-break, with a CI;
  - say whether it is an optimum or a saturation point.
- **T5 (P3). Predictor on balanced tasks.**
  - `sft_mixed` spread ≥ 0.25 sd, checked.
  - Scores: drift, logit-drift, rarity, p_init, ‖∇p‖, and a logistic combination.
  - ΔAUC beyond rarity + p_init, with run fixed effects and a task bootstrap.
  - Checkpoint windows contain ≥ 20 events.
- **T6. Ablations under realistic training.**
  - The existing conditions, plus PPO clipping (ε = 0.2), μ ∈ {1, 4} updates per batch, and
    gradient-norm clipping.
  - Tabular, 10 seeds.
- **T7. H1 medium tier.** A parametric-bootstrap null, checked on an exact-enumeration case. Drop
  the tier if it still miscalibrates.
- **T8. KL switching without noise.** Exact expected update with the analytic KL gradient
  (tabular), on the bistable configurations at η ∈ {3, 10}.
- **T9. All-wrong-group fixes:** Heckel weights, a virtual correct sample, ±α on uniform groups,
  negative-sample reinforcement.
  - Add them to `synthetic_rl/trainer.py` `calculate_advantages` and `synthetic_rl/theory.py`.
  - Derive each predicted change in W_G first, then confirm on the T1 and T2 absorbing
    configurations.
  - Reference resets: dropped, with the reason given.

**Code touched:**
- `synthetic_rl/synthetic_task.py` (class offset b);
- `experiments/run.py` (persistent-absorption logging, B without replacement, KL in
  `expected_step` for tabular, PPO clipping and μ);
- `experiments/grids.py` (`v3_*` grids);
- `experiments/analyze.py` (T0 metrics, Kaplan–Meier, Holm, the tie-break);
- `tests/`.

## Phase C: small LLMs on the 12 RTX 4090s (Days 1–5)

**Hardware and setup:**
- 24 GB GPU, 62 GB RAM and 1.7 TB `/tmp` per machine. The home disk is 91% full.
- Weights are cached in `/tmp` and re-fetched after the 03:00 wipe. Checkpoints every ≤ 30 min,
  with resume. Only results go home.
- Use only machines with no other logged-in user.

**Loop:**
- a new `llm/` package in `981_project`;
- vLLM sampling and a PyTorch update;
- **our own `calculate_advantages`**;
- sequence-mean loss aggregation (stated, not TRL's default);
- binary exact-match verifier with a lenient answer parser;
- defaults: G = 8, 64 prompts per step, no KL, μ = 1, 300–500 steps;
- learning rate set by the C0 pilot.

**Models:**
- Qwen2.5-0.5B-Instruct (full fine-tuning);
- SmolLM2-1.7B-Instruct (LoRA).

### Experiments

- **C0. Feasibility and measurement pilot (Days 1–2).**
  - Throughput for sampling and training on one 4090.
  - A learning-rate pilot: the smallest lr at which mean p̂ moves in 300 steps.
  - Problem selection: base p̂ in [0.1, 0.9] per model.
  - **Decline test:** per-problem binomial likelihood ratio, N = 32 for all problems, then
    N = 256 adaptively for candidates, with BH-FDR at 10%.
  - The p̂ transition matrix.
  - The format-failure (unparseable answer) rate, reported separately.
- **C1 (P4). Lost-problem census.**
  - 2 families × 3 seeds.
  - Evaluation on 1,000 train + 300 held-out problems.
  - Measures: FDR-controlled declines; pass@1 against pass@64 (unbiased estimator); train against
    held-out.
  - A null result is reported as such.
- **C2 (P5). Minority-procedure collapse.** Toy answer class ↔ LLM solution procedure.
  - **Task:** a template generator with two procedures that share surface templates. A difficulty
    pilot tunes the operand size until base p̂ is in 0.3–0.7 for both procedures, and instances are
    stratified on base p̂.
  - **Arms:** m ∈ {0.5, 0.8, 0.95} × both role assignments × 3 seeds, on Qwen 0.5B.
  - **Controls:** the base model's rate of majority-procedure errors, and SFT on the same mix.
  - **Measures:** the minority decline rate against m, and the share of lost minority problems whose
    new answer equals the majority procedure's output.
- **C3. Natural-data rarity (no training).**
  - Pre-specified rarity measures: answer format, strategy cluster (k fixed by silhouette first),
    and operation sequence.
  - Logistic regression of decline on rarity, controlling for p̂_before and length.
- **C5. Loss drivers.** From the C1 configuration: verifier flip 20%, the sign rule, lr ×10. Qwen
  0.5B, 3 seeds.

### C7. LLM stretch goal: does the KL penalty behave as the toy predicts? (exploratory, Days 4–5)

C7 runs only once P4 and P5 are done, and only if GPUs are free. Qwen2.5-0.5B, full fine-tuning,
reverse KL against the frozen initial model (the standard per-token k3 estimator), summed over the
sequence to match the toy's sequence-level KL.

**Grid:**
- β ∈ {0 (reuses C1), 0.02, 0.1, 0.5};
- G ∈ {4, 16} at a fixed 512 samples per step;
- 2 seeds;
- about 12 new runs;
- 800 steps, so that runs have time to settle.

**Questions, each with a toy prediction:**
1. **Does KL prevent losses?** Toy: KL holds p near p_ref and stops absorption except at very large
   steps. Measure the FDR-controlled declines against β. Prediction: monotone fewer losses as β
   grows.
2. **Does the per-problem end state follow the finite-G fixed point?**
   - For each problem, compute p*(p_ref = p̂_before, β_eff, G, rule) with `theory.fixed_points`.
   - β_eff is fitted with one free scale per run, because the per-token KL weight maps to an
     effective sequence β only up to a constant.
   - Measure: calibration slope and R² of p̂_after against p*.
   - Prediction: a positive relation with slope near 1 if each problem behaves roughly
     independently. Sharing could break this, which is itself informative because it extends H4
     beyond tabular models.
3. **Finite-G signature:**
   - At the same β, G = 16 should push low-p_ref problems further up than G = 4 (GRPO's W_G grows
     with G at low p).
   - Look for two clusters in p̂_after among problems with similar p̂_before at G = 16 (a weak test
     of bistability; reported descriptively).

**Cost:** about 12 runs × about 2–3 GPU-hours each, plus evaluation at N = 32. That is about 1 day
on 6 GPUs.

**Not claimed:** exact fixed-point agreement (the β mapping is approximate), and bistability (too
weak a test). C7 can only support or weaken "KL dynamics follow the per-prompt theory in LLMs".

**GPU budget:**
- core (C0–C5): about 120–180 GPU-hours;
- C7: about 30–40 GPU-hours;
- total: about 15–18 hours of wall clock on 12 GPUs.

## Writing (Day 5)

- **Related work:**
  - Schaul et al. 2019 "Ray interference" (read before any T1 or C2 claim);
  - the RLVR diversity-collapse and pass@k narrowing papers (Yue et al.);
  - the 16 missing keys from `c.md`, plus Song et al. 2607.01083;
  - merge the bib files.
- **Framing:** an exact, controlled study of how prompts get lost under binary rewards, plus a
  transfer test on two small-LLM families.
- **Limitations:** graded rewards; the autoregressive toy, the LLM predictor and the LLM group-size
  test (all cut); reference resets; KL in LLMs beyond C7's approximate β mapping.
- **Milestone wording fixes:** lines 258–260, 289–292, 309–311, 316–319 and 322–331.
- **One final adversarial review.**

## Schedule

| Day | CPU (toy) | GPU (LLM) |
|---|---|---|
| 1 | T0 fixes and tests; A1–A4, A6 | Environment build; C0 throughput, lr pilot, format check |
| 2 | Batch 1: T1, T2, T4, T6, T7, T8 | C1 (both families, 3 seeds) |
| 3 | T9 (analytic first), T5, T3 analysis | C2 (after its difficulty pilot); C3 analysis |
| 4 | Analysis of everything; Holm on P1–P5 | C5; **C7 if GPUs are free** |
| 5 | Writing | C7 finishes; final evaluations |

**Cut order if time slips:** C7, then C5, then C3, then T8, then T7.

## Decision rules

| If | Then |
|---|---|
| T1 balanced arms show no absorption | Headline is "minority-class collapse"; C2 is the transfer test |
| T1 balanced arms absorb | General interference; P3 tests drift |
| P4 is null | Report it; C2 and C5 test whether the toy drivers induce losses |
| SmolLM2 base p̂ is too extreme | Switch to Llama-3.2-1B-Instruct (needs Hugging Face access) |
| vLLM fails on the lab image | Pure-PyTorch sampling, Qwen 0.5B only, C5 and C7 cut |

## Verification

- All unit and golden-file tests pass.
- P1–P5 predictions and pass rules are committed before each launch, and the commit hash appears in
  the findings.
- Holm table for P1–P5; C7 is labelled exploratory.
- Every LLM loss count comes with its FDR level and format-failure rate.
- Toy batches: result count equals job count, 0 failed. LLM runs: each logs its config and seed, and
  resumes cleanly after a forced restart (tested once in C0).
- `HANDOFF.md` is updated at the end of each day.
