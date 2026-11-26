# Pre-registration: C1 (P4) and C5, small-LLM lost-problem census

This file is written and committed before any C1 or C5 training run starts. The code is on branch
`llm-transfer`, and each run records its commit in `$OUT/COMMIT`. The analysis (`extra/llm/analyze_c1.py`)
must implement exactly these tests. Any later deviation is reported as a deviation.

**What is already known.** The C0 pilots are known and disclosed in `extra/analysis/llm/c0_qwen.md`
(branch `analysis-v3`):
- At 150 steps, Qwen at lr 1e-6 had 0 BH-significant declines at N = 32. At lr 3e-6 it had 2 (1
  lost). At lr 1e-5 it collapsed (168 lost).
- N = 32 cannot detect drops from p̂ < 0.375.
- The SmolLM2 pilots were invalid because of a weight-sync bug (fixed in `33a6264`). They are rerun
  in wave G0 (tag `c0d`), before C1.

## Design

- **Models.**
  - Qwen2.5-0.5B-Instruct, full fine-tuning, fp32 master weights, 8-bit AdamW.
  - SmolLM2-1.7B-Instruct, LoRA r = 64 (α = 128) on all attention and MLP projections, AdamW.
- **Training.**
  - GRPO with our `calculate_advantages`, G = 8, 64 prompts per step, 500 steps.
  - β = 0, one update per batch, temperature 1, at most 512 new tokens.
  - Sequence-level loss −Σ A · log π(completion) / (64 · 8).
- **Problems.** 1,000 training problems and 300 held-out problems per model. Both are drawn with seed
  0 from the model's pool: GSM8K train problems with base p̂ in [0.1, 0.9], pooled over two N = 32
  evaluations. The id lists are in `extra/llm/pools/<model>_c1_{train,heldout}.json`.
- **Arms and seeds.**
  - Qwen: lr 1e-6 is the **primary arm**; 3e-6 is a secondary arm. Seeds 0, 1 and 2 for each.
  - SmolLM2: one arm, seeds 0, 1 and 2. Its lr is set by the G0 pilot rule, fixed before C1 launches:
    the smallest lr in {1e-5, 3e-5, 1e-4, 3e-4} at which the mean p̂ on the training problems rises
    by ≥ 0.05 within 150 steps, with ≤ 2 BH-significant declines at N = 32.
- **Measurement.** For every problem in the 1,300, the number correct out of N = 256 samples at
  temperature 1, from:
  - before: the base model, seed 0;
  - null: the base model, seed 1;
  - after: the final model, seed 0.
  - Also N = 32 at step 250 (a trajectory only).

## Definitions

- **Decline:** a one-sided Fisher exact test of p_after < p_before on (correct, 256) counts, with
  Benjamini–Hochberg at q = 0.10 across the 1,000 training problems. Held-out problems are tested
  separately with their own BH.
- **Lost:** a significant decline that ends at p̂_after ≤ 0.05.
- **Null losses** L₀: the same test applied to base seed 0 against base seed 1, on the same 1,000
  problems. Null rate r₀ = (L₀ + 1) / 1001 (add-one, so r₀ > 0).

## P4 (primary)

- **Per run:** L = the number of lost training problems. One-sided binomial test of L against
  Binomial(1000, r₀).
- **Per family:** p_family = the second-smallest of the three seed p-values in the primary arm. The
  family passes if p_family < 0.05, i.e. at least 2 of 3 seeds are significant.
- **P4 holds** if both families pass. The p-value passed to Holm (across P1–P5) is
  max(p_Qwen, p_SmolLM2).
- **If P4 fails:** that is reported as a null result for small-LLM GRPO at this lr and length. C2
  and C5 then test whether the toy's drivers induce losses.

## Secondary and descriptive (exploratory)

- **Base p̂ of lost problems.**
  - Prediction from the toy: losses concentrate at low base p̂. The median base p̂ of lost problems
    is below the median of the 1,000 problems.
  - Test: one-sided Mann–Whitney per family, pooled over seeds.
- **Cross-seed stability.** Overlap of the lost sets between seed pairs, with a hypergeometric test
  against independent draws. A significant overlap means the losses are problem-specific, not random.
- **Held-out problems.** The loss count on held-out problems, against training problems.
- **Every run also reports:**
  - pass@1 against pass@64 (unbiased, from N = 256), per problem and in total;
  - BH-significant gains;
  - the p̂ transition matrix;
  - format failures, before and after;
  - completion length.
- **Qwen lr 3e-6:** the same tests, reported beside the primary arm and not fed to Holm.
- **Trajectory:** the step-250 evaluation (N = 32): mean p̂, and the problems below 0.05 at step 250.

## C5: loss drivers (exploratory)

Qwen at the C1 primary configuration, 3 seeds each, with one change per arm:
- verifier flip 20%;
- the sign rule;
- lr × 10 (1e-5).

**Predictions from the toy (T6):**
- the flip and the sign rule increase losses over C1 at the same lr;
- lr × 10 collapses, as in C0.

**Measures:** the same per-run measures as C1, compared with the C1 primary arm by a one-sided
Mann–Whitney over seeds (n = 3 against 3, so only descriptive).

## Deviations

Any departure from this file, whether a different N, a failed run, or a changed problem set, is
listed in the C1 write-up under "Deviations", with its reason.
