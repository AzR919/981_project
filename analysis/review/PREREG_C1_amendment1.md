# PREREG_C1, amendment 1 (2026-11-22)

Committed before any C1 decline analysis. It amends `PREREG_C1.md`; everything not mentioned here
stands.

## What has been seen

- **C1 Qwen runs** (lr 1e-6 and 3e-6, 3 seeds each) and **C5 runs** (9): their training logs (reward,
  length), and that each finished. No before/after comparison has been computed for any of them.
- **SmolLM2 pilots:** at lr 1e-5, 3e-5 and 1e-4 the reward rose to 0.7–0.8, then collapsed after
  about 100 steps. lr 3e-4 had not collapsed by step 150. The pre-registered lr rule returned NONE.
- **Model probes:** Llama-3.2-1B-Instruct trains with full fine-tuning on one GPU (base p̂ 0.30 on 64
  problems; sampler–trainer gap 0.14 nats).
- **Reviews** of Yang (arXiv 2609.22547) and Qin et al. (arXiv 2606.03087), both preprints
  (`papers/reviews/deep/`). Neither has a per-problem test or a calibrated null. Their zero-count
  "locked" problems are partly explained by a no-change null (reviewer's check).

## Changes

1. **Second family.** Llama-3.2-1B-Instruct replaces SmolLM2-1.7B, following the pre-registered gate.
   - It is trained with full fine-tuning (fp32 master weights, 8-bit AdamW), the same method as Qwen,
     so family is no longer confounded with LoRA.
   - **Learning rate:** the same rule as before, over {1e-6, 3e-6, 1e-5} (150 steps on train problems
     0–999). Take the smallest lr at which mean p̂ rises by ≥ 0.05 with ≤ 2 BH-significant declines
     at N = 32.
   - **Pool:** train problems with pooled base p̂ in [0.1, 0.9] from two N = 32 evaluations with
     **seeds 2 and 3**. These are independent of the N = 256 "before" (seed 0) and null (seed 1)
     draws. The Qwen pool used seeds 0 and 1; see change 5.
   - **Second "after" draw:** each Llama run also gets a second N = 256 evaluation (seed 1). Report
     how many lost problems are still ≤ 0.05 in it.
2. **Family rule.** "At least 2 of 3 seeds" is tested with the partial-conjunction p-value
   p_family = min(1, 2·p₍₂₎). It is valid under arbitrary dependence between seeds, which share the
   base evaluation. The first-registered p₍₂₎ is reported beside it.
   - Proof: under the null at most one seed is non-null.
     - If one seed is non-null, p₍₂₎ ≤ t needs one of two true nulls at ≤ t: probability ≤ 2t.
     - If none is, it needs two of three: by Markov, probability ≤ 3t/2.
   - P4 holds if p_family < 0.05 in both families. The value passed to Holm is the larger p_family.
3. **Wording.** P4 is reported as "losses in excess of sampling noise". Lost problems under RLVR are
   known (Yue et al. 2025). What C1 adds is a per-problem test with a calibrated null on train and
   held-out problems.
4. **Reporting, for every run:**
   - **A turnover table** on train and held-out problems: gained, declined and lost (BH at 10%);
     strict-lost (0 of 256 after, base p̂ ≥ 0.1); Yang's newly solved (0 → ≥ 1) and locked (≥ 1 → 0).
     Each count has its base-against-base null count beside it.
   - **95% intervals from resampling problems** for the mean change and the lost count.
   - **The paired pass@k gap** for k = 1…256 with a simultaneous 95% band (prompt-level multiplier
     bootstrap).
   - **Modal wrong answers:** the share of completions on the most frequent wrong answer for lost and
     not-lost problems, and whether it is the base model's most frequent wrong answer.
   - **The detection limit:** with 256 samples, a fall to ≤ 0.05 is detectable from base p̂ ≳ 0.15.
5. **Regression to the mean (Qwen).** The Qwen pool was selected with the same seeds (0 and 1) as its
   N = 256 draws. The analysis reports the lost count with base seed 1 as "before", and the number
   lost under both choices.
6. **Not available for the finished Qwen runs:** truncation counts and reference answers were not
   stored, and the trained models were in `/tmp` (wiped). Format-failure rates and mean completion
   length are reported in their place. New runs store truncation counts.

## Unchanged

The primary arm for Qwen (lr 1e-6), the definitions of decline and lost, the binomial test per run
against r₀ = (L₀ + 1)/1001, and the C5 comparisons.
