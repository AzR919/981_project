# Pre-registration: v3c (pass@k rules, toy) and C8 and C9 (LLM KL direction, loss aggregation)

Written and committed before batch `v3c` and the C8 and C9 GPU jobs start. All are **exploratory**,
so none enters the Holm family P1–P5. The code is on `toy-v3` (`7759008`) and `llm-transfer`
(`4e575cc`, `7efbe5b`).

**Disclosures:**
- One local 200-step smoke run per (level, rule) checked only that the code runs; outcomes were not
  inspected.
- `rloo_k`'s W_G(p) = (1 − p)^(k−1) was derived and tested before any run.
- **KL fix.** `llm/train.py` previously differentiated the k3 estimator, which gives the
  forward-KL gradient. No run with β > 0 had been made. `--kl reverse` (the default) now uses k2.

## v3c: pass@k training rules (T10)

**Grid:**
- rules grpo, mean, passk (GRPO_k, biased) and rloo_k (RLOO_k, unbiased), with k ∈ {2, 4, 8};
- G ∈ {8, 32} at B·G = 256;
- levels tab (spread), lin and m64 (balanced, `sft_mixed`), and the minority task (linear, ρ = 0.9,
  m = 0.9);
- 3 η per level × 10 seeds or task seeds;
- 5,000 steps.

**Measures** (at the end):
- exact pass@1, pass@8 and pass@64 (mean over prompts of 1 − (1 − p)^K);
- persistent absorption at τ = 1e-3;
- for the minority task, absorption among minority prompts.

**Analytic predictions:**
- rloo_k: W = (1 − p)^(k−1).
- passk: W falls from about 2.5–5 at p = 0.01 to about 0 at p = 0.9 for k = 8.

**Predicted outcomes**, comparing each rule with grpo at the same (level, G, η), paired over seeds:
1. **Minority task.** rloo_k and passk absorb fewer minority prompts than grpo, with the effect
   growing in k. This follows from the ratio of low-p to high-p weight, which is ≫ grpo's 1.12.
2. **Shared levels.** pass@k rules give a higher final pass@64 and a lower final pass@1 than grpo.
3. **tab.** No rule absorbs (the tabular proposition), so there are no differences in absorption.
4. **Speed.** rloo_k at k = 8 has the smallest final pass@1 on the minority and shared tasks, since its
   weight on high-p prompts vanishes.

**Tests:** one-sided Wilcoxon per cell, paired over seeds, with the share of cells that agree.
No multiplicity correction; these are exploratory.

## C8: forward-KL penalty in the LLM

**Runs:** Qwen-0.5B, full fine-tuning, lr 1e-6, G = 16, 32 prompts per step, 800 steps,
β ∈ {0.1, 0.5}, `--kl forward`, seeds 0 and 1.

**Comparison:** the C7 reverse-KL runs at the same (β, G = 16, seed).

**Predictions:**
- At equal β, forward KL keeps more problems near their base p̂ than reverse KL, with fewer
  BH-significant declines and losses at N = 256.
- The toy's forward-KL fixed points (`theory.fixed_points(direction="forward")`) lie closer to p_ref
  than the reverse ones for small p_ref.

## C9: loss aggregation over tokens in the LLM

**Runs:** Qwen-0.5B with the C1 primary settings (lr 1e-6, G = 8, 64 prompts per step, 500 steps),
`--agg token`, seeds 0–2.

**Scaling:** the loss is divided by total completion tokens, rescaled by the step-1 mean length.

**Comparison:** C1 Qwen at lr 1e-6, seeds 0–2.

**Predictions:**
- The completion-length trajectory differs from C1's. Token aggregation down-weights long answers'
  sequence log-probability per token, so answer length grows more under it.
- The loss count, measured as in PREREG_C1, is reported beside C1's with no directional prediction.

## Deviations

Any departure from this file is listed in the write-up with its reason.
