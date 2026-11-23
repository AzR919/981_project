# v3f: do pass@k rules protect the minority at matched progress? (T16)

All 480 runs returned and none failed. They were launched from a clean commit (`toy-v3` `e45b533`;
job digest `879c78be6482`). Pre-registered in `PREREG_v3d_addendum2.md` (`review` `a9c0980`);
exploratory. `experiments/analyze_v3f.py` regenerates `FINDINGS.md`.

- **Task:** linear, ρ = 0.9, m = 0.9; sampled updates; 20,000 steps (4× v3c).
- **Matched point:** the first logged step at which the majority class mean reaches 0.80.

| Rule | Majority reaches 0.80 | Minority mean at that point | Final majority / minority mean (η = 1) | Minority prompts absorbed |
|---|---|---|---|---|
| grpo, mean | by step 100 | 0.0001–0.0004 | 1.00 / 0.000 | 100% |
| pass@2 (both forms) | by step 100 | **0.03–0.08** | 0.98 / 0.79–0.87 | 0–12% |
| pass@4 (both forms) | steps 350–3,900 (not at η = 0.03) | **0.31–0.42** | 0.92–0.95 / 0.67–0.72 | 0% |
| pass@8, biased form | steps 8,000–12,000 at η = 1 only (7–8 of 10 tasks) | 0.46–0.48 | 0.80–0.81 / 0.54–0.56 | 0% |
| pass@8, unbiased form | never within 20,000 steps | – | 0.75 / 0.50 | 0% |

## Predictions

1. **Protection at matched progress** (minority above 0.2 when the majority reaches 0.80): **12 of 36
   cells.**
   - Holds for k = 4 wherever the majority gets there.
   - **Fails for k = 2:** the minority dips to 0.03–0.08, i.e. it is pushed down almost as under grpo.
     But it is not absorbed, and it recovers afterwards.
   - Under grpo it is below 0.01 in 6 of 6 cells, as predicted.
2. **Not just slow** (k ≥ 4 at η = 1 reaches 0.80 with no absorbed minority prompt): **4 of 8 cells.**
   Holds for k = 4; fails for k = 8, which is too slow to reach 0.80 in most runs.
3. **Cost grows with k:** 2 of 12 cells by the registered test, because most k = 8 cells never reach
   the matched point. The direction is clear: step 100 for k = 2, 350–3,900 for k = 4, and 8,000 or
   never for k = 8.

## Reading

- **This revises the v3c summary.** "pass@2 prevents the collapse at no speed cost" was too strong.
  - pass@2 does not stop the early push-down. It keeps the minority off the floor (success 0.03–0.08
    instead of 0.0001), which lets it recover: final minority success is 0.79–0.87 at η = 1, but only
    0.24–0.50 at η = 0.03 after 20,000 steps.
  - pass@4 does protect the minority while the majority is being learned (0.31–0.42 at the matched
    point), at the price of a majority that is 4–40 times slower to reach 0.80.
  - pass@8 protects by hardly moving.
- **The practical statement:** hard-prompt weighting trades majority speed for minority retention.
  At moderate k it prevents absorption and allows recovery. It is a trade-off, not a free fix.
- **The mean rule** behaves like grpo here: 100% of minority prompts absorbed.

## Deviations

None. Predictions 1–3 are reported as partly failing.
