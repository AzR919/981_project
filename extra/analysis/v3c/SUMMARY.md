# v3c: pass@k training rules (exploratory, PREREG_v3c.md)

All 1,920 runs returned and none failed. `extra/experiments/analyze_v3c.py` regenerates `FINDINGS.md`.

| Prediction | Result |
|---|---|
| 1. Minority task: pass@k rules absorb fewer minority prompts than grpo | **Confirmed, strongly.** grpo absorbs the whole minority set (share absorbed 1.00). Both pass@k rules at every k absorb 0–5% (6/6 cells per rule, p < 0.05) |
| 2. Shared levels: pass@k rules give higher pass@64 and lower pass@1 | **Half confirmed.** The unbiased rloo_k raises pass@64 in 12/12 cells (+0.15 to +0.20). But pass@1 also rises (+0.13 to +0.18), so there is no speed cost at these settings |
| 3. Tabular: no absorption under any rule | **Confirmed** (0 everywhere) |
| 4. rloo_k at k = 8 learns slowest | **Confirmed on the minority task.** Mean pass@1 is 0.56 for rloo_k8, 0.75 for rloo_k4 and 0.88 for rloo_k2, against 0.90 for grpo. Within the run, k ≥ 4 trades speed for protection |

## The headline

On the minority task, grpo stalls at pass@1 = 0.897: it solves the 90% majority and loses every
minority prompt. passk2 and rloo_k2 at the largest η reach 0.94 and 0.92 while absorbing only 2–5% of
minority prompts. **Rules that weight hard prompts by about k(1 − p)^(k−1) prevent minority collapse.
At k = 2 they do so without slowing down.** This is the toy-level counterpart of the analytic
prediction: the ratio of low-p to high-p weight is ≫ grpo's 1.12.

## Caveats

- **Larger k is slow.** At k ≥ 4, part of the "no absorption" may be slow learning within the
  horizon. Compare at matched pass@1 before calling it protection.
- **Different overall scales.** The rules differ in overall weight (rloo_k's W ≤ 1 against grpo's
  2–3), so at the same η the effective step differs. The minority result holds at every η tested.

## Later correction (v3f, 2026-11-22)

At matched majority progress, pass@2 does not stop the early push-down of the minority; it prevents
absorption and allows recovery. pass@4 protects during learning at a speed cost. See
`extra/analysis/v3f/SUMMARY.md`.
