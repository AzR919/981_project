# v3n: shared skills (T26)

All 320 runs returned and none failed (`toy-v3` `c991964`). Pre-registered in `PREREG_v3d_addendum8.md`
(`review` `2334a5d`); exploratory. `extra/experiments/analyze_v3n.py` regenerates `FINDINGS.md`, `runs.csv`, `cells.csv`.

Setup: 4-token answers; k positions carry the answer class, 4 − k follow one balanced rule shared by all prompts.

| Prediction | Result |
|---|---|
| 1. Step-0 push ratio within 5% of the computed value | **Holds: 320 of 320** (median error 0.2%). The first-step identity extends to multi-token answers |
| 2. Where shared positions make the cross-class sum positive, the majority helps the minority | **Untested: 0 such tasks** |
| 3a. Absorption falls as sharing grows | **Holds: 8 of 8 cells** |
| 3b. With 3 shared positions, ≤ 10% absorbed in ≥ 9 of 10 tasks | **Fails: 0 of 8 cells.** Absorption stays 98–100% in 6 cells (0.24 and 0.77 in linear ρ = 0.5) |
| 4. Majority-only step raises the minority later (reported) | Never at step 0; by 50% of training in 1–4 of 10 tasks, MLP cells only |

## Reading

- **Design limitation (found after the run):** at a shared position the correct token depends on the prompt's
  features, so a majority and a minority prompt agree there only at chance (≈ 1/K). The expected overlap of a
  shared position is then about zero: sharing *dilutes* the push (median R falls from 3.5–17 at k = 4 to 2–9 at
  k = 1) but cannot reverse it. Prediction 2 therefore never applied.
- So T26 does **not** test the LLM explanation (positive transfer through shared steps) fairly; it shows only
  that neutral shared parts dilute the competition without preventing collapse.
- **Follow-up:** shared positions on which majority and minority prompts agree well above chance (e.g. one
  token correct for every prompt, or a rule with high agreement), so the cross-class overlap is positive.
