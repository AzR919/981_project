# C7 / C8: the KL penalty in the LLM (final)

All 20 planned runs finished (queue `r2`). C7 is reverse KL against the frozen base model, β ∈ {0, 0.02, 0.1, 0.5},
G ∈ {4, 16} at 512 samples per step. C8 is forward KL at β ∈ {0.1, 0.5}, G = 16. All runs: Qwen2.5-0.5B, lr 1e-6,
800 steps, 2 seeds per setting. Plan: `NEXT_PLAN.md` (C7, exploratory); C8 is pre-registered in `PREREG_v3c.md`.

Generated tables:
- `c78.md` and `c78_runs.csv` (`llm/analyze_c1.py`): census and turnover per run;
- `c78_fixed_points.md` (`llm/analyze_kl_fixed_points.py`): question 2.

`c78_partial.md` was the first look, with 11 of 20 runs.

**Runs used:**
- KL runs broken by the unbounded log-ratio were replaced: r2_117 for r2_87, and r2_118 for r2_99. r2_111–116
  replace r2_105–110, which were stopped and rerun on the fixed code (`llm-transfer` `a7bd0aa`).
- r2_101 is r2_98 resumed (see HANDOFF).

## Results

Training success is on the 1,000 training problems, held-out success on 300 unseen problems. Declines and losses
are BH-tested at 10%, at 256 samples per problem.

| Setting | Training success | Held-out success | Held-out declines | Held-out lost |
|---|---|---|---|---|
| no KL, G = 4 | 0.99, 0.99 | 0.84, 0.83 | 15, 14 | 4, 3 |
| no KL, G = 16 | 0.99, 0.99 | 0.82, 0.84 | 26, 15 | 10, 7 |
| reverse 0.02, G = 4 | 0.92, 0.91 | 0.82, 0.80 | 0, 4 | 0, 0 |
| reverse 0.02, G = 16 | 0.94 (one run; the other broke) | 0.83 | 1 | 0 |
| reverse 0.1, G = 4 | 0.78, 0.79 | 0.71, 0.72 | 1, 0 | 0, 0 |
| reverse 0.1, G = 16 | 0.83, 0.82 | 0.75, 0.74 | 0, 0 | 0, 0 |
| reverse 0.5, G = 4 | 0.61, 0.62 | 0.58, 0.60 | 0, 0 | 0, 0 |
| reverse 0.5, G = 16 | 0.64, 0.64 | 0.60, 0.61 | 0, 0 | 0, 0 |
| forward 0.1, G = 16 | 0.83, 0.83 | 0.75, 0.75 | 0, 3 | 0, 0 |
| forward 0.5, G = 16 | 0.65, 0.65 | 0.61, 0.62 | 0, 0 | 0, 0 |

Training problems lost: 0 or 1 in every usable run.

## The questions

1. **Does KL prevent losses?** **Yes; prediction holds.** Without KL, 14–26 held-out problems decline and 3–10 are
   lost. With any β ≥ 0.02, there are 0–4 declines and 0 losses. The cost grows with β, in both training and
   held-out success. β = 0.02 keeps held-out success within about 0.02 of no KL while removing the declines.
2. **Does each problem end at its finite-G fixed point?** Partly: the predicted end state tracks the observed one,
   but adds nothing beyond base success (`c78_fixed_points.md`).
   - **Holds as stated:** the predicted p* and the observed end success are positively related. In logit space the
     slope is 1.04–1.09 at G = 4 and 0.74–0.97 at G = 16 (forward KL: 1.14–1.57). R² is 0.71–0.96 and Spearman
     0.86–0.98.
   - **But it beats no baseline:** fitting end success on base success alone gives the same R² in every run (to
     within 0.03). Here the fixed-point formula is in effect a monotone map of p_ref, so it carries no information
     about individual problems beyond their starting success.
   - **The β mapping is not one constant:** the fitted β_eff = s·β is 0.47–0.63 at β = 0.02, 1.0–1.3 at β = 0.1 and
     2.1–2.8 at β = 0.5. It grows roughly like √β, not in proportion. The plan's assumption of one free scale is
     wrong for this per-token KL estimator.
   - **Reading:** the H4 resting-point theory gives the right overall level and ordering in the LLM. It is not shown
     to predict individual problems, and its β does not translate to the per-token KL by a constant.
3. **Finite-G signature.** At every β, G = 16 settles higher than G = 4. Training success: 0.94 against 0.91–0.92
   at β = 0.02, 0.82–0.83 against 0.78–0.79 at β = 0.1, 0.64 against 0.61–0.62 at β = 0.5. That is the predicted
   direction: GRPO's weight W_G at low p grows with G, so the reward pushes harder against the same KL. This is
   aggregate only; the per-problem version (low-p_ref problems) and the bistability check were not run.

**C8 prediction (forward keeps more problems near base than reverse, with fewer declines): not supported.** At
equal β and G = 16, forward and reverse KL give the same training success (0.83 against 0.82–0.83 at β = 0.1;
0.65 against 0.64 at β = 0.5) and the same near-zero declines. This fits the toy picture: the two directions
differ only for answers whose probability goes to zero, and in the LLM nothing collapses.

## Numerical failures (deviation)

- **Reverse KL β = 0.02 at G = 16 broke in 2 of its 3 attempts:**
  - r2_99: NaN at step 713, on the old code;
  - r2_114: on the fixed code, at step 645, one update had a gradient norm of 1.1e7 (normal: 100–200, KL 2,310
    against about 7). Two steps later every answer ran to the 512-token limit and success was 0.
  - Only r2_118 finished, and its KL rose at the end (13.5 at step 800 against about 7).
- The fix in `a7bd0aa` bounds the log-ratio and skips non-finite updates, but training deliberately does not clip
  the gradient norm (`train.py:474`), so one extreme but finite update passes through.
- r2_114 is reported as a numerical failure and left out of the settings table.
- The forward-KL β = 0.1 seed-0 run broke the same way on the old code (r2_87). Its rerun (r2_117) is clean.
- **Reading:** small-β KL runs drift far from the reference (5–8 nats per sequence), and the per-token estimator
  then occasionally produces extreme values. A practical setup would clip gradients; we did not, to stay close to
  the theory.

## What this adds to the story

- In the LLM, the KL penalty does what the toy predicts for prompts that are *not* collapsing: it holds problems
  near the base model and removes the small held-out declines, at a cost that grows with β.
- The toy's forward-against-reverse difference needs collapsing answers, which the LLM does not have, so the
  directions behave alike here.
- G = 16 settling above G = 4 at equal β is the LLM's only H4-style finite-G signature, and it points the
  predicted way.
