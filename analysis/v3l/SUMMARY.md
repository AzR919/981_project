# v3l: the late phase, prompt by prompt (T24)

All 120 runs returned and none failed (`toy-v3` `5487480`, clean). Pre-registered in
`PREREG_v3d_addendum6.md` (`review` `93743f0`), with the prediction rule fixed before launch.
`experiments/check_late_time.py` regenerates `FINDINGS.md`, `runs.csv` and `prompts.csv`.

Setup: linear policy, SGD, exact update, K = 3, ρ ∈ {0.5, 0.7, 0.9}, m ∈ {0.7, 0.9}, η ∈ {0.1, 1},
100,000 steps, 10 tasks; every prompt's success stored at log-spaced steps. 118 of 120 runs count.

## Predictions

| Prediction | Result |
|---|---|
| 1. The reduced model predicts each prompt's final success (median Spearman ≥ 0.8) | **Fails: 0.28** |
| 2. It beats persistence, push-only and one-γ baselines in ≥ 80% of runs | **Fails:** 69%, 67%, 53% |
| 3. Collective restoring beats self-only restoring in ≥ 70% of recovering runs | **Fails narrowly:** 67% of 36 |
| 4. Recovery counts within 25% in ≥ 70% of recovering runs | **Holds:** 72% |
| 5. The run's smallest γ ranks its late margin growth (within-cell median ≥ 0.6) | **Holds: 0.86** (12 cells) |

## Reading

- **What holds, and replicates on fresh pre-registered runs:** the majority's max-margin geometry
  sets how fast a run's minority sinks late in training (prediction 5; 0.78 on T12, 0.86 here).
- **The late phase barely reorders prompts.** In cells where few recover, a prompt's success at the
  start of the late phase predicts its final success with Spearman 0.97–1.00. Which prompts end up
  deepest is decided before the late phase, during the fall.
- **The per-prompt model is not right quantitatively.** Where most prompts stay down it is close in
  absolute terms (0.4–0.6 decades of error, against 3.6–4.9 for persistence), but simple baselines
  are nearly as close. At ρ = 0.9, m = 0.7, where γ ≈ 1, its collective restoring term tips about 900
  prompts into a predicted recovery that does not happen (simulated: 0).
- **Consequence for the theory:** the late phase is described at the level of the run (its sinking
  rate), not of the prompt. The per-prompt outcome is set in phase 1; a theory of the fall is the
  missing piece.

## Deviations

None from the prediction rule. The analysis also reports, descriptively, the per-prompt rank
correlation of the state at τ0 with the final state (not a registered prediction).
