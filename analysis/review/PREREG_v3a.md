# Pre-registration: v3 batch 1 (T1, T2, T4, T6, T8)

Written and committed before the batch launches. The code is on branch `toy-v3`; its commit hash is
recorded in each result's `DONE` file. Analysis code (branch `analysis-v3`) must implement exactly
these tests. Any later deviation is reported as a deviation.

**Disclosure.** Before writing this, the main session ran an exploratory local probe on the T1 task:
linear and MLP, ρ = 0.9, m ∈ {1/3, 0.9}, η ∈ {0.1, 1, 10}, `expected_full`, one task seed.
- At m = 0.9, about 10.4% of prompts (the minority share) were persistently absorbed at every η.
- At m = 1/3, nothing was absorbed at η ≤ 1, and much was absorbed at η = 10.

That probe is not part of the evidence.

## Shared definitions (T0)

- **Eligible prompt:** p_init ≥ 10τ.
- **Persistently absorbed at τ:** max p over the last 20% of steps < τ (`tail_max`).
- **Primary τ:** 1e-3. τ = 1e-2 is reported as a sensitivity check.
- **Outcome per run:** the persistent-absorption fraction among eligible prompts.
- **Unit of replication:** the task seed (10 per cell). Bootstrap CIs resample task seeds, 10,000
  draws.
- **Multiplicity:** Holm–Bonferroni across the final one-sided p-values of the primary hypotheses
  P1–P5 (P3–P5 are pre-registered before their own batches). Everything else is secondary and
  labelled exploratory.

## P1 (T1): absorption rises with majority share at fixed geometry

- **Cells:** arch ∈ {linear, MLP-64} × ρ ∈ {0.5, 0.9} × optimizer ∈ {SGD, Adam} × update ∈
  {`expected_full`, `sampled`}.
- **No-overshoot η set:** per cell, the η values at which the balanced arm (m = 1/3) has median
  persistent absorption ≤ 0.02 over tasks. If the set is empty, the cell is reported and excluded.
- **Test, per cell, over its no-overshoot η:**
  - (a) paired one-sided Wilcoxon signed-rank test over (task, η) pairs: absorption at m = 0.9 >
    absorption at m = 1/3;
  - (b) Spearman correlation between m and absorption over all (task, η, m) > 0.
- **P1 holds** if (a) and (b) both have p < 0.05 in every included SGD cell **and** every included
  Adam cell.
  - The pooled p-value fed to Holm is the largest (a) p-value over the included cells.
  - "Holds under SGD only" is reported if the Adam cells fail.
- **Descriptive:**
  - the share of absorbed prompts whose answer is a minority class;
  - the supervised control's absorption, predicted ≈ 0;
  - the per-class margin distributions.

## P2 (T2): kick size, not response noise, at equal mean drift

- **Cells:** level ∈ {lin, c05, m64, bal} × 2 η values (fixed per level), 10 tasks.
- **(a) Kick size.** For `expected`, the Spearman correlation between B ∈ {8, 32, 128, 512} and
  absorption is < 0 with p < 0.05, in ≥ 3 of the 4 levels (pooled over η and tasks within a level).
- **(b) Response noise.**
  - For every (level, B): mean over tasks and η of absorption(`sampled`) − absorption(`expected`).
  - Its 95% bootstrap CI must lie within ±0.05 (TOST equivalence) in every level, pooling B.
- **P2 holds** if (a) and (b) both hold. The p-value fed to Holm is the largest of the level-wise
  (a) p-values among the levels counted.
- **Descriptive:** `expected_full` at the same η, the kick → 0 limit.

## Secondary (exploratory)

- **T4 (H5 rerun).**
  - Best G by a deterministic tie-break: the smallest G whose median samples-to-target is within the
    tracking resolution (1 step × 256 samples) of the minimum, with a bootstrap CI over seeds.
  - Best η must be interior to the grid. If not, the cell is reported as edge.
  - Prediction: in tabular GRPO, the best G does not increase as p₀ increases (Spearman ≤ 0). Mean-rule
    curves are flatter than GRPO's.
- **T6 (ablations).**
  - Per condition: the share of runs with a material drop (max_drop > 1e-3), and the persistent
    absorption.
  - Predictions:
    - control: 0 material drops;
    - flip05, flip20 and sign: material drops and absorption at large η;
    - momentum, Adam and gradclip: no material drops;
    - ppo4 and multi4 (off-policy reuse): material drops;
    - kl01 and kl1: drops only towards or below p_ref, as A2 defines them.
- **T8 (KL without noise).**
  - Under `expected_full` there is no noise, so starts at offset ≠ 0 converge to the basin predicted
    by the side of the unstable root (share above = 0 for offsets < 0, 1 for > 0) at η ∈ {1, 3}.
  - At η = 10, report whether the deterministic update itself leaves the upper basin.
