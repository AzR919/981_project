# PREREG_v3d, addendum 9 (2026-11-24): confirmatory replication battery (A1–A5)

Committed before batches `v3o` (toy, `toy-v3` `4391e1a`) and `v3q` (transformer environment, `enum-env`
`v3_a4`) launch. **Confirmatory.** Five tests, one Holm family at α = 0.05. Fresh tasks: toy task seeds
10–29 and transformer partition seeds 10–29; none has been run before. Unit of replication: the task seed.
Each test reuses the original batch's grid and definitions unchanged; only the seeds differ.

| Test | Original (exploratory) | Grid |
|---|---|---|
| A1 | P1, v3a: absorption rises with the majority share | `v3_a1` = `v3_t1` (9,280 runs) |
| A2 | T17, v3g: forward KL prevents absorption | `v3_a2` (640) |
| A3 | T16, v3f: pass@4 protects the minority at matched progress | `v3_a3` (360) |
| A4 | T20, v3h: under Adam the minority is absorbed at a high share and not when balanced | `v3_a4` (480) |
| A5 | T23, v3k: a warm start at p0 = 0.5 protects the minority at m = 0.7 | `v3_a5` (160) |

Absorbed: tail max below 1e-3 over the last 20% of steps, among prompts starting at or above 1e-2 (as in
the originals).

## Tests and p-values

- **A1.** The P1 task-level test of `analysis/v3a/P1_TASK_LEVEL.md`, unchanged: per cell (architecture, ρ,
  optimizer, update; 16 cells), each task is averaged over the cell's no-overshoot η set; one-sided sign
  test that the per-minority-prompt absorption rate at m = 0.9 exceeds that at m = 1/3, across 20 tasks.
  A1's p-value is the largest of the 16 cell p-values (all cells must hold).
- **A2.** Per (architecture, update, η) cell (8 cells) and task: minority absorption share at β = 0 minus
  the largest share among forward KL β ∈ {0.03, 0.1, 0.3}. One-sided sign test (> 0) over the 20 tasks
  pooled over cells by taking each task's median over cells. Secondary, reported: share of forward-KL
  runs with no minority prompt absorbed at 1e-3 and at 1e-2 (the "both thresholds" of the original).
- **A3.** Matched point: first logged step at which the majority class mean reaches 0.80. Per task: the
  minority class mean at the matched point under pass@4 (best of the two forms, per (G, η) cell, median
  over cells) minus under GRPO. Runs where the majority never reaches 0.80 count as 0 for pass@4 (against
  it). One-sided sign test (> 0) over 20 tasks.
- **A4.** Adam, lr 1e-3, shared transformer, exact update, both algorithms. Per partition seed: the
  fraction of high-share tasks (share ≥ 0.875) whose minority is absorbed (at least half of its prompts
  below 1e-3 while class 0 is not) minus that fraction for balanced tasks. One-sided sign test (> 0) over
  20 seeds.
- **A5.** Per (architecture, update) cell and task: minority absorption share at the uniform start minus
  at p0 = 0.5 (m = 0.7). Per task the median over the 4 cells; one-sided sign test (> 0) over 20 tasks.

Holm over the five p-values at α = 0.05. Ties (difference exactly 0) are dropped from a sign test.

## Reading

Each test that survives Holm becomes a confirmatory result in the report; each that fails is reported as
"did not replicate on fresh tasks", with the exploratory result kept as exploratory.

## Deviations

Listed in the v3o/v3q write-up.
