# v3o and v3q: confirmatory replication battery (A1–A5)

**Confirmatory.** Pre-registered in `PREREG_v3d_addendum9.md` (`review` `e9510e4`) before launch:
- five tests, one Holm family at α = 0.05;
- on tasks that had never been run: toy task seeds 10–29 and transformer partition seeds 10–29;
- each test reuses the original batch's grid and definitions; only the seeds differ.

`extra/experiments/analyze_v3o.py` regenerates `FINDINGS.md`, `runs.csv`, `results.csv` and the per-test tables
(`a1_cells.csv` … `a5_cells.csv`).

## Jobs

| Batch | Commit | Grid | Runs | Failed |
|---|---|---|---|---|
| v3o (toy) | `toy-v3` `4391e1a`, clean, job digest `8b81b8a2e7c5` | `v3_a1` = `v3_t1` | 9,280 | 0 |
| | | `v3_a2` ⊂ `v3_t17` (β = 0 and forward KL β ∈ {0.03, 0.1, 0.3}) | 640 | 0 |
| | | `v3_a3` ⊂ `v3_t16` (GRPO, pass@4, unbiased pass@4) | 360 | 0 |
| | | `v3_a5` ⊂ `v3_t23` (m = 0.7; uniform and p0 = 0.5) | 160 | 0 |
| v3q (transformer) | `enum-env` `4b271a0`, clean, job digest `299bb326e002` | `v3_a4` | 440 | 0 |

- v3o has 10,440 jobs.
- v3q has 680 jobs: 440 for A4 and 240 for J (`v3_t29`, exploratory, analysed in `extra/analysis/v3q/SUMMARY_J.md`).
- The `v3_a4` grid is 11 tasks × 2 algorithms × 20 seeds = 440 runs. The addendum's 480 was an arithmetic error
  (deviation 8).

## Results

**All five tests are significant after Holm.** In every test all 20 tasks (or seeds) go in the predicted direction,
with no ties. The exact one-sided sign-test p is 2⁻²⁰ = 9.5e-7 in each test, and the Holm-adjusted p is 4.8e-6.

| Test | Claim | Statistic (fresh, 20 units) | Same statistic, original 10 units | p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| A1 | Absorption per minority prompt rises with the majority share (m = 0.9 against 1/3) | 20 of 20 tasks up in every one of the 16 cells | (original test: Wilcoxon, all cells p ≤ 0.0098) | 9.5e-7 (max over cells) | 4.8e-6 | **confirmed** |
| A2 | Forward KL (β 0.03–0.3) prevents absorption at 1e-3 | 20/20 up; median task difference 1.00 | 10/10, 1.00 | 9.5e-7 | 4.8e-6 | **confirmed** |
| A3 | pass@4 protects the minority at matched progress | 20/20 up; median difference in the minority mean 0.39 (range 0.27–0.45) | 10/10, 0.40 | 9.5e-7 | 4.8e-6 | **confirmed** |
| A4 | Adam lr 1e-3, transformer: minority absorbed at share ≥ 0.875, not when balanced | 20/20 up; mean seed difference 0.89 | 10/10, 0.86 | 9.5e-7 | 4.8e-6 | **confirmed** |
| A5 | A warm start at p0 = 0.5 protects the minority at m = 0.7 | 20/20 up; median task difference 1.00 (lowest 0.79) | 10/10, 1.00 | 9.5e-7 | 4.8e-6 | **confirmed** |

## Effect sizes against the originals

### A1 (P1, v3a)

Per minority prompt, the absorption rate rises monotonically with m in 16 of 16 cells (15 of 16 originally). At
m = 0.9 it is 0.37–1.00 (originally 0.54–1.00). At balance it is at most 0.045 (originally at most 0.037).

The effect is smaller on the fresh tasks in three cells:
- linear, ρ = 0.5, Adam (exact and sampled): m = 0.9 rate 0.37 and 0.41, against 0.54 originally;
- MLP, ρ = 0.5, Adam, exact update: 0.70, against 0.84.

The two linear ρ = 0.5 SGD cells are also a little lower (0.77 against 0.87). Elsewhere the rate is within 0.03 of the
original.

The original P1 statistic (the absorbed fraction of all prompts, Wilcoxon) also holds on the fresh tasks in all 16
cells, with largest p 0.0023. Its weakest cell is linear/0.9/sgd/sampled, with 13 of 20 tasks up.

### A2 (T17, v3g)

Secondary measures, over forward-KL runs at β ∈ {0.03, 0.1, 0.3}:

| | Fresh (480 runs) | Original (240 runs) |
|---|---|---|
| No minority prompt absorbed at 1e-3 | **480/480** | 237/240 |
| No minority prompt absorbed at 1e-2 | **442/480** (β = 0.03: 123/160; 0.1: 159/160; 0.3: 160/160) | 217/240 (β = 0.03: 57/80; 0.1 and 0.3: all) |
| Both thresholds | 442/480 | 217/240 |
| Minority share absorbed at 1e-2, β = 0.03 | 7.8% (worst cell: MLP, sampled, η = 1: 0.46) | 10.9% (worst cell 0.66) |
| Minority mean when the majority reaches 0.80 (median) | 0.021 (β 0.03 / 0.1 / 0.3: 0.007 / 0.022 / 0.063) | 0.020 (0.006 / 0.021 / 0.062) |
| T17 prediction 3 (none absorbed, majority reaches 0.80, minority below 0.2 there) | 23 of 24 cells | 23 of 24 cells |
| No-anchor runs fully absorbed | 160/160 | 80/80 |

- The registered claim is about the 1e-3 threshold.
- At 1e-2, the qualification from the v3g corrections still applies: β = 0.03 keeps the minority above a floor, but
  not always above 1e-2.
- **Forward KL keeps the minority alive but does not shield it.** The minority still falls to about 0.02 while the
  majority is learned, and 55 of 480 forward-KL runs end with the minority below 0.05.

### A3 (T16, v3f)

At the matched point the minority mean under pass@4 is 0.28–0.44 per cell (originally 0.31–0.42). Under GRPO it is
5e-5 to 5e-3, below 0.01 in 6 of 6 cells (originally 6 of 6). No minority prompt is absorbed under pass@4, and all are
absorbed under GRPO, as in the original.

The original prediction 1 (minority above 0.2 in ≥ 90% of the runs that reach the matched point) holds in 9 of 12
pass@4 cells (10 of 12 originally). It fails in:
- the two unbiased cells at η = 0.03, which never reach 0.80 (as in the original);
- one new cell: biased pass@4, G = 32, η = 1, with 14 of 20 runs above 0.2.

The cost is unchanged. At η = 0.03 the biased form's majority reaches 0.80 in only 2/20 (G = 8) and 11/20 (G = 32)
runs, and the unbiased form's never does. Those 67 runs count as 0 against pass@4 in the test.

### A4 (T20, v3h)

- **The test's class-level rule:** the minority is absorbed while class 0 is not in 152 of 160 high-share runs
  (originally 79 of 80) and in 5 of 80 balanced runs (originally 5 of 40).
- **Per prompt, under the exact update:** the minority rate is 0.20 at balance and 0.91 at high share (originally 0.18
  and 0.94). Class 0 is untouched at high share (0.0002 per prompt).
- **The v3h correction still applies:** at balance about a fifth of *each* class's prompts are absorbed (class 0:
  0.20). So "not when balanced" holds at the class level (half the prompts), not per prompt.
- **With sampled GRPO the balanced arm is a little worse:** 0.26 per prompt; 4 of 40 runs lose the minority class and
  7 of 40 lose class 0.

### A5 (T23, v3k)

- At m = 0.7 the uniform start absorbs every minority prompt in all 80 runs.
- At p0 = 0.5 the share absorbed is 0.022 (linear, both updates) and 0.0003–0.003 (MLP), against originally 0.07–0.14
  and 0–0.004.
- With p0 = 0.5 the minority ends at 0.75–0.76 (linear) and 0.99–1.00 (MLP), against originally 0.78 and 1.00.
- The pre-set reading "gone or much reduced" holds in all 4 cells, as originally.

## What is now confirmatory

All five exploratory findings replicated on fresh tasks and become confirmatory results:

1. **A1:** at fixed feature geometry, the per-prompt absorption rate of the minority rises with the majority share, in
   every (architecture, ρ, optimizer, update) cell.
2. **A2:** a forward-KL anchor to the start policy at β ∈ {0.03, 0.1, 0.3} prevents absorption at the 1e-3
   threshold. The scope is the same as in the original: the reference still solves the minority (see also G in
   `extra/analysis/v3p/SUMMARY.md`), and at 1e-2 the protection is partial at β = 0.03.
3. **A3:** pass@4 keeps the minority at about 0.4 when the majority reaches 0.80, where GRPO has it near 0. The cost is
   a slower majority, and at η = 0.03 the majority often never gets there.
4. **A4:** in the transformer environment under Adam (lr 1e-3), the minority class is absorbed at a high majority
   share and not at balance, by the class-level rule.
5. **A5:** a class-balanced warm start at p0 = 0.5 removes almost all of the absorption at m = 0.7.

**None failed to replicate.**
- **Effect sizes held,** with one exception: the A1 rate at m = 0.9 is smaller in three ρ = 0.5 cells, though still
  far above the balanced rate in every task.
- **Secondary points that only nearly hold** are the same as before: A2 at 1e-2, A3 at η = 0.03, A4 per prompt at
  balance. There is also one new failing A3 cell (pass@4, G = 32, η = 1, by the original 90% rule).

## Deviations and readings

1. **A1's test statistic.** The original task-level test (`analyze_v3.p1_task_level`) used an exact Wilcoxon on the
   absorbed fraction of *all* prompts. The addendum registers a sign test on the *per-minority-prompt* rate (absorbed
   minority prompts ÷ minority prompts, as `minority_rate` in that function). That sign test is the confirmatory
   statistic. The original statistic is reported alongside it (`persist_up`, `persist_wilcoxon_p`). A1's p is the
   largest p over the 16 cells, as registered.
2. **A1's η sets** are recomputed on the fresh tasks with the original rule: the η where the balanced arm's median
   persistent absorption is ≤ 0.02, now over 20 tasks. They match the original sets except in two cells, which each
   gain one larger η (linear/0.9/sgd/sampled: η = 3; mlp/0.9/sgd/expected_full: η = 1). The original sets give the
   same result (16/16 cells, 20 of 20 tasks).
3. **A2's measures.** "Minority absorption share" is `minority_absorbed` from `analyze_v3g.t17_row` (absorbed
   minority prompts ÷ minority prompts). The original code had no 1e-2 measure, so the secondary uses v3k's
   definition: tail max below 1e-2 among prompts starting at or above 1e-1. The original's 1e-2 numbers were
   recomputed from the raw v3g runs with that definition, and they reproduce the 11% at β = 0.03 in the v3g
   corrections.
4. **A3's medians.** The medians over the 6 (G, η) cells are taken separately for pass@4 (best of the two forms per
   cell) and for GRPO, and then differenced. The other reading, the median of the per-cell differences, gives the
   same 20/0. No GRPO run failed to reach 0.80, so that unregistered case never arose.
5. **A4's definitions.** "Exact update, both algorithms" is read as both algorithms in the `v3_a4` grid: `pg` (exact)
   and `grpo` (sampled, G = 8). The exact update alone gives the same result (20/0, mean difference 0.89). The rest
   follows `analyze_v3h`:
   - the minority is all non-zero classes pooled (`absorbed_b`);
   - "absorbed" means `absorbed_b ≥ 0.5` while `absorbed_a < 0.5`;
   - high share means n_major/64 ≥ 0.875 (n_major 56 and 60, for K = 2 and 3);
   - balanced means n_major = 22 at K = 3 and 32 at K = 2, the definition of v3h prediction 1b.
6. **Labels.** Each toy task's class labels are built once per (ρ, m, task seed) with `analyze_v3.t1_labels` and
   shared across tests. On one A5 MLP warm-start task, they match the per-run rebuild in `analyze_v3k.read_one`.
   The original per-run readers (`t17_row`, and the `read_one` of v3f, v3k and v3h) run unchanged, with the cached
   labels replacing their per-run rebuilds. A1 repeats the T1 branch of `analyze_v3.read_one`, because that function
   only accepts the hypothesis name `v3_t1`.
7. **Original comparison data:**
   - `extra/analysis/v3f/runs.csv`, `extra/analysis/v3h/runs.csv` and `extra/analysis/v3k/runs.csv`;
   - the raw v3g runs, for A2's 1e-2 measure;
   - `extra/analysis/v3a/P1_TASK_LEVEL.md` and `extra/analysis/v3a/tables/v3_t1_cells.csv`, for A1.
8. **A4's run count.** The addendum states 480 runs for `v3_a4`; the grid as committed has 440 (11 × 2 × 20). The
   difference is an arithmetic error in the addendum, not a change to the grid.
