# v3 batch 1: what the results say

All 17,279 runs returned and none failed. The tests follow `analysis/review/PREREG_v3a.md` (branch
`review`, commit `dc5736e`), and the code ran from `toy-v3` at `108055d`. The numbers are in
`FINDINGS.md`, which `experiments/analyze_v3.py all` regenerates from the raw results.

## Verdicts

| Test | Pre-registered verdict | In one line |
|---|---|---|
| **P1** (T1) | **PASS**, in all 16 cells | With the geometry fixed, persistent absorption rises with the majority share, under both SGD and Adam |
| **P2** (T2) | **FAIL** | Kick size matters (3 of 4 levels). Response noise is **not** negligible at B = 8 |
| T4 (exploratory) | prediction holds | Tabular GRPO's best G is 2 at every p₀; the old 32/16/4/2 pattern was an η-grid artifact |
| T6 (exploratory) | 2 predictions wrong | PPO-style batch reuse, clipped or not, causes **no** drops in the tabular case |
| T8 (exploratory) | **PASS**, 60/60 | Without noise, KL runs settle on the side of the unstable root they start on, even at η = 10 |

Holm runs across P1–P5 once P3–P5 are in. For now, P1's p-value is 3.0e-8, so even multiplied by
5 (Bonferroni, a bound on Holm) it is 1.5e-7. P2 failed its pre-registered rule, so its p-value
(0.0027, from the kick-size part) does not matter.

## P1: imbalance causes minority-answer collapse

- **What gets absorbed.** When absorption happens and m > 1/3, it takes the **whole minority-answer
  set** and nothing else:
  - absorbed ÷ minority share has a median of exactly 1 for every m and both optimizers;
  - 100% of absorbed prompts are minority-class at m = 0.7 and m = 0.9, and 97.8% at m = 0.5.
- **What drives it.** It appears at the smallest η and under the deterministic flow
  (`expected_full`), so it is a mean-drift effect, not noise. At ρ = 0.9 even m = 0.5 loses the
  whole minority half.
- **The balanced arm** (m = 1/3) absorbs nothing until η overshoots. Past that point it loses about
  two thirds of prompts, which means the policy collapses onto a single answer.
- **Supervised control.** Cross-entropy on the same tasks absorbs nothing (0 in all 16 cells).
- **Margins.** Minority-class prompts have smaller margins (top score minus second), and the gap
  widens with m. At ρ = 0.9 and m = 0.9 the median is 0.52 against 2.11.
- **Caveat.** The no-overshoot η set removes the large-η cells, where the balanced arm absorbs too.
  The claim covers small steps only.

## P2: kick size matters, but so does response noise at small batches

- **(a) Kick size: holds in 3 of 4 levels.** With the mean step held equal, absorption falls as B
  grows in lin, m64 and bal (Spearman −0.87, −0.72, −0.31).
  - c05 is the exception: its absorption is flat in B and present under `expected_full` (0.06–0.22).
  - This fits the review's finding that c05 absorption is minority collapse, which is driven by the
    mean step, not the kick.
- **(b) Response noise: fails in 3 of 4 levels.** `sampled` minus `expected` averages +0.03 to
  +0.04, and its 95% CI crosses +0.05 in bal, c05 and m64.
  - All of the excess is at **B = 8**: +0.11 on average (+0.08 to +0.14 per level).
  - At B ≥ 32 the mean gap is +0.008.
- **Consequence.** The review claim that "response sampling does not matter" (SUMMARY item 3) is
  true only from B = 32 up. At B = 8, sampled responses add absorption on top of the kick.
- **Consistency check.** At B = 512 without replacement, `expected` is the full expected step, so it
  should match `expected_full` exactly. It does in 66 of 80 runs. The 14 mismatches are all in
  overshooting runs: bal at η = 10, and lin at η ≥ 300 by ≤ 0.008. That is floating-point divergence
  in unstable runs, as with h4/01500.

## T4: best group size (H5 rerun)

- **Tabular:** with η now up to 1000, the best G for GRPO is 2 at every p₀, with CIs within {2, 4}.
  The earlier "smallest tied G" table (32, 16, 4, 2) came from an η grid that was too narrow.
- **Not a true optimum.** At p₀ ≤ 0.1, small G still has its best η at the top of the grid. So
  G = 2 is "fastest up to η = 1000", a saturation point rather than an interior optimum.
- **Mean rule against GRPO.** Their spreads over G are identical in the tabular case. With every
  prompt at the same p₀, GRPO's normalisation is close to a constant rescale, and tuning η absorbs
  it. The "mean rule is flatter" prediction held in only 17% of (level, p₀) and is not meaningful
  here.
- **Shared levels** (lin, m64): the best G is 2–8 with wide CIs. There is no clean trend.

## T6: ablations of the tabular "no decrease" result

- **No drops anywhere in:** control, momentum, gradient clipping, Adam, β = 0.1, **4 reused
  epochs**, and **PPO clipping with 4 epochs**.
  - The last two were predicted to cause drops; they don't.
  - Those runs did learn (final mean 0.95), so the result is not a stalled run.
  - Monotonicity in the tabular case survives off-policy reuse of the batch.
- **Drops and absorption appear with:**
  - a noisy verifier: flip 5%, drops in 74% of runs; flip 20%, in 90%;
  - the sign rule: 68%;
  - β = 1, but only at η ≥ 30, which is overshoot.
- **Absorption grows with η** in each of those conditions.
- **The KL prediction cannot be tested as written.** The reference is the initial policy, so
  p_ref = p_init, and every logged drop below p_init is a drop below p_ref. Telling the two kinds
  apart needs the running maximum of p, which is not logged.

## T8: KL bistability without noise

- In all 99 runs, the final p is on the same side of the unstable root as the start (offset 0
  falls to the lower basin). That covers η = 10 too.
- So the deterministic update never switches basins on its own. The switching seen in the stretch
  batch needs sampling noise.

## Deviations from the pre-registration

1. **T4 bootstrap:** 1,000 resamples of seeds instead of 10,000, because of cost. It has 5 seeds,
   not the 10 the plan asked for (the grid had 5), and no saturation stop rule. Cells whose best η is
   at the grid edge are listed in the `edge_eta` column.
2. **T4 verdict.** When the best G is the same at every p₀, Spearman is undefined. The script then
   counts "does not increase" as holding.
3. **T6 KL** cannot be tested as written (see above).
4. **T1 init seeds:** one init seed per task (seed 0), as in the grid. The plan said 2. The
   pre-registration's unit of replication, the task seed, is unaffected.
5. **P2(b):** uses the 95% CI as pre-registered, not the 90% CI of a standard TOST. Per-B values are
   added as a description.

## What this changes for the next batches

- **C2** (LLM minority procedure) is the right transfer test. The toy result is now a clean
  confirmatory pass, under both SGD and Adam.
- **Fix the noise wording.** "Kick size, not response noise" becomes "kick size, and response noise
  at small batches". C1/C2 use 64 prompts × G = 8, so B = 64, the range where noise added little in
  the toy.
- **T6/C5.** PPO clipping is not a loss driver in the tabular toy. A noisy verifier and the sign rule
  are, which supports keeping the verifier flip and the sign rule in C5.
- **T5, T7, T9 and the P3 pre-registration are still to build.** P3 has to be committed before T5
  launches.
