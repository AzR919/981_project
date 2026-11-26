# v3p: a collapsed KL reference (G, T27) and Adam's epsilon (H, T28)

All 820 runs returned and none failed (260 G, 560 H). Launched from `toy-v3` `4391e1a`. Pre-registered in
`PREREG_v3d_addendum10.md` (`review` `e9510e4`); exploratory. `extra/experiments/analyze_v3p.py` regenerates
`FINDINGS.md`, `runs.csv`, `cells.csv` and `h_gamma_corr.csv` (the per-cell γ correlations behind H2).

## G (T27): forward KL toward a collapsed reference

**Setup:**
- the T1 minority task (K = 3, ρ = 0.9, m = 0.9), linear and MLP, SGD η = 0.1, exact update, 20,000 steps,
  10 tasks;
- from 0.5%, 5% or 25% of training (steps 100, 1,000, 5,000), a forward KL term with β ∈ {0.01, 0.1, 1} pulls toward
  the policy as it was at that step;
- controls: no anchor, and forward KL toward the start policy from step 0 (T17).

**The fall is fast.** Without an anchor, the median minority prompt drops below 1e-2 by step 15, and the last one by
step 100 (linear) or 30 (MLP). So in all 180 late-anchor runs the reference had already collapsed. At the 0.5%
anchor the minority mean is a median 1.3e-3 (linear) or 2.5e-4 (MLP), and it is lower at later anchors.

| | Minority prompts absorbed (mean share) | Runs with none absorbed | Minority at the end (median) | Majority at the end (median) |
|---|---|---|---|---|
| no anchor | 1.00 | 0 of 20 | 4e-7 to 7e-6 | 1.00 |
| start anchor, β = 0.01 | 0.01–0.06 | 13 of 20 | 0.004 (linear), 0.53 (MLP) | 1.00 |
| start anchor, β = 0.1 | **0** | **20 of 20** | 0.36 (linear), 0.97 (MLP) | 0.95–0.97 |
| start anchor, β = 1 | **0** | **20 of 20** | 0.28 (linear), 0.73 (MLP) | 0.69 (linear), 0.73 (MLP) |
| anchor at 0.5%, any β | 0.86–1.00 | 0 of 60 | 1e-6 to 6e-4 | 1.00 |
| anchor at 5% or 25%, any β | 1.00 | 0 of 120 | 5e-7 to 4e-5 | 1.00 |

| Prediction | Pass rule | Result |
|---|---|---|
| G1. Anchors taken after the fall do not restore the minority | Minority at the end ≤ 10 × its value at the anchor step, in ≥ 80% of late-anchor runs below 1e-2 at the anchor | **Holds: 180 of 180 (100%)**; every arch × anchor × β cell is 10 of 10 |
| G2. The start anchor prevents absorption at β ≥ 0.1 | No minority prompt absorbed in ≥ 90% of runs | **Holds: 40 of 40 (100%)** |
| G3. Does an early anchor (0.5%) protect like the start anchor? (reported) | none | **No.** It absorbs 86–100% of minority prompts at every β (0 of 60 runs absorb none), against 0% for the start anchor at β ≥ 0.1. But 0.5% is not before the fall (deviation 1) |

### Reading

- **By the registered reading, forward KL protects only while the reference still solves the minority.** Prediction
  1 holds. Toward a collapsed reference forward KL does nothing useful.
- At β = 1 it slows the further sinking: the minority ends near its value at the anchor, instead of 10–100 times
  below it. It never brings the minority back.
- In this toy a reference is still "good" only for the first 15–100 of 20,000 steps, so any reference taken from a
  trained policy is already collapsed.
- T17's start-anchor protection replicates exactly (G2), and so does its cost: at β = 1 the majority is held to
  0.69–0.73.

## H (T28): Adam's epsilon

**Setup:**
- the T1 task, linear and MLP, ρ ∈ {0.5, 0.9}, m ∈ {0.7, 0.9} (8 cells), exact update, 10,000 steps, 10 tasks;
- Adam with ε ∈ {1e-8, 1e-5, 1e-3, 1e-2, 0.1, 1} at lr = max(1e-3, 0.1 ε): lr is 1e-3 up to ε = 1e-2, then 0.01 and
  0.1;
- control: SGD at η = 0.1;
- **recovered:** the minority mean fell below 1e-3 at some logged step, and is above 0.1 at the end.

Share of runs recovered (of 10):

| Cell | ε = 1e-8 | 1e-5 | 1e-3 | 1e-2 | 0.1 | 1 | SGD |
|---|---|---|---|---|---|---|---|
| linear, ρ 0.5, m 0.7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| linear, ρ 0.5, m 0.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| linear, ρ 0.9, m 0.7 | 0.2 | 0.2 | 0 | 0 | 0 | 0 | 0 |
| linear, ρ 0.9, m 0.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| MLP, ρ 0.5, m 0.7 | 0.5 | 0.6 | 0.5 | **0** | 0.2 | 0.3 | 0.2 |
| MLP, ρ 0.5, m 0.9 | 0.6 | 0.5 | 0.1 | 0 | 0.1 | 0.1 | 0.1 |
| MLP, ρ 0.9, m 0.7 | 0.3 | 0.1 | 0 | 0 | 0 | 0 | 0 |
| MLP, ρ 0.9, m 0.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

| Prediction | Pass rule | Result |
|---|---|---|
| H1. The recovery share falls from ε = 1e-8 to ε = 1 and ends within 0.1 of SGD | Share(ε = 1) < share(ε = 1e-8), and within 0.1 of SGD, in ≥ 6 of 8 cells | **Fails: 4 of 8.** "Within 0.1 of SGD" holds in 8 of 8, "decreases" in 4 of 8. The 4 failing cells have no Adam recovery at ε = 1e-8 either (0 of 10), so there is no gap to close. With ≤ in place of <, 8 of 8 |
| H2. Late exponent and γ_min under each ε, with the ℓ∞ max-margin γ (reported) | none | See below |

**Per-prompt absorption** tells the same story in more detail:
- MLP at ρ = 0.5, m = 0.7: the share of minority prompts absorbed is 0.23–0.30 for ε ≤ 1e-3, and 0.63–0.64 for
  ε ≥ 1e-2 and under SGD.
- ρ = 0.9, m = 0.7: 0.70 (MLP) and 0.84 (linear) at ε = 1e-8, and 0.95–1.00 from ε = 1e-5 (MLP) or 1e-3 (linear) on.

**H2:**
- **The late exponent g** (T12's M2: the minority's margin growth relative to the majority's over the last decade)
  is 0.98–1.03 at ρ = 0.9 under every ε and SGD. At ρ = 0.5 it is near 0 or negative for small ε (recovering
  runs), and returns to SGD's value from ε = 1e-3 or 1e-2.
- **γ_min** depends only on the task, so it is the same under every ε: median over tasks 0.80–1.01 (ℓ2) and
  0.77–1.05 (ℓ∞). Across tasks the two γ_min have Spearman 0.66–0.82.
- **Which γ ranks the late exponent better** within a cell (linear, median over the 4 (ρ, m) cells):
  - under Adam at ε ≤ 1e-3: ℓ∞ 0.89–0.92 against ℓ2 0.77–0.79;
  - at ε ≥ 0.1 and under SGD: ℓ2 0.81–0.84 against ℓ∞ 0.72–0.78.
- That is the direction expected if Adam follows the ℓ∞ geometry (Zhang, Zou and Cao 2024) and SGD the ℓ2 one. But
  each correlation rests on 10 tasks, the two γ are correlated, and the gap is not consistent from cell to cell: a
  hint, not a result.

### Reading

- **As registered, H1 fails.** The data come close to the hypothesis, but not to what was written down.
- Wherever Adam recovers more than SGD (4 cells), the advantage disappears as ε grows and reaches the SGD value by
  ε = 1. In the other 4 cells Adam does not recover at small ε either.
- **The clearest evidence for the per-coordinate normalisation comes from a comparison the prediction did not name.**
  ε = 1e-3 and ε = 1e-2 share the same lr (1e-3). Between them, the MLP at ρ = 0.5, m = 0.7 goes from 5 of 10 runs
  recovered to 0 of 10, while absorption doubles. The switch happens over one decade of ε at fixed lr, which
  points to the normalisation rather than the step size.
- That cell's non-monotone tail (0, 0.2, 0.3 at ε = 1e-2, 0.1, 1) comes with a tenfold lr increase at each step. It
  is within what 10 runs can resolve.

## Deviations

1. **G3's premise does not hold.** The pre-registration calls the 0.5% anchor "before the fall". The fall ends by
   step 100 (linear) or 30 (MLP), so every anchor in the grid comes after it. G3 therefore compares the start anchor
   with an already collapsed reference. Whether an anchor taken after step 0 but before the fall protects is not
   tested.
2. **G1's denominator.** "Such runs" was read as late-anchor runs whose minority class mean (class B,
   `history[].class_mean`) is below 1e-2 at the logged anchor step: all 180. The test is pooled; every cell gives
   10 of 10.
3. **G2** uses T17's rule: no minority prompt absorbed (tail max below 1e-3, among prompts that start at or above
   1e-2), counted per run and pooled over arch and β ∈ {0.1, 1}.
4. **H1's share** is over all 10 runs of a cell; a run that never falls below 1e-3 counts as not recovered.
   "Decreases" was read strictly, and that strict part is what fails. The share among runs that fell (in
   `FINDINGS.md`) gives the same verdicts.
5. **H2.** "Late exponent" = T12's M2 measure g. γ is computed on the input features for both architectures, which
   describes the MLP only loosely. The ℓ∞ direction comes from an LP (min ‖u‖∞ s.t. φ_A u ≥ 1, scipy HiGHS) that
   need not have a unique solution; the solver's vertex is used.

## What it means for the project

**G closes a gap in the KL story.**
- Forward KL toward a reference protects the minority only while that reference still solves it.
- In the toy only the start policy qualifies, because the minority collapses within the first 0.5% of training.
- For LLMs, the protection therefore depends on the base or SFT reference still solving the minority. An anchor
  refreshed from a trained checkpoint protects nothing.

**H does not confirm its registered prediction, but it narrows down Adam's recovery advantage.**
- The advantage exists only in some cells, mostly the MLP at ρ = 0.5.
- There it switches off between ε = 1e-3 and 1e-2 at fixed lr, which supports per-coordinate normalisation as the
  mechanism.
- The report should cite the strict failure (4 of 8) and present the fixed-lr comparison as post hoc.
