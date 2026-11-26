# Late phase: what decides whether the collapse lasts? (working note, 2026-11-23)

**Status: a hypothesis, partly supported and partly refuted.** Nothing here is a claim yet.

## The idea

The first-step identity (`odds_law.md`) says whether the minority starts to fall; it does not
predict the outcome (T22, T23). For the outcome, the late phase matters:

- Late in training the majority's prompts are nearly solved, so the majority's updates behave like
  a loss with an exponential tail on separable data. For the linear policy, gradient descent then
  converges in direction to the **maximum-margin separator** û of the majority's prompts, with the
  weights growing like log t (Soudry, Hoffer, Nacson, Gunasekar and Srebro, "The implicit bias of
  gradient descent on separable data", JMLR 2018). The group rule's weight changes only the time
  scale (for pass@k the tail is e^{−kM}).
- A minority prompt y is then pushed toward the majority's answer at rate γ_y = û·φ_y (û scaled so
  the majority's closest prompts have margin 1), so its success decays like t^{−γ_y}, while its
  own restoring signal fades with its success (Schaul et al.'s f(0) = 0).
- **Per-prompt prediction:** y stays absorbed if γ_y > 1 and recovers if γ_y < 1.

## Test on existing data (T12: 400 linear-policy runs at 10× the horizon; no new runs)

`experiments/probe_late_time.py` (rows in `late_time_probe_rows.csv`). û is the hard-margin
solution on the majority's features (dual solved with nonnegative L-BFGS).

1. **Run level: supported.** The smallest γ_y among a run's minority prompts ranks the measured
   margin growth g of M2 (`analysis/v3d/SUMMARY.md`): Spearman 0.89 (SGD, exact update), 0.85 (SGD,
   sampled), 0.76 and 0.84 (Adam). The block average k_BA/k_AA does worse (0.50–0.70).
2. **Prompt level: refuted as stated.** Among minority prompts that fell below 1e-3:

   | Regime | γ_y < 1 | recovered (final > 0.1) | γ_y ≥ 1 | recovered |
   |---|---|---|---|---|
   | SGD, exact | 147 | **0%** | 9,434 | 12% |
   | SGD, sampled | 149 | 1% | 9,413 | 11% |
   | Adam, exact | 90 | 78% | 5,743 | 33% |
   | Adam, sampled | 99 | 9% | 5,944 | 9% |

   Under SGD, the regime the derivation is about, the per-prompt criterion points the wrong way.
3. Values: γ_y has median 1.2–1.6 and ranges from 0.2 to 2.9; at ρ = 0.9 the smallest values are
   0.83–0.88, where M2 measured g ≈ 1.

## Reading and next step

- The majority's max-margin geometry seems to set how fast the minority sinks, as a whole.
- Whether a prompt comes back is not decided by its own γ_y. The likely missing piece is collective:
  a minority prompt's restoring signal comes mostly from its classmates (the kernel within its
  class times their success), so recovery is a class-level event. The next version should treat
  the minority class as a block with its own restoring term and predict class-level recovery,
  tested on T12 and T18.
- Until that test passes, the report should state only the run-level association, as an
  observation, and the first-step identity as the theory.

## Update, 2026-11-23 evening: three further tests

1. **Formal and symbolic checks.** The first-step algebra is proved in Lean 4 with Mathlib
   (`formal/Formal/RLVR.lean`: tabular monotonicity, the cross-class factor, the push ratio in block
   and odds form). The late-phase algebra is checked in SymPy (`tests/test_theory_symbolic.py`),
   including an exact solution of the scalar late model dL/dτ = γ/τ − c e^{−L}: with y = e^L the
   equation is linear, and permanent absorption needs γ > 1 *and* a deep enough fall.
2. **The depth comparison above was partly circular** (a prompt that never recovered is deepest at the
   end by construction) and is withdrawn as evidence. T24 replaces it.
3. **T24 (pre-registered, `analysis/v3l/SUMMARY.md`).** The run-level association replicates on fresh
   runs (within-cell Spearman 0.86). The per-prompt reduced model fails its main predictions. The late
   phase preserves the order of prompts (Spearman 0.97–1.00 between the state at the start of the late
   phase and the final state, where few recover): the per-prompt outcome is decided during the fall.
4. **Stretch hypothesis, optimizer geometry (`experiments/probe_optimizer_geometry.py`): not
   supported.** Under SGD the ℓ2 max-margin geometry predicts the late sinking better than the ℓ∞ one
   (within-cell 0.81 against 0.72), as expected. Under Adam the ℓ∞ geometry adds nothing (0.60 against
   0.62 with the exact update). The two geometries correlate at 0.75 across tasks, so the test had
   power. The planned optimizer batch (T25) was not run.

## Status of the theory

| Piece | Status |
|---|---|
| First-step identity (odds law, kernel correction) | Proved (Lean), checked (SymPy), verified on 1,080 + T20 + T22 runs |
| Late sinking rate set by the majority's max-margin geometry (implicit bias) | Supported at the run level, replicated on pre-registered runs |
| Per-prompt late model; "γ < 1 recovers" | Not supported |
| What decides each prompt's outcome | The state reached during the fall (order preserved late); no theory yet |
| Adam's ℓ∞ geometry explains its recoveries | Not supported |

