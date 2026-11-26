# Pre-registration: v3d: who pushes whom (T11), plateau or permanent (T12), projection control (T13)

Written and committed before batch `v3d` launches. The code is on `toy-v3` (`dda0b74`). All three
tests are **exploratory**, so none enters the Holm family P1–P5. The T0 definitions of
`PREREG_v3a.md` apply: persistent absorption at τ = 1e-3; task seeds are the unit of replication
(10 per cell).

**Setting.** The T1 tasks: correlated features, ρ ∈ {0.5, 0.9}; K = 3, L = 1; linear and MLP-64;
uniform start (every p = 1/3). Class A is answer class 0, with share m; class B is the other two
classes.

**Disclosures.**
- **P1 results are known.** Under plain GRPO the minority set is absorbed at m ≥ 0.7.
- **Local smoke runs** (400 steps, one task, ρ = 0.9, m = 0.9, one per job kind) checked that the
  code runs. The class means of one of them were seen: MLP, `shortened`, η = 0.01, where the minority
  mean fell from 0.333 to 0.244 in 400 steps. No branch measurements were inspected.
- **Unit tests** confirm the analytic control: with two contexts at angle θ and A-share f, the mixed
  step moves B's logit in proportion to (1 − f) + f·cos θ.
- **Motivation.** The test separates three readings of minority collapse that end states cannot tell
  apart:
  - B gets too little signal;
  - A slows B;
  - A actively reverses B.
- **Prior work.** Schaul et al. (2019, "ray interference") describe winner-take-all plateaus under
  shared parameters and predict delayed learning, not permanent loss. T12 tests that reading.

## T11: one-step branches from a common state

At steps 0, 0.1T, 0.5T and 0.9T of each run, from copies of the policy and optimizer state, take
one step along each of:
- zero;
- g_A (the majority's part of the exact expected update);
- g_B (the minority's part);
- g_A + g_B (the actual step).

Record the exact change in mean success of each class.

**Arms:**
- SGD at η = 0.01 with `expected_full`;
- SGD at η = 0.01 with `sampled`, with 16 sampled batches per branch point (same samples across
  branches);
- Adam at η ∈ {1e-3, 1e-4} with `expected_full`.

**Grid:** m ∈ {1/3, 0.7, 0.9}.

**Predictions** (SGD, `expected_full`; a cell agrees if ≥ 9 of 10 tasks agree):
1. **Active reversal before collapse.** At steps 0 and 0.1T with m ∈ {0.7, 0.9}:
   - the A-only step lowers B's mean success (Δ_B(A) < 0);
   - the B-only step raises it (Δ_B(B) > 0);
   - the mixed step lowers it (Δ_B(mixed) < 0).
   - So the majority overpowers a useful minority signal.
   - cos(g_A, ∇p̄_B) < 0.
2. **Balanced control.** At m = 1/3 and step 0, Δ_B(mixed) > 0.
3. **Starvation after collapse.** At 0.5T and 0.9T in cells where the minority is absorbed,
   |Δ_B(B)| is below 1% of its step-0 value: the minority's own signal has vanished because
   ∇p_B ∝ p_B.
4. **Dose.** The ratio −Δ_B(A)/Δ_B(B) at step 0 is larger at m = 0.9 than at m = 0.7 (paired over
   tasks).
5. **Sampled.** The mean sampled Δ_B for each branch has the same sign as the exact one at step 0.

**Adam arms:** descriptive (actual displacements; no linearity assumed).

**Mixed-branch check:** it must reproduce `expected_step` (`mixed_check` < 1e-10).

## T12: plateau or permanent

The T1 minority tasks (m ∈ {0.7, 0.9}) at 10× the T1 horizon, `expected_full` and `sampled`:

| Optimizer | η | Steps |
|---|---|---|
| SGD | 0.01 | 100,000 |
| SGD | 0.1 | 10,000 |
| SGD | 1 | 10,000 |
| Adam | 1e-4 | 50,000 |
| Adam | 1e-3 | 50,000 |

**Recovery:** the minority class mean exceeds 0.01 at the final step, after having been below 1e-3.

**Predictions:**
1. **No recovery.** In SGD cells, at most 5% of runs whose minority mean ever falls below 1e-3 show
   recovery. This is the permanent reading; a plateau reading predicts recovery in most runs.
2. **Linear Adam is a horizon effect.** For linear policies with Adam at η = 1e-4, which showed
   almost no collapse at 5,000 steps: at 50,000 steps the minority-prompt absorption rate exceeds 0.5
   in ≥ 8 of 10 tasks at ρ = 0.9.
3. **Matched progress.** Take the first logged step where the majority mean exceeds its start by
   +0.10. At that step the minority mean is below its start, for both SGD at η = 0.01 and Adam at
   η = 1e-4 (in ≥ 8 of 10 tasks per cell at m = 0.9).

## T13: projection control

**Updates:**
- `projected`: the majority part with its component along ∇p̄_B removed whenever that component is
  negative, plus the unchanged minority part;
- `shortened`: the majority part scaled to the projected norm, plus the minority part;
- `expected_full`: the unmodified update.

**Grid:** SGD, η ∈ {0.01, 0.1}, m ∈ {0.7, 0.9}.

**Predictions** (paired one-sided Wilcoxon over tasks, per cell):
1. `projected` has lower minority absorption than `shortened`.
2. `shortened` does not differ from `expected_full` by more than 0.05 in minority absorption, i.e. a
   shorter majority step alone does not protect the minority.
3. The final majority mean under `projected` is within 0.05 of `expected_full`.

**Interpretation.** T13 is interpreted only if T11 prediction 1 holds. The projection uses the
minority's mean success gradient, which is privileged information. It is a mechanism test, not a
proposed method.

## Deviations

Any departure from this file is listed in the write-up with its reason.
