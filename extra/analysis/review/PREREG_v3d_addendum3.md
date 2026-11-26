# PREREG_v3d, addendum 3 (2026-11-23): H4 and H5 as prevention levers (T17, T18, T19)

Committed before batch `v3g` launches (code: `toy-v3`, "Toy v3g" commit). Exploratory.

**Why.** H4 (KL penalty) and H5 (group size) are milestone hypotheses whose original tests did not
connect to the minority collapse. These three experiments ask each the same question: does the knob
protect the minority class, and at what cost to the majority?

**Disclosed before writing.**
- Outcomes seen: v3d, v3e and v3f (`extra/analysis/v3d`, `v3e`, `v3f`), and T8 (KL bistability in the
  tabular policy at p_ref ≤ 0.2).
- Only unit tests were run with the new code (`extra/tests/test_v3.py::ReleaseTests`); no T17–T19 run was
  inspected.
- The theory values below were computed from `synthetic_rl/theory.py` before launch.

**Common task and measures.** Class-share task with K = 3, uniform start, majority class A, SGD.
The KL reference is the start policy (p_ref = 1/3 for every prompt). Measures as in addendum 2:
- matched point: the first logged step where the majority class mean reaches 0.80;
- the minority class mean at the matched point and at the end;
- the share of minority prompts persistently absorbed (τ = 1e-3, last 20% of steps).

## T17: KL anchor on the minority task (H4)

- **Runs.** Linear and MLP, ρ = 0.9, m = 0.9; β ∈ {0, 0.01, 0.03, 0.1, 0.3, 1, 3}; reverse and
  forward KL; `expected_full` and `sampled` (G = 8, B = 32); η ∈ {0.1, 1}; 10 task seeds; 20,000
  steps. 1,040 runs.
- **Theory used** (per-prompt fixed points at p_ref = 1/3, G = 8, `fixed_points`): one stable point
  in every case.

| β | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|
| reverse | → 1 | → 1 | → 1 | → 1 | 0.806 | 0.481 |
| forward | 0.997 | 0.992 | 0.974 | 0.920 | 0.728 | 0.488 |

- **Heuristic for the minority.** Let c be the majority's push on a minority logit, in units of the
  reward weight (order 1–3 at m = 0.9).
  - Reverse KL pulls a logit back in proportion to β·(z − z_ref), so the minority settles near
    z_ref − c/β.
  - Forward KL pulls in proportion to β·(p_ref − p)/p, so it settles near p ≈ β·p_ref/c.
- **Predictions** (a cell is one arch × update × η; "holds" means in ≥ 9 of 10 tasks):
  1. **Monotone protection.** For each direction, the absorbed minority share does not increase
     with β (allowing 0.02 of slack), in every cell.
  2. **Majority cost.** The majority reaches 0.80 for both directions at β ≤ 0.3, and never at
     β = 3. β = 1 is borderline and is reported, not predicted.
  3. **Forward KL keeps the minority alive without protecting it.** At β ∈ {0.03, 0.1, 0.3} no
     minority prompt is absorbed and the majority reaches 0.80, but the minority mean at the matched
     point is below 0.2.
  4. **Reverse KL does not prevent absorption where the majority still learns.** At β ≤ 0.3 the
     absorbed minority share stays above 0.5.
  5. **At equal β, forward protects at least as well as reverse:** absorbed share (forward) ≤
     absorbed share (reverse) + 0.02, in every cell and β.

## T18: release test (H4: why collapsed prompts stay down)

- **Runs.** Linear and MLP, ρ = 0.9, m = 0.9, η = 0.1, `expected_full`, 20,000 steps, 10 tasks.
  Phase 1 is plain GRPO (β = 0). At the release step (100, 1,000 or 5,000) the majority's reward
  term is dropped; the minority's term is unchanged. After the release: no KL; reverse KL at
  β ∈ {0.01, 0.1}; forward KL at β ∈ {0.01, 0.1}. Control: no release. 320 runs. All arms of a
  task share the same state at the release.
- **Return** means the minority class mean at the end is ≥ 0.2 (it starts at 1/3).
- **Correction to the plan.** The plan expected "two resting points" to explain why collapsed prompts
  stay down. The fixed-point calculation above says otherwise for this task: with p_ref = 1/3 there
  is one resting point under either direction. Bistability (T8) needs a low p_ref. What theory
  predicts here is a difference in speed: the minority's own reward gradient and the reverse-KL
  gradient both vanish in proportion to p as p → 0; the forward-KL gradient does not.
- **Predictions:**
  1. **Control:** no return in ≥ 9 of 10 tasks (as in T12 at ρ = 0.9).
  2. **β = 0 after the release:** return depends on depth. The return rate does not increase with the
     release step, and at release step 5,000 there is no return in ≥ 9 of 10 tasks. Among returning
     runs, the steps to return rise with 1/p at the release (Spearman > 0, pooled).
  3. **Reverse KL** behaves like β = 0 at the two deeper releases: no return in ≥ 9 of 10 tasks at
     release steps 1,000 and 5,000, for both β.
  4. **Forward KL at β = 0.1:** return in ≥ 9 of 10 tasks at every release step.
  5. Forward KL at β = 0.01 is reported, not predicted.
- **Reading.** If 2–4 hold, collapsed prompts stay down because the forces that could restore them
  shrink with p, not because of a second resting point; and the KL direction decides whether an
  anchor can rescue them.

## T19: group size and the collapse threshold (H5)

- **Runs.** The T15 share grid at K = 3, extended to m = 0.883 (offsets −0.05 … 0.55 from 1/3);
  linear and MLP; ρ ∈ {0.5, 0.9}; η = 0.01; 10,000 steps; 10 tasks.
  - `expected_full`, grpo, G ∈ {2, 4, 8, 16, 64};
  - `expected_full`, mean rule, G ∈ {2, 64};
  - `sampled`, grpo, the same G at a fixed budget B·G = 256, ρ = 0.9 only.
  4,180 runs.
- **Theory used.** At a uniform start every prompt has the same p, so the reward weight is a common
  factor and G cannot change the direction of the first step. Later, G matters through the ratio of
  the weight at low p to the weight at high p, W_G(0.01)/W_G(0.9):

| G | 2 | 4 | 8 | 16 | 64 |
|---|---|---|---|---|---|
| grpo | 1.00 | 1.02 | 1.12 | 1.31 | 2.07 |
| mean | 1.00 | | | | 1.00 |

- **Definitions** as in addendum 1: m\* is where the median one-step change of the minority under
  the actual update crosses zero; `absorption_from` is the smallest grid m with mean minority
  absorption ≥ 0.05.
- **Predictions** (exact arm; a cell is one arch × ρ):
  1. **m\* does not depend on G:** the range of m\* across G is below 0.005 in every cell.
  2. **The absorption threshold rises with G under grpo:** `absorption_from` is non-decreasing in G
     in every cell, and larger at G = 64 than at G = 2 in ≥ 3 of 4 cells.
  3. **Not under the mean rule:** `absorption_from` is the same grid value at G = 2 and G = 64 in
     ≥ 3 of 4 cells.
  4. **Size.** Any shift is small against the range of m: the G = 64 threshold is at most 0.2 above
     the G = 2 threshold. Group size is a weak lever.
- **Sampled arm:** descriptive. It shows whether small batches at large G (B = 4 at G = 64) add
  absorption through the large-step route.

## Deviations

Listed in the v3g write-up.
