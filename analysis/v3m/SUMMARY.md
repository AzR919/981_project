# v3m: permanent or plateau? Exact long-time flow (T25)

Pre-registered in `PREREG_v3d_addendum7.md` (`review` `15a86b0`), with the prediction rule fixed before the
integrations; exploratory. Code: `experiments/exact_flow.py`, the exact gradient flow of the linear policy under the
exact expected GRPO update, integrated in log time (`toy-v3`; matches discrete SGD to within 0.015 decades).
`experiments/check_permanence.py` regenerates `FINDINGS.md`, `runs.csv` and `prompts.csv` from the saved
integrations (`~/rlvr-data/v3m/*.npz`; archive `v3m_raw.tar.gz`).

**Setup:** linear policy, K = 3, ρ ∈ {0.3, 0.5, 0.7, 0.9, 0.95}, m ∈ {0.7, 0.9}, 10 tasks each (100 tasks), flow time
0.1 to 1e14. Normal training reaches flow time about 1e3–1e4.

**Integration:**
- 98 tasks were integrated with LSODA.
- Two stiff tasks stalled: `rho0.7_m0.9_t9` was finished with Radau (8.5 min) and `rho0.9_m0.7_t5` with BDF
  (16.5 min, looser tolerance 1e-6).
- All 100 finished, with finite values.

## Predictions (per prompt, from the exact solution of the scalar late model)

| Prediction | Pass rule | Result |
|---|---|---|
| 1. Prompts predicted never to recover stay absorbed | ≤ 5% recovered by 1e14 | **Fails: 48% of 4,403 recovered** |
| 2. Prompts predicted to recover before 1e12 do | ≥ 80% recovered by 1e14 | **Fails: 67% of 4,179** |
| 3. Timing of recovery | Spearman ≥ 0.7 and median error ≤ 1 decade | **Fails: Spearman 0.34, median error 2.9 decades** |
| 4. Run-level late exponent within 0.1 of γ_min | ≥ 80% of qualifying runs | 3 of 3, but only 3 runs qualified: **not informative** |
| 5. How many tasks end recovered (reported) | none | See below |

**Registered reading:** 1–3 fail, and 4 has too few runs to hold or fail. So the per-prompt late model does not
predict individual outcomes over long times, consistent with T24. Each prompt's fate depends on the collective
dynamics: recovering classmates pull the rest up, which the one-prompt model leaves out.

## What the long-time flow shows (prediction 5, reported)

Tasks whose minority class mean is back above 0.5 at flow time 1e14 (of 10), and the recovery time of the prompts
that recovered:

| ρ | m | Tasks recovered | Prompts recovered (of fallen) | Median recovery time [10th–90th percentile] |
|---|---|---|---|---|
| 0.3 | 0.7 | 10 | 160 of 167 | 1e4.9 [1e4.7, 1e4.9] |
| 0.3 | 0.9 | 8 | 445 of 524 | 1e3.4 [1e2.4, 1e5.1] |
| 0.5 | 0.7 | 9 | 1,201 of 1,360 | 1e2.4 [1e2.0, 1e7.0] |
| 0.5 | 0.9 | 6 | 309 of 526 | 1e4.1 [1e2.8, 1e13.4] |
| 0.7 | 0.7 | 8 | 1,188 of 1,504 | 1e3.9 [1e2.8, 1e6.6] |
| 0.7 | 0.9 | 3 | 229 of 526 | 1e6.4 [1e3.9, 1e7.4] |
| 0.9 | 0.7 | 6 | 890 of 1,504 | 1e7.5 [1e6.2, 1e13.6] |
| 0.9 | 0.9 | 1 | 85 of 526 | 1e9.6 [1e9.6, 1e14.0] |
| 0.95 | 0.7 | 3 | 450 of 1,504 | 1e11.9 [1e9.6, 1e12.1] |
| 0.95 | 0.9 | **0** | **0 of 523** | — |

**γ_min decides it at the task level (post hoc):**
- Across the 100 tasks, γ_min (the minority's smallest projection on the majority's max-margin direction) ranks the
  final minority success at Spearman −0.70.
- Tasks that recover have median γ_min 0.89; tasks that don't, 1.03 (Mann–Whitney p = 1e-11).
- With γ_min ≥ 1, 3 of 30 tasks recover. With γ_min < 1, 51 of 70 do.
- This is the scalar model's threshold (γ ≥ 1: no recovery), holding for whole tasks although it fails prompt by
  prompt.

## Reading, and the wording for the report

- **Mostly a plateau, sometimes effectively permanent.** With weakly similar features the minority recovers, at
  flow times 1e2–1e5, comparable to or longer than training. With strongly similar features (ρ ≥ 0.9) recovery, when
  it comes at all, takes 1e7.5–1e12: 10⁴ to 10⁸ times longer than training.
- **At ρ = 0.95 and m = 0.9 nothing recovers by 1e14.** "Permanent" is the right word there, up to the horizon
  computed.
- **Which applies is predicted by γ_min:** below 1, a plateau; at or above 1, permanent to within 1e14, in 27 of 30
  tasks. This is a task-level, post hoc reading of the pre-registered quantity γ_min, not a registered test.
- **The per-prompt late model fails over long times, as in T24.** Prompt-level outcomes need the collective
  dynamics; the class-level fall model (`analysis/theory/fall.md`) is the step in that direction.

## Deviations

- Two tasks needed stiffer solvers (Radau, BDF) after LSODA stalled. The solver changes the accuracy, not the
  dynamics. `exact_flow.flow` gained a `method` option; the default is unchanged.
- The γ_min threshold analysis is post hoc.
- Prediction 4 is uninformative: only 3 runs met its qualifying rule (the minority mean below 1e-3 for two decades
  before recovering).
