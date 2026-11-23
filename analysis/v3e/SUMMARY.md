# v3e: where does the majority start to push the minority down? (T15)

All 1,080 runs returned and none failed. They were launched from a clean commit (`toy-v3` `779297f`;
job digest `1dcf342238dd`). Pre-registered in `analysis/review/PREREG_v3d_addendum1.md` (`review`
`2401287`); exploratory. `experiments/analyze_v3e.py` regenerates `FINDINGS.md`.

**Setup:** K answer classes; class A has share m and the other K − 1 classes split the rest; uniform
start; SGD at η = 0.01 with the exact expected update.

**Definitions:**
- R = −Δ_B(A)/Δ_B(B) at step 0: the majority's push on the others over their own gain.
- m\*: where the one-step change of the other classes under the actual update crosses zero.

| Prediction | Result |
|---|---|
| 1. m\* is strictly above 1/K (at m = 1/K the other classes still gain in ≥ 9 of 10 tasks) | **8 of 12 cells.** Holds for K = 3 and K = 5 (m\* − 1/K = 0.02–0.11). Fails for K = 2, where m\* = 0.498–0.529: at 1/K with two classes, whichever class is larger in the sample wins |
| 2. R scales with the odds m/(1 − m) | **12 of 12 cells**; the slope of log R on logit m is 0.98–1.05 |
| 3. Location for K = 3 | **Both hit.** Linear at ρ = 0.5: m\* = 0.386 (predicted 0.37–0.42). The other three cells: 0.351–0.355 (predicted 0.34–0.37) |
| 4. m\* − 1/K is smaller at ρ = 0.9 than at ρ = 0.5 | **6 of 6** (K, architecture) pairs |
| 5. Persistent absorption only at or above m\* | **12 of 12 cells** |

## Reading

- **The push is proportional to the odds.** R(m) = c · m/(1 − m), with c set by the task: c = R at
  m = 1/K times (K − 1), and R at 1/K is 0.55–1.01. The mixed step starts to lower the other classes
  where R = 1.
- **The step-0 threshold is just above 1/K.** It is exactly 1/K for two classes, and 0.02–0.05 above
  for K = 3 and 5 in most cells (0.11 for the linear task at ρ = 0.5 and K = 5). This is close to the
  reading of Li et al.'s leading-order result, with a task-dependent correction that shrinks as
  feature correlation grows.
- **Persistent absorption needs clearly more imbalance than the first decline:**

| K | Step-0 threshold m\* | Absorption from (ρ = 0.9) | Absorption from (ρ = 0.5) |
|---|---|---|---|
| 2 | 0.50–0.53 | 0.62 | 0.85 |
| 3 | 0.35–0.39 | 0.45–0.53 | 0.68 (MLP); none up to 0.68 (linear) |
| 5 | 0.22–0.31 | 0.32–0.40 | 0.55 (MLP); none up to 0.55 (linear) |

- **Between the two thresholds** the minority dips and comes back. That is the plateau regime seen
  in T12.

So the statement for the report has two parts:
1. The majority's update lowers the other classes as soon as its share exceeds about 1/K, with a
   strength proportional to the odds.
2. Whether that becomes a collapse lasting the whole run depends on how far past the threshold the
   share is, and on feature correlation.

## Deviations

None. Prediction 1 is reported as failing for K = 2.
