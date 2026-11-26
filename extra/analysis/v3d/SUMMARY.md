# v3d: who pushes whom, plateau or permanent, projection control

All 1,760 runs returned and none failed. They were launched from a clean commit (`toy-v3` `7a49348`;
job digest `d497e591db3a`). The tests follow `extra/analysis/review/PREREG_v3d.md` (`review` `f22540c`) and
are exploratory. `extra/experiments/analyze_v3d.py` regenerates `FINDINGS.md`.

Class A is answer class 0 (share m). Class B is the rest: the minority when m > 1/3.

## T11: the majority actively reverses the minority (all five predictions hold)

From copies of one state, one step along the majority's part, the minority's part, or both
(SGD, exact expected update).

| Prediction | Result |
|---|---|
| 1. Before collapse (steps 0 and 0.1T, m ∈ {0.7, 0.9}): majority-only lowers minority success, minority-only raises it, the mixed step lowers it | **16/16 cells**, 10/10 tasks in each |
| 2. Balanced control (m = 1/3): the mixed step raises class B | **4/4 cells** |
| 3. After collapse: the minority's own step is below 1% of its step-0 size | **125/125** branch points (median ratio 7e-6) |
| 4. The push grows with imbalance | Ratio −Δ_B(A)/Δ_B(B) at step 0: **3.6–4.8 at m = 0.7, 10.7–17.1 at m = 0.9** (10/10 tasks in each cell). It is 0.78–0.93 at m = 1/3, just under the break-even value of 1 |
| 5. Sampled GRPO agrees in sign | 100% (majority-only), 100% (minority-only), 89% (mixed) of 120 runs |

- **Mixed-branch check.** The mixed branch reproduces the simulator's own step to 6e-17.
- **Reading.** Collapse is not starvation. At the start the minority's own update is useful, and the
  majority's update outweighs it by 4–17 times. Starvation only sets in afterwards: once p_B is near
  zero the minority's own signal vanishes (∇p ∝ p), which is what holds it there.
- **The interference is direct.** The cosine between the majority's step and the minority's success
  gradient is negative in every task of every cell, including the balanced one (about −0.85 at step 0
  in the linear task).
- **Adam.** The same signs hold for actual displacements at η = 1e-3 and 1e-4 (table in FINDINGS).

## T12: plateau or permanent, at 10× the horizon (prediction 1 fails)

- **Prediction 1 (no recovery under SGD): FAIL.** 34 of the 409 runs that fell below 1e-3 recovered
  (8.3%; predicted ≤ 5%). Recovery is confined to the weakly correlated tasks (ρ = 0.5):
  - MLP at ρ = 0.5 and m = 0.7: 20 of 50 runs that fell below 1e-3 recover, 11 of 18 at η = 1;
  - MLP at ρ = 0.5 and m = 0.9: 7 of 60;
  - linear at ρ = 0.5 and m = 0.9: 7 of 47;
  - in the linear task at ρ = 0.5 and m = 0.7 the minority dips to 0.011 and returns to 0.88–0.96.
    That is a plateau, with no absorption left at the end.
- **No recovery at ρ = 0.9:** 0 of 240 runs at 10× the horizon; minority means stay at 1e-5 to 1e-8.
- **So the collapse is a plateau whose length depends on feature correlation and imbalance.** That
  is ray interference's reading (Schaul et al.). At ρ = 0.9 it outlasts 10× the horizon. "Permanent"
  is not supported as a general statement.
- **Prediction 2 (linear Adam is a horizon effect): holds in 3 of 4 cells.** At 50,000 steps, linear
  policies with Adam at η = 1e-4 absorb the minority in 10/10 tasks (m = 0.9; m = 0.7 sampled) and
  7/10 (m = 0.7 exact; the prediction was ≥ 8). The weak collapse at 5,000 steps was a short horizon.
- **Prediction 3 (matched progress): holds in 16/16 cells.** When the majority has gained +0.10, the
  minority is below its start in every task, under both SGD and Adam. Adam is gentler at that point
  in the linear task (minority mean 0.25–0.27, against 0.003–0.011 under SGD).

## T13: projection control (prediction 1 holds in 8 of 16 cells and is a tie at zero in the other 8; predictions 2 and 3 fail)

Share of minority prompts absorbed (mean over 10 tasks; η ∈ {0.01, 0.1} agree):

| Task | Unmodified | Majority step shortened | Majority step projected |
|---|---|---|---|
| m = 0.7, 4 cells | 0.34–1.00 | 0.00 | 0.00 |
| m = 0.9, 4 cells | 0.81–1.00 | 0.47–0.89 | **0.00–0.01** |

- **Prediction 1.** Projection removes the collapse in all 16 cells. It beats the shortened step in
  all 8 cells at m = 0.9 (p ≤ 0.002). At m = 0.7 both reach zero, so the test is a tie there.
- **Prediction 2: FAIL.** "A shorter majority step does not protect" is false. Shortening the
  majority's part to the projected norm removes the collapse at m = 0.7 and reduces it at m = 0.9.
  - Reason: with cos ≈ −0.85, the projected norm is about half of the majority's part, so shortening
    acts as down-weighting the majority class.
  - Class re-weighting is therefore a second way to prevent collapse at moderate imbalance.
- **Prediction 3: FAIL for linear policies.** Under projection the majority's final success falls
  from 1.00 to 0.45–0.86 in the linear task; the minority reaches 0.95–0.98. In the MLP both classes
  end at ≥ 0.97 (minority 0.91–1.00).
  - So the conflict is forced in a low-capacity policy, and avoidable in a larger one.

## M2 (addendum 1): is the minority still sinking at 10× the horizon?

g is the growth of the minority's wrong-side margin relative to the majority's margin, over the last
decade of training (descriptive; written after T12 was seen).

| Cell | Median g | Reading |
|---|---|---|
| ρ = 0.9, SGD (all 12 cells) | 0.99–1.01 | The minority sinks as fast as the majority improves. No run is recovering (0 of 120) |
| ρ = 0.9, Adam | 1.00–1.05 (0.49 for MLP at m = 0.7, η = 1e-4) | Same, except that cell, where 5 of 10 runs are recovering |
| ρ = 0.5, m = 0.9, SGD | 0.87–1.01 | Mostly still sinking; 1–3 of 10 runs recovering |
| ρ = 0.5, m = 0.7 | −0.5 to 0.4 | Recovering in 4–8 of 10 runs |
| ρ = 0.5, Adam | negative | Recovering in 5–9 of 10 runs; final minority success 0.26–0.99 |

- **At strong feature correlation**, g sits at 1 and cannot separate "permanent" from "a plateau far
  longer than the run".
- **At weak correlation** the plateau ends inside 10× the horizon, sooner under Adam.

## M4 (addendum 1): imbalance, not margin (prediction holds)

Share of prompts absorbed under the unmodified update, by class and by tercile of score margin:

- **Majority prompts:** 0.000 in every tercile of every cell, including the lowest-margin third.
- **Minority prompts:** similar across terciles. For example, in the linear task at ρ = 0.9 and
  m = 0.7: 0.96, 0.95, 0.95. In the MLP at ρ = 0.5 and m = 0.7: 0.88, 0.85, 0.83.
  - The terciles are within 0.15 of each other in 7 of 8 cells.
  - The exception is the linear task at ρ = 0.5 and m = 0.9: 0.81, 0.88, 0.67.

Which class a prompt belongs to decides whether it is absorbed; its margin does not.

## What this changes

1. **Headline mechanism: active reversal, then starvation.** Direct evidence, in every task tested.
2. **Wording:** "collapses for the whole training horizon, and for at least 10× longer when features
   are strongly correlated". Not "permanently".
3. **Prevention.** Three routes now work in the toy: pass@k weighting (v3c), down-weighting the
   majority's step, and projecting out the harmful component (which needs privileged information).
4. **The break-even ratio.** The ratio −Δ_B(A)/Δ_B(B) crosses 1 between m = 1/3 and m = 0.7. Its
   exact location is the threshold test (T15).

## Deviations

- **T12 prediction 2** was stated for both m values, and is reported per cell.
- **T13** is interpreted because T11 prediction 1 holds, as pre-registered.

## Corrections after the review of 2026-11-23 (`extra/analysis/review/ADVERSARIAL_v5.md`)

- **Recovery and the optimizer.** At ρ = 0.9, 24 of 80 runs with the exact update and Adam recover
  within 10× the horizon; none of the sampled Adam runs and none of the 240 SGD runs do
  (`tables/t12_cells.csv`). The wording for the report is "for the whole training horizon, and for
  at least 10× longer under SGD or with sampled updates when features are strongly correlated".
