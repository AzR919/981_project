# v3j: where the wrong mass sits (T22, junk answers)

All 1,280 runs returned and none failed. Launched from a clean commit (`toy-v3` `f0bf112`).
Pre-registered in `PREREG_v3d_addendum4.md` (`review` `9458401`); exploratory.
`experiments/analyze_v3j.py` regenerates `FINDINGS.md`, `runs.csv` and `cells.csv`.

Setup: two answer classes have prompts (class 0 with share m, class 1 the minority); the other
K − 2 tokens are never correct ("junk"). K ∈ {2, 3, 6, 20}; uniform start; ρ = 0.9; SGD, η = 0.03;
10,000 steps; 10 tasks.

## Result in one line

**Junk answers move the first step exactly as the formula says, and do not protect the minority.**

## The first step: both predictions hold

| K (junk tokens) | 2 (0) | 3 (1) | 6 (4) | 20 (18) |
|---|---|---|---|---|
| Leading-order threshold (K − 1)/K | 0.500 | 0.667 | 0.833 | 0.950 |
| Predicted from the task's features (linear / MLP) | 0.505 / below 0.5 | 0.673 / 0.673 | 0.839 / 0.837 | 0.955 / 0.951 |
| Measured m\* (linear / MLP) | 0.505 / below 0.5 | 0.666 / 0.671 | 0.831 / 0.835 | 0.954 / 0.950 |
| R at m = 0.9, measured (linear / MLP) | 8.33 / 8.49 | 4.28 / 4.25 | 1.85 / 1.80 | 0.45 / 0.45 |

- **Prediction 1** (measured m\* within 0.02 of the prediction): **8 of 8 cells.**
- **Prediction 2** (R shrinks as 1/(K − 1)): **holds.** The ratio of R at K = 20 to K = 2 is 0.054
  and 0.053; 1/19 is 0.053.

## The outcome: both predictions fail

Share of minority prompts absorbed (exact update, linear; the MLP is the same or stronger):

| K | m = 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|
| 2 | 0.00 | 0.56 | 0.93 | 0.99 | 1.00 | 1.00 |
| 3 | 0.00 | 0.48 | 0.91 | 1.00 | 1.00 | 1.00 |
| 6 | 0.00 | 0.79 | 1.00 | 1.00 | 1.00 | 1.00 |
| 20 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

- **Prediction 3** (with K = 20, no absorption up to m = 0.9): **fails, in 10 of 12 cells.** With 18
  junk answers the minority is fully absorbed from m = 0.6, although at the first step the mixed
  update *raises* it for every m below 0.95.
- **Prediction 4** (absorption only at or above the first-step threshold): **fails, in 6 of 8
  cells.** Absorption starts at m = 0.6 for every K.
- More junk makes the absorption more complete at m = 0.6 (0.56 with none, 1.00 with 18), not less.
- The sampled update gives the same table to within 0.1.

## Reading

- **The first-step threshold does not predict the outcome.** With junk answers the two classes do
  not compete at the start, because most of each prompt's wrong mass is on junk. Training then
  moves that mass onto the two class answers, and from there the classes compete as in the
  no-junk case, where the majority wins from about m = 0.6. Junk delays the competition; it does
  not remove it.
- **This is the limit of the odds law.** It is exact for the first step and silent about the rest
  (`analysis/theory/odds_law.md`, corrections). Here the first step and the outcome disagree in
  sign for every share between 0.6 and 0.95 at K = 20.
- **The outcome threshold looks like a property of the two competing classes alone:** about 0.5–0.6
  whatever the number of junk answers.
- **For the LLM result this removes one explanation, and I had proposed it.** "The LLM's wrong
  answers are spread over a large space, so the classes do not compete" is not supported: in the
  toy a large junk space does not prevent the collapse. With K = 20 a minority prompt starts with
  5% on the majority's answer, close to the LLM's 0.6–5.4%, and still collapses.
- **What is left untested** is the other half of the reviewer's point: every toy run starts from a
  policy that does not yet tell the classes apart, while the LLM already answers minority problems
  correctly about half the time. T23 (warm start) tests that.

## Deviations

One host (`secb1013-d01`) rebooted four minutes after launch and lost its 106 runs. They were rerun
as batch `v3j2` from the same commit and job list.
