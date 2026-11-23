# v3i: prevention levers in the transformer environment (T21)

All 560 runs returned and none failed. They were launched from a clean commit (`enum-env`
`351d722`). Pre-registered in `PREREG_T20_addendum1.md` (`review` `f1f97f0`); exploratory.
`experiments/analyze_v3i.py` regenerates `FINDINGS.md`, `runs.csv` and `cells.csv`.

Setup: Li, Pandey and Chen's environment with our class verifier (see `analysis/v3h/SUMMARY.md`).
Two tasks (K = 3 with 56 of 64 prompts in class 0; parity, 48 of 64), exact and sampled updates,
Adam at lr 1e-3 and SGD at lr 0.3; 10 seeds; 8 cells. Without a lever the minority is absorbed
(mean share ≥ 0.5) in 7 of the 8 cells; the eighth is the parity task with the exact update and
Adam, which recovers on its own.

## Result

Mean share of minority prompts absorbed, over the 7 collapsing cells:

| Lever | Adam cells (3) | SGD cells (4) | Class 0 at the end | Minority's lowest mean during training |
|---|---|---|---|---|
| none | 0.85–1.00 | 0.79–1.00 | 0.89–1.00 | below 1e-6 |
| reverse KL, β = 0.03 | 0.88–1.00 | 0.89–1.00 | 0.92–1.00 | below 1e-6 |
| reverse KL, β = 0.3 | 0.00–0.84 | 0.60–0.95 | 0.42–1.00 | below 1e-2 |
| forward KL, β = 0.03 | 0.00–0.06 | 0.00–0.28 | 0.99–1.00 | 1e-7 to 0.007 |
| forward KL, β = 0.3 | 0.00 | 0.00 | 0.92–0.98 | 0.005–0.15 |
| pass@2 | 0.00–0.03 | 0.54–0.90 | 0.92–1.00 | 0.004–0.32 (Adam) |
| pass@4 | 0.00 | 0.79–0.94 | 0.87–0.99 | 0.16–0.45 (Adam) |

| Prediction | Result |
|---|---|
| 1. Forward KL prevents absorption (≤ 0.05 at both β, every cell) | **6 of 8 cells.** At β = 0.3: none absorbed in all 8. At β = 0.03 two cells miss: 0.06 (K = 3, exact, Adam) and 0.28 (K = 3, exact, SGD) |
| 2. Forward KL at β = 0.03 costs little (class 0 ends ≥ 0.8) | **8 of 8** |
| 3. Reverse KL at β = 0.03 does not prevent it | **7 of 7 collapsing cells** |
| 4. pass@4 lowers absorption by at least 0.3 | **3 of 7: fails.** It removes absorption in every Adam cell and does nothing in the SGD cells |

## Reading

- **The KL direction result transfers.** Forward KL prevents the collapse with a transformer on
  token prompts, under both step rules and under sampling; reverse KL at the same strength does
  not. At β = 0.3 forward KL leaves no minority prompt absorbed in any cell, with class 0 at
  0.92–0.98 and the minority at 0.91–0.98.
- **pass@k weighting depends on the optimizer here.** With Adam, pass@2 and pass@4 remove the
  absorption and the minority ends at 0.97–1.00. With plain SGD at lr 0.3 they do not help. In the
  toy the pass@k runs used SGD and worked, so this is a difference between the environments, and
  the prediction fails as registered.
- **Shielding against bringing back, as in the toy.** pass@4 with Adam keeps the minority's mean at
  0.16–0.45 throughout. Forward KL lets it fall (to 0.005–0.15 at β = 0.3, lower at 0.03) and brings
  it back.
- **Reverse KL at β = 0.3** helps in some Adam cells (it removes the absorption on the sampled
  parity task) and costs class 0 under sampled SGD (0.42–0.44). It is not a dependable lever.

## What the two environments now agree on

| | Toy (features, linear or MLP) | Transformer on token prompts |
|---|---|---|
| The push follows the odds law | yes, parameter-free | yes, parameter-free |
| A majority absorbs a correct minority, only with sharing | yes | yes |
| Forward KL prevents absorption | yes, from β = 0.01 | yes, fully at β = 0.3, mostly at 0.03 |
| Reverse KL at small β does not | yes | yes |
| pass@k weighting prevents absorption | yes (SGD) | with Adam only |
| Sampled updates block recovery that the exact update allows | not tested this way | yes (parity task) |

## Deviations

None. Predictions 1 and 4 are reported as partly failing.
