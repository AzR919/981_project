## H4 stretch: two stable KL fixed points (594 runs)

GRPO with KL(π‖π_ref), G = 64, tabular, 64 identical prompts per run (64 replicates). The policy starts at p_start, a logit offset from the predicted unstable root, while the reference stays at p_ref. A crossing is a replicate passing the unstable root between snapshots (every 100 steps).

![basin of the upper fixed point against the starting offset](figures/s_eq_basins.png)

| p0 | beta | eta | -3.0 | -2.0 | -1.0 | -0.5 | -0.25 | 0.0 | 0.25 | 0.5 | 1.0 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.50 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.44 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.01 | 0.50 | 0.30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.45 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.01 | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.47 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.03 | 0.50 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.03 | 0.50 | 0.30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.03 | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.51 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.20 | 1.00 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.20 | 1.00 | 0.30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.47 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.20 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.47 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Mean crossings per run (64 replicates, 10,000 steps):

| p0 | beta | 0.1 | 0.3 | 1.0 | 3.0 | 10.0 |
|---|---|---|---|---|---|---|
| 0.0 | 0.5 | 26.7 | 18.3 | 9.5 | 5.2 | 3.2 |
| 0.0 | 0.5 | 35.4 | 29.9 | 19.2 | 14.8 | 43.2 |
| 0.2 | 1.0 | 29.3 | 25.4 | 14.8 | 8.2 | 5.5 |

End states at η ≤ 1: median |error| to the lower root 0.0025 (max 0.5363), to the upper root 0.0075 (max 0.2417).

Mean-rule control (single predicted root): 1 stable root(s); share of runs ending within 0.01 of it: 1.00.

**Verdict: PASS** (at η ≤ 1 the upper-state share is < 0.1 for starts ≤ −0.5 logit and > 0.9 for starts ≥ +0.5 logit from the predicted unstable root)

**Verdict: PASS** (end states within 0.01 of the predicted stable roots (median))

**Verdict: FAIL** (crossings increase with η in every setting (Spearman > 0))

*Post hoc.* The crossing count above is dominated by replicates started at the unstable root (offset 0) slowly leaving it, which is slower at small η. Switching between the two stable states, counted in the second half of training for starts away from the unstable root (total over runs):

| p0 | beta | 0.1 | 0.3 | 1.0 | 3.0 | 10.0 |
|---|---|---|---|---|---|---|
| 0.01 | 0.5 | 0 | 0 | 0 | 0 | 0 |
| 0.03 | 0.5 | 0 | 0 | 0 | 6 | 538 |
| 0.2 | 1 | 0 | 0 | 0 | 0 | 0 |

Noise-driven switching appears only where the barrier is small (p_ref 0.03: stable 0.699, unstable 0.757) and grows steeply with η, as a barrier-crossing picture predicts.

**Verdict: PASS** (the mean-rule control ends at its single root from every start)

