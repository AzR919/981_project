## Follow-up (c): T6 KL conditions with the decline above p_ref

| ablation | eta | below_share | above_share | max_drop_above | persist |
|---|---|---|---|---|---|
| control | 0.3 | 0 | 0 | 0 | 0 |
| control | 3 | 0 | 0 | 0 | 0 |
| control | 30 | 0 | 0 | 0 | 0 |
| control | 300 | 0 | 0 | 0 | 0 |
| kl01 | 0.3 | 0 | 0 | 0.000648 | 0 |
| kl01 | 3 | 0 | 1 | 0.0236 | 0 |
| kl01 | 30 | 0 | 1 | 0.212 | 0 |
| kl01 | 300 | 0 | 1 | 0.883 | 0 |
| kl1 | 0.3 | 0 | 1 | 0.00584 | 0 |
| kl1 | 3 | 0 | 1 | 0.144 | 0 |
| kl1 | 30 | 0.117 | 1 | 0.604 | 0 |
| kl1 | 300 | 1 | 1 | 0.982 | 0.0343 |

**Verdict: PASS** (no material drop below p_init at η ≤ 3 for KL: True; control has no decline from a peak: True; drops below p_init only at η ≥ 30: True)

