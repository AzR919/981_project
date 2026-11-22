## H3 stretch: ablations of the tabular proposition (558 runs)

Tabular, starting p spread over logits in [−4, 3], GRPO, B = 32, 5000 steps; one condition of the proposition broken at a time. *max drop* = largest p_init − min p over prompts.

| ablation | runs | runs_with_decreases | median_decreases | max_drop | mean_absorbed | max_absorbed | mean_final |
|---|---|---|---|---|---|---|---|
| control | 72 | 0 | 0 | 0 | 0 | 0 | 0.881 |
| flip05 | 72 | 72 | 1.45e+03 | 0.75 | 0.0162 | 0.15 | 0.859 |
| flip20 | 72 | 72 | 8.41e+03 | 0.946 | 0.0422 | 0.352 | 0.806 |
| momentum | 72 | 44 | 1.5 | 0 | 0 | 0 | 0.99 |
| adam | 54 | 49 | 4 | 0 | 0 | 0 | 0.895 |
| kl01 | 72 | 72 | 1.26e+05 | 0 | 0 | 0 | 0.877 |
| kl1 | 72 | 72 | 9.15e+04 | 0.455 | 0.0136 | 0.17 | 0.69 |
| sign | 72 | 69 | 932 | 0.433 | 0.0242 | 0.266 | 0.86 |

Largest drop and absorbed fraction by L:

| ablation | max_absorbed L=1 | max_absorbed L=2 | max_absorbed L=4 | max_drop L=1 | max_drop L=2 | max_drop L=4 |
|---|---|---|---|---|---|---|
| control | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| flip05 | 0.016 | 0.092 | 0.150 | 0.346 | 0.587 | 0.750 |
| flip20 | 0.059 | 0.211 | 0.352 | 0.910 | 0.906 | 0.946 |
| momentum | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| adam | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| kl01 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| kl1 | 0.170 | 0.037 | 0.002 | 0.455 | 0.233 | 0.144 |
| sign | 0.006 | 0.096 | 0.266 | 0.128 | 0.414 | 0.433 |

By η (largest drop / largest absorbed fraction):

| ablation | 0.001 | 0.01 | 0.1 | 0.3 | 3.0 | 30.0 | 300.0 |
|---|---|---|---|---|---|---|---|
| control | nan | nan | nan | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| flip05 | nan | nan | nan | 0.001 / 0.000 | 0.009 / 0.000 | 0.116 / 0.006 | 0.750 / 0.150 |
| flip20 | nan | nan | nan | 0.003 / 0.000 | 0.029 / 0.002 | 0.207 / 0.051 | 0.946 / 0.352 |
| momentum | nan | nan | nan | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| adam | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | nan | nan | nan | nan |
| kl01 | nan | nan | nan | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| kl1 | nan | nan | nan | 0.000 / 0.000 | 0.000 / 0.000 | 0.001 / 0.000 | 0.455 / 0.170 |
| sign | nan | nan | nan | 0.001 / 0.000 | 0.010 / 0.000 | 0.089 / 0.031 | 0.433 / 0.266 |

**Verdict: PASS** (the control never lowers p, and every broken condition lowers p in at least one run)

