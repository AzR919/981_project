# v3o: why losing the minority is harmful (pass@k, coverage, worst class)

Collapse onto the correct answer is the goal of RL with a verifier, so absorption alone does not show harm. This note
measures what the confirmatory runs of battery A lose when the minority is absorbed, with three measures from published
work: pass@k against the start policy, coverage, and worst-class accuracy. It uses only the saved runs in
`~/rlvr-data/v3o` (A1, A2, A3); nothing was retrained.

**Status:** descriptive analysis of the confirmatory runs, not a registered test. No p-values.

`experiments/analyze_harm.py` regenerates `harm_cells.csv` (all four majority shares, every measure, with medians and
ranges over tasks; per-run values in `~/rlvr-data/v3o/cache/harm_runs.csv`).

## Definitions

Per run, from the per-prompt success probability p_x at the start (`p_init`) and at the end of training (the snapshot
at the last step), over the 512 training prompts. The majority is class 0; the minority is classes 1 and 2 pooled.

| Measure | Definition | Source |
|---|---|---|
| pass@k | mean over prompts of 1 − (1 − p_x)^k, k ∈ {1, 2, 4, 8, 16, 32, 64} | Yue et al., NeurIPS 2025 (RL models beat the base model at k = 1 and lose to it at large k) |
| Crossover k* | smallest k in the list at which the start policy's pass@k is at least the final policy's; "none" if the final policy wins at every k | from Yue et al. |
| Coverage C | mean over prompts of 1/p_x, the expected number of samples to see a correct answer; reported as log10 C | Huang et al. 2026, Definition 1.1 |
| Worst-class pass@1 | smallest class mean of p_x at the end | Sagawa et al., ICLR 2020 (worst-group accuracy) |

- **Float32 floor.** Snapshots are stored as float32 with denormals flushed: the smallest stored non-zero p is
  1.2e-38, and any p below 1.18e-38 reads as 0. The reported log10 C clips p at **1e-30**, so it does not depend on the
  storage floor; it is a lower bound wherever prompts are below 1e-30. "Minority at floor" is the share of minority
  prompts stored as 0. The CSV also has log10 C clipped at the float32 floor (`logC_*`) and the median of 1/p over
  minority prompts (`log_median_inv_final_min`).
- **Aggregation.** A1: per cell (architecture/ρ/optimizer/update) and majority share m, each task is the mean over the
  cell's A1 η set (`analyze_v3o.a1`, the η where the balanced arm's median persistent absorption is ≤ 0.02). A2 and
  A3: per arm, each task is the mean over that grid's cells (A2: 2 architectures × 2 updates × η ∈ {0.1, 1}; A3:
  G ∈ {8, 32} × η ∈ {0.03, 0.3, 1}). Then the median over the 20 tasks, with the range over tasks in brackets. k* is
  computed on each task's mean pass@k curves; the table gives its lower median over tasks (so a k in the list,
  with "none" as infinite) and the number of tasks with a crossover.
- **Runs.** 9,280 `v3_a1` (8,960 RL runs; the 320 supervised runs are left out, as in A1), 640 `v3_a2`, 360
  `v3_a3`, as in `SUMMARY.md`. Every run starts uniform.

**Check:** the start policy's pass@k equals 1 − (2/3)^k in all 10,280 runs, to within 1.3e-8 (float32 storage of 1/3).

## A1: m = 0.9 against the balanced share 1/3

Start: pass@1 = 0.333, pass@8 = 0.961, pass@64 = 1 − 5e-12, log10 C = log10 3 = 0.48 in every run.

| cell | m | pass@1 start | pass@1 final | worst-class pass@1 final | pass@8 start → final | pass@64 start → final | k* (tasks) | log10 C start → final | minority at floor |
|---|---|---|---|---|---|---|---|---|---|
| linear/0.5/adam/expected_full | 0.333 | 0.333 | 0.874 (0.827–0.892) | 0.85 (0.79–0.88) | 0.961 → 0.995 | 1.000 → 1.000 | 16 (20/20) | 0.48 → 0.1 | 0.00 |
| linear/0.5/adam/expected_full | 0.9 | 0.333 | 0.928 (0.878–0.951) | 0.15 (0.00019–0.54) | 0.961 → 0.941 | 1.000 → 0.951 | 8 (20/20) | 0.48 → 5.1 | 0.00 |
| linear/0.5/adam/sampled | 0.333 | 0.333 | 0.818 (0.755–0.840) | 0.78 (0.69–0.82) | 0.961 → 0.991 | 1.000 → 1.000 | 16 (20/20) | 0.48 → 0.2 | 0.00 |
| linear/0.5/adam/sampled | 0.9 | 0.333 | 0.921 (0.868–0.943) | 0.02 (0.00021–0.44) | 0.961 → 0.930 | 1.000 → 0.945 | 8 (20/20) | 0.48 → 5.0 | 0.00 |
| linear/0.5/sgd/expected_full | 0.333 | 0.333 | 0.893 (0.822–0.930) | 0.86 (0.63–0.92) | 0.961 → 0.990 | 1.000 → 0.998 | 16 (20/20) | 0.48 → 0.4 | 0.00 |
| linear/0.5/sgd/expected_full | 0.9 | 0.333 | 0.900 (0.865–0.926) | 0.00041 (2.2e-06–0.24) | 0.961 → 0.902 | 1.000 → 0.907 | 8 (20/20) | 0.48 → 10.0 | 0.11 |
| linear/0.5/sgd/sampled | 0.333 | 0.333 | 0.888 (0.719–0.925) | 0.84 (0.62–0.91) | 0.961 → 0.981 | 1.000 → 0.993 | 16 (20/20) | 0.48 → 1.8 | 0.00 |
| linear/0.5/sgd/sampled | 0.9 | 0.333 | 0.901 (0.865–0.925) | 0.0004 (2.2e-06–0.23) | 0.961 → 0.905 | 1.000 → 0.910 | 8 (20/20) | 0.48 → 10.1 | 0.11 |
| linear/0.9/adam/expected_full | 0.333 | 0.333 | 0.809 (0.748–0.832) | 0.78 (0.69–0.82) | 0.961 → 0.991 | 1.000 → 1.000 | 16 (20/20) | 0.48 → 0.2 | 0.00 |
| linear/0.9/adam/expected_full | 0.9 | 0.333 | 0.896 (0.865–0.921) | 0.00063 (0.00023–0.0011) | 0.961 → 0.897 | 1.000 → 0.899 | 8 (20/20) | 0.48 → 3.9 | 0.00 |
| linear/0.9/adam/sampled | 0.333 | 0.333 | 0.710 (0.627–0.737) | 0.66 (0.5–0.71) | 0.961 → 0.981 | 1.000 → 0.999 | 16 (20/20) | 0.48 → 0.3 | 0.00 |
| linear/0.9/adam/sampled | 0.9 | 0.333 | 0.896 (0.864–0.921) | 0.00067 (0.00023–0.0011) | 0.961 → 0.897 | 1.000 → 0.899 | 8 (20/20) | 0.48 → 3.9 | 0.00 |
| linear/0.9/sgd/expected_full | 0.333 | 0.333 | 0.778 (0.680–0.818) | 0.75 (0.61–0.8) | 0.961 → 0.988 | 1.000 → 0.999 | 16 (20/20) | 0.48 → 0.2 | 0.00 |
| linear/0.9/sgd/expected_full | 0.9 | 0.333 | 0.896 (0.865–0.922) | 7.6e-05 (1.3e-05–0.00019) | 0.961 → 0.897 | 1.000 → 0.897 | 8 (20/20) | 0.48 → 4.6 | 0.00 |
| linear/0.9/sgd/sampled | 0.333 | 0.333 | 0.745 (0.602–0.799) | 0.64 (0.43–0.73) | 0.961 → 0.951 | 1.000 → 0.978 | 8 (20/20) | 0.48 → 0.9 | 0.00 |
| linear/0.9/sgd/sampled | 0.9 | 0.333 | 0.896 (0.865–0.922) | 7.4e-05 (1.3e-05–0.00019) | 0.961 → 0.897 | 1.000 → 0.897 | 8 (20/20) | 0.48 → 4.6 | 0.00 |
| mlp/0.5/adam/expected_full | 0.333 | 0.333 | 0.999 (0.989–1.000) | 1 (0.98–1) | 0.961 → 1.000 | 1.000 → 1.000 | 32 (15/20) | 0.48 → 1.6 | 0.00 |
| mlp/0.5/adam/expected_full | 0.9 | 0.333 | 0.924 (0.887–0.955) | 0.093 (8.9e-10–0.48) | 0.961 → 0.930 | 1.000 → 0.931 | 8 (20/20) | 0.48 → 21.0 | 0.31 |
| mlp/0.5/adam/sampled | 0.333 | 0.333 | 0.979 (0.938–0.993) | 0.96 (0.85–0.99) | 0.961 → 0.987 | 1.000 → 0.989 | 16 (20/20) | 0.48 → 5.9 | 0.00 |
| mlp/0.5/adam/sampled | 0.9 | 0.333 | 0.908 (0.865–0.940) | 3.5e-06 (3.5e-08–0.23) | 0.961 → 0.908 | 1.000 → 0.909 | 8 (20/20) | 0.48 → 14.6 | 0.13 |
| mlp/0.5/sgd/expected_full | 0.333 | 0.333 | 0.996 (0.875–0.998) | 0.99 (0.81–1) | 0.961 → 1.000 | 1.000 → 1.000 | 32 (12/20) | 0.48 → 0.5 | 0.00 |
| mlp/0.5/sgd/expected_full | 0.9 | 0.333 | 0.896 (0.865–0.922) | 1.8e-05 (2.1e-07–0.00021) | 0.961 → 0.896 | 1.000 → 0.897 | 8 (20/20) | 0.48 → 6.1 | 0.00 |
| mlp/0.5/sgd/sampled | 0.333 | 0.333 | 0.963 (0.835–0.975) | 0.93 (0.71–0.95) | 0.961 → 0.972 | 1.000 → 0.974 | 16 (20/20) | 0.48 → 6.2 | 0.00 |
| mlp/0.5/sgd/sampled | 0.9 | 0.333 | 0.896 (0.865–0.922) | 2.6e-05 (2.3e-07–0.00028) | 0.961 → 0.896 | 1.000 → 0.897 | 8 (20/20) | 0.48 → 5.5 | 0.00 |
| mlp/0.9/adam/expected_full | 0.333 | 0.333 | 0.998 (0.882–1.000) | 1 (0.83–1) | 0.961 → 0.999 | 1.000 → 0.999 | 32 (16/20) | 0.48 → 3.1 | 0.00 |
| mlp/0.9/adam/expected_full | 0.9 | 0.333 | 0.900 (0.865–0.924) | 5.2e-07 (3.3e-08–0.12) | 0.961 → 0.900 | 1.000 → 0.900 | 8 (20/20) | 0.48 → 15.2 | 0.33 |
| mlp/0.9/adam/sampled | 0.333 | 0.333 | 0.934 (0.842–0.962) | 0.88 (0.75–0.93) | 0.961 → 0.969 | 1.000 → 0.976 | 16 (20/20) | 0.48 → 5.8 | 0.00 |
| mlp/0.9/adam/sampled | 0.9 | 0.333 | 0.896 (0.865–0.922) | 1.5e-06 (3.4e-07–8.6e-06) | 0.961 → 0.896 | 1.000 → 0.896 | 8 (20/20) | 0.48 → 11.3 | 0.15 |
| mlp/0.9/sgd/expected_full | 0.333 | 0.333 | 0.887 (0.761–0.967) | 0.79 (0.46–0.93) | 0.961 → 0.936 | 1.000 → 0.955 | 8 (20/20) | 0.48 → 2.4 | 0.00 |
| mlp/0.9/sgd/expected_full | 0.9 | 0.333 | 0.896 (0.865–0.922) | 9.7e-06 (2e-06–2.6e-05) | 0.961 → 0.896 | 1.000 → 0.897 | 8 (20/20) | 0.48 → 4.4 | 0.00 |
| mlp/0.9/sgd/sampled | 0.333 | 0.333 | 0.896 (0.737–0.940) | 0.81 (0.43–0.89) | 0.961 → 0.946 | 1.000 → 0.962 | 8 (20/20) | 0.48 → 2.7 | 0.00 |
| mlp/0.9/sgd/sampled | 0.9 | 0.333 | 0.896 (0.865–0.922) | 1.2e-05 (2.3e-06–3e-05) | 0.961 → 0.896 | 1.000 → 0.897 | 8 (20/20) | 0.48 → 4.2 | 0.00 |

## A2 and A3 arms (end of training, each grid's own cell: ρ = 0.9, m = 0.9, SGD)

A2 pools linear and MLP, exact and sampled updates; A3 is linear, sampled. `passk4` is the biased pass@4 weight,
`rloo_k4` the unbiased one.

| grid | cell | m | η | arm | runs | pass@1 final | minority pass@1 final | minority pass@64 final | worst-class pass@1 final | pass@8 start → final | pass@64 start → final | k* (tasks) | log10 C start → final | minority at floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3_a2 | linear/mlp/0.9/sgd/expected_full+sampled | 0.9 | 0.1 1 | forward 0.03 | 160 | 0.949 (0.933–0.968) | 0.67 (0.37–0.85) | 0.936 | 0.6 | 0.961 → 0.984 | 1.000 → 0.993 | 16 (20/20) | 0.48 → 0.24 | 0.00 |
| v3_a2 | linear/mlp/0.9/sgd/expected_full+sampled | 0.9 | 0.1 1 | forward 0.1 | 160 | 0.921 (0.909–0.938) | 0.58 (0.41–0.75) | 0.956 | 0.52 | 0.961 → 0.982 | 1.000 → 0.996 | 16 (20/20) | 0.48 → 0.18 | 0.00 |
| v3_a2 | linear/mlp/0.9/sgd/expected_full+sampled | 0.9 | 0.1 1 | forward 0.3 | 160 | 0.855 (0.779–0.885) | 0.45 (0.34–0.62) | 0.955 | 0.41 | 0.961 → 0.977 | 1.000 → 0.995 | 16 (20/20) | 0.48 → 0.30 | 0.00 |
| v3_a2 | linear/mlp/0.9/sgd/expected_full+sampled | 0.9 | 0.1 1 | none | 160 | 0.896 (0.865–0.922) | 2.4e-06 (5.3e-07–7.4e-06) | 0.000154 | 1.9e-06 | 0.961 → 0.896 | 1.000 → 0.896 | 8 (20/20) | 0.48 → 5.95 | 0.00 |
| v3_a3 | linear/0.9/sgd/sampled | 0.9 | 0.03 0.3 1 | grpo | 120 | 0.896 (0.865–0.922) | 7e-06 (1.6e-06–2.2e-05) | 0.000448 | 6e-06 | 0.961 → 0.896 | 1.000 → 0.897 | 8 (20/20) | 0.48 → 5.21 | 0.00 |
| v3_a3 | linear/0.9/sgd/sampled | 0.9 | 0.03 0.3 1 | passk4 | 120 | 0.850 (0.802–0.863) | 0.6 (0.52–0.64) | 1 | 0.58 | 0.961 → 0.999 | 1.000 → 1.000 | 64 (12/20) | 0.48 → 0.09 | 0.00 |
| v3_a3 | linear/0.9/sgd/sampled | 0.9 | 0.03 0.3 1 | rloo_k4 | 120 | 0.816 (0.769–0.831) | 0.54 (0.46–0.58) | 1 | 0.51 | 0.961 → 0.998 | 1.000 → 1.000 | 64 (18/20) | 0.48 → 0.11 | 0.00 |

## Reading

- **At m = 0.9 the start policy overtakes the trained one at k* = 8 in all 16 cells and all 20 tasks.** Trained pass@1
  is 0.90–0.93 against 0.333, but pass@8 falls from 0.961 to 0.896–0.941 and pass@64 from 1.000 to 0.896–0.951. The
  trained policy's pass@k stops near the majority share; the minority's pass@64 ends at 5e-5 to 0.52 (cell medians).
- **At m = 1/3 the trained policy also falls behind at large k, but by less.** k* is 16 or 32 in 13 cells and 8 in
  three ρ = 0.9 SGD cells (linear sampled, both MLP); three MLP exact-update cells have no crossover in 4–8 of 20
  tasks. pass@64 stays at 0.955–1.000: the start leads by at most 0.045 there, against 0.049–0.104 at m = 0.9.
- **Worst-class pass@1 separates the two shares; mean pass@1 does not.** Mean pass@1 at m = 0.9 (0.90–0.93) is higher
  than at 1/3 in 10 of 16 cells (all linear cells and the two MLP ρ = 0.9 SGD cells) and lower by up to 0.10 in the
  other six. The worst class ends below 1e-3 in 13 of 16 cells at m = 0.9 (at most 0.15), and at 0.64–1.00 at m = 1/3.
- **Coverage rises from C = 3 to 10^3.9–10^21 at m = 0.9.** At m = 1/3 log10 C is 0.1–6.2; the values above 1 come from
  a few prompts (C is a mean of 1/p), since the median of 1/p over minority prompts stays at most 1.4 at m = 1/3 against
  10^1.5–10^18 at m = 0.9.
- **The forward-KL anchor keeps coverage.** At β ∈ {0.03, 0.1, 0.3} log10 C ends at 0.18–0.30 (below the start's 0.48),
  pass@64 at 0.993–0.996 and the minority's pass@1 at 0.45–0.67. The start still wins from k = 16, by 0.004–0.013. The
  no-anchor arm matches A1 (k* = 8, pass@64 0.896, log10 C 5.95).
- **pass@4 keeps coverage at a cost in pass@1.** Both forms end with minority pass@1 0.54–0.60, pass@64 = 1.000 and
  log10 C 0.09–0.11, against GRPO's 7e-6, 0.897 and 5.21; mean pass@1 is 0.82–0.85 against 0.896. Their k* = 64 is a
  tie at numerical precision: start minus final pass@64 is below 5e-7 in every pass@4 run.

## Limits

- **Float32 floor.** Final p is stored as exactly 0 in 1,077 of the 9,960 RL runs (2.6% of all final per-prompt
  values). In six m = 0.9 cells the median over tasks of the minority share at 0 is 11–33%. log10 C with p clipped at 1e-30 is then a lower
  bound, and its value in those cells is set by the clip, not by the policy.
- **Exact, not sampled.** pass@k and C are computed from the exact per-prompt probabilities of the final policy, with
  independent samples assumed. A sampled evaluation would add estimator noise and could not resolve p near 1e-30.
- **The start policy is uniform by construction,** so its pass@k is 1 − (2/3)^k exactly and its coverage is 3. A
  pretrained base model would not be uniform, and k* depends on this baseline: at m = 0.9, k* = 8 is where
  1 − (2/3)^k first passes the trained policy's plateau near 0.9.
- **k is limited to the list** {1, …, 64}, so k* is resolved only to a power of two.
- **Pooling.** The A2 and A3 rows average over architectures, updates, G and η within each task; the per-run values
  for finer splits are in `harm_runs.csv`.
