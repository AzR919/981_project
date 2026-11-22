# Bay & Yearick's group-size law against exact theory, learned policies and our speed results

Fidelity φ(G, p) = E[ĝ] / √(p(1 − p)) = W_G(p) / W_∞(p) for GRPO (the identity of our milestone, eq. 1, is their group-standard-deviation identity in expectation). Interior law: φ ≈ 1 − 1/(8Gp(1 − p)); budget for fidelity 1 − ε: G* = 1/(8εp(1 − p)).

## (a) Exact against the closed form

| G | p | phi_exact | phi_law | abs_err |
|---|---|---|---|---|
| 2 | 0.02 | 0.14 | -2.19 | 2.33 |
| 2 | 0.05 | 0.218 | -0.316 | 0.534 |
| 2 | 0.1 | 0.3 | 0.306 | 0.00554 |
| 2 | 0.2 | 0.4 | 0.609 | 0.209 |
| 2 | 0.5 | 0.5 | 0.75 | 0.25 |
| 4 | 0.02 | 0.241 | -0.594 | 0.836 |
| 4 | 0.05 | 0.373 | 0.342 | 0.0306 |
| 4 | 0.1 | 0.507 | 0.653 | 0.146 |
| 4 | 0.2 | 0.663 | 0.805 | 0.142 |
| 4 | 0.5 | 0.808 | 0.875 | 0.067 |
| 8 | 0.02 | 0.36 | 0.203 | 0.158 |
| 8 | 0.05 | 0.539 | 0.671 | 0.132 |
| 8 | 0.1 | 0.698 | 0.826 | 0.128 |
| 8 | 0.2 | 0.843 | 0.902 | 0.0597 |
| 8 | 0.5 | 0.928 | 0.938 | 0.00967 |
| 16 | 0.02 | 0.505 | 0.601 | 0.0969 |
| 16 | 0.05 | 0.712 | 0.836 | 0.123 |
| 16 | 0.1 | 0.854 | 0.913 | 0.0588 |
| 16 | 0.2 | 0.938 | 0.951 | 0.0127 |
| 16 | 0.5 | 0.967 | 0.969 | 0.00169 |
| 32 | 0.02 | 0.667 | 0.801 | 0.134 |
| 32 | 0.05 | 0.859 | 0.918 | 0.0587 |
| 32 | 0.1 | 0.943 | 0.957 | 0.0134 |
| 32 | 0.2 | 0.974 | 0.976 | 0.0018 |
| 32 | 0.5 | 0.984 | 0.984 | 0.00039 |
| 64 | 0.02 | 0.821 | 0.9 | 0.0794 |
| 64 | 0.05 | 0.945 | 0.959 | 0.0138 |
| 64 | 0.1 | 0.976 | 0.978 | 0.00187 |
| 64 | 0.2 | 0.987 | 0.988 | 0.000368 |
| 64 | 0.5 | 0.992 | 0.992 | 9.46e-05 |

- Where 8Gp(1 − p) ≥ 4 (law ≥ 0.75): max |exact − law| = 0.250. Below that the expansion breaks down (it even goes negative), as they note near p ∈ {0, 1}.

## (b) Learned policies (base H2 bins, GRPO, SGD)

φ measured from the one-step mean update (H2) against the exact φ and the interior law.

| level | G | bins | med_rel_err_vs_exact | med_abs_err_vs_law_interior |
|---|---|---|---|---|
| lin | 2 | 30 | 0.00432 | nan |
| lin | 4 | 30 | 0.00681 | 0.077 |
| lin | 16 | 30 | 0.00634 | 0.00796 |
| lin | 64 | 30 | 0.00626 | 0.00631 |
| m64 | 2 | 30 | 0.014 | nan |
| m64 | 4 | 30 | 0.018 | 0.0611 |
| m64 | 16 | 30 | 0.0435 | 0.044 |
| m64 | 64 | 30 | 0.0498 | 0.0495 |
| tab | 2 | 30 | 0.000122 | nan |
| tab | 4 | 30 | 0.000718 | 0.108 |
| tab | 16 | 30 | 0.00683 | 0.0261 |
| tab | 64 | 30 | 0.00904 | 0.00958 |

- Interior bins pooled: median |measured − law| = 0.041.
- Measured φ matches the exact φ to within 0.9% (tabular) and 0.7% (linear, shared); the MLP deviates by 1.4–5%, through cross-prompt terms in the mean update (H2) that a single-prompt law does not model.
- The interior law itself is off by 0.06–0.11 at G = 4, where 8Gp(1 − p) is small; from G = 16 it is within 0.01–0.05 of the measurement.

## (c) Their budget against our speed optimum (GRPO, T4 + T4e, samples to mean p = 0.5)

G* (ε = 0.1) is the group size their law says is needed for 90% gradient fidelity; `samples_at_G_star` uses the grid G nearest to G*.

| level | p0 | best_G_speed | samples_at_best | samples_at_G_star | G_star_eps01 | slowdown_at_G_star |
|---|---|---|---|---|---|---|
| lin | 0.01 | 4 | 2.3e+03 | 2.59e+04 | 126 | 11.2 |
| lin | 0.03 | 2 | 768 | 3.84e+03 | 43 | 5 |
| lin | 0.1 | 2 | 256 | 768 | 13.9 | 3 |
| lin | 0.3 | 2 | 256 | 256 | 5.95 | 1 |
| m64 | 0.01 | 2 | 768 | 4.95e+04 | 126 | 64.5 |
| m64 | 0.03 | 2 | 256 | 7.94e+03 | 43 | 31 |
| m64 | 0.1 | 8 | 256 | 1.79e+03 | 13.9 | 7 |
| m64 | 0.3 | 2 | 1.02e+03 | 1.02e+03 | 5.95 | 1 |
| tab | 0.01 | 2 | 3.58e+03 | 7.42e+03 | 126 | 2.07 |
| tab | 0.03 | 2 | 1.54e+03 | 2.05e+03 | 43 | 1.33 |
| tab | 0.1 | 2 | 512 | 768 | 13.9 | 1.5 |
| tab | 0.3 | 2 | 256 | 256 | 5.95 | 1 |

- Per sample, the fastest group is far smaller than G*: fidelity per update is not efficiency per sample. A small group gives a noisier but much cheaper update, and at a tuned η that wins. The law answers 'how many samples make one update faithful', not 'which G learns fastest for a fixed budget'.
