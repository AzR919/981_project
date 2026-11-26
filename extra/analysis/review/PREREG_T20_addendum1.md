# PREREG_T20, addendum 1 (2026-11-23): prevention levers in the transformer environment (T21)

Committed before batch `v3i` launches. Code: `enum-env`, "T21" commit. Exploratory.

## Why

T20 showed the minority collapse in Li, Pandey and Chen's environment (`extra/analysis/v3h/SUMMARY.md`).
In our toy, a forward KL anchor and pass@k weighting prevent absorption, and a reverse KL anchor at
usual strengths does not (`extra/analysis/v3g/SUMMARY.md`, `extra/analysis/v3f/SUMMARY.md`). T21 asks whether
the same levers behave the same way with a transformer on token prompts.

## What has been seen

T20 in full, including: under Adam (lr 1e-3) and SGD (lr 0.3) the minority is absorbed at share
0.875 while class 0 is not; on the parity task the exact update with Adam recovers fully and
sampled GRPO does not. No T21 run has been made; only unit tests of the new options.

## Design

- **Tasks:** K = 3 with 56 of 64 prompts in class 0 (random assignment); the parity task (48 of 64).
- **Policy and steps:** shared transformer; exact update and sampled GRPO (G = 8); Adam at lr 1e-3
  and SGD at lr 0.3 (the two step rules that did not break the whole policy in T20); 2,000 steps;
  10 seeds.
- **Arms:** no lever; reverse KL, KL(π ‖ base), at β ∈ {0.03, 0.3}; forward KL, KL(base ‖ π), at
  β ∈ {0.03, 0.3}; pass@k weighting, each prompt's loss times k(1 − p)^(k−1) with its exact success
  p, at k ∈ {2, 4}. 560 runs.
- **Measures:** share of minority prompts absorbed (success below 1e-3 over the last 20% of steps);
  final class means; the minority's lowest class mean.

A cell is one task × update × step rule (8 cells). The "collapsing cells" are those whose no-lever
arm has a mean absorbed share of at least 0.5.

## Predictions

1. **Forward KL prevents absorption:** mean absorbed share at most 0.05 at both β, in every cell.
2. **At β = 0.03 forward KL costs little:** class 0 ends at 0.8 or above (median) in every cell.
3. **Reverse KL at β = 0.03 does not prevent it:** mean absorbed share at least 0.5 in every
   collapsing cell.
4. **pass@4 lowers absorption:** below the no-lever arm by at least 0.3 in every collapsing cell.
5. **Reported, not predicted:** reverse KL at β = 0.3 (in the toy it helped only from about this
   strength); pass@2; whether any lever keeps the minority up during learning (lowest mean above
   0.2) or only brings it back.

## Reading

- **1–4 hold:** the levers transfer; the prevention section of the report can cite both
  environments.
- **1 fails under sampled GRPO only:** the anchor needs more strength when the reward term is noisy;
  report with the strengths tried.
- **3 fails (reverse KL at 0.03 protects):** the KL direction matters less with a transformer than
  in the toy; report as a difference between the environments.

## Deviations

Listed in the T21 write-up.
