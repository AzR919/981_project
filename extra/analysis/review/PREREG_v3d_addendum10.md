# PREREG_v3d, addendum 10 (2026-11-24): caveats raised by the review (G, H, J)

Committed before batches `v3p` (toy, `toy-v3` `4391e1a`: `v3_t27`, `v3_t28`) and `v3q` (`enum-env`:
`v3_t29`) launch. Exploratory.

## G (T27): forward KL toward a collapsed reference

T17's protection used a reference (the start policy) that still solves the minority. In practice the
reference may already be collapsed. Linear and MLP, ρ = 0.9, m = 0.9, SGD η = 0.1, exact, 20,000 steps; anchor
taken at 0.5%, 5% or 25% of training, forward β ∈ {0.01, 0.1, 1}; controls: no anchor, and forward KL toward
the start (T17).

1. Anchors taken after the minority has fallen below 1e-2 do not restore it: the minority at the end is
   within a factor 10 of its value at the anchor step, or lower, in ≥ 80% of such runs.
2. The start anchor (T17 control) prevents absorption at β ≥ 0.1 (as in T17) in ≥ 90% of runs.
3. Reported: whether an early anchor (0.5%, before the fall) protects like the start anchor.

Reading: if 1 holds, forward KL protects only while the reference still solves the minority, which limits
the practical claim.

## H (T28): Adam's epsilon

Adam recovers the minority more often than SGD (T12). If the per-coordinate normalisation is the cause,
the difference should vanish as ε grows (Adam tends to momentum SGD with step η/ε; η is scaled with ε).

1. The share of runs whose minority recovers (class mean back above 0.1 at the end after falling below
   1e-3) decreases from ε = 1e-8 to ε = 1, and at ε = 1 is within 0.1 of the SGD control, in ≥ 6 of 8
   (architecture, ρ, m) cells.
2. Reported: the late exponent and γ_min under each ε, with the ℓ∞ max-margin γ (Zhang, Zou, Cao 2024)
   computed for comparison.

## J (T29): the transformer environment under SGD at small step sizes

T20's share effect was clean under Adam; at SGD 1 and 3 both classes were absorbed. Small steps:
SGD η ∈ {0.03, 0.1}, flow time matched to SGD 0.3 × 2,000 steps; K = 3, six shares; both algorithms.

1. With η ≤ 0.1, class 0 is never absorbed at share ≥ 0.875, and the minority is absorbed in ≥ 80% of
   high-share runs and in ≤ 20% of balanced-share runs (n_major = 22).

## Deviations

Listed in the v3p/v3q write-ups.
