# A minimal example of minority collapse: two prompt kinds, two answers

## Status

A derivation with numerical checks, written on 2026-11-26. It is **not a registered test**: the numbers of the
example were seen while it was being built (`PREREG_v3d_addendum11.md`, "Not registered"). The tables come from
`experiments/minimal_example.py` (`analysis/theory/minimal_example/*.csv`) and the checks from
`experiments/check_minimal_example.py` (`minimal_example_check.md`, `minimal_example/phase.csv`). Nothing is fitted.

| Part | Status |
|---|---|
| Section 1: non-negative interactions are sufficient for no prompt to ever fall; the K = 2 converse | Proved below (three lines); the algebra is checked in SymPy and on random policies |
| Section 2: the first-step threshold R > 1 | Proved (an identity); agrees with the exact flow in 3,263 of 3,263 runs with R ≠ 1 |
| Section 4: the tail system, its threshold γ > α/β, the saddle, the separatrix, the late exponent | Proved for the tail system (an asymptotic approximation of the flow); SymPy; checked against the exact flow |
| Where the phase boundary lies (the depth a fall from the uniform start reaches) | Observed only (exact flow), no closed form |
| Section 5: what each lever does | Fixed points and the reset step derived; everything else observed |
| The reset threshold in N | Observed; the guessed formula 1/(ηβλ) has the right β scaling and the wrong constant |

## Setting

- K = 2 answers. Two kinds of prompts. Majority prompts have features a = (1, 0), correct answer 0 and share m.
  Minority prompts have features b = (γ, 0.5), correct answer 1 and share 1 − m. γ = a·b/|a|² is the overlap of the
  minority's features with the majority's. |b|² = γ² + 1/4, and b_⊥ = (0, 0.5) is the part of b orthogonal to a.
- Policy: π(·|x) = softmax(Wφ(x)) with W (2 × 2) starting at 0, so both kinds start at success 1/2. Only
  w = W_0 − W_1 matters: the majority's success is p_A = σ(u) with margin u = w·a, the minority's is p_B = σ(s) with
  margin s = −w·b. v = −s = w·b is the minority's deficit; σ is the logistic function.
- A reward rule acts through its weight W(p) (the advantage gap of `synthetic_rl/theory.py`): the expected update is
  Σ_x share_x W(p_x)∇p_x. Define h(z) = W(σ(z))σ(z)σ(−z). In the time t of a direct logistic model in w,

  du/dt = m h(u)|a|² − (1 − m) h(s)(a·b),  ds/dt = −m h(u)(a·b) + (1 − m) h(s)|b|².

  The simulator's flow time τ (η × steps) runs twice as fast, t = 2τ, because both rows of W move. A direct model
  with this factor agrees with `exact_flow` to 6e-9 decades (`minimal_example_check.md`).
- Rules: the mean rule (Dr. GRPO, `mean`; W = 1 − 1/G, 0.875 at G = 8), GRPO with standard-deviation normalisation
  (`grpo`, ε = 1e-5 in the denominator), GRPO in the large-group limit (G = ∞, W = 1/√(p(1 − p) + ε)), and unbiased
  pass@k (`rloo_k`, W = (1 − p)^(k−1)). (The comment `# mean: W = 1 for every G` in `minimal_example.py` should read
  "W = 1 − 1/G, the same at every p".)
- "Stays down" means the minority's success is still below 1/2 at flow time 1e14; normal training reaches 1e3–1e4.

## 1. A sufficient condition: non-negative interactions

**Statement.** Take any softmax-linear policy π_θ(y|x) ∝ exp(θ·φ(x, y)) with one correct answer y\*_x per prompt and
let d(x, y) = φ(x, y\*_x) − φ(x, y). If ⟨d(x, y), d(x′, y′)⟩ ≥ 0 for every pair of different prompts x ≠ x′ and all
answers y, y′, then along the flow θ̇ = Σ_x′ share_x′ W(p_x′)∇p_x′ with W ≥ 0 no prompt's success ever decreases.

**Proof.**
1. ∇ log π(y\*|x) = φ(x, y\*) − Σ_y π(y|x)φ(x, y) = Σ_y π(y|x) d(x, y), so ∇p_x = p_x Σ_y π(y|x) d(x, y).
2. κ(x, x′) = ⟨∇p_x, ∇p_x′⟩ = p_x p_x′ Σ_{y,y′} π(y|x)π(y′|x′)⟨d(x, y), d(x′, y′)⟩ ≥ 0 for x ≠ x′, at every θ, and
   κ(x, x) = |∇p_x|² ≥ 0 always.
3. dp_x/dt = ⟨∇p_x, θ̇⟩ = Σ_x′ share_x′ W(p_x′) κ(x, x′) ≥ 0.

The weights of GRPO at any G, the mean rule, RLOO and pass@k (biased and unbiased) are non-negative: the smallest
value on a grid of p from 1e-12 to 1 − 1e-12 is 1e-84, for unbiased pass@8 near p = 1. The statement is about the
expected-update flow. It does not cover finite steps (a second-order term), sampling noise or Adam's preconditioner.

**Tabular parameters** are the special case φ(x, y) = e_x ⊗ e_y, where every cross-prompt inner product is 0.

**The class-label toy.** With φ(x, y) = e_y ⊗ φ(x), ⟨d(x, y), d(x′, y′)⟩ = ⟨e_c − e_y, e_c′ − e_y′⟩·⟨φ(x), φ(x′)⟩, where
c and c′ are the correct classes. Over wrong answers the first factor takes the values 2 (same class, K = 2), 1 and 2
(same class, K ≥ 3), −2 only (different classes, K = 2), and −2, −1 and 1 (different classes, K = 3; also 0 for
K ≥ 4; SymPy test `test_class_label_interaction_factor`). So the condition needs within-class inner products ≥ 0 and,
across classes, inner products equal to 0 for K ≥ 3 and ≤ 0 for K = 2. In the example it reads γ ≤ 0.

**The K = 2 converse.** At the uniform start, ∇p_x = d_x/4 for K = 2, so prompt x with n′ copies of a prompt x′ changes
at a rate proportional to W(1/2)(|d_x|² + n′⟨d_x, d_x′⟩). If ⟨d_x, d_x′⟩ < 0, prompt x falls as soon as
n′ > |d_x|²/|⟨d_x, d_x′⟩|. In the example this threshold is |b|²/γ; the simulator's first step falls exactly when it is
exceeded in 12 of 12 cases tested around it.

**Checks.** On 300 random softmax-linear policies (K = 3, 6 prompts, d ≥ 0 entrywise) the gradient formula matches
central differences to 2.6e-7 (relative), and the smallest κ and the smallest dp_x/dt under eight rules are
non-negative. A flow of one such problem under GRPO to t = 1e8 has no decrease at all. In the phase table below, all
506 runs with γ ≤ 0 never fall, and no step of either prompt's log success decreases.

**Credit.** The kernel κ and the term "prompt interference" are from Barakat et al. (NeurIPS 2026, arXiv 2602.21189,
Def. 3.1). The condition is the instructor's (unpublished). Neither is ours; what this section adds is the short proof
for the softmax-linear case, the class-label reading and the K = 2 converse.

## 2. The first step

At the uniform start u = s = 0, both weights equal h(0) and ds/dt = h(0)[(1 − m)|b|² − mγ]. The minority starts to
fall exactly when

  R = γm / ((1 − m)|b|²) > 1.

This is the odds law (`odds_law.md`) for K = 2: R = c·m/(1 − m) with c = k_BA/k_BB = γ/|b|². Because W(1/2) is a common
factor, R is the same for every rule. Across the 3,266 runs of the phase table the sign of the minority's change at
τ = 1e-5 agrees with R > 1 in all 3,263 runs with R ≠ 1. The other three have R = 1 to rounding, where the first-order
change is zero; all three never fall. On this grid a minority that does not start to fall never falls later
(3,263 of 3,263).

## 3. Three regimes

Mean rule, G = 8, m = 0.9, exact flow to τ = 1e14 (`regimes.csv`):

| γ | R | lowest minority success | outcome |
|---|---|---|---|
| −0.5 | −9 | 0.5 (start) | never falls |
| 0 | 0 | 0.5 (start) | never falls |
| 0.25 | 7.2 | 10^−0.45 | falls, back above 1/2 at τ = 112 |
| 0.5 | 9.0 | 10^−0.78 | falls, back at τ = 282 |
| 0.75 | 8.3 | 10^−1.6 | falls, back at τ = 1.3e3 |
| 0.9 | 7.6 | 10^−3.3 | falls, back at τ = 5.3e4 |
| 1.0 | 7.2 | 10^−9.2 | falls, back at τ = 4.2e10 |
| 1.1 | 6.8 | 10^−15.5 | stays down (still falling at 1e14) |
| 1.2 | 6.4 | 10^−17.0 | stays down |
| 1.5 | 5.4 | 10^−21.3 | stays down |
| 2.0 | 4.2 | 10^−28.4 | stays down |
| 3.0 | 2.9 | 10^−42.6 | stays down |

The three regimes are: γ ≤ 0, nothing ever falls (section 1); 0 < γ ≤ 1 with R > 1, the minority falls and comes back
(a plateau, longer as γ approaches 1); γ > 1 and a deep enough fall, it stays down (section 4). R does not order the
outcome: R is largest at γ = 0.5, which recovers fastest.

**How the boundary moves with m** (`phase.csv`: 71 values of γ in [−0.5, 3] × 46 values of m in [0.50, 0.95], mean
rule, τ = 1e14):

| m | γ that falls | γ that stays down |
|---|---|---|
| 0.50 | none | none |
| 0.55 | 0.30–0.95 | none |
| 0.60 | 0.20–1.30 | none |
| 0.65 | 0.15–1.70 | none |
| 0.67 | 0.15–1.85 | 1.40–1.60 |
| 0.70 | 0.15–2.20 | 1.25–2.10 |
| 0.75 | 0.10–2.90 | 1.15–2.85 |
| 0.80 | 0.10–3.00 | 1.10–3.00 |
| 0.85 | 0.05–3.00 | 1.05–3.00 |
| 0.90 | 0.05–3.00 | 1.05–3.00 |
| 0.95 | 0.05–3.00 | 1.00–3.00 |

- Nothing stays down below m = 0.67, although γ > 1 and R > 1 together are possible from m > 5/9 (R = 1 at γ = 1
  needs m = 5/9). The fall must also be deep enough.
- The stays-down region has two edges. The lower edge comes down toward γ = 1 as m grows (1.40 at m = 0.67, 1.05 at
  m = 0.9). At moderate m there is also an upper edge: for large γ, R = γm/((1 − m)(γ² + 1/4)) falls back toward 1,
  the fall is shallow and the minority recovers (at m = 0.7, γ = 2.2 recovers and γ ≥ 2.25 never falls).
- The four runs at γ = 1.0 that stay down (m = 0.92–0.95) and one at γ = 1.05, m = 0.85 are on the recovering side of
  the tail system (section 4) and are predicted to recover after 1e14; the other 971 runs that stay down are past the
  separatrix at 1e14.

**Group size and the mean rule give the same picture.** In `regimes.csv` (m ∈ {0.5, 0.7, 0.9}, 12 values of γ) the
outcome class is the same under the mean rule, GRPO at G = 2 and 8, and GRPO at G = ∞ in all 36 cells. Only the depth
changes, a little (at m = 0.9, γ = 1.2: 10^−17.0 for the mean rule, 10^−17.5 for GRPO G = 8, 10^−20.0 for G = ∞). So
neither finite G nor the standard-deviation normalisation is needed for the collapse.

**Sampled runs** (`sampled.csv`; GRPO G = 8, B = 32, SGD η = 0.5, 5,000 steps, 20 seeds, m = 0.9): at γ = −0.5 no seed
ends below 1e-3 and the median lowest success is 0.5; at γ = 0.5 none ends below 1e-3 (median lowest 0.15, median final
0.995); at γ = 1.2 all 20 end below 1e-3 (median final 1.6e-5). The exact flow agrees with small-step exact SGD
(η = 0.01) to 6.4e-4 decades up to τ = 500 (`agreement.csv`).

## 4. Why it stays down: tails

**Tail weights.** On the collapsed branch u → ∞ and v → ∞. Write h(u) ≈ C_A e^{−αu} for the majority and
h(−v) ≈ C_B e^{−βv} for the minority. Local exponents measured from the code's weights between z = 25 and 30:

| Rule | α (majority) | β (minority) | α/β |
|---|---|---|---|
| mean rule, REINFORCE | 1 | 1 | 1 |
| GRPO, G = 2 and 8 | 1 | 1 | 1 |
| GRPO, G = ∞ without ε | 1/2 | 1/2 | 1 |
| GRPO, G = ∞ as coded (ε = 1e-5) | 0.49 for z ≈ 3–6, then 1 | same | 1 |
| unbiased pass@k | k | 1 | k |
| supervised cross-entropy (W = 1/p) | 1 | 0 | ∞ |

With ε = 1e-5 the weight saturates at 1/√ε ≈ 316 once p(1 − p) ≪ 1e-5, so deep in the tail the coded G = ∞ rule has
α = β = 1. Under supervised cross-entropy the minority's own pull does not fade (β = 0).

**The tail system.** With X = e^{αu}, Y = e^{βv}, g = a·b and the tail weights, the equations of the setting become

  dX/dt = α[mC_A|a|² − (1 − m)C_B g X/Y],  dY/dt = β[mC_A g Y/X − (1 − m)C_B|b|²].

The right-hand sides are homogeneous of degree 0, so there are self-similar solutions X = ξ\*t, Y = ζ\*t with

  ζ\* = β(1 − m)C_B|b_⊥|² / (βγ/α − 1),  ξ\* = αβ m C_A |a|²|b_⊥|² / (β|b|² − αg).

ζ\* > 0 only when γ > α/β, and then ξ\* > 0 as well. So a collapsed branch exists only when γ > α/β: the threshold is 1
for the mean rule, REINFORCE and GRPO, k for pass@k, and there is none for supervised learning.

**Saddle and separatrix.** In log time with x = X/t, y = Y/t the fixed point (ξ\*, ζ\*) has eigenvalues −1 (along the
ray through it, a shift of time) and (βg − α|a|²)(β|b|² − αg)/(αβ|a|²|b_⊥|²), which is positive on the branch: a saddle.
More than that holds. The ratio ρ = Y/X obeys

  dρ/dt = [mC_A(βg − α|a|²)ρ − (1 − m)C_B(β|b|² − αg)] / X,

which is linear in ρ. So the stable manifold is exactly the ray ρ = ρ\* = ζ\*/ξ\*, and in the tail the separatrix is the
straight line βv − αu = log ρ\*. Above it ρ grows, the minority's pull becomes negligible, X ≈ αmC_A|a|²t,
dv/dt ≈ γ/(αt), so v ≈ (γ/α) log t: Y grows like t^{βγ/α} and the minority's success decays like t^{−γ/α}. Below it
ρ falls and the minority climbs out. On the self-similar solution itself the minority's success decays like t^{−1/β}.
At γ = α/β exactly, dρ/dt < 0 for every ρ: the minority always recovers, more slowly than any power of t. All of this
is in `tests/test_theory_symbolic.py::MinimalExampleTests`.

So permanence needs two things: γ > α/β, and a fall that ends above the line βv − αu = log ρ\*. log ρ\* falls as m
grows, which is why the lower edge of the stays-down region comes down with m. Within the tail the depth condition is
in closed form; what is not is whether a fall from the uniform start reaches the right side of the line.

**Checks against the exact flow** (`minimal_example_check.md`):
- Late slopes d log p_B/d log τ over τ ∈ [1e12, 1e14] equal −γ/α to three digits: −1.2, −1.5, −2.0 and −3.0 for the mean
  rule at γ = 1.2, 1.5, 2 and 3; −1.5 and −3.0 for GRPO G = 8 and for coded GRPO G = ∞ (as predicted for α = 1, not
  −2γ); −1.5 and −2.5 for pass@2 at γ = 3 and 5; −1.5 and −2.0 for pass@4 at γ = 6 and 8 (started past pass@k's
  separatrix). The majority's 1 − p_A decays like τ^{−1/α} (slopes −1, −0.50, −0.25). GRPO at G = ∞ without ε (direct
  model) gives −3.0 and −6.0 at γ = 1.5 and 3, that is −2γ.
- The separatrix: flows started deep in the tail (u0 = 15 for the mean rule and GRPO, 10 for pass@2, 8 for pass@4) with
  βv0 − αu0 = log ρ\* + δ, δ ∈ {−0.5, −0.1, 0.1, 0.5}, for the mean rule (γ = 1.2, 2), GRPO G = 8 (γ = 1.5), pass@2
  (γ = 3) and pass@4 (γ = 6), each to τ = 1e14 after the start. In 20 of 20 the outcome is the one predicted: every run
  with δ > 0 ends further above the line and stays down; every run with δ < 0 recovers.
  A fast version is `tests/test_minimal_example.py::TailTests`.

## 5. What each lever does

### pass@k (unbiased, `rloo_k`)

- **First step:** unchanged (W(1/2) is a common factor in R).
- **Branch:** removed when γ < k (threshold α/β = k). From the uniform start at m = 0.9 (`passk.csv`): pass@2 stays down
  at γ = 2.5, 3 and 5 (10^−18.1, 10^−21.8, 10^−36.4 at 1e14) and recovers at γ ≤ 1.5 (lowest 10^−1.28 at γ = 1.5).
- **Shallower fall:** the majority's weight vanishes like (1 − p_A)^{k−1}, so its push dies early. pass@4 does not
  collapse at any γ tested, including γ = 5 > 4 (lowest success 10^−0.39 = 0.41); pass@8 never goes below 0.43.
- **After a collapse** (`switch.csv`: mean rule or GRPO down to 1e-3 or 1e-6, then pass@k to τ = 1e14 after the
  switch): the minority stays down exactly when γ > k, in 72 of 72 rows. Every state with γ > k was past pass@k's
  separatrix at the switch (v − ku − log ρ\*_k between 1.9 and 9.7), as the tail theory needs. In the 48 rows that recover,
  the majority dips (its lowest success is 0.51–0.995) while the minority climbs.
- **Constructed failure:** γ > k from the uniform start (pass@2 at γ = 2.5), or γ > k after a collapse (pass@4 at
  γ = 5 and 6).

### Fixed reverse KL(π‖π_start)

- With strength β_KL, each prompt's one-prompt fixed point is logit p = W/β_KL (`synthetic_rl/theory.py`). Because a and
  b are linearly independent, w ↦ (u, s) is invertible, and the joint resting point is each prompt at its own fixed
  point. The majority stops at u\* = W/β_KL, where its net force (reward minus KL) and hence its push on the minority
  vanish; the minority then returns.
- Direct model, mean rule, γ = 1.2, m = 0.9, to τ = 1e14: β_KL = 1 and 0.3 end with both prompts at the fixed point
  (0.706 and 0.949, equal to `fixed_points`); β_KL = 0.1 falls to 10^−3.7 and is back at τ = 5.0e4; β_KL = 0.03 and 0.01
  are still falling at 1e14 (10^−14.9 and 10^−16.7), because the majority, whose margin grows like log τ, has not yet
  reached u\* (28.9 of 29.2, and 32.2 of 87.5). In `resets.csv` (τ = 3e4): β_KL ≤ 0.1 is still down (10^−5.5 to 10^−3.5),
  β_KL ≥ 0.3 is back, and β_KL = 1 and 3 cap the majority at 0.706 and 0.572.
- "The minority then returns at a rate proportional to its own success" is approximately right. The pull at the bottom
  is about p_B(W + β_KL log(1/p_B)), but at the minority's lowest point the majority is still short of u\*
  (u = 0.72–0.87 u\*) and still pushing, so the climb takes 4–15 times the bottom-only estimate.
- **Constructed failure:** small β_KL (the fall is deep and the return is beyond any training horizon), or a reference
  taken after the collapse. Anchored at the state where the minority first reaches 1e-6, the minority's log-odds can
  only gain W/β_KL over the reference: it ends at 10^−2.2, 10^−5.6 and 10^−6.0 for β_KL = 0.1, 1 and 10, equal to the
  one-prompt fixed point relative to the reference to eight digits.

### Fixed forward KL(π_start‖π)

- The pull on a prompt's margin is β_KL(p_ref − p), which tends to β_KL p_ref = β_KL/2 as p → 0: a floor. Direct model,
  γ = 1.2: the lowest minority success is 10^−3.7, 10^−2.5, 10^−1.9 and 10^−1.3 for β_KL = 0.001, 0.01, 0.03 and 0.1,
  against 10^−17 at 1e14 without KL; all are back by τ = 5e4 and end at the one-prompt fixed point (0.999 to 0.946).
  In `resets.csv` every fixed forward KL is back by τ = 3e4 at both γ.
- **Constructed failure:** a reference taken after the collapse. Then the floor is p_ref. For the mean rule the
  minority's net force near the bottom is about p_B(W − β_KL) + β_KL p_ref: with β_KL > W it stays at about
  p_ref·β_KL/(β_KL − W) (10^−5.1 for β_KL = 1, 10^−6.0 for β_KL = 10, from 10^−6.0), with β_KL < W it climbs to about
  1 − β_KL/W (0.886 for β_KL = 0.1). All three match the one-prompt fixed point to eight digits.

### Reference resets (KL to the latest policy, reverse direction as in Huang et al., arXiv 2611.14467)

- **The exact inner step.** Solved exactly, each inner problem max_π F(p) − β_KL KL(π‖π_old) gives, per prompt, the
  mirror-descent step π_new ∝ π_old exp(W·r/β_KL); for K = 2 its margin moves by z − z_old = W(σ(z))/β_KL. It is
  representable because a and b are linearly independent, so the reward is linear in the features (their Assumption
  2.1). Each prompt's gain depends only on its own state, never on the other prompt: no interference. Under the mean
  rule both gain exactly W/β_KL per reset; under GRPO the gain depends on the prompt's own p. The end point of the
  inner gradient flow in w matches the one-prompt roots to 4e-12 in 36 cases.
- **N steps per reset.** With N gradient steps the inner problem is only partly solved: the KL gradient is zero at the
  reference, so the first steps follow the plain gradient and the minority falls; the correction comes later. N = 1 is
  exactly the unregularised run (`tests/test_minimal_example.py::test_reset_every_step_is_the_unregularised_run`).
- **The threshold in N** (`resets.csv`, η = 0.5, 60,000 steps, τ = 3e4; "back" = final minority success above 0.1):

  | γ | rule | KL | β_KL | still down at N = | back at N = |
  |---|---|---|---|---|---|
  | 1.2 | mean | reverse | 0.3 / 1 / 3 | 100 / 30 / 10 | 300 / 100 / 30 |
  | 1.2 | mean | forward | 0.03 / 0.1 / 0.3 / 1 / 3 | 1000 / 300 / 100 / 30 / 10 | 3000 / 1000 / 300 / 100 / 30 |
  | 1.2 | mean | reverse | 0.01 / 0.03 / 0.1 | 3000 (all) | none up to 3000 |
  | 0.9 | mean | reverse | 0.03 / 0.1 / 0.3 / 1 / 3 | 300 / 30 / 10 / 3 / 1 | 1000 / 100 / 30 / 10 / 3 |

  N\* ∝ 1/β_KL holds within the grid's resolution (a factor of about 3) over 1 to 2 decades of β_KL. The guess
  N\* ≈ 1/(ηβ_KL λ), with λ = 2λ_min(Σ_x share_x p_x(1 − p_x)φ_xφ_xᵀ) at the uniform start (0.0114 at γ = 0.9,
  0.0107 at γ = 1.2), has the right scaling in β_KL but not the right constant: ηβ_KL λ N\* lies in [0.16, 0.54] at
  γ = 1.2 and in [0.017, 0.057] at γ = 0.9, not near 1, and the threshold changes tenfold between γ = 0.9 and 1.2 while
  λ changes by 6%. So λ_min is not what sets it; a formula for the constant is open. At γ = 0.9 "still down" is a
  horizon effect (the plain run is back at τ = 5.3e4 > 3e4), and at γ = 0.9 GRPO is back by τ = 3e4 in every column,
  so those rows carry no information.
- **Constructed failure:** too few steps per reset (N below the threshold, N = 1 in the limit) or a weak β_KL
  (β_KL ≤ 0.1 at γ = 1.2 under reverse KL: no threshold up to N = 3000).

## 6. The ladder

What can be removed and the collapse still happen, from the full setting down to this example:

| What is removed | Still collapses? | Evidence |
|---|---|---|
| nothing (MLP, sampled GRPO, Adam) | yes | A1 (`analysis/v3o/SUMMARY.md`) |
| the nonlinearity (linear policy) | yes | A1 |
| sampling noise (exact expected update) | yes | A1 and H3 (`analysis/base/FINDINGS.md`) |
| the step size (gradient flow) | yes | T25 (`analysis/v3m/SUMMARY.md`) |
| finite G (G = ∞) | yes | this note, section 3 |
| the standard-deviation normalisation (mean rule) | yes | v3f (`analysis/v3f/SUMMARY.md`) and this note |
| everything but two prompt kinds, two answers, two weights | yes | this note |
| negative interactions (γ ≤ 0 here) | no | section 1, and at full size T30: 60 of 60 block-feature tasks with no fall at all (`analysis/v3r/SUMMARY.md`) |
| sharing (tabular) | no | Lean proof (`formal/`) and 2,268 runs |

## Relation to prior work

- **Schaul et al. (2019), ray interference.** A two-context bandit with a shared bias, where learning one context
  stalls the other and produces plateaus. Theirs: the mechanism and the plateaus. This note adds a correct minority
  answer pushed toward probability 0, the γ > α/β condition for permanence, and the tail separatrix.
- **Li, Pandey and Chen (2026).** The push-down of answers that are rarely rewarded, through shared parameters. This
  note's first step (section 2) is the K = 2 case of the odds law, whose decomposition follows theirs.
- **Barakat et al. (2026).** The kernel κ, prompt interference (Def. 3.1), and pass@k against pass@1 training. This
  note uses their kernel in section 1 and adds the tail exponent α = k that makes pass@k's threshold γ > k.
- **Huang et al. (2026).** Coverage collapse without parameter sharing, and reference resets with their
  representability assumption (Assumption 2.1). This note shows that in the example the exact reset step has no
  interference, and measures how many inner steps that needs.
- **Mei et al. (NeurIPS 2023).** Ordering-based feature conditions for softmax policy gradient with linear features in
  a single-prompt bandit. The condition of section 1 is of a different kind: across prompts, on inner products.
- **Soudry et al. (2018).** The implicit bias of gradient descent on separable data (margins growing like log t). The
  tail system of section 4 is the two-prompt version of that regime, with a second prompt whose own pull competes.

- **The instructor's submitted paper (unpublished, not seen by us).** He said on 2026-11-26 that it has the
  non-negative-interaction condition and a counterexample in which GRPO collapses with linearly separable features.
  This example may overlap with that counterexample; we cannot check, so the list below is what this note contains,
  not a claim of priority.

Ours: the two-kind example as a tabulated object, the three-line proof and the K = 2 converse of section 1, the tail
system with its threshold γ > α/β, the exact straight-line separatrix and the exponent −γ/α, the phase table, and the
per-lever tables with a constructed failure for each.

## What this does and does not show

- It is one stylised setting: two prompt kinds, K = 2 answers, a linear softmax policy from a uniform start. It shows
  exactly where the collapse comes from in that setting; it does not show that a larger model fails for the same
  reason, although the ladder above says nothing needed for the collapse was lost on the way down.
- The depth condition is numerical. Inside the tail it is a closed-form line, but where a fall from the uniform start
  lands relative to that line, and hence the phase boundary in (γ, m), is only computed, not derived.
- "Stays down" is measured at τ = 1e14. Five runs that stay down to 1e14 are predicted to recover later.
- The GRPO weight in the code has ε = 1e-5, which changes its tail (α = β = 1 instead of 1/2 for G = ∞) but not the
  threshold α/β = 1.
- The reset threshold in N follows 1/β_KL, but its constant is unexplained.
- Section 1 is a statement about the expected-update flow: it does not cover finite steps, sampled updates or Adam.
- Nothing here is registered.
