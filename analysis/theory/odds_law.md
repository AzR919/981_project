# The odds law: how hard the majority pushes the minority at the start of training

**Status:** derived on 2026-11-23 and checked numerically against 1,080 simulator runs
(`odds_law_check.md`). The algebra has not yet been checked by hand by a second person.

## Setting

- n prompts x with features φ(x). K answer classes. Prompt x has correct class c(x).
- Class A is the majority (class 0), with share m = n_A/n. The other K − 1 classes ("B") have equal
  sizes n_B/(K − 1), where n_B = n − n_A.
- Policy: π(·|x) = softmax(W φ(x)), with W = 0 at the start, so π = 1/K for every prompt and class.
  For the MLP, φ is the last hidden layer plus a constant 1: the output layer starts at zero, so at
  the start only the output layer has a non-zero gradient.
- Success of prompt x: p_x = π(c(x)|x). Kernel: k(x, y) = φ(x)·φ(y).
- Update: one step of size η on the exact expected GRPO update, θ ← θ + η·(1/n) Σ_x w ∇p_x. At the
  uniform start every prompt has the same p, so the weight w = W_G(1/K) is a common factor.

## Derivation

1. For the softmax, ∇_W p_x = p_x (e_{c(x)} − π(·|x)) φ(x)ᵀ. At the start this is
   (1/K)(e_{c(x)} − 1/K) φ(x)ᵀ, where 1 is the all-ones vector.
2. So ⟨∇p_y, ∇p_x⟩ = (1/K²) (e_{c(y)} − 1/K)·(e_{c(x)} − 1/K) k(y, x)
   = (1/K²) (δ_{c(y)c(x)} − 1/K) k(y, x).
   Two prompts of the same class help each other with weight 1 − 1/K. Two prompts of different
   classes hurt each other with weight 1/K.
3. To first order in η, a step along a set S of prompts changes p_y by
   η (w/n) Σ_{x∈S} ⟨∇p_y, ∇p_x⟩.
4. Average over the minority prompts y ∈ B, and write
   - S_BA = Σ_{y∈B} Σ_{x∈A} k(y, x),
   - S_same = Σ_{y∈B} Σ_{x∈B, c(x)=c(y)} k(y, x) (self-pairs included),
   - S_diff = Σ_{y∈B} Σ_{x∈B, c(x)≠c(y)} k(y, x).

   Then, with s = η w / (n n_B K²):
   - majority-only step: Δ_B(A) = −s · S_BA / K;
   - minority-only step: Δ_B(B) = s · [(1 − 1/K) S_same − S_diff / K].

## Result

**Push ratio.** R = −Δ_B(A)/Δ_B(B) = S_BA / [(K − 1) S_same − S_diff].

**Odds form.** With the kernel means k_BA = S_BA/(n_B n_A), k_same = S_same·(K − 1)/n_B², and
k_diff = S_diff·(K − 1)/(n_B²(K − 2)):

  R = c · m/(1 − m), with c = (K − 1) k_BA / [(K − 1) k_same − (K − 2) k_diff].

This is an identity, not an approximation. The majority's push relative to the minority's own gain
is the odds of the majority share, times a constant c set by the feature geometry.

**Threshold.** The mixed step lowers the minority exactly when R > 1, that is when
m > m\* = 1/(1 + c).

- If features carry no class information (k_BA = k_same = k_diff), then c = K − 1 and m\* = 1/K.
  This is the leading-order threshold that follows from Li, Pandey and Chen's decomposition.
- If same-class prompts are more similar than cross-class prompts (k_same > k_BA), then c < K − 1
  and m\* > 1/K. The threshold moves up by an amount fixed by the kernel means.
- For K = 2 there is no k_diff term and c = k_BA/k_same.

c depends weakly on m, because moving prompts between classes changes the kernel means. The
threshold is the solution of m = 1/(1 + c(m)).

## Check against the simulator

`experiments/check_odds_law.py` computes the prediction from each task's features and compares it
with the step-0 changes the simulator measured in the v3e runs (K ∈ {2, 3, 5}, linear and MLP,
ρ ∈ {0.5, 0.9}, nine values of m, ten tasks). Nothing is fitted.

- **R:** median relative error 0.14%, 95th percentile 1.0%, largest 1.7%, over 1,080 runs. The
  residual is the second-order term in η (the simulator takes a finite step of η = 0.01).
- **m\*:** predicted and measured agree within 0.0002 in all 12 cells.
- **c at m = 1/K** (medians): 0.88–1.00 for K = 2, 1.65–1.97 for K = 3, 2.33–3.81 for K = 5. The
  value K − 1 is approached as feature correlation rises, which is why the measured threshold sits
  closest to 1/K at ρ = 0.9.

## What this does and does not show

- It explains two things measured earlier and reported as empirical: the slope of log R on logit m
  is 1 (v3e, 12 of 12 cells), and the threshold sits just above 1/K, further above at lower feature
  correlation.
- It is a statement about the first step from a uniform start, for a softmax output layer on fixed
  features. It is exact for the linear policy and for an MLP whose output layer starts at zero.
- It says nothing about the outcome of training. Persistent absorption starts at a clearly larger
  share than m\* (`analysis/v3e/SUMMARY.md`). A formula for that second threshold is open.
- Away from the uniform start the weights W_G(p_x) differ across prompts and the class factor
  (δ − 1/K) becomes (e_c − π_y)·(e_{c′} − π_x); the same steps give a weighted version of the
  result, not derived here.

## Relation to prior work

- Schaul et al. (2019) derive interference through a shared bias in a two-context bandit.
- Li, Pandey and Chen (2026) state the decomposition behind step 2 for wrong strings that a
  verifier accepts, and its leading-order sign.
- New here: the closed form for a *correct* minority class with the kernel correction, the odds law
  as an identity, and a parameter-free check against exact simulation.
