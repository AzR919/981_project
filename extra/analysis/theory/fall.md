# The fall: a reduced class-level ODE for phase 1 (working note, 2026-11-24)

**Status:** derivation and model. Predictions fixed before comparison are in section 5; the comparison is in
`fall_results.md`. Code: `extra/experiments/fall_ode.py` (model), `extra/experiments/check_fall.py` (comparison),
`extra/tests/test_fall_ode.py`.

## 1. Why phase 1

The first-step identity (`odds_law.md`) says whether the minority starts to fall. The late phase
(`late_time.md`, v3l) sets the run's sinking rate and preserves the order of prompts, so which prompts end up
absorbed is decided while the majority is being learned. This note writes down the smallest model of that
phase: the trajectory of the class success levels from the start until the majority is learned.

## 2. From the exact flow to a flow on class means

Linear policy, logits z_y = W φ_y, one answer token, K answers, U ≤ K of them with prompts (the other
J = K − U are "junk", never correct). Under the exact expected update with plain SGD, in flow time τ = η t
(`extra/experiments/exact_flow.py`):

  dW/dτ = (1/n) Σ_x w(p_x) (e_{c(x)} − π_x) φ_xᵀ,  w(p) = W_G(p) · p,

so every logit vector moves as

  dz_y/dτ = (1/n) Σ_x w(p_x) (e_{c(x)} − π_x) k(x, y),  k(x, y) = φ_x · φ_y.

**Closure (class symmetry).** Inside each class a, replace the per-prompt factor w(p_x)(e_a − π_x) by its value
at the class-mean logit z̄_a = W φ̄_a, where φ̄_a is the class-mean feature. Then, with f_a = n_a/n,

  dz̄_b/dτ = Σ_a f_a k_ab w(p_a) (e_a − π_a),  k_ab = φ̄_a · φ̄_b.

k_ab is exactly the mean of k(x, y) over the block x ∈ a, y ∈ b (self-pairs included): the same block means
as the odds law. Equivalently W = W0 + Σ_a C_a φ̄_aᵀ with

  dC_a/dτ = f_a w(p_a) (e_a − π_a),  π_a = softmax(W0 φ̄_a + Σ_b k_ab C_b),

which is the exact flow on U weighted points (the "class model", U × K numbers). It is exact when the
features inside each class coincide (checked to solver accuracy against `exact_flow.flow` in the tests).
What it drops is the spread inside a class (the Jensen terms of w and π across prompts).

**Test particles.** In the class model's field a prompt y has logits W0 φ_y + Σ_a (φ̄_a · φ_y) C_a(τ). This
gives a per-prompt prediction at no extra cost, without feeding the prompt's own spread back into the field.

## 3. The class-symmetric form

Class A (index 0) has share m; the M = U − 1 minority classes are pooled with share (1 − m)/M each, and the
blocks are pooled into k_AA, k_AB (mean over minority classes), k_same (minority diagonal) and k_diff
(minority off-diagonal). By symmetry the state is five log-odds:

- A prompts: u_A = z_A(A) − z_A(b) (against each minority answer), v_A = z_A(A) − z_A(junk);
- B prompts: L_B = z_B(A) − z_B(own), r_B = z_B(other minority) − z_B(own), j_B = z_B(junk) − z_B(own).

Write p_A, s_A, t_A for an A prompt's mass on A's answer, on one minority answer and on one junk answer, and
p_B, q_B, o_B, ι_B for a B prompt's mass on its own answer, on A's answer, on one other minority answer and on one
junk answer (so 1 − p_A = M s_A + J t_A and 1 − p_B = q_B + (M − 1) o_B + J ι_B). With g_A = m w(p_A) and
g_B = ((1 − m)/M) w(p_B):

  du_A/dτ = g_A k_AA (1 − p_A + s_A) − g_B k_AB [M q_B + (1 − p_B) − (M − 1) o_B]
  dv_A/dτ = g_A k_AA (1 − p_A + t_A) − g_B k_AB M (q_B − ι_B)
  dL_B/dτ = g_A k_AB (1 − p_A + s_A) − g_B k_same (1 − p_B + q_B) − g_B k_diff (M − 1)(q_B − o_B)
  dr_B/dτ = − g_B (k_same − k_diff)(1 − p_B + o_B)
  dj_B/dτ = g_A k_AB (s_A − t_A) − g_B k_same (1 − p_B + ι_B) + g_B k_diff (M − 1)(o_B − ι_B)

The start is the pooled class-mean logits of W0 (zero for the uniform start; the warm-started W0 for v3k).

## 4. What the equations say

1. **Transfer identity.** The majority's own term moves the minority's log-odds L_B and its own log-odds u_A
   in the fixed ratio κ = k_AB/k_AA:

     dL_B/dτ |_A = κ · du_A/dτ |_A.

   The majority transmits to the minority κ times the log-odds it learns for itself. The fall is this push
   minus the minority's own restoring term.
2. **The restoring term is self-limiting.** It is −g_B [k_same (1 − p_B + q_B) + …] with g_B ∝ W_G(p_B) p_B.
   For GRPO W_G(0+) is finite (a lone correct sample has a finite advantage), so the restoring force is
   proportional to p_B once p_B is small: the deeper the minority, the less it resists.
3. **First step.** At the uniform start d log p_B/dτ = −(dL_B + (M − 1) dr_B)/K, which is negative exactly
   when c m/(1 − m) > 1 with c = M k_AB / (M k_same − (M − 1) k_diff): the odds law (test in
   `extra/tests/test_fall_ode.py`).
4. **Late time.** Once p_A → 1, (1 − p_A) ≈ M e^{−u_A} and du_A/dτ ≈ m k_AA W_G(1) e^{−u_A}(M + 1), so
   u_A ≈ log τ + const and dL_B/dτ ≈ κ/τ − c_B e^{−L_B}: the scalar late model of `late_time.md` with
   γ = κ. For class means the majority's max-margin direction is φ̄_A, so κ is the class-level version of
   γ_min. With y = e^{L_B} the late equation is linear, y(τ) = τ^κ [y_1 τ_1^{−κ} − c_B (τ^{1−κ} − τ_1^{1−κ})/(1 − κ)]:
   for κ > 1 the minority stays down if the fall reached deep enough when the late phase began, and comes back
   otherwise; for κ < 1 it eventually comes back. The finite horizon decides what the runs see.
5. **Junk answers.** At a uniform start s_A = t_A, so the majority's push does not separate a minority prompt's
   own answer from junk; the minority's own term drains junk into its own answer (j_B falls), and the
   majority's term drains it into A's answer (L_B rises). The two classes start to compete in earnest once
   the junk mass is gone, which is the delay seen in v3j.

## 5. Predictions fixed before comparison

Written and committed before any model output was compared with a measured value. Caveat: the batch summaries
(v3e, v3g, v3j, v3k, v3l) were read before the model was written, so their headline tables were known. The
model has no fitted parameter: everything comes from each run's own features, shares, group rule, start and
horizon T = η · steps.

**Models.** Primary: the class-symmetric model (`sym`). Secondary: the class model (`cls`, unpooled blocks)
and the test particles (`tp`).

**Measured quantities (per run).**
- Depth D: the median over minority prompts of log10 of each prompt's lowest success over the run (`min_p`,
  observed every `track_every` steps).
- Absorbed: the share of minority prompts with p_init ≥ 1e-2 whose success stays below 1e-3 over the last
  20% of steps (`tail_max`), as in the batch analyses; the run counts as absorbed when that share is ≥ 0.5.

**Predicted quantities.** `sym`/`cls`: log10 of the lowest minority success on [0, T]; absorbed when the
minority success stays below 1e-3 over [0.8 T, T]. `tp`: the median over minority prompts of the predicted
per-prompt minimum; absorbed when ≥ 50% of the minority prompts are predicted to stay below 1e-3 over
[0.8 T, T].

**Scope.** Linear policy, exact update: v3e (T15), v3g T19, v3j (T22), v3k (T23), v3l (T24). Linear runs with
the sampled update are reported separately (the model has no sampling noise). MLP runs are an out-of-scope check
(the model uses the MLP's last hidden layer at the start, held fixed); no prediction is registered for them.

**Predictions (primary model, per batch, linear exact runs):**

| # | Statement | Pass if |
|---|---|---|
| P1 | The predicted depth ranks the measured depth across the batch's runs | Spearman ≥ 0.8 in each of the 5 batches |
| P2 | The predicted depth is right in level | median of (predicted − measured) within ±0.5 decades in each batch |
| P3 | It ranks runs within a cell (same configuration, different tasks) | median within-cell Spearman ≥ 0.5 (cells with ≥ 5 runs and some spread), pooled over batches, reported per batch |
| P4 | It predicts which runs end absorbed | accuracy ≥ 0.85 in each batch |
| P5 | It predicts v3e's absorption threshold in m (smallest grid m where more than half of the 10 tasks are absorbed; "none" counts as a value) | equal, or one grid step off, in ≥ 5 of the 6 linear cells |
| P6 | Per prompt: test particles order the minority prompts by depth (runs whose median minority prompt falls below 1e-2) | median within-run Spearman ≥ 0.6 |

**Expected failure modes, stated in advance.** (a) The closure drops the spread inside a class; the
restoring force of the real minority is a mean of w(p_y) over prompts and is dominated by its least-fallen
prompts, so the model may predict falls that are too deep and absorption that is too frequent near the
threshold. (b) At ρ = 0.5 the class means are less representative (larger spread), so the error should be
larger there than at ρ = 0.9. (c) Horizon-limited runs (v3e at τ = 100) are mostly phase 1, which is where the
model should do best.
