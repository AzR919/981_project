/-
Copyright (c) 2026 CMPT 981 project team. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: CMPT 981 project team
-/
import Mathlib

/-!
# Algebraic facts behind the first-step identity

Formal checks of three facts used in `analysis/theory/odds_law.md`.

1. Tabular monotonicity: for one prompt with a softmax policy, the exact group-centred update
   direction on
   the logits does not lower the correct answer's probability to first order.
2. Cross-class factor: the inner product of two prompts' softmax success directions.
3. First-step identity: the push ratio in block form, and its odds form.
-/

open Finset BigOperators

namespace RLVR

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Indicator of `j = c` as a real number. -/
def ind (c j : ι) : ℝ := if j = c then 1 else 0

/-- **Tabular monotonicity.** With probabilities `π`, correct answer `c` and `p = π c`, the
first-order change of `p` along the group-centred direction `g j = π j * (ind c j - p)` is
`p^2 (1 - p)^2 + p^2 ∑_{j ≠ c} π j ^ 2`, hence nonnegative. -/
theorem tabular_first_order (π : ι → ℝ) (c : ι) :
    (∑ j, (π c * (ind c j - π j)) * (π j * (ind c j - π c))) =
      π c ^ 2 * (1 - π c) ^ 2 + π c ^ 2 * ∑ j ∈ univ.erase c, π j ^ 2 := by
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ c)]
  have hc : ind c c = 1 := by simp [ind]
  have hj : ∀ j ∈ univ.erase c, ind c j = 0 := by
    intro j hj
    simp [ind, (Finset.mem_erase.mp hj).1]
  rw [hc, Finset.mul_sum]
  congr 1
  · ring
  · refine Finset.sum_congr rfl (fun j hjm => ?_)
    rw [hj j hjm]
    ring

theorem tabular_first_order_nonneg (π : ι → ℝ) (c : ι) :
    0 ≤ ∑ j, (π c * (ind c j - π j)) * (π j * (ind c j - π c)) := by
  rw [tabular_first_order]
  have : 0 ≤ ∑ j ∈ univ.erase c, π j ^ 2 := Finset.sum_nonneg (fun j _ => sq_nonneg (π j))
  positivity

/-- **Cross-class factor.** `(e_a - πx) · (e_b - πy) = ind a b - πy a - πx b + πx · πy`. -/
theorem cross_class_factor (πx πy : ι → ℝ) (a b : ι) :
    (∑ j, (ind a j - πx j) * (ind b j - πy j)) =
      ind a b - πy a - πx b + ∑ j, πx j * πy j := by
  have h1 : (∑ j, ind a j * ind b j) = ind a b := by
    simp only [ind, ite_mul, one_mul, zero_mul]
    simp [eq_comm]
  have h2 : (∑ j, ind a j * πy j) = πy a := by simp [ind, Finset.sum_ite_eq']
  have h3 : (∑ j, πx j * ind b j) = πx b := by simp [ind, Finset.sum_ite_eq']
  have expand : ∀ j, (ind a j - πx j) * (ind b j - πy j) =
      ind a j * ind b j - ind a j * πy j - πx j * ind b j + πx j * πy j := fun j => by ring
  simp only [expand, Finset.sum_add_distrib, Finset.sum_sub_distrib, h1, h2, h3]

/-- At a uniform start the cross-class factor is `ind a b - 1/K`. -/
theorem cross_class_factor_uniform (a b : ι) (hK : (Fintype.card ι : ℝ) ≠ 0) :
    (∑ j, (ind a j - 1 / (Fintype.card ι : ℝ)) * (ind b j - 1 / (Fintype.card ι : ℝ))) =
      ind a b - 1 / (Fintype.card ι : ℝ) := by
  rw [cross_class_factor (fun _ => 1 / (Fintype.card ι : ℝ))
    (fun _ => 1 / (Fintype.card ι : ℝ)) a b]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  field_simp
  ring

/-- **First-step identity, block form.** With `s > 0`, the push ratio
`R = -Δ_B(A) / Δ_B(B)` where `Δ_B(A) = -s S_BA / K` and
`Δ_B(B) = s((1 - 1/K) S_same - S_diff / K)`
equals `S_BA / ((K - 1) S_same - S_diff)`. -/
theorem push_ratio_block (s K sBA sSame sDiff : ℝ) (hs : s ≠ 0) (hK : K ≠ 0)
    (hD : (K - 1) * sSame - sDiff ≠ 0) :
    (s * sBA / K) / (s * ((1 - 1 / K) * sSame - sDiff / K)) = sBA / ((K - 1) * sSame - sDiff) := by
  have hD' : (1 - 1 / K) * sSame - sDiff / K = ((K - 1) * sSame - sDiff) / K := by field_simp
  rw [hD']
  field_simp

/-- **Odds form.** Writing the block sums through the kernel means (`n_A = m n`,
`n_B = (1 - m) n`, equal minority classes), `R = c m / (1 - m)` with
`c = (K - 1) k_BA / ((K - 1) k_same - (K - 2) k_diff)`. -/
theorem push_ratio_odds (K n m kBA kSame kDiff : ℝ) (hK1 : K - 1 ≠ 0) (hn : n ≠ 0)
    (hm : m ≠ 0) (hm1 : 1 - m ≠ 0) (hc : (K - 1) * kSame - (K - 2) * kDiff ≠ 0) :
    let nA := m * n
    let nB := (1 - m) * n
    let sBA := kBA * nB * nA
    let sSame := kSame * nB ^ 2 / (K - 1)
    let sDiff := kDiff * nB ^ 2 * (K - 2) / (K - 1)
    sBA / ((K - 1) * sSame - sDiff) =
      ((K - 1) * kBA / ((K - 1) * kSame - (K - 2) * kDiff)) * (m / (1 - m)) := by
  intro nA nB sBA sSame sDiff
  have hnB : nB ≠ 0 := mul_ne_zero hm1 hn
  have hden : (K - 1) * sSame - sDiff = nB ^ 2 * ((K - 1) * kSame - (K - 2) * kDiff) / (K - 1) := by
    simp only [sSame, sDiff]
    field_simp
  rw [hden]
  simp only [sBA, nA, nB]
  field_simp

end RLVR
