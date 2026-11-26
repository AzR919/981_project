# formal

Lean 4 proofs (with Mathlib) of the algebraic facts behind the first-step identity in
`extra/analysis/theory/odds_law.md`. See `Formal/RLVR.lean`:

- `tabular_first_order`, `tabular_first_order_nonneg`: with a softmax policy on one prompt, the
  exact group-centred update direction does not lower the correct answer's probability to first
  order (why a tabular policy never absorbs a prompt).
- `cross_class_factor`, `cross_class_factor_uniform`: the inner product of two prompts' softmax
  success directions, and its value `ind a b - 1/K` at a uniform start.
- `push_ratio_block`, `push_ratio_odds`: the push ratio in block form and its odds form
  `c · m / (1 - m)`.

The asymptotic (late-phase) results are checked symbolically in `extra/tests/test_theory_symbolic.py`,
not here.

Build: install `elan`, then `lake exe cache get && lake build` in this folder (Mathlib's prebuilt
cache is about 8 GB and lives in `.lake/`, which git ignores).
