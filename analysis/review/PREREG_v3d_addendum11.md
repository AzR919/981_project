# PREREG_v3d, addendum 11 (2026-11-26): a stylized ladder, a one-knob prevention table, pass@k (T30–T33)

Committed before batch `v3r` launches (code: `toy-v3`, commit named in the launch record). Prompted by the
instructor's feedback of 2026-11-26: one simple setting with one cause; the "non-negative interactions" feature
condition; KL to the latest policy ("reference resets", Huang et al., arXiv 2611.14467); pass@k training measured on
pass@1 (Barakat et al., arXiv 2602.21189). T30 has one confirmatory prediction (its P1 is a theorem); everything else
is exploratory unless marked.

**Seen before this was written:** the summaries of v3f, v3g, v3m and v3o (final values, matched-point tables), and
two unsaved numerical checks of a two-prompt example (direct logistic model). **Not seen:** any block-feature run,
any reset run in the standard toy, any pass@k exact flow, any pass@k trajectory, any switch run.

**Not registered:** the two-prompt example (`experiments/minimal_example.py`). It is a derivation with a numerical
check, because its numbers were seen first. It is reported in `analysis/theory/minimal_example.md`.

Task seeds 30–49 (never run) unless stated. "Absorbed" is per minority prompt, persistent over the last 20% of
training, at 1e-3 unless stated. Class 0 is the majority.

## T30: non-negative interactions at full size (block features)

**Claim under test.** With d(x, y) = φ(x, y*) − φ(x, y): if every ⟨d(x, y), d(x′, y′)⟩ ≥ 0, then for a softmax-linear
policy κ(x, x′) = ⟨∇p_x, ∇p_x′⟩ ≥ 0 at every policy, so under any update Σ W(p_x′)∇p_x′ with W ≥ 0 no prompt's
success decreases along the gradient flow. The condition is the instructor's; the kernel is Barakat et al.'s.

**Task.** `features="block"`: K = 3, 512 prompts, `block_dim` = 5. A prompt of class c has |g| (g standard normal)
in block c, zeros in the other blocks, and one last coordinate equal to `shared_scale` = s for every prompt. At s = 0
the condition holds exactly; at s > 0 every cross-class pair violates it (inner product s²). The share of a prompt's
squared norm that is shared is about s²/(5 + s²).

**Grid.** Linear policy, GRPO G = 8. s ∈ {0, 0.5, 1, 2, 4, 8}; m ∈ {1/3, 0.7, 0.9}; 20 tasks.
- Exact flow (`exact_flow.py`) to flow time 1e8.
- Discrete runs, exact (`expected_full`) and sampled (B = 32): SGD, η = η̃/(5 + s²) with η̃ ∈ {0.5, 1.5} (the same
  step relative to the feature norm at every s), 20,000 steps.
- Exploratory: MLP (width 64), s ∈ {0, 8}, m = 0.9, exact and sampled, η̃ = 0.5. The theorem does not cover it.

**Predictions (linear).**
1. **(Confirmatory; a theorem.)** At s = 0, in the exact flow, no prompt's success falls more than 1e-6 below its
   running maximum, in 60 of 60 tasks. A violation means the proof or the code is wrong.
2. At s = 0, discrete exact runs: no minority prompt is absorbed, and every class mean ends above 0.99, in ≥ 95% of
   runs.
3. Dose–response at m = 0.9 (exact, η̃ = 1.5): the absorbed share of minority prompts is 0 at s = 0 in 20 of 20
   tasks; at s = 8 it is above the s = 0 value in ≥ 18 of 20 tasks, with a median ≥ 0.5. Reported: the whole curve,
   the smallest s with a median ≥ 0.05, and R(s) and γ_min(s) per task.
4. At m = 1/3 the median absorbed share is ≤ 0.05 at every s.
5. Sampled GRPO (secondary): median absorbed share ≤ 0.02 at s = 0 and ≥ 0.5 at s = 8, at m = 0.9.
6. Reported: the first-step sign against R > 1 (an identity).

## T31: prevention with one knob changed (the standard cell)

**Cell.** Linear, ρ = 0.9, m = 0.9, K = 3, 512 prompts, uniform start, exact update, SGD η = 0.1, 100,000 steps
(flow time 1e4), class means logged every 100 steps, per-prompt success at steps 2,000 and 100,000. Every arm differs
from plain GRPO (G = 8) in exactly the settings named.

| Arm | Settings |
|---|---|
| plain | GRPO; and the mean rule as a second baseline |
| fixed reference, reverse KL(π‖π_start) | β ∈ {0.03, 0.1, 0.3, 1, 3} |
| fixed reference, forward KL(π_start‖π) | β ∈ {0.03, 0.1, 0.3, 1} |
| resets, reverse KL(π‖π_latest) (Huang et al.'s direction) | β ∈ {0.3, 1, 3} × `reset_every` ∈ {1, 10, 100, 1000, 5000} |
| resets, forward KL | β ∈ {0.1, 0.3, 1} × `reset_every` ∈ {10, 100, 1000, 5000} |
| pass@k (unbiased form, `rloo_k`) | k ∈ {2, 4, 8} |

Secondary: the reverse-reset arms at β = 1 under the mean rule (Huang et al.'s setting). Robustness (exploratory):
MLP with the exact update, and linear with sampled updates (B = 32), for plain, reverse fixed β ∈ {0.3, 1}, forward
fixed β = 0.1, reverse resets β = 1 with `reset_every` ∈ {10, 100, 1000}, and pass@4.

**Yardstick, the same for every arm,** at step 2,000 and at the end: minority class mean (also its lowest logged
value); majority class mean and the first logged step at which it reaches 0.99; mean pass@1; pass@4 and pass@16;
median of 1/p over minority prompts; absorbed share at 1e-3 and 1e-2.
- **Protected and learned** (per task): final minority mean ≥ 0.9, final majority mean ≥ 0.9, and no minority prompt
  absorbed at 1e-2.

**Predictions.**
1. (Identity.) `reset_every` = 1 reproduces plain GRPO: class means differ by ≤ 1e-6 at every logged step.
2. Plain GRPO: the minority mean ends below 1e-3 in ≥ 19 of 20 tasks.
3. Fixed reverse KL: no β is protected and learned in ≥ 16 of 20 tasks (small β collapses, large β caps the majority).
4. Fixed forward KL: no minority prompt absorbed at 1e-3 at β ≥ 0.03 in ≥ 18 of 20 tasks (as A2), but protected and
   learned in ≤ 4 of 20 tasks at every β.
5. **Resets, reverse KL: the open question, two-sided.** Let c(β, N) be the number of tasks protected and learned.
   - "Protects": some cell has c ≥ 16. "Fails": every cell has c ≤ 4. Otherwise "partial".
   - No theory covers this cell: the reward is not linear in the features (Huang et al.'s Assumption 2.1 fails).
   - My expectation, for the record: "protects", with moderate confidence, at β ≥ 1 and `reset_every` ≥ 1000; cells
     with `reset_every` ≤ 10 collapse like plain GRPO.
   - Because the best cell is chosen after seeing 15 cells, it is rerun on seeds 50–69. The claim "resets protect at
     (β, N)" is made only if c ≥ 16 there too.
6. At β = 1 (reverse resets) the final minority mean at `reset_every` = 1000 is above that at 10 in ≥ 16 of 20 tasks.
7. pass@k, k ∈ {2, 4}: the final minority mean is ≥ 0.5 in ≥ 18 of 20 tasks; for every k the majority reaches 0.99
   later than under plain GRPO in 20 of 20 tasks.
8. Reported: forward resets, the mean-rule resets, the robustness rows.

## T32: pass@k over long times (exact flow)

The 100 tasks of T25 (linear, K = 3, ρ ∈ {0.3, 0.5, 0.7, 0.9, 0.95}, m ∈ {0.7, 0.9}, task seeds 0–9), whose GRPO flows
are saved (54 of 100 recover; γ_min ranges up to 1.29). The same flow with unbiased pass@k weights, k ∈ {2, 4}, to
flow time 1e14, T25's solver settings; a task that LSODA cannot finish in 30 minutes is rerun with Radau, then BDF,
and counts as not recovered if all three fail.

**Derivation.** The majority's push fades like e^(−k·margin) under pass@k, k times faster in the exponent than the
minority's pull. The scalar late model's threshold moves from γ ≥ 1 to γ ≥ k. Every task has γ_min < 2.

1. Tasks recovered at 1e14 (minority class mean > 0.5): ≥ 85 of 100 under pass@2 and ≥ 95 of 100 under pass@4.
2. Among the 30 tasks with γ_min ≥ 1 (3 recover under GRPO): ≥ 24 recover under pass@2.
3. The lowest minority class mean is higher than under GRPO in ≥ 95 of 100 tasks, for each k.
4. Among tasks that recover under both pass@2 and GRPO, pass@2 recovers earlier in ≥ 90%.
5. Reported: the 2 × 2 table (γ_min < k against recovered) for k = 1, 2, 4; the majority class mean at flow times
   1e2, 1e3, 1e4 (the cost).

## T33: pass@k training measured on pass@1

**(a) Existing sampled runs** (T16 in `v3f`, 480 runs; A3 in `v3o`, 360 runs; uniform start; class means every 100
steps). Only final values and the matched-point table have been read.
1. From a uniform start pass@k slows the majority and does not reverse it: the majority class mean never falls more
   than 0.05 below its running maximum, in ≥ 90% of pass@k runs (k = 2, 4, 8, both forms). The same in the exact
   pass@k arms of T31 with a bound of 0.02, in ≥ 18 of 20 tasks per k.
2. Early cost: at the first logged step where plain GRPO's majority mean is ≥ 0.99 (same task, G and η), mean pass@1
   under pass@k is below GRPO's in ≥ 90% of pairs, for k ∈ {4, 8}. Reported for k = 2.
3. Reported: the first step from which mean pass@1 under pass@k stays above GRPO's, per k, G and η.

**(b) Switching to pass@k after a collapse** (new; the T31 cell; plain GRPO up to step 100 or 1,000, then the
unbiased pass@k rule, k ∈ {2, 4, 8}; `switch_at`). Barakat et al. predict that pass@k ascent can lower pass@1 when
hard prompts interfere negatively with easy ones.
4. Under k = 8 (switch at step 100) the majority class mean falls at least 0.05 below its value at the switch, at
   some later logged step, in ≥ 16 of 20 tasks.
5. The median largest fall of the majority is ordered k = 2 < k = 4 < k = 8.
6. Under k = 8, mean pass@1 falls below its value at the switch while mean pass@8 is above its value at the switch,
   at some logged step, in ≥ 16 of 20 tasks.
7. Rescue: under k = 4 (switch at step 100) the minority class mean is back above 0.5 by the end in ≥ 16 of 20
   tasks. Reported for the other k and for the switch at step 1,000.

Under the exact update with a small step, the sign of the change in mean pass@1 is the sign of Barakat et al.'s
inner product ⟨∇J_k, ∇J_1⟩ to first order, so no separate gradient measurement is made.

## Deviations

Listed in `analysis/v3r/SUMMARY.md`.

## Amendment 1 (2026-11-26, before any `v3r` run or flow)

Written after the simulator options were committed (`toy-v3` `8e6b0aa`) and the two-prompt example was tabulated
from that code (not a registered test), and after the harm analysis of the final snapshots of A1–A3
(`analysis/v3o/HARM.md`; no trajectories were read). Nothing from T30–T33 has been run.

1. **Flows use `stable_tail=True`.** `exact_flow.flow` then computes 1 − π_c as the sum of the other answers'
   probabilities. It is the same right-hand side, evaluated without cancellation; without it LSODA stalls once some
   prompt has 1 − p below about 1e-13. For T32 the GRPO baseline is recomputed with the same settings on the 100
   tasks and compared with the saved T25 flows before any pass@k flow is read; the differences are reported, and the
   paired comparisons of T32 use the recomputed baseline.
2. **Normalisation of the unbiased pass@k rule.** `rloo_k`'s expected weight is (1 − p)^(k−1), which is 1/k of the
   pass@k gradient's weight. T32's thresholds and times are in the code's flow time under this normalisation, as
   registered. At the uniform start the weight is 0.667, 0.296, 0.059 for k = 2, 4, 8, against GRPO's 1.929 (G = 8),
   so part of pass@k's slowness in steps is this constant. T31 therefore gains three reported arms with a matched
   initial step: `rloo_k`, k ∈ {2, 4, 8}, at η = 0.1 × 1.929/weight = 0.289, 0.651, 3.30. Prediction 7 stays as
   registered, for the unscaled arms.
3. **Correction to T32's derivation.** In the two-prompt example, γ > k is necessary for the collapsed state to be
   stable under pass@k, and it is not sufficient for a collapse from the uniform start: pass@4 never collapsed at
   m = 0.9 for γ up to 8, because its majority weight (1 − p)³ switches the push off before the minority is deep.
   The paragraph should read "the collapsed state is stable only when γ > k". No registered prediction changes:
   every T32 task has γ_min < 2.
4. T33(b) runs also store every prompt's success at the switch step.
