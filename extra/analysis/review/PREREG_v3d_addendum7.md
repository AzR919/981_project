# PREREG_v3d, addendum 7 (2026-11-24): permanent or plateau? (T25, exact long-time flow)

Committed before the T25 runs. Code: `toy-v3` `3bf676f` (`extra/experiments/exact_flow.py`). Exploratory, with
a fixed prediction rule.

## Why

"Is the collapse permanent or a long plateau?" is still open: within 10× the training horizon
(T12) no run at ρ = 0.9 recovered. The linear policy's exact gradient flow can now be integrated to
flow times of 1e14 in seconds per task (checked against discrete SGD to within 0.015 decades).

## What has been seen

All earlier results, including T24 (the per-prompt late model failed within flow time 1e4–1e5) and
the exact solution of the scalar late model. **One look:** while checking the integrator, one task
(ρ = 0.5, m = 0.9, task 0) was integrated to 1e10: its minority was absorbed at 1e6 (mean 2e-7) and
47% of its prompts had recovered by 1e10. No other task has been integrated.

## Runs

Linear policy, exact expected GRPO update (G = 8), uniform start, K = 3,
ρ ∈ {0.3, 0.5, 0.7, 0.9, 0.95}, m ∈ {0.7, 0.9}, task seeds 0–9 (100 tasks), flow time 1e-1 to 1e14
(80 log-spaced points). Per prompt: log-success at every point.

## Prediction rule (fixed now)

- τ0: the first stored time at which the majority's class mean is at least 0.99. L_y(τ0) = −log p_y(τ0).
- γ_y = û·φ_y, with û the majority's ℓ2 max-margin direction (unit margin on its closest prompts);
  c_y = (W_G(0)/n) Σ_{y' minority} k(y, y')·s(y, y'), s = 2 for classmates (including y) and 1 for the
  other minority class; W_G(0) = 2.6456 (GRPO, G = 8).
- Exact solution of dL/dτ = γ/τ − c e^{−L} from (τ0, L_y(τ0)): with A = τ0^{1−γ} + (1−γ) e^{L_y(τ0)} τ0^{−γ}/c_y,
  the predicted recovery time is τ\*_y = A^{1/(1−γ)} when A > 0 (for γ < 1, A > 0 always); for γ > 1 and
  A ≤ 0 the prompt is predicted never to recover. (γ = 1: τ\* = τ0 exp(e^{L} / (c τ0)).)
- Observed recovery time: the first stored time at which p_y ≥ 0.5.

## Predictions

1. **Permanence:** among minority prompts predicted never to recover, at most 5% have recovered by
   1e14.
2. **Recovery:** among prompts predicted to recover before 1e12, at least 80% have recovered by 1e14.
3. **Timing:** among prompts that recover, Spearman(predicted log τ\*, observed log recovery time) is
   at least 0.7, and the median |log10(predicted/observed)| is at most 1.
4. **Run level:** in runs whose minority mean stays below 1e-3 for at least two decades after τ0, the
   late exponent of the minority's class mean (−Δ log p̄ / Δ log τ over the two decades before its
   recovery, or the last two decades) is within 0.1 of γ_min in at least 80% of runs.
5. **Reported, not predicted:** how many tasks end with their minority recovered by 1e14, by ρ and m.

## Reading, fixed in advance

- **1–3 hold:** the late phase has a quantitative theory at the per-prompt level once the horizon is
  long enough: the T24 failure was a horizon effect, and "permanent vs plateau" is decided by γ_y and
  the depth reached.
- **4 holds, 1–3 fail:** only the run-level rate is understood; the outcome needs the collective
  (classmate) dynamics.
- **5 settles the wording** for the report: "permanent" only where tasks stay absorbed to 1e14;
  otherwise "a plateau of length …".

## Deviations

Listed in the T25 write-up.
