# PREREG_v3d, addendum 4 (2026-11-23): where the wrong mass sits (T22, junk answers)

Committed before batch `v3j` launches. Code: `toy-v3`, "Toy v3j" commit. Exploratory.

## Why

The adversarial review of 2026-11-23 (finding 5) pointed out that every toy run and every run in
the transformer environment starts with a minority prompt giving the *majority's* answer with
probability 1/K, while the base LLM in C2 gives the other procedure's answer on only 0.6% and 5.4%
of samples. The summaries read the LLM result as a boundary set by shared skills. The start is an
untested alternative.

The first-order algebra (`analysis/theory/odds_law.md`, step 2) makes this precise. For a majority
prompt x and a minority prompt y, the factor in ⟨∇p_y, ∇p_x⟩ is π_x·π_y − π_x(b) − π_y(a): the
overlap of the two response distributions, minus the mass each puts on the other's answer. Mass on
answers that belong to no class ("junk") adds to the overlap and to nothing else. With two answer
classes and a uniform start over K tokens, the factor is −1/K across classes and 1 − 1/K within a
class, so

  R = S_BA / ((K − 1)·S_same) = κ/(K − 1) × m/(1 − m), κ = k_BA/k_same,

and the first-step threshold is m\* = (K − 1)/(K − 1 + κ): 1/2 with no junk, and close to
(K − 1)/K when κ ≈ 1.

## What has been seen

All earlier toy results; T20, T21; C2 and C10; the review. Unit tests of the new option, including
one that checks R against the kernel formula for K = 2 and K = 6 on one task. No T22 run.

## Runs

Two answer classes with prompts (class 0 has share m) and K − 2 junk tokens: K ∈ {2, 3, 6, 20}.
m ∈ {0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98}. Linear and MLP; ρ = 0.9; uniform start; SGD,
η = 0.03; 10,000 steps; exact expected update and sampled updates (G = 8, B = 32); 10 task seeds.
1,280 runs.

Measures as before: the exact one-step changes of the minority's mean success at step 0 (exact
arm); m\*, where their median crosses zero; the share of minority prompts persistently absorbed
(τ = 1e-3, and τ = 1e-2 as the sensitivity threshold); final class means.

## Predictions (exact arm; a cell is one architecture × K)

1. **The threshold moves as the formula says.** In every cell, the measured m\* is within 0.02 of
   the value computed from the task's kernel sums, with nothing fitted. If the median does not cross
   zero on the grid, the cell is reported as "above 0.98" or "below 0.5" and compared on that basis.
2. **The push shrinks as 1/(K − 1).** The ratio of the median R at m = 0.9 for K = 20 to that for
   K = 2 is between 0.03 and 0.08 (1/19 = 0.053), for both architectures.
3. **Absorption follows.** With K = 2 the minority is absorbed (mean share ≥ 0.5) at every
   m ≥ 0.8. With K = 20 no minority prompt is absorbed at any m ≤ 0.9, in ≥ 9 of 10 tasks.
4. **Absorption only at or above m\*,** in every cell.
5. **Reported, not predicted:** the sampled arm (with K = 20 most groups are all wrong at the
   start); whether the majority itself learns within the horizon at K = 20; the τ = 1e-2 counts.

## Reading

- **1–4 hold:** whether a majority suppresses a correct minority depends on where a prompt's wrong
  mass sits. On the other class's answer, the classes compete. On answers belonging to no class,
  the push weakens in proportion and the threshold moves toward 1. This gives the LLM result a
  reading that does not rest on "shared skills": an LLM's wrong answers are spread over a large
  space and are rarely the other procedure's answer.
- **This stays a toy statement about the first step and about uniform starts.** It does not show
  that the LLM would collapse if its wrong mass sat on the other procedure's answer; that needs an
  LLM pair built to do so (review finding 5, the one-token-cue design).
- **3 fails with 1–2 holding:** the first step is as predicted but absorption has another route in
  these tasks; report which.

## Deviations

Listed in the v3j write-up.
