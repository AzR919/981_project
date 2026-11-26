# PREREG_C2, amendment 1 (2026-11-22)

Committed before any C2 result is looked at. It amends `PREREG_C2.md`; everything not mentioned
stands.

## What has been seen

- **Base evaluations:** the N = 32 base evaluation of the 6,000 candidates (used to build the sets).
  The two N = 256 base evaluations have finished but have not been opened.
- **One training run** (m = 0.5, majority "each", seed 0) finished under the old code. Its output has
  not been opened. It is **excluded** and rerun.
- **Reviews** (`papers/reviews/deep/`): Multi-Task GRPO (ICML 2026), Li, Galhotra & Cardie (preprint)
  and Zhou & Li (preprint). They name three rival explanations for a minority decline. The reviewer's
  points used here:
  - a minority starved of signal stops improving, but does not by itself fall below base;
  - the group-size arm planned earlier cannot separate sampling chance from interference, because
    expected correct minority samples per step do not depend on G at fixed samples per step.

## Primary test (P5), replacing the GEE

- **Panel.** The **300 held-out problems of the minority procedure**, the same problems in every
  run of a role.
- **Per-run outcome.** The decline rate on the panel: BH at 10% over the panel; one-sided Fisher
  against base seed 0 at N = 256.
- **Test.** Within each role (8 runs: m = 0.5 × 3 seeds, 0.8 × 2, 0.95 × 3), an exact permutation
  test of the Spearman correlation between m and the decline rate. All 560 assignments of m labels
  to runs are enumerated; one-sided, positive.
- **P5 holds** if p < 0.05 in both roles. The larger p goes to Holm.
- **Untestable rule.** If a role has fewer than 10 declines in total over its 8 runs, P5 is reported
  as untestable for that role: no measurable minority decline at this learning rate and length.

## Secondary (exploratory)

- **Fall below base.** The mean change in p̂ on the panel at m = 0.95, with a 95% interval from
  resampling problems within runs. A negative interval shows a fall below base, which starvation
  alone does not predict.
- **Trend in mean change** against m. Reported, with the caveat that less improvement (starvation)
  and decline (interference) both produce it.
- **Switch share.** The share of declined panel problems whose most frequent answer after training
  is the other procedure's answer, against the same share for the base model.
- **Trained minority problems** against held-out ones.
- **Majority problems:** their decline rate against m.

## Controls for the three rivals

1. **Signal share (Multi-Task GRPO).**
   - (a) From the group log, per procedure and per step: sampled prompts; informative (mixed-reward)
     groups; all-wrong and all-correct groups.
   - (b) **Matched-minority arm.** Every step draws 4 minority prompts and n_A ∈ {4, 16, 76} majority
     prompts (m = 0.5, 0.8, 0.95), from the same 500 + 500 training problems, with the loss divided
     by 80 × G in all arms. Both roles, seeds 0 and 1: 12 runs.
     - The minority's exposure is identical across arms; only the amount of majority data changes.
     - Prediction: under interference, the panel decline rate rises with n_A; under starvation, it
       does not depend on n_A.
     - Test: the permutation test above, on these 6 runs per role (90 assignments; smallest p =
       0.011).
2. **Sampling chance (Li, Galhotra & Cardie).** At m = 0.95, majority "each": 16 and 256 prompts per
   step, 3 seeds each, at the same learning rate and number of steps. The 64-prompt main runs are the
   middle point.
   - Prediction: under sampling-chance suppression, the decline is larger with fewer prompts per
     step; under mean-drift interference, the decline does not shrink as prompts per step grow.
   - Also reported: whether the minority's decline starts before its expected number of correct
     samples per step falls below 1.
3. **Access against ability (Zhou & Li).** On the 600 held-out problems, for the base model and each
   trained main run, N = 64 with:
   - no hint (the main evaluation);
   - a neutral hint;
   - the problem's own first step;
   - the other procedure's first step.
   - Reported, paired by problem: the recovery of minority success under the own-first-step hint,
     relative to the base model under the same hint.
   - The neutral hint is a negative control. The wrong-step hint measures how easily the cue is
     overridden.
   - Not done: the token-likelihood split at the first operation. The reason is that our reference
     solutions are templated, not model-written.
4. **Per-prompt tilt (Yang).** The base model's share of other-procedure answers per problem is
   reported. Only cross-prompt coupling predicts a dependence of the decline on m.

## Delay against loss

- Every main run is evaluated at step 250 (N = 32) and step 500 (N = 256); step 0 is the base
  evaluation.
- One long arm: m = 0.95, both roles, seed 0, 1,500 steps, with N = 256 evaluations at 500, 1,000 and
  1,500.

## Supervised control

Fine-tuning on the model's own correct samples (rejection sampling):
- up to 4 correct base completions per training problem;
- the same optimizer and learning rate as the GRPO runs (1e-6), 500 steps of 64 examples;
- m ∈ {0.5, 0.95}, majority "each", seeds 0–2.

Its majority-procedure gain is reported beside the GRPO runs' gain. If it is less than half of GRPO's,
the control is described as too weak to compare. The templated-solution SFT runs are dropped.

## Unchanged

The problems, the mixes, the training settings of the 16 main runs (now with drift and group logging,
which do not change the update), and the definitions of decline and lost.
