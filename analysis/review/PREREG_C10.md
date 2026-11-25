# Pre-registration: C10, who pushes whom in the LLM (2026-11-23)

Committed before any C10 run starts. Code: branch `llm-transfer` ("C10 branch test" commit);
analysis `llm/analyze_branch.py`. Exploratory.

## Why

In the toy (T11), from one saved state the majority-only update lowers minority success and the
minority-only update raises it. C2 and its matched-minority arm test this in the LLM only through
the outcome of full training runs. C10 tests it directly: which class's updates lower the minority?

## What has been seen

- Toy: T11–T16 (`analysis/v3d`, `v3e`, `v3f`).
- LLM: C1 and C5 (`analysis/llm/C1_C5_SUMMARY.md`); training curves of the C2 main runs (reward and
  drift only). No C2 evaluation has been analysed, and no C10 run exists.

## Design

- **Model and settings:** Qwen2.5-0.5B-Instruct from the base weights; GRPO as in C2 (G = 8,
  lr 1e-6, 8-bit AdamW, full fine-tuning).
- **Problems:** the balanced C2 training pool (500 per procedure, `mix_m0.5_each_s0`), sampled by
  procedure with `--strata`. This is the composition of the matched-minority arm at its largest
  imbalance.
- **Arms** (the denominator of the loss is 80 prompts in every arm, so the two single-class arms
  are the two parts of the mixed update):

| Arm | Majority prompts per step | Minority prompts per step |
|---|---|---|
| mixed | 76 | 4 |
| majority-only | 76 | 0 |
| minority-only | 0 | 4 |

- **Roles and seeds:** both role assignments (majority `each`, majority `total`) × seeds 0–2:
  18 runs of 100 steps, with a checkpoint at step 50.
- **Measurement:** the 600 held-out problems (300 per procedure), 128 samples each, at steps 50
  and 100. Reference: the base model's 256-sample evaluation (seed 0). Null: base seed 1 minus base
  seed 0.
- **Statistic:** the change in mean success on the 300 held-out minority problems, with a
  problem-level bootstrap interval per run (2,000 draws).

This is a short-horizon branch, not a single step: the three arms drift apart over 100 steps. The
plan's 61 + 3 composition was replaced by 76 + 4 so that the mixed arm matches the matched-minority
arm.

## Predictions (step 100)

1. **Majority-only training lowers the minority.** The interval is below zero in all 3 seeds of
   both roles.
2. **Minority-only training raises it.** The interval is above zero in all 3 seeds of both roles.
3. **Additivity.** Per role (seed means), the mixed change differs from the sum of the two parts by
   at most half of the larger part.

Reported beside them: R_LLM = −(majority-only change)/(minority-only change); the same quantities
at step 50; the change on the majority panel in each arm.

## Readings, fixed in advance

- **1 holds:** the majority's updates actively lower the minority procedure in the LLM. This is the
  toy's mechanism, shown directly.
- **Majority-only raises the minority** (interval above zero in most runs): the two procedures
  share more skill than they compete for. At this horizon there is positive transfer, not
  interference, and any decline in C2 would have to come from something else.
- **No change** (intervals include zero): at lr 1e-6, 100 steps do not move the minority; the test
  is uninformative at this horizon and is reported so.
- 1 holding in one role only is reported as role-dependent, not as support.

## Deviations

Listed in the C10 write-up.
