# C2: does a majority procedure collapse a minority procedure in the LLM? (P5)

All 16 main runs and the 12 matched-minority runs finished (queue `r2`, 0 failed; code commit
`8002345`). Tests as pre-registered in `PREREG_C2.md` and amendment 1; `llm/analyze_c2.py`
regenerates `c2.md` and `c2_runs.csv`.

Setup: Qwen2.5-0.5B, GRPO, 500 steps, lr 1e-6. Two procedures share one problem template. The
majority procedure has share m ∈ {0.5, 0.8, 0.95} of the training problems. Everything below is
measured on the same 300 held-out problems of the minority procedure (256 samples each).

## Result: P5 fails

| Majority procedure | Runs | Declines on the panel | Trend of the decline rate with m | Exact one-sided p |
|---|---|---|---|---|
| `each` (minority `total`) | 8 | 200 | Spearman 0.64 | 0.0536 |
| `total` (minority `each`) | 8 | 66 | Spearman 0.86 | 0.0071 |

P5 needs p < 0.05 in both roles. The larger p is 0.0536, so **P5 fails as registered.**

P5 fails narrowly on the registered test, and both correlations are positive. The exploratory evidence against an effect of the majority is in the level of the outcome and in the matched-minority arm (6 runs per role):

- **The minority procedure gets much better, not worse.** Its mean held-out success rises by
  0.36–0.45 in 15 of 16 runs (from about 0.52–0.55 to 0.91–0.97), including at m = 0.95, where it
  gets about 3 prompts per step. The remaining run (majority `each`, m = 0.95, seed 1) rises by 0.13.
- **Declines are few.** 2–9% of the 300 problems decline significantly (null: 0), against 100%
  absorption in the toy at the same share. One run has 28%.
- **No switch to the majority's procedure.** Among declined problems, the share whose most common
  answer is the other procedure's answer is 0 in every run.

## The controls

| Rival explanation | Control | Result |
|---|---|---|
| The majority's updates push the minority down (the toy's mechanism) | Matched-minority arm: 4 minority prompts per step in every run, 4, 16 or 76 majority prompts | **No dependence on the number of majority prompts** (Spearman −0.36 and −0.72; p = 0.80 and 0.97). The mean gain is the same at 4 and at 76 majority prompts |
| The minority gets too few informative groups | Group log | Informative minority groups per step fall from 2.3–2.8 (m = 0.5) to 0.3–0.6 (m = 0.95). The weak trend in the main runs follows this count |
| The skill remains but is not started | Hints | After training, success on the minority panel is the same with a neutral hint, the right first step or a wrong first step (within 0.01) |

So the small rise of the decline rate with m in the main runs goes with the minority's own sample
count, not with pressure from the majority.

## What the declines are

- **They are a held-out effect.** On the minority problems the run trained on, 0–0.8% decline
  (one run: 4%). On held-out minority problems, 2–9% do. The same was seen in C1.
- **They are confident wrong answers.** A problem that ends at success ≤ 0.05 gives one wrong answer
  in 69–95% of samples. It is not the other procedure's answer and not a format failure (every
  answer parses).
- **One run degraded more broadly:** majority `each`, m = 0.95, seed 1 has 84 declines and 45 lost
  problems on the minority panel, and 17 lost on the majority panel. Its training reward still
  ended at 0.98. It is one seed of three and is reported, not explained.

## Multiple testing over the five pre-registered predictions

| Prediction | Outcome | p passed to Holm | Holm-adjusted |
|---|---|---|---|
| P1 Class imbalance causes minority collapse (toy) | holds | 0.0098 | 0.049 |
| P5 Minority decline rises with m (LLM) | fails | 0.0536 | 0.21 |
| P2 Response noise does not matter (toy) | failed its registered rule | 1 | 1 |
| P3 Initial drift predicts absorption (toy) | fails | 1 | 1 |
| P4 Lost training problems in the LLM census | fails | 1 | 1 |

After Holm's correction only P1 holds at 0.05.

## Reading

- **The toy's collapse does not transfer to this LLM setting.** With a pretrained model, two
  procedures that share most of their steps help each other: training almost only on one raises the
  other. In the toy the two classes compete for the same output from an equal start.
- **This is the decision gate "a controlled mechanism and its boundary"** (see the corrections: why the boundary is where it is remains a hypothesis)**.** The report states the
  mechanism and its law in the toy, and the LLM result as the boundary: the collapse needs classes
  whose updates oppose each other, and these two procedures do not.
- **The odds law gives the language for the boundary.** The push is R = c·m/(1 − m), and c carries
  the similarity between the classes' gradients. C10 measures the sign of the majority's effect on
  the minority in the LLM directly.
- **What remains unexplained in the LLM:** a few percent of held-out problems are lost to a confident
  wrong answer in every run, at every m, while trained problems are not. The rejection-sampling
  control (queued) tests whether that is specific to GRPO.

## Deviations

- Run `r2_10` (majority `total`, m = 0.8, seed 0) was stopped after its 256-sample evaluation to
  free a machine for another user, and its hint evaluations were run afterwards from the same model.
- Two runs (`r2_26`, `r2_27`) were cut off by a reboot at 03:00 and restarted from their
  checkpoints by the queue.

## Corrections after the review of 2026-11-23 (`analysis/review/ADVERSARIAL_v5.md`)

1. **Why there is no collapse is not settled.** Two candidate reasons fit the data: the procedures
   share most of their skills, so their updates are aligned; and the base model already separates
   them, giving the other procedure's answer on only 0.6% and 5.4% of samples, where the toy starts
   with 1/K. By the first-step algebra the second alone predicts a weak push. No LLM pair with
   opposing updates was tested. T22 tests the second reason in the toy; an LLM pair told apart by
   one token is in the to-do.
2. **Holm input for P2.** The table enters P2 with p = 1; the registered input is its largest
   kick-size p-value, 0.0027. With that, P1's adjusted p is 0.039 instead of 0.049. P1 holds either
   way, and P2 still fails its registered rule.
3. **Scope.** One model, 500 steps, lr 1e-6.

## Seed replication of the degraded run (added 2026-11-23)

`c2_seeds.md`, `c2_seeds_runs.csv`; pre-registered in `PREREG_C2_amendment2.md`; exploratory. **P5 is
unchanged: it failed as registered.** The problem draw of the degraded run (draw 1, majority
`each`) was rerun with five new training seeds at m = 0.95 and at m = 0.5.

| | m = 0.5 (seeds 3–7) | m = 0.95 (seeds 3–7) | The original degraded run |
|---|---|---|---|
| Held-out minority problems lost (of 300) | 4, 5, 3, 5, 5 | 11, 9, 17, 6, 11 | 45 |
| Held-out minority declines | 11, 9, 6, 10, 9 | 19, 18, 23, 17, 21 | 84 |
| Mean change of the minority panel | +0.40 to +0.41 | +0.36 to +0.39 | +0.13 |
| Majority-panel decline rate | 0.027–0.037 | 0.017–0.033 | 0.083 |

- **Broad degradation is rare.** By the registered definition (at least 30 lost), 0 of 5 new runs at
  m = 0.95 degrade, and 0 of 5 at m = 0.5. With the three original seeds, it is 1 run in 8 at
  m = 0.95: an outlier of the training seed.
- **A moderate effect of the share does show up on this draw.** Every run at m = 0.95 has more
  held-out minority declines than every run at m = 0.5 (17–23 against 6–11), and about twice as many
  lost problems. This was not a registered test; it agrees in direction with P5's near miss.
- **It is not the majority's pressure.** The matched-minority arm (above) shows no dependence on
  the number of majority prompts, and C10 shows majority-only training raising the minority. What
  changes between m = 0.5 and m = 0.95 is how many minority problems the run trains on (about 32
  against 3 per step): the minority panel still gains +0.36 or more, but unseen minority problems
  decline a little more often when few like them are trained.
