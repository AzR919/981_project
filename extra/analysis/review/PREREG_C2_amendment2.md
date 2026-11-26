# PREREG_C2, amendment 2 (2026-11-23): how often does a run degrade broadly at m = 0.95?

Committed before the replication runs start. Exploratory. **It does not change P5**, which failed
as registered (`extra/analysis/llm/C2_SUMMARY.md`) and stays failed whatever these runs show.

## What has been seen

All C2 main and matched-minority results. One of the three runs at m = 0.95 with majority `each`
(problem draw 1, training seed 1) lost 45 of the 300 held-out minority problems and 17 of the 300
held-out majority problems. In the other 15 main runs the number of lost minority problems is
4–11. The two matched-minority runs at 76 + 4 prompts did not degrade.

## Question

Was that run a rare event of the training seed, or does that problem draw at m = 0.95 degrade often?
Three seeds cannot tell a 1-in-3 event from a 1-in-20 one.

## Runs

The problem draw of the degraded run (`mix_m0.95_each_s1`) and, as the reference, the same draw at
m = 0.5 (`mix_m0.5_each_s1`); five new training seeds each (3–7); otherwise identical to the C2
main runs, without the hint evaluations. 10 runs.

## Definition and reading, fixed in advance

- **Broadly degraded:** at least 30 of the 300 held-out minority problems are lost (a significant
  decline ending at success ≤ 0.05). This sits between the degraded run (45) and the largest count
  among the other 15 runs (11).
- Report the number of degraded runs out of 5 at each share, together with the original three
  seeds.
- **0 of 5 at m = 0.95:** a rare event of the training seed; the report describes the original run
  as one outlier in eight.
- **2 or more of 5 at m = 0.95 and 0 of 5 at m = 0.5:** a share-dependent instability; reported as
  an exploratory finding, with the caveat that it was looked for after seeing the data.
- **Any at m = 0.5:** the instability is not tied to the share.
- 1 of 5 at m = 0.95 is reported as inconclusive.
- Also reported for each run: lost problems on the majority panel, and the share of all-wrong
  minority groups per step.
