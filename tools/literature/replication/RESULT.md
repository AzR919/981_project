# Replication of Li, Pandey and Chen, "Wrong the Same Way" (NeurIPS 2026 workshop)

2026-11-23. Their code at the pinned commit `05ef005` (MIT), run with their own scripts on the SECB
lab CPUs under torch 2.15.0 (the version they pin), single-threaded, each job in a fresh copy of the
code (`launch_wsw.sh`, `wsw_jobs.py`). All 11 experiments plus `theory.py` and `exp11_kernel.py`:
326 jobs (part 1: 256; part 2: E1 and E8 seeds 5–9, because their README uses ten seeds and the
script default is five). Every job exited cleanly. Results: `~/rlvr-data/wsw`, `~/rlvr-data/wsw2`.

## Control

Their `figures.py`, run here on their own published `results/`, reproduces all **494 of 494**
numbers in their `numbers.json` exactly (`numbers_control.md`). Any difference below comes from the
experiments, not from the figure code or library versions (matplotlib 3.9.2, numpy 1.26.4 here).

## Our rerun

`figures.py` on our results, against their published `numbers.json` (`numbers_comparison.md`;
`compare_numbers.py`): of 506 numbers, **130 identical, 193 within 2% (or 0.01), 183 different.**

**Why not bit-identical.** The base models differ already: no base model's hack fraction is
identical (median difference 8×10⁻⁴), on the same torch version, single-threaded. The supervised base
construction is not bit-reproducible across CPUs (their laptop, our lab machines), and the
difference compounds through training: final hack fractions differ by a median of 0.016 (Adam) and
0.044 (SGD), and a few runs end in a different outcome (largest difference 1.0).

**The headline claim replicates.** Mean drift of the hack fraction from the base model, by the
number of prompts m that share a leak (64 prompts in total):

| | m = 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Shared transformer, Adam, published | +0.058 | +0.076 | +0.110 | +0.163 | +0.236 | +0.331 | +0.384 |
| Shared transformer, Adam, ours | +0.069 | +0.097 | +0.122 | +0.204 | +0.191 | +0.330 | +0.374 |
| Tabular, Adam, published | +0.092 | +0.082 | +0.084 | +0.097 | +0.089 | +0.095 | +0.095 |
| Tabular, Adam, ours | +0.089 | +0.095 | +0.091 | +0.094 | +0.090 | +0.089 | +0.093 |
| Shared transformer, SGD, published | +0.003 | +0.134 | +0.431 | +0.464 | +0.543 | +0.583 | +0.593 |
| Shared transformer, SGD, ours | +0.096 | +0.161 | +0.460 | +0.454 | +0.595 | +0.671 | +0.560 |

With shared parameters the drift rises with m (one non-monotone step in ours, at m = 16 under Adam
and at m = 8 and 64 under SGD); without sharing it is flat in both. Their central claim, that the
dependence on universality comes from parameter sharing, holds in our rerun.

## Not yet done

- A number-by-number account of the 183 different entries (which are qualitative changes, which are
  seed noise around a bistable outcome); the per-prompt-copies and mask-only controls checked one by
  one against the paper's statements.
