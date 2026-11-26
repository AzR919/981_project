# To-do (not scheduled)

## MATH subset for models of 3B and above (shelf item 3)

**Why.** GSM8K is nearly solved at 3B and above. Base p̂ on the first 64 train problems (model probes,
round 1):

| Model | Base p̂ |
|---|---|
| Qwen2.5-1.5B | 0.70 |
| Qwen2.5-3B | 0.88 |
| SmolLM3-3B | 0.885 |
| Qwen2.5-7B | 0.96 |

So the pool of problems with p̂ in [0.1, 0.9], where losses are measurable, would be thin.

**Needed:**
1. A MATH answer checker: normalise `\boxed{}` strings and compare with sympy where possible. The
   current `extra/llm/data.py` parser is numeric only.
2. A MATH subset, e.g. levels 3–5 of the MATH train split, with a fixed seed.
3. Base-p̂ calibration (N = 32, two seeds) for Qwen2.5-3B, Llama-3.2-3B and SmolLM3-3B, then the C0
   steps.

**Gate.** Go ahead only if the int8 or micro-batch-1 probes show that 3B trains on one 4090 (round 2
of the probes), and the C1 results justify a size arm.

## Follow-ups suggested by the results of 2026-11-22/23 (not scheduled)

1. **Prevention in the transformer environment (gated on T20).** If T20 shows the minority collapse
   in Li, Pandey and Chen's environment, rerun its collapsing cells with the levers that worked in
   the toy: forward and reverse KL (their loss has reverse KL only; forward is a small addition in
   `extra/experiments/enum_env.py`) and pass@k weighting. CPU, minutes. Pre-register as an addendum to
   `PREREG_T20.md`. If T20 shows no collapse, drop this.
2. **How often does a run degrade broadly at 95% majority? (LLM, exploratory.)** One of three seeds
   of C2 at m = 0.95 with majority `each` lost 45 of 300 held-out minority problems and 17 majority
   problems; its two sibling seeds and both matched-minority runs at 76 + 4 did not. Three seeds
   cannot tell a 1-in-3 event from a 1-in-20 one. Five more seeds at m = 0.95 and five at m = 0.5,
   majority `each`, without hint evaluations: 10 runs of about 80 minutes. This does not change the
   P5 verdict, which stays "fails"; it is reported separately.
3. **Read the KL runs (C7, C8) for the held-out declines, not only for training losses.** No new
   runs: the toy now predicts that the KL direction matters, and the one LLM effect we see is on
   held-out problems.

Considered and not added:
- a rerun of the group-size grid at matched flow time (it would refine a negative result);
- more majority shares or a longer horizon for C2 (the long arm is already queued);
- C11, a fix in the LLM (its gate is not met).

## From the adversarial review of 2026-11-23 (`extra/analysis/review/ADVERSARIAL_v5.md`), not scheduled

1. **An LLM pair told apart by one token** (finding 5, the LLM-side test). C2's procedures differ by
   a cue sentence and the base model almost never confuses them. Build a pair whose prompts differ
   by one token, check that the base model then gives the other procedure's answer often, and rerun
   C2 at m = 0.5 and 0.95: about 8 GPU runs of 80 minutes, plus a pilot for the pool. Do this after
   T22, which tests the same idea in the toy.
2. **P1 part (b) at the task level.** The Spearman test is still computed on (task, η) pairs.
3. **Cite the KL-direction papers** already in `papers/notes` (2509.07430, 2510.20817) and Schaul et
   al.'s non-vanishing-pull remedy where the forward-KL result is described.
4. **T21 at the 1e-2 threshold** (needs the per-prompt tail values, which the runs store).
5. Not planned: a KL anchor to a reference that has already lost the minority. By the same algebra
   it holds the minority down; state this instead of running it.

## Experiment rigour pass (a refinement step, after all planned experiments are done)

Requested by the user on 2026-11-23. For every place where the report compares our result with
another paper's, or uses our numbers to comment on theirs, check that the two are comparable, and
where they are not, see whether a comparable run or re-analysis is possible on our hardware (one
24 GB GPU per machine; models up to about 1.5B with full fine-tuning). Fix what we can; state the
rest as a limitation.

Checklist per comparison: samples per problem; model and size; decoding (temperature, top-p, token
limit); problem set and how it was filtered; number of seeds; the definition of the statistic.

Known cases to start from:
- **Yang (2609.22547), locked prompts.** Our "83 of 3,000" is at 32 samples per problem; Yang uses
  128–256. Not comparable, and no longer used against that paper. Done instead: a re-analysis of
  Yang's own released counts at n = 128 (`extra/tools/literature/checks/yang_2609_22547_check2_exact_null.py`
  on `literature`): 33 locked against 15–18 expected with no change. Possible addition: evaluate
  our base model twice at 256 samples on unfiltered problems (2 GPU jobs of about 2 hours).
- **Ramesh et al. (Multi-Task GRPO).** They use Qwen2.5-3B and real task mixtures; we use 0.5B and
  two templated procedures. Is a 1.5B run of C2 feasible?
- **Li, Galhotra & Cardie (replay).** Their toy varies samples per group; our samples-per-step arm
  varies prompts per step. State the difference.
- **Zhou & Li (access against ability).** Our hints are three fixed templates at 64 samples.
- **Li, Pandey & Chen.** T20 runs in their environment at their settings; their published cells
  were reproduced within 0.016 on another torch version. Rerun the reproduction on torch 2.15.
- **Davis & Recht; Vojnovic & Yun.** H1, H2 and H4 checks: confirm the settings match the theorems'
  assumptions (clipping off, fresh on-policy samples).

## Other open items

- **Restart C1 Qwen lr 3e-6, seed 0** (queue job `q_00`, c12). It hit a CUDA illegal-address fault at
  step 363. Restart on c12 with `extra/llm/relaunch.sh`; the checkpoint path is host-specific.
- **Analyses not yet run:**
  - A2 (ablation re-score), A3 (H4 fixed-β slopes), A6 (Barakat write-up);
  - T3 (attribution of lost prompts): descriptive only, optional;
  - Kaplan–Meier in flow time: dropped. Our own plan critique found it the wrong tool (first-passage
    curves measure dips, not persistent absorption), and T12/T18 answer the timing question;
  - A3 is done as a number (at fixed β, Var(logit p) ∝ η with slope 0.93–1.17, `analysis/base`); it
    needs one sentence in the report, no new analysis;
  - the per-prompt rarity + margin fit for T1 (plan critique).
- **Writing:** related work (add Davis & Recht as the source of H1 and H2, plus Barakat, Vojnovic,
  Li, Song, Baudry, Mei, Yu, Zhang-EMA, Ghosh, Zheng, Huang, and Zhang 2505.17508 for KL design;
  check Schaul et al. 2019), milestone wording fixes, a one-page narrative mapping each kept test to
  the collapse claim.
