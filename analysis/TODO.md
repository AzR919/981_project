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
   current `llm/data.py` parser is numeric only.
2. A MATH subset, e.g. levels 3–5 of the MATH train split, with a fixed seed.
3. Base-p̂ calibration (N = 32, two seeds) for Qwen2.5-3B, Llama-3.2-3B and SmolLM3-3B, then the C0
   steps.

**Gate.** Go ahead only if the int8 or micro-batch-1 probes show that 3B trains on one 4090 (round 2
of the probes), and the C1 results justify a size arm.

## Other open items

- **Restart C1 Qwen lr 3e-6, seed 0** (queue job `q_00`, c12). It hit a CUDA illegal-address fault at
  step 363. Restart on c12 with `llm/relaunch.sh`; the checkpoint path is host-specific.
- **Analyses not yet run:**
  - A2 (ablation re-score), A3 (H4 fixed-β slopes), A6 (Barakat write-up);
  - T3 (attribution of lost prompts), Kaplan–Meier in flow time;
  - the per-prompt rarity + margin fit for T1 (plan critique).
- **Writing:** related work (add Davis & Recht as the source of H1 and H2, plus Barakat, Vojnovic,
  Li, Song, Baudry, Mei, Yu, Zhang-EMA, Ghosh, Zheng, Huang, and Zhang 2505.17508 for KL design;
  check Schaul et al. 2019), milestone wording fixes, a one-page narrative mapping each kept test to
  the collapse claim.
