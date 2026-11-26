# PREREG_C1, amendment 2 (2026-11-22): a control for the held-out declines

Committed before the control runs start. Exploratory.

## What has been seen

The C1 Qwen results (`extra/analysis/llm/C1_C5_SUMMARY.md`, `analysis-v3` `f25100f`):
- At lr 1e-6, GRPO lost no training problem.
- On the 300 held-out problems, 8–12 declined significantly and 1–5 were lost (null: 0).
- Mean held-out success rose from 0.51 to 0.84.

The Llama learning-rate pilots (lr 1e-6 chosen by the rule).

## Question

Are the held-out declines specific to GRPO, or does any fine-tuning on the same 1,000 problems
produce them?

## Control

- **Data.** Qwen2.5-0.5B-Instruct, the same 1,000 training problems. Up to 4 correct completions per
  problem, sampled from the base model (16 samples per problem, seeds 10–12).
- **Training.** Cross-entropy on those completions (rejection-sampling fine-tuning). Same optimizer and
  learning rate as C1 (8-bit AdamW, 1e-6), 8 epochs of batches of 64 (about 500 steps). Seeds 0–2.
- **Evaluation.** As for C1: 256 samples per problem on the 1,300 problems, analysed with
  `extra/llm/analyze_c1.py`.

## Reading, fixed in advance

- Let D_GRPO be the number of BH-significant held-out declines per run (8, 9 and 12 in the three C1
  runs), and D_RFT the same for the control.
- **Specific to GRPO** if all three control runs have D_RFT ≤ 4, i.e. half of the smallest GRPO count
  or less.
- **Generic to fine-tuning** if at least two control runs have D_RFT ≥ 8.
- Otherwise, undecided.
- **Caveat on training gain.** The control's gain on the training problems is reported beside the
  counts. If its mean training success ends below 0.75 (GRPO: 0.98), the comparison is described as
  confounded by a smaller training effect.
