# Pre-registration: C2 (P5), minority-procedure collapse in a small LLM

This file is written and committed before any C2 training run starts. The code is on branch
`llm-transfer`: the generator `extra/llm/gen_procedures.py`, the sets `extra/llm/make_c2_sets.py`, and the SFT
control `extra/llm/sft.py`. The analysis must implement exactly these tests.

**What is already known:**
- **Difficulty pilot** (C0, tag `c0d`, base Qwen, N = 32, 200 problems per procedure and range):
  - no single operand range put both procedures in 0.3–0.7;
  - base Qwen gives the other procedure's answer on 1–8% of samples.
- **The "ml" range** was added before any training run, and the candidates are stratified on base p̂.
- **P1** (toy) passed. The toy analogue predicts that the minority answer class is absorbed under
  GRPO but not under supervised training.

## Design

- **Problems.** Two procedures share each surface template:
  - per unit ("each"): a·(b − c);
  - total: a·b − c.
  - One cue sentence differs between them. The answers always differ.
- **Candidates.** 3,000 per procedure (range `ml`, seed 1), base-evaluated at N = 32.
  - Kept if base p̂ is in [0.15, 0.85].
  - Stratified into 7 p̂ bins with equal counts per procedure.
  - Per procedure: a universe of 1,000 and 300 held out.
- **Mixes.** 1,000 training problems with share m from the majority procedure.
  - m ∈ {0.5, 0.8, 0.95};
  - majority ∈ {each, total} (the role swap);
  - mix seeds 0 and 1 for every cell, plus seed 2 at m ∈ {0.5, 0.95}.
  - That is 16 GRPO runs. The run seed equals the mix seed.
- **Training.** Qwen2.5-0.5B-Instruct, C1 primary settings: full fine-tuning, lr 1e-6, GRPO, G = 8,
  64 prompts per step, 500 steps, β = 0.
- **SFT control.** Cross-entropy on worked solutions, m ∈ {0.5, 0.95}, majority = each, seed 0, 2
  epochs, lr 1e-5.
- **Measurement.** N = 256 at temperature 1 on the run's 1,000 training problems plus all 600 held-out
  problems, keeping answer histograms.
  - Before: base Qwen, seed 0, N = 256, on all 2,600 problems.
  - Null: base seed 1.

## Definitions

- **Minority problems.** Problems of the run's non-majority procedure. That covers the minority
  problems in its training mix and the 300 held-out problems of that procedure. At m = 0.5 the
  "minority" is the procedure not named as majority, which serves as the balanced control.
- **Decline.** One-sided Fisher, base against after, N = 256. BH at q = 0.10 across all problems the
  run evaluated.
- **Lost.** A significant decline ending at p̂ ≤ 0.05.
- **Switch.** A lost problem whose most frequent parsed answer after training is the other
  procedure's answer.

## P5 (primary)

**Model.** A GEE logistic regression (exchangeable correlation within run) of each minority problem's
decline indicator on m (numeric), using all 16 GRPO runs.

**Fit.** It is fitted separately within each role: majority = each, and majority = total.

**P5 holds if:**
- the coefficient on m is > 0 with a one-sided p < 0.05 in **both** roles.

**Holm p-value.** The value passed to Holm (P1–P5) is the larger of the two role p-values.

**Null calibration.** The base-against-base decline rate is reported. If it exceeds 1% of problems,
the null is flagged.

## Secondary (exploratory)

- **Switch share** among lost minority problems, against the base rate of other-procedure answers on
  the same problems.
  - Prediction: above the base rate, since collapse goes toward the majority procedure.
- **Majority problems:** decline rate against m. Prediction: flat or falling.
- **Trained minority against held-out minority** problems.
- **SFT control.** Minority decline rate at m = 0.95 against m = 0.5. Prediction: no increase, which
  mirrors the toy's supervised control.
- **Mean p̂ per procedure** before and after, pass@64, and format failures.

## Deviations

Any departure from this file is listed in the C2 write-up under "Deviations", with its reason.
