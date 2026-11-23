# C1 (Qwen) and C5: what the results say

- **Pre-registration:** tests follow `analysis/review/PREREG_C1.md` and `PREREG_C1_amendment1.md`
  (`review` branch, `2bdc507`), committed before any comparison was computed.
- **Generated tables:** `c1_qwen.md` and `c5_qwen.md`, produced by `llm/analyze_c1.py` (`llm-transfer`).
- **Setup:** Qwen2.5-0.5B-Instruct, full fine-tuning, GRPO with G = 8 and 64 prompts per step, 500
  steps on 1,000 GSM8K training problems. 256 samples per problem before and after, on the 1,000
  training problems and 300 held-out problems.

## P4 for Qwen: FAIL (no lost training problems at the primary learning rate)

| Arm | Seeds | Lost (train) | Declined (train) | Mean p̂ (train) | Lost (held-out, of 300) | Declined (held-out) |
|---|---|---|---|---|---|---|
| Null: base against base | – | 0 | 0 | 0.49 → 0.49 | 0 | 0 |
| **lr 1e-6 (primary)** | 3 | **0, 0, 0** | 0, 0, 0 | 0.49 → 0.98 | 1, 2, 5 | 12, 8, 9 |
| lr 3e-6 (secondary) | 3 | 15, 12, 1 | 18, 24, 1 | 0.49 → 0.91–0.97 | 20, 12, 11 | 39, 39, 26 |

- **Primary arm.** GRPO lost no training problem in any seed, and 999–1,000 of 1,000 improved
  significantly. The partial-conjunction p-value is 1. Since P4 needs both families, **P4 fails**
  whatever Llama shows.
- **Why no losses on training problems.** Each problem is sampled about 32 times and training
  reaches 0.98, so the training set is close to memorised. This is unlike the toy, where minority
  prompts start at the same success rate as the rest and are pushed down from the first steps.
- **Held-out problems do get worse** (exploratory). At the primary learning rate, 8–12 of 300 unseen
  problems decline significantly and 1–5 are lost, against 0 for the null. Training on other problems
  costs about 3% of unseen problems, while mean held-out success rises from 0.51 to 0.84.
- **Secondary arm (lr 3e-6):** 12–15 training problems are lost in two seeds, and 1 in the third.
  - **One repeated wrong answer.** Lost problems end with 61–79% of their 256 completions on a single
    wrong answer, against 2–4% for other problems. That answer is the base model's most frequent
    wrong answer only 17–27% of the time.
  - **Low base success.** Lost problems had lower base success (median 0.23 against 0.50; p = 2e-5).
  - **No overlap between seeds** (0 shared problems). Which problems are lost is not a property of
    the problem. This is unlike the toy's minority collapse, where the lost set is fixed by the task.
  - **Sensitivity.** The counts are unchanged when base seed 1 is used as "before".
- **pass@k.** The paired gap is positive at every k up to 64 at lr 1e-6. Coverage did not narrow on
  these problems, which all had base pass@64 ≈ 1.

## C5 loss drivers (exploratory; lr 1e-6 unless stated; 3 seeds each)

| Arm | Lost (train) | Mean p̂ after (train) | Lost (held-out) | Note |
|---|---|---|---|---|
| Control | 0, 0, 0 | 0.98 | 1, 2, 5 | |
| Verifier flips 20% | 1, 0, 0 | 0.90–0.91 | 3, 0, 2 | Learns less; no extra losses |
| Sign rule | 0, 0, 0 | 0.90–0.93 | 1, 5, 0 | Learns less; answers grow longer (330–370 tokens against 260) |
| lr 1e-5 | 1000, 511, 1000 | 0.00, 0.26, 0.00 | 300, 200, 300 | Collapse: two seeds produce no parsable answer at the 512-token limit |

- **The toy's predictions for the flip and sign arms are not supported** at this learning rate: both
  reduce how much is learned, but neither produces lost problems.
- **lr × 10 collapses**, as in the C0 pilot. This is instability (unparsable, maximum-length output),
  not selective loss.

## Reading

1. Under settings that train well, small-LLM GRPO does not lose training problems. The toy's minority
   collapse does not show up as a census effect on GSM8K, which has no controlled minority structure.
   C2 is the test that has one.
2. Losses appear with a 3× larger learning rate. They look like the toy's large-step route (random
   across seeds, collapse onto one wrong answer), not the structural minority route.
3. The cost that does appear at the primary setting is on unseen problems.

## Deviations and limits

- **Truncation counts and reference answers** were not stored for these runs (see the amendment). The
  most frequent wrong answer is identified by excluding the answer whose count equals the correct
  count.
- **The cross-seed overlap test** is uninformative for the lr 1e-5 arm (everything is lost).
- **One lr 3e-6 run** (seed 2) lost its final evaluation to the nightly `/tmp` wipe. It was resumed
  from its step-500 checkpoint and re-exported, so the evaluated model is the same.

## Second family: Llama-3.2-1B (added 2026-11-23)

`c1_llama.md`, `c1_llama_runs.csv`. Full fine-tuning at lr 1e-6, chosen by the pre-registered rule
(`PREREG_C1_amendment1.md`); 3 seeds; 1,000 training and 300 held-out problems from a pool of 2,157.

| | Qwen2.5-0.5B, lr 1e-6 | Llama-3.2-1B, lr 1e-6 |
|---|---|---|
| Training problems lost (of 1,000) | 0, 0, 0 | 0, 0, 0 |
| Mean training success, before → after | 0.49 → 0.98 | 0.42 → 0.99 |
| Held-out declines (of 300; null 0) | 8–12 | 10, 12, 13 |
| Held-out lost (of 300; null 0) | 1–5 | 4, 5, 9 |
| Mean held-out success after | 0.84 | 0.85–0.86 |

- **P4 fails in both families** (partial-conjunction p = 1 in each): at a learning rate that learns
  well, GRPO loses no training problem.
- **The held-out declines replicate.** In both families a few percent of unseen problems decline,
  and 1–3% end at success ≤ 0.05, while every training problem improves.
