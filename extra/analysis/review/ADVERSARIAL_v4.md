# Adversarial review v4 (2026-11-21, 00:00–00:40): verified verdicts

**Reviewers.** Three AI reviewer agents read a frozen pack of summaries, pre-registrations, code and the
literature graph (R1: assumptions and consistency; R2: placement in the literature; R3: method). The
main session checked every high-severity claim before recording it.

**Verdicts:** ✅ confirmed · ◐ partly · ✗ rejected.

**State at review time.** The LLM results (C1–C9) were not yet analysed. C1 Qwen at lr 1e-6 had
finished training, but no decline analysis was run.

## Confirmed and already acted on

| # | Claim (source) | Check | Action |
|---|---|---|---|
| 1 | **The KL penalty gradient was the forward KL** (R3 3.10). Differentiating k3 on-policy gives ∇KL(ref ‖ π), not ∇KL(π ‖ ref) | ✅ Exact check on a categorical: k3 matches forward, k2 matches reverse | Fixed in `llm-transfer` `4e575cc` with `tests/test_llm_kl.py`. No run with β > 0 had happened; C7 now uses the reverse gradient, and C8 tests forward on purpose |
| 2 | **P1's p-value is pseudo-replicated** (R1, R3 1.1). The test paired (task, η) while the unit is the task | ✅ | Recomputed with the task as unit: **all 16 cells still pass**, max p 0.0098, ×5 = 0.049 (`analysis/v3a/P1_TASK_LEVEL.md`, `fa77a29`). This replaces 3.0e-8 for Holm |
| 3 | **"Absorption rises with m" is not a dose-response in the absorbed fraction** (R1, R3 1.3) | ✅ for the fraction (m = 0.7 > 0.9 in 14/16 cells) | Use the per-minority-prompt rate, which rises monotonically in 15/16 cells. Reword the claim: "imbalance causes the minority class to be absorbed, more completely as m grows" |
| 4 | **H1 identity and H2 surrogate are known** (R2): Davis & Recht prove the finite-G identity and the finite-G surrogate | ✅ (note `2025_Davis…md:26-27`) | Report H1 and H2 as verifications, crediting Davis & Recht; our additions are the calibrated test (T7) and λ(G) with CIs |
| 5 | **Schaul et al. 2019 (ray interference) is not in our notes or graph** (R2) | ✅ (no match) | Read it before any claim about T1 or C2 (TODO) |
| 6 | **Zhang et al. 2505.17508 is missing from the milestone references** (R1, R2; the professor asked for it) | ✅ | Cite for KL-estimator design. It is also the right citation for item 1 |

## Confirmed, needing a design change (open decisions)

| # | Claim | Check | Recommendation |
|---|---|---|---|
| 7 | **The P4 null is sampling noise only** (R1, R3 2.2). r₀ = (L₀ + 1)/1001 ≈ 0.001, so a handful of drift-induced losses "pass" | ✅ (arithmetic) | Amend PREREG_C1 **before** any C1 decline analysis. Keep P4 as stated, but report it as "excess declines over sampling noise". Add secondary nulls: the decline rate on held-out problems, and a "strict lost" (0/256 after, base p̂ ≥ 0.1) |
| 8 | **The family contrast is confounded with method** (R3 3.3): full fine-tuning against LoRA, different optimizer and lr | ✅ | Add a Qwen-0.5B LoRA arm (about 20 GPU-hours), or call the two arms "two configurations" |
| 9 | **The sync check is weak** (R3 3.1): one 64-token sample at steps 1–3, when LoRA's delta is near 0 | ◐ The unit test fails on the old code, and the lr 1e-3 GPU smoke tracked a large change; but the routine check would not catch drift later | Log vLLM log-probs of every training sample (`logprobs=0`) and the per-step mismatch. Optionally add truncated importance sampling (TIS) |
| 10 | **Sampler–trainer mismatch without correction** (R3 3.4) | ✅ and **observed**: all three SmolLM2 pilots at lr 1e-5, 3e-5 and 1e-4 reach reward 0.7–0.8, then collapse after about 100 steps (gradient norm 17 → 299, answers at the 512-token cap, reward 0). lr 3e-4 had not collapsed by step 150. The pre-registered lr rule gave NONE | As in 9, then rerun the SmolLM2 pilots with mismatch logging (and TIS as an arm). Following the gate, Llama-3.2-1B replaces SmolLM2 as P4's second family; amend PREREG_C1 first |
| 11 | **C2 SFT control not comparable** (R3 3.9): different loss, lr, format and one seed | ✅ | Replace with rejection-sampling fine-tuning on the model's own correct samples (3 seeds), or report SFT as descriptive only |
| 12 | **C2 GEE weaknesses** (R3 2.5): 8 clusters per role, and the minority set's composition changes with m | ✅ | Add a run-level permutation trend test and an analysis of the 300 held-out minority problems alone (constant across m). Amend PREREG_C2 before C2 results |
| 13 | **Shared "before" sample** (R3 2.1): all runs compare with base seed 0 | ✅ | Report the base-against-base overlap null, and use problem fixed effects in C2 |
| 14 | **C2 generator** (R3 3.8): the cues are not minimal pairs, and operands are drawn separately per procedure | ◐ The operand distributions match; the cue wording differs by more than one token | Accept for now. A paired generator (same template and operands, cue only) is a cheap improvement before C2 runs; C2 is queued and not started |
| 15 | **C7 premise is tabular-only** (R1). Shared-parameter data show non-monotone absorption in β | ✅ (base H4 table) | Restate the C7 prediction as exploratory, with no monotone claim |
| 16 | **P3 entered Holm as a near-certain rejection although it failed** (R3 1.6) | ✅ | For Holm, P3's claim (ΔAUC ≥ 0.03) is not rejected; set its p to 1 |

## Partly confirmed or rejected

- ◐ **T7 "holds"** (R1, R3 2.7): lin has 8.75% of ranks above 0.95, and m64 KS p = 0.055. Reword as
  "not rejected".
- ◐ **"Response noise only at B = 8"** (R1): bal at B = 512 (+0.053) and c05 at B = 32 (+0.050) are
  also at the margin. Reword as "mostly at B = 8".
- ◐ **T4 verdict** (R1, R3): the half of the prediction about the mean rule being flatter was not
  met. Report T4 as undetermined; the T4e interior optimum is now descriptive.
- ✗ **"The probe already showed P1"** (R3 1.4) as a flaw: it was disclosed in PREREG_v3a. Keep the
  wording "confirmed on fresh seeds after a disclosed exploratory probe".
- ◐ **Regression to the mean from selection seeds** (R3 3.7): the pool used N = 32 seeds 0 and 1, and
  the N = 256 base also uses seeds 0 and 1 in vLLM, so the samples may overlap. Low impact, but use
  seed 2 for any new pool.

## What the reviewers agree is defensible

> A controlled dissection of a failure LLM papers only observe. With exact gradients and a
> deterministic binary verifier, parameter sharing makes GRPO absorb the minority answer class,
> increasingly as imbalance grows. This holds under SGD and Adam, with the mean step at small η, and
> supervised training does not do it. Noise adds to it only at small batches. The finite-G identity
> (Davis & Recht) is verified with a calibrated test. The KL basins are bistable and switch only
> through noise. Fixes for all-wrong groups (Heckel) help only at small steps.

Known results (H1, H2, the pass@k narrowing) are framed as verifications. The LLM arms test "excess
declines" (C1) and "procedure imbalance" (C2); they do not test the toy's mechanism.
