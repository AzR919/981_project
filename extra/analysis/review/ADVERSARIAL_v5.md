# Adversarial review v5 (2026-11-23): verified verdicts

**Reviewer.** One AI reviewer agent read the promise ledger, every results summary with its generated
tables and CSVs, all pre-registrations, the previous review and the prior-art reviews. Its report
is `ADVERSARIAL_v5_reviewer.md`. The main session recomputed every high-severity finding from the
data before recording it here.

**State reviewed.** `analysis-v3` at `d7c5feb`, `review` at `f1f97f0`. Still running at review
time: the C1 and C2 rejection-sampling controls, the C2 seed replication, the C2 long and
samples-per-step arms, and the KL runs in the LLM.

**Verdicts:** ✅ confirmed · ◐ partly · ✗ rejected.

## High severity

| # | Finding | Check | Action |
|---|---|---|---|
| 1 | **"Forward KL prevents absorption at every strength" holds only at the 1e-3 threshold.** At the registered sensitivity threshold 1e-2, forward KL at β = 0.01 leaves 34% of minority prompts absorbed, and 11% at β = 0.03 | ✅ Recomputed from the raw T17 runs: 0.339 and 0.109 at 1e-2 against 0.011 and 0.001 at 1e-3; none at β ≥ 0.1. 58 of 320 forward-KL runs at β ≤ 0.3 end with the minority below 0.05. In the MLP sampled cell at η = 1 the minority ends at 0.005–0.05, not at the resting points quoted | Correction added to `analysis/v3g/SUMMARY.md` and the ledger: forward KL keeps the minority above a floor that grows with β; it removes absorption at both thresholds from β = 0.1 |
| 2 | **"The KL direction decides" was shown only with a reference that still gives the minority's answer 1/3 (toy) or 0.3 (transformer environment).** A pull toward such a reference is the remedy Schaul et al. describe; two KL-direction papers in our notes are not cited | ✅ By construction: the reference is the start policy in T17, T18 and T21 | Reworded as "an anchor to a reference that still solves the minority"; citations added to the to-do. A run with a collapsed reference is not scheduled: its outcome follows from the same algebra |
| 3 | **T20's balanced control.** "Neither class absorbed in 18 of 20 (Adam 1e-3)" used a class-level rule (half of a class) that the pre-registration does not use. Per prompt, every balanced Adam run absorbs prompts; and SGD lr 0.3 loses a class at balance too | ✅ Recomputed from `v3h/runs.csv`: at balance, Adam 1e-3 absorbs 18–20% of each class's prompts in 20 of 20 runs; SGD lr 0.3 absorbs 48–51%, and at high share it absorbs class-0 prompts in 11 of 40 runs | Correction added to `analysis/v3h/SUMMARY.md`: only Adam 1e-3 isolates the share effect, as a rise of the per-prompt rate from about 0.2 at balance to 0.94 at share ≥ 0.875. Prediction 1a is worded "passed as registered, but the registered cells break the majority too; the isolating cell was not named in advance" |
| 4 | **The headline "drives the minority to near-zero success (P1; 16 of 16 cells)".** P1 tests that absorption rises with the share, not how complete it is | ✅ `P1_TASK_LEVEL.md`: the per-prompt rate at m = 0.9 is 0.54 in the two linear, ρ = 0.5, Adam cells | Ledger and story reworded: "absorbs the minority, more completely as the share grows (54–100% of its prompts at m = 0.9)" |
| 5 | **The class-agnostic start is an untested alternative for the LLM null.** Toy and transformer runs start with a minority prompt giving the majority's answer 1/K of the time; the C2 base model does so on 0.6% and 5.4% of samples | ✅ (`c2.md`, base rates). The algebra agrees: the cross-class factor is the overlap of the two response distributions minus the mass each puts on the other's answer | LLM readings reworded from "the boundary is shared skills" to a hypothesis with two candidate reasons. **New experiment T22** (junk answers; `PREREG_v3d_addendum4.md`), launched as batch `v3j`. The LLM-side test (a pair of procedures told apart by one token) is in the to-do |

## Medium severity

| # | Finding | Check | Action |
|---|---|---|---|
| 6 | Novelty: multi-step loss of correct contexts, the vanishing signal at low success, 1/ε plateaus and the remedy of a non-vanishing pull are in Schaul et al.; zero first-order drift at m = 1/K and its step-size dependence are in Li, Pandey and Chen | ✅ against our own deep reviews | "New:" labels removed from `v3h`; ledger H3 reworded. What remains ours: class share as a controlled driver for a correct minority under GRPO with a dose–response; the kernel correction c; levers compared at matched progress; a correct-answer verifier in the enumerable transformer environment; the calibrated LLM census and the positive-transfer result |
| 7 | The odds law is a first-order identity verified numerically, not a law of training; in T20 c = K − 1 by construction | ✅ | Reworded to "first-step identity" throughout the ledger and `odds_law.md` |
| 8 | Recovery under the exact update appears only with Adam (parity task; T12 at ρ = 0.9: 24 of 80 exact Adam runs recover, 0 of 240 SGD runs) | ✅ `v3d/tables/t12_cells.csv`, `v3h/cells.csv` | `v3h`, `v3i` and `v3d` corrected: "with Adam the exact update recovers and sampling blocks it; with SGD neither recovers" |
| 9 | C10: the arms are not additive under 8-bit AdamW, and the toy ratio quoted (4–17) is for m = 0.7–0.9, not C10's 0.95 | ✅ | `C10_SUMMARY.md` corrected; the sign result stands (18 of 18 intervals above zero) |
| 10 | LLM scope: "loses no problem it trains on" holds at lr 1e-6 only; held-out declines recur in number but are mostly different problems per seed | ✅ (C1 at lr 3e-6 loses 15, 12 and 1) | Story and C1/C2 readings scoped |
| 11 | P5 is a near miss and both correlations are positive | ✅ | `C2_SUMMARY.md`: "fails narrowly"; the counter-evidence is the exploratory matched-minority arm |
| 12 | Two definitions of the absorption threshold (v3e: median > 0.5; v3g: mean ≥ 0.05) | ✅ | Noted in both summaries; conclusions unchanged under either |
| 13 | T18's Spearman 0.81 is pooled over release steps; within a step it is 0.38 and 0.47 (p = 0.10, 0.07) | ✅ recomputed | `v3g` corrected |

## Low severity

All confirmed and corrected in place: the uncorrected P1 p-value in `v3a/SUMMARY.md`; the Holm input
for P2 (1 used; the registered input is 0.0027, under which P1's adjusted p is 0.039, so P1 holds
either way); number slips in `v3e`, `v3f`, `v3g`, C1 and C2; the stale "Open" list in the ledger;
results that hold by construction (tabular monotonicity, m\* independent of G) now presented as
sanity checks; T13 prediction 1 holds in 8 of 16 cells and ties at zero in the rest.

Not corrected: P1 part (b), the Spearman test, is still computed with (task, η) pairs; redoing it
at the task level is in the to-do.

## Checked by the reviewer and found sound

Both P5 permutation p-values (exact), the Holm arithmetic, the C1 held-out counts from raw data,
the C10 values, the T20 and T21 counts under their stated definitions, the T11, T15, T16, T17 and
T18 tables, the odds-law algebra, and the margin control.

## Not checked

T21 at the 1e-2 threshold; anything that depends on the controls still running.
