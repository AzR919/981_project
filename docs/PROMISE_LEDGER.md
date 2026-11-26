# Promise ledger: what the milestone promised, what happened, where it goes in the report

Status on 2026-11-23. Sources: `latex/milestone.tex` (hypotheses H1–H5, lines 306–319), the
pre-registrations in `extra/analysis/review/`, and the summaries in `analysis/`. Failed predictions stay
in the report as failed.

## Milestone hypotheses

| | Promised in the milestone | Result | Already published? | Place in the report |
|---|---|---|---|---|
| **H1** | The identity E[ĝ] = W_G(p)∇p holds at finite G; rules whose W_G differ by a constant match after rescaling the step | **Holds.** No bias at any tier with the precision to detect one; the calibrated null is not rejected in tabular runs (KS p = 0.73) | Yes: Davis & Recht prove the finite-G identity. The milestone (lines 258–260, 322) wrongly says they cover only infinite groups; the report corrects this | "The simulator matches known theory" (verification) |
| **H2** | Group size alone moves GRPO's surrogate from p at G = 2 toward 2 arcsin √p | **Holds without sharing** (tabular runs sit on the W_G line). With sharing a prompt's drift is not set by its own W_G; the full shared-gradient drift explains the change (R² 0.999) | Yes (Davis & Recht) | Verification; the sharing part leads into the mechanism |
| **H3** | Stability is set by one effective step, η W_G(p) p(1 − p)/B, together with interference through shared parameters | **The single-variable claim fails.** No one variable lines up the sharing levels (shifts up to 4.7 decades). **The interference part holds and became the main result:** with shared parameters, and from a start where every prompt gives every class the same probability, a majority answer class absorbs the minority, more completely as its share grows (P1 holds in 16 of 16 cells; 54–100% of minority prompts absorbed at m = 0.9). Without sharing, success cannot decrease (2,268 runs, a sanity check) | Schaul et al. (2019) already show multi-step loss of correct contexts through shared parameters, the vanishing signal at low success and the slow recovery; Li, Pandey & Chen (2026) give the first-step decomposition and its sign at m = 1/K. Ours: class share as a controlled driver for a correct minority under GRPO, with a dose–response; the correction c from the feature geometry; levers compared at matched progress | The core of the report: mechanism, who pushes whom, the odds law, when it lasts |
| **H4** | SGD with a KL penalty settles at the finite-G fixed point, not Mroueh's limit | **Fixed points hold** (median error 1e-4). Two stable points at low reference success (60 of 60). Our own scaling claim (spread set by η/β) **fails** | Fixed-point equations: Vojnovic & Yun | Verification, plus a new result that ties H4 to the main finding: an anchor to a reference that still solves the minority keeps it above a floor, and the forward direction does so at far smaller strength than the reverse (T17, T18, T21) |
| **H5** | Group-size scaling as a separate study | **The milestone pattern was an artefact.** With a wide enough learning-rate grid the best G is 2 almost everywhere. Fixes for all-wrong groups help only at small steps (Heckel) | Related analyses exist (Bay; Nie) | A short negative result: group size does not protect the minority (T19) |

## Pre-registered predictions (v3)

| | Prediction | Outcome | p passed to Holm | Holm-adjusted |
|---|---|---|---|---|
| **P1** | Class imbalance causes minority collapse (toy) | holds | 0.0098 | 0.049 |
| **P2** | Response noise does not matter | fails: noise adds 11 points of absorption at B = 8 | 1 | 1 |
| **P3** | Initial drift predicts which prompts are absorbed | fails: ΔAUC 0.009 against a bar of 0.03 | 1 | 1 |
| **P4** | GRPO loses training problems in a small LLM | fails in both families: 0 lost of 1,000 in every seed | 1 | 1 |
| **P5** | The minority procedure's decline rate rises with the majority share (LLM) | fails: p = 0.0536 and 0.0071; the minority improves at every share | 0.0536 | 0.21 |

## What was added beyond the milestone, and why

| Addition | What it settles |
|---|---|
| Branch test (T11) | The majority's update lowers the minority; the minority's own update raises it. It is active pushing, not starvation |
| Long horizon and release tests (T12, T18) | The collapse is a long plateau, not a second resting point. It lasts because the majority keeps pushing and because restoring forces shrink with success |
| Threshold grid and the odds law (T15, `extra/analysis/theory/odds_law.md`) | At the first step from a uniform start, the push is R = c·m/(1 − m), with c from the feature geometry, and the first decline starts where m = 1/(1 + c(m)). This is a first-order identity; it was verified with nothing fitted against the simulator's first step in 1,080 runs. It is not a law of the whole training run |
| Margin check (M4) | Class decides, not prompt difficulty |
| Prevention levers (T13, T16, T17, T19) | Hard-prompt weighting and a forward KL anchor (to a reference that still solves the minority) prevent absorption at different costs; forward KL at small strength leaves the minority at a low floor. Reverse KL at usual strengths and group size do not help |
| LLM census with a calibrated null (C1, both families; census calibration) | At lr 1e-6, no training problem is lost in either family (at lr 3e-6, Qwen loses 15, 12 and 1); a few percent of unseen problems decline in every run, mostly different problems in each seed. At 32 samples per problem, a naive "locked problem" count flags 83 of 3,000 problems in an unchanged model; our test flags none. This is about our setting and that sample count; it is not a statement about other papers' counts |
| Controlled imbalance in the LLM (C2 and its controls) | No collapse: a minority procedure improves at every majority share, and majority-only training raises it (C10). Why is open: the procedures share skills, and the base model almost never gives the other procedure's answer (0.6% and 5.4%), unlike the toy's start (T22 tests the second reason in the toy) |

## Open on 2026-11-23

- T22 (junk answers: where the wrong mass sits): batch `v3j` running.
- Rejection-sampling controls for the held-out declines; the C2 seed replication; the long arm; KL
  runs in the LLM.
- An LLM pair whose two procedures are told apart by one token (the LLM-side test of the start).
- P1 part (b) at the task level.
- A hand check of the odds-law derivation by a team member.

## The story in one paragraph

Under GRPO with parameters shared across prompts, and from a start that does not yet tell the answer
classes apart, a majority answer class absorbs a correct minority class, more completely as its
share grows. The first step obeys a closed-form identity in the class share and the feature
geometry, and the collapse then lasts as a long plateau. It can be prevented, at a cost that
differs by method. The same holds for a small transformer on token prompts. In two small LLMs at a
learning rate that learns well we do not see it: no training problem is lost, and a minority
procedure improves alongside the majority, even under majority-only training. Whether that is
because the procedures share skills or because the base model already separates them is not
settled. What the LLMs do show is a small loss of unseen problems to confident wrong answers, at a
rate that repeats across seeds and families.
