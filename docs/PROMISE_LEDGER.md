# Promise ledger: what the milestone promised, what happened, where it goes in the report

Status on 2026-11-23. Sources: `latex/milestone.tex` (hypotheses H1–H5, lines 306–319), the
pre-registrations in `analysis/review/`, and the summaries in `analysis/`. Failed predictions stay
in the report as failed.

## Milestone hypotheses

| | Promised in the milestone | Result | Already published? | Place in the report |
|---|---|---|---|---|
| **H1** | The identity E[ĝ] = W_G(p)∇p holds at finite G; rules whose W_G differ by a constant match after rescaling the step | **Holds.** No bias at any tier with the precision to detect one; the calibrated null is uniform in tabular runs (KS p = 0.73) | Yes: Davis & Recht prove the finite-G identity. The milestone (lines 258–260, 322) wrongly says they cover only infinite groups; the report corrects this | "The simulator matches known theory" (verification) |
| **H2** | Group size alone moves GRPO's surrogate from p at G = 2 toward 2 arcsin √p | **Holds without sharing** (tabular runs sit on the W_G line). With sharing a prompt's drift is not set by its own W_G; the full shared-gradient drift explains the change (R² 0.999) | Yes (Davis & Recht) | Verification; the sharing part leads into the mechanism |
| **H3** | Stability is set by one effective step, η W_G(p) p(1 − p)/B, together with interference through shared parameters | **The single-variable claim fails.** No one variable lines up the sharing levels (shifts up to 4.7 decades). **The interference part holds and became the main result:** with shared parameters a majority answer class drives the minority to near-zero success (P1; 16 of 16 cells). Without sharing, success never decreases (2,268 runs) | The one-step mechanism is in Schaul et al. (2019) and Li, Pandey & Chen (2026). The multi-step outcome for a correct class, and its law in the class share, are not | The core of the report: mechanism, who pushes whom, the odds law, when it lasts |
| **H4** | SGD with a KL penalty settles at the finite-G fixed point, not Mroueh's limit | **Fixed points hold** (median error 1e-4). Two stable points at low reference success (60 of 60). Our own scaling claim (spread set by η/β) **fails** | Fixed-point equations: Vojnovic & Yun | Verification, plus a new result that ties H4 to the main finding: the KL direction decides whether the minority survives (T17, T18) |
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
| Threshold grid and the odds law (T15, `analysis/theory/odds_law.md`) | The push is R = c·m/(1 − m), with c from the feature geometry; the first decline starts at m = 1/(1 + c). Parameter-free, checked on 1,080 runs |
| Margin check (M4) | Class decides, not prompt difficulty |
| Prevention levers (T13, T16, T17, T19) | Hard-prompt weighting and a forward KL anchor prevent absorption at different costs; reverse KL at usual strengths and group size do not |
| LLM census with a calibrated null (C1, both families; census calibration) | No training problem is lost at a learning rate that learns well; a few percent of unseen problems are. A naive "locked problem" count flags 83 of 3,000 problems in an unchanged model |
| Controlled imbalance in the LLM (C2 and its controls) | The toy collapse does not transfer: two procedures that share most steps help each other |

## Open on 2026-11-23

- C10 (who pushes whom in the LLM): 18 runs in the queue.
- T20 (a correct minority in Li, Pandey and Chen's environment): batch `v3h` running.
- Rejection-sampling controls for the held-out declines; the long arm; KL runs in the LLM.
- A hand check of the odds-law derivation by a team member.

## The story in one paragraph

Under GRPO with parameters shared across prompts, a majority answer class pushes a correct minority
class to near-zero success. The push at the start of training obeys a closed-form odds law, and
the collapse lasts as a long plateau. It can be prevented, at a cost that differs by method. In two
small LLMs the same pressure does not appear: training loses no problem it trains on, and a
minority procedure improves alongside the majority. What the LLMs do show is a small, repeatable
loss of unseen problems to confident wrong answers.
