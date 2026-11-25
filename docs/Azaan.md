# Project status and next steps (Azaan, 2026-11-25)

This note covers everything merged up to `4840338` (PRs #10–#18, Nov 22–25), on top of the side note at the
end of `TODO.md` (Nov 24). It is a reading guide: the source for each point is named so it can be checked.
The main sources are `docs/PROMISE_LEDGER.md` (milestone promises against results), the `SUMMARY.md` of each
batch under `analysis/`, `analysis/theory/`, `analysis/llm/*_SUMMARY.md` and
`analysis/review/ADVERSARIAL_v5.md`.

---

## 1. What happened since Nov 22, in one table

| Area | New work | Main outcome |
|---|---|---|
| Toy mechanism | v3d–v3n: T11–T26 (about 12k runs) | The majority actively pushes the minority down, then the minority starves. The collapse is a long plateau, not a second resting point. Class decides, not margin |
| Theory | `analysis/theory/`: odds law, late phase, fall model; Lean 4 proofs and SymPy checks | First-step push ratio R = c·m/(1 − m), exact and parameter-free. The late phase is explained at the run level only, not per prompt. A class-level model of the fall has been written, but its test hasn't been run |
| Prevention levers | T13, T16, T17, T18, T19, T21 | Forward KL to a reference that still solves the minority, pass@k weighting, and down-weighting or projecting the majority's step all prevent absorption, at different costs. Reverse KL at small β and group size do not |
| Second environment | Li, Pandey & Chen's transformer on token prompts: a replication of their paper, then T20 and T21 | Their headline replicates. With our correct-class verifier, the minority collapse and the odds law carry over to a shared transformer |
| LLM | C1 (Qwen and Llama), C2, C5, C7–C10, extras | No collapse. GRPO loses no training problem at lr 1e-6. A minority procedure *improves*, even under majority-only training. A few percent of held-out problems are lost in every run, and KL removes that |
| Simon | `experiments/h1.py` with an MLP option; `docs/Simon.md` | Adds an MLP version of the H1 check. The old tabular plots were deleted and no new plots are committed yet |
| Pending | v3o and v3q (confirmatory replication battery A1–A5, about 11k runs); v3p; the fall model's held-out test | Pre-registered, launched or ready; no results in the repo yet |

---

## 2. The story now

> Under GRPO with parameters shared across prompts, and from a start that does not yet tell the answer
> classes apart, a majority answer class absorbs a correct minority class, more completely as its share
> grows. The first step obeys a closed-form identity in the class share and the feature geometry; the
> collapse then lasts as a long plateau. It can be prevented, at a cost that differs by method. The same
> holds for a small transformer on token prompts. In two small LLMs it does not appear.

(Condensed from `PROMISE_LEDGER.md`, "The story in one paragraph".)

How this relates to the milestone:
- **H1, H2 and H4(a) hold** as verifications of known theory (Davis & Recht; Vojnovic & Yun).
- **H3's single-step law failed.** Its interference half became the main result.
- **H5 became a short negative result:** group size is not a lever.

### Mechanism, step by step (toy)

1. **First step: the push.**
   - What happens:
     - The majority's update lowers the minority, and the minority's own update raises it.
     - The ratio between them is R = c·m/(1 − m).
     - c comes from the feature geometry, from kernel means between and within classes.
   - Evidence:
     - T11: 16 of 16 cells, all tasks.
     - The odds law checked against 1,080 runs with nothing fitted (median error 0.14%).
     - The algebra is proved in Lean 4.
   - Caveat: this is a first-step identity, not a law of the whole run.
2. **The fall.**
   - The minority drops while the majority is learned.
   - Which prompts end deepest is decided here: the late phase barely reorders them (Spearman 0.97–1.00, v3l).
   - A class-level model of this phase is written up in `analysis/theory/fall.md`. Its predictions were fixed in advance, and its held-out test on v3o is pending.
3. **The plateau.**
   - Once success is near 0, the minority's own signal vanishes (∇p ∝ p), and the majority keeps pushing.
   - The release test (T18): remove the majority's reward and the minority returns. The time it takes grows with 1/p at release.
   - Under SGD at ρ = 0.9 the plateau outlasts 10× the horizon (0 of 240 runs recover). At ρ = 0.5 it often ends.
   - With Adam and the exact update, some runs recover.
4. **What doesn't decide it.**
   - Prompt margin doesn't (M4): class decides.
   - The first-step sign doesn't predict the outcome:
     - T22 (junk answers): the first step can raise the minority and it still collapses;
     - T23 (warm start): the first step lowers it and it still survives.

### Levers (toy; linear task, ρ = 0.9, m = 0.9 unless stated)

| Lever | Effect | Cost |
|---|---|---|
| Forward KL to the start policy | No absorption from β = 0.1 at both thresholds. At β = 0.01–0.03 there is a low floor (absorption only by the 1e-2 threshold). Brings a released minority back in 100–600 steps | Caps final success as β grows. The reference must still solve the minority |
| Reverse KL | Nothing at β ≤ 0.1; works only in a narrow window, β = 0.3–1 | Caps the majority at β = 1 |
| pass@k weighting | k = 2 keeps the minority off the floor and lets it recover. k = 4 protects it while the majority is learned | The majority learns 4–40× slower at k = 4; k = 8 barely moves. In the transformer it works only with Adam |
| Shorten or project the majority's step | Removes the collapse at m = 0.7. Projection also removes it at m = 0.9 | Projection needs class labels. In the linear policy the majority then ends at 0.45–0.86 |
| Warm start (head start) | p0 = 0.5 protects at m = 0.7 but not at m = 0.9 | — |
| Group size G | Not a lever (T19) | — |
| Heckel weights | Help at small η only | Double absorption with large steps |

### LLM, in short

- **C1 census:** 0 training problems lost in Qwen and Llama at lr 1e-6. At lr 3e-6 Qwen loses 15, 12 and 1, onto one repeated wrong answer, mostly different problems in each seed.
- **Held-out problems:** a few percent decline in every run, in both families. Fine-tuning on the model's own correct answers (rejection sampling) shows none, but that control learns much less.
- **C2 (P5) fails:** the minority procedure improves by +0.36 to +0.45 at every share.
- **C10:** majority-only training *raises* the minority, in 18 of 18 runs.
- Why there is no collapse is open. Two candidate reasons:
  - the procedures share skills, so their updates are aligned;
  - the base model already separates them: it gives the other procedure's answer only 0.6–5.4% of the time.
- **C7/C8 (KL):**
  - Any β ≥ 0.02 removes the held-out declines.
  - Forward and reverse KL behave alike, as the toy predicts when nothing collapses.
  - G = 16 settles above G = 4.
  - The per-problem fixed points add nothing beyond base success.
- **All five pre-registered predictions after Holm:** only P1 holds.

---

## 3. Where we stand

**Strong and ready to write up:**
- H1, H2 and H4(a) as verifications. Simon's H1 figure plus the vector test from the base batch.
- The minority-collapse mechanism (P1 plus T11–T18), with the odds law as its first-step theory.
- The lever comparison (forward KL vs. reverse KL vs. pass@k vs. projection).
- Carry-over to the transformer environment.
- The LLM as the boundary: a clean negative result with controls.

**Weak or open:**
- **Confirmatory replication (A1–A5):** launched or ready, no results yet. These are the tests that make the headline claims confirmatory rather than exploratory.
- **Theory of the fall:** written, not tested. Two per-prompt late-phase models already failed.
- **Hand check:** the ledger asks for a team member to re-derive the odds law; nobody has yet.
- **Why the LLM shows no collapse:** unsettled. The planned test, a pair of procedures told apart by one token, hasn't been run.
- **Simon's MLP H1:** code is in, plots aren't. The old tabular plots were removed in `0130bca`.
- **Prior art:**
  - Much of the mechanism is in Schaul et al. 2019: loss of correct contexts, vanishing signal, plateaus, the non-vanishing-pull remedy.
  - Li, Pandey & Chen give the first-step decomposition and its sign at m = 1/K.
  - What is ours: class share as a controlled driver for a *correct* minority under GRPO, the kernel correction c, levers compared at matched progress, and the calibrated LLM census.
- **Definitions:** two definitions of "absorption threshold" are in use (v3e and v3g). The report must pick one.

**Scale problem.** There are now roughly 50k toy runs, more than 26 named tests (T1–T26, A1–A5) and 11 LLM experiments. That is far more than a course report can carry, and the instructor's first comment was "trying to do too much". The work is careful, but most of it has to be cut from the report or moved to an appendix.

---

## 4. What I think we should do next

### Stop launching new experiments
Let the batches already launched (v3o, v3p, v3q) finish and be analysed, then freeze. Specifically, don't start:
- the MATH subset or 3B models (`analysis/TODO.md`);
- the one-token LLM pair;
- T25 (an optimizer batch), or any new T-numbers.

Each of these would answer a side question rather than the report's main one.

### Finish what the report depends on
1. **Replication battery (A1–A5):** run its analysis as pre-registered, with Holm. The headline rests on it.
2. **The fall model's held-out test** on v3o. If it fails, the report keeps the first-step identity as its only theory.
3. **The odds-law derivation** (`analysis/theory/odds_law.md`): one of us re-derives it by hand. It's short. I can do it, or Simon can.
4. **Simon's MLP H1:** run it and commit the tabular and MLP plots. This is the H1 figure.

### Write the final report around one question
Suggested outline: the tabular → linear/MLP → transformer → LLM ladder from the TODO note.
1. **Setup and identity.** H1 and H2 as verifications, crediting Davis & Recht for the finite-G versions.
2. **Without sharing, nothing collapses.** Tabular runs, plus H4's fixed points and bistability.
3. **With sharing, the majority absorbs the minority.**
   - P1, with A1 as its confirmation.
   - T11, the push.
   - The odds law.
   - T12/T18, the plateau.
4. **Levers.** One table (forward KL / reverse KL / pass@k / projection / group size), at matched progress.
5. **Carry-over to a shared transformer** (T20/T21), in one paragraph.
6. **Boundary: small LLMs.** C1, C2, C10 and C7/C8 in one section: no collapse; held-out declines that KL removes; the open "why".
7. **Limitations:** from `analysis/LIMITATIONS.md`, plus the failed predictions (P2–P5, H3's single-step law, the late-phase models).

Everything else goes in an appendix or is left out:
- T13 details, T22–T26;
- the late-phase and optimizer-geometry probes;
- C5, C9 and the extras;
- the replication of Li et al.

### Writing tasks to split
- **Related work:**
  - Schaul et al. 2019, cited prominently;
  - Li, Pandey & Chen;
  - Zhang et al. 2505.17508 and the two KL-direction papers (2509.07430, 2510.20817);
  - the missing keys in `analysis/review/c.md`;
  - merge the two bib files.
- **Fix the milestone's claims** about Davis & Recht (finite G) and H3.
- **Choose one absorption definition** and one threshold (1e-3, with 1e-2 as sensitivity) for the whole report.
- **Do the team walkthrough and the AI-use statement at the end,** as agreed.

### Decisions for the team
1. **Freeze date for experiments.** I suggest freezing once A1–A5 are analysed.
2. **How much LLM to keep.** I suggest one section, framed as the boundary.
3. **Who writes which section.** Natural split:
   - Ankush: mechanism, levers, LLM;
   - Simon: H1/H2 and the transformer environment;
   - Azaan: introduction, theory (odds law), related work, synthesis.

---

## 5. Against the original plan (milestone + `TODO.md` + guide), toy only

Decision (2026-11-25): the LLM work is set aside for now. This section covers only what the milestone,
`TODO.md` sections 1–3 and `docs/project_guide.html` planned. Sources: `analysis/base/FINDINGS.md` (the H1–H5
batch, about 17k runs), `analysis/stretch/FINDINGS.md`, `analysis/phase_a/`, `analysis/v3a`–`v3g`.

### Hypotheses (milestone section 3)

| | Planned | Done? | Result |
|---|---|---|---|
| **H1** | The identity E[ĝ] = W_G(p)∇p holds at finite G, in every configuration. Rules whose W_G differ by a constant match after rescaling the step (RLOO at η = mean-only at ηG/(G−1); GRPO = RLOO at G = 2) | Identity: **yes.** Rescaling equivalence: **not run** as a same-seed trajectory test | **Holds.** 414 runs comparing full vectors (mean, RLOO, GRPO, pass@k; tab/lin/m64; L = 1, 2, 4; G = 2–32): median z² ≈ 0.93–0.98, cosine ≥ 0.989. The calibrated tighter test (T7) is not rejected. Simon's norm plots (tabular; MLP code added, plots pending) give the figure. Known result (Davis & Recht) |
| **H2** | Group size alone moves GRPO's surrogate from p (G = 2) toward 2 arcsin√p. Per-prompt progress follows W_G(p) (warm-started MLP, mixed difficulty) | **Yes** | **Holds without sharing.** Tabular runs sit on the W_G line (slope 1.000, R² 1.000). λ(G) = 0, 0.015, 0.22, 0.63 at G = 2, 4, 16, 64, matching exact theory. **With sharing, more than 90% of a prompt's drift comes from other prompts,** so its own W_G no longer sets its progress; the full shared drift explains it (R² 0.999) |
| **H3** (main) | Stability is set by one effective step s = ηW_G p(1−p)/B, together with interference. Collapse plot: absorbed fraction against s̄ × interference. Grid: η, B, G (fixed-budget pairs), rule, L, sharing levels | **Yes,** 13,608 runs over the full planned grid (6 sharing levels × L × 7 (B, G) pairs × 3 rules × 12 η × 3 seeds) | **The single-variable claim fails:** no candidate lines up the levels (off by up to 4.7 decades on held-out levels). **Tabular never loses a prompt** (2,268 runs, 0 decreases). With sharing, absorption has two routes: (1) large per-prompt kicks η/B at small batches; (2) at correlated features, collapse already at the smallest η, which turned out to be minority-answer collapse. Route 2 became the main result (P1 and T11–T18; section 2 above) |
| **H4** | (a) KL sets where training stops: the finite-G p\*, reverse and forward. (b, c) KL raises the stability threshold, more so for forward KL | **Yes** | **(a) Holds:** median error 1e-4 over 960 runs, both directions; the guide's 0.109 reproduced. **(b) Fails:** the spread did not scale as η/β. New: at G ≥ 16, with a low reference, there are two stable resting points, and SGD lands on the side of the unstable root it starts on (60 of 60). **(c) Forward vs. reverse** answered on the minority task instead: forward KL prevents absorption at far smaller β than reverse (T17, T18) |
| **H5** | Group size, budget N = BG and escape from p ≈ 0, as a separate study (p₀ × G under fixed budget, tabular then shared) | **Yes** | The first best-G table was a tie-break artefact. With a wide enough η grid, the best G is 2 almost everywhere, often with η at the grid edge, so it's undetermined. Group size does not move the collapse threshold (T19). Fixes for all-wrong groups (Heckel) help only at small steps (T9). Bay & Yearick's group-size law checked against exact theory (Phase A) |

Other promises in the milestone's synthesis:
- **Heckel weights as a test of the absorption mechanism:** done (T9). They help at small η only.
- **Huang et al.'s reference resets:** dropped, because their collapse needs graded rewards.
- **The arcsin shift:** delivered (λ(G), above).

### Code planned in `TODO.md` section 3

| Item | Status |
|---|---|
| Tabular parameterization, reproducing `main.py` at L = 1 | ✅ Simon (`architecture="linear"` + one-hot prompts); `tests/test_tabular_bandit.py` passes |
| Warm-start to a target p₀, before the `Trainer` is built | ✅ `synthetic_rl/warmstart.py`; `sft_up`, `sft_down`, `sft_mixed` and tabular p₀ in `experiments/run.py` |
| Exact W_G(p) and diagnostics (per-prompt p, gradient overlap, absorbed fraction, informative fraction) | ✅ `synthetic_rl/theory.py` and the run logs (cross-share, absorption at several thresholds, persistent absorption) |
| Pass@k rules (RLOO_k, GRPO_k) | ✅ `passk2`, `passk4`, `rloo_k` |
| Forward KL alongside reverse | ✅ |
| Sweep runner (configs in, per-prompt logs out, seeds, SGD) | ✅ `experiments/run.py`, `experiments/grids.py` |
| Plotting script for the collapse plots | ✅ `experiments/analyze*.py` (figures under `analysis/*/figures`) |
| Cluster scripts | ✅ for the SFU lab machines (`cluster/`), not DRAC |
| Tests for the new options | ✅ (`tests/`) |

### Decisions planned in `TODO.md`

- **Keep H2 or H5 if time is short:** both were kept.
- **Default task size:** base batch K = 4, L ∈ {1, 2, 4}, d = 16; the v3 batches use L = 1, K = 3.
- **Symbol for response length:** L in the code. Still to fix in the report.
- **Optimizer:** plain SGD as planned, with ≥ 3 seeds. Adam was added later as a second arm.

### Housekeeping still open

- `torch` is not pinned in `requirements.txt`.
- The commented-out `logits` and `log_ref` lines are still in `main.py`.
- The pilot scripts were not moved into the repo (`run_pilot.py` is at the root).

### In short

Everything the plan asked for was run on the planned toy. The verifications (H1, H2, H4(a)) hold.
The main hypothesis (H3) failed as stated, and following its interference half led to the
minority-collapse result, which goes beyond the plan. Still missing from the plan:
- the H1 rescaling equivalence as a trajectory test;
- Simon's MLP plots;
- the housekeeping items.
