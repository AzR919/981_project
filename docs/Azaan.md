# Project status and next steps (Azaan, 2026-11-25)

This note covers everything merged up to `4840338` (PRs #10–#18, Nov 22–25), on top of the side note at the
end of `TODO.md` (Nov 24). It is a reading guide: the source for each point is named so it can be checked.
The main sources are `docs/PROMISE_LEDGER.md` (milestone promises against results), the `SUMMARY.md` of each
batch under `analysis/`, `analysis/theory/`, `analysis/llm/*_SUMMARY.md` and
`analysis/review/ADVERSARIAL_v5.md`.

---

## H3 in detail: what it was meant to test, what we ran, what we found

Sources:
- the plan: `docs/project_guide.html` (H3 box) and `latex/milestone.tex`;
- the run: `experiments/grids.py` (`h3`);
- the results: `analysis/base/FINDINGS.md` (section H3), `analysis/review/SUMMARY.md`, `analysis/v3a/SUMMARY.md`, `analysis/v3d/SUMMARY.md`, `analysis/theory/odds_law.md`.

### Where we started

- **The identity.** For one prompt, the expected GRPO update is E[ĝ_x] = W_G(p_x)∇p_x (H1). So the size of a prompt's step is set by η, the batch size B, and the factor W_G(p)·p(1−p).
- **Suk & Duan.** For a *single* prompt, a step above a threshold (η ≲ Δμ(1−p)/T) can drive p to 0 even when every update points the right way. They leave open what happens when many prompts share parameters.
- **Our pilot.** The failure we saw was *absorption*: prompts pushed to a confident wrong answer (p ≈ 0). From there no sampled group contains a correct answer, so the prompt gets no learning signal. With tabular parameters this cannot happen, because a prompt's own update never lowers its own success. So we expected absorption to need parameter sharing.

### The hypothesis, as planned (guide, H3, "main study")

- **(a)** The outcome of training depends on η, B, G and the advantage rule mainly through **one effective step** at the start, s̄ = (η/B)·mean over prompts of W_G(p)·p(1−p).
- **(b)** The threshold exists only with sharing. Tabular runs never collapse, and the threshold in s̄ drops as the **overlap between prompts' gradients** (interference) grows.
- **(c)** Response length L and starting success p₀ enter only through quantities measurable at initialization: ‖∇z‖² and that overlap.
- **The planned test:** a single "collapse plot" of absorbed fraction against s̄ × interference, with every configuration on one curve.
- **Refuted if:** tabular runs collapse, or configurations with equal s̄ and interference end in clearly different places.

### What we ran (base batch, 13,608 runs)

**Task.**
- Each prompt is a feature vector x ∈ ℝ¹⁶, and there are 512 training prompts.
- The correct answer at each of the L positions is set by a hidden linear rule (argmax of a fixed linear score). So the task is solvable by every policy in the ladder, and any collapse is an optimization effect.
- K = 3 tokens, L ∈ {1, 2, 4}. The reward is 1 only if all L tokens are correct.
- Every run starts from the uniform policy, so every prompt starts at the same p = 3^(−L), i.e. 0.33, 0.11 or 0.012. Any difference in a prompt's fate therefore comes from sharing.

**Policies (the sharing ladder).**

| Level | Policy | Features | Sharing |
|---|---|---|---|
| `tab` | linear, no bias | one-hot | none |
| `lin` | linear | Gaussian | shared W |
| `c05`, `c09` | linear | correlated, ρ = 0.5, 0.9 | shared, prompts alike |
| `m64`, `m256` | MLP, width 64 or 256 | Gaussian | shared, nonlinear |

**Grid.**
- 6 levels × L ∈ {1, 2, 4} × 3 rules (mean-only, RLOO, GRPO) × 3 seeds.
- 12 step sizes η from 0.003 to 1000.
- 7 (B, G) pairs. Five of them have a fixed budget B·G = 256: (128, 2), (64, 4), (32, 8), (16, 16), (4, 64). The other two are (8, 8) and (128, 8).
- Sampled GRPO-style updates, plain SGD, no KL, 5,000 steps.

**Measured.**
- **Absorbed fraction:** the share of prompts that started at p ≥ 0.01 and ended below 0.01.
- Final success.
- At initialization: ‖∇z‖², the cosines between prompts' gradients, and each prompt's predicted first-step drift.

**Pre-registered test.**
- Four candidate x-axes:
  - (i) s̄;
  - (ii) s̄·‖∇z‖²;
  - (iii) (ii)·|cos|;
  - (iv) (ii)·(1 + B·cos).
- For each, a logistic fit of absorption against log x, pooled over levels and checked by leaving one sharing level out.
- **Pass:** an interference-weighted candidate beats s̄ alone on held-out levels, *and* no level's curve is shifted by more than one decade.

### What the base batch showed

1. **(b) holds: without sharing, nothing collapses.**
   - 2,268 tabular runs, at every η, including η = 1000.
   - 0 runs with any per-prompt decrease of p, and 0 absorbed prompts.
   - This is a sanity check more than a finding, since it holds by construction.
2. **(a) fails: no single variable lines up the levels.**
   - The best candidate on held-out levels was plain s̄. Adding the interference terms made things worse.
   - Even s̄ leaves the levels' curves far apart. In decades of x, relative to the pooled fit:

     | Candidate | lin | c05 | c09 | m64 | m256 |
     |---|---|---|---|---|---|
     | (i) s̄ | +1.5 | +3.5 | **+4.7** | −2.0 | −2.0 |
     | (iv) with interference | +0.8 | +1.6 | **+5.7** | −3.4 | −2.9 |

   - Restricted to L ≤ 2 (at L = 4 a dip from 0.012 to 0.01 already counts as absorbed), the largest shift is still 2.4 decades. **Verdict: FAIL.**
3. **(c) is not supported either.** The initialization quantities ‖∇z‖² and the mean cosine are exactly what candidates (ii)–(iv) add, and they don't fix the shifts.
4. **Partial support for the η/B scaling.** Across 34 groups, the slope of the threshold η* against B has median 0.83 (s̄ ∝ η/B predicts 1).
5. **Group size at a fixed budget.** Larger G lowers the threshold η*, more so for GRPO. Final success barely changes.
6. **Absorption is early.** 88% of absorptions happen in the first 500 of 5,000 steps. In absorbing runs, pass@64 also drops, e.g. from 0.81 to 0.26 for the MLPs.

### Why it failed: two different routes to absorption

The post-hoc analysis (base findings, section 4b) and the four-way review found that the levels fail in two different ways. No single step variable can describe both.

| | Route 1: large kicks | Route 2: drift onto the majority answer |
|---|---|---|
| Where | `lin`, `m64`, `m256` | `c05`, `c09` (correlated features) |
| Needs a large η? | Yes; nothing at small η | No: `c09` absorbs about 20% already at η = 0.003, flat over 7 decades of η |
| Threshold moves with B? | Yes, close to η/B | Hardly (slope 0 for c05) |
| Does the first-step drift predict which prompts? | Weakly (AUC 0.62–0.77) | Almost perfectly (AUC 0.88 for c05, 1.000 for c09) |
| What it is | Each sampled prompt gets a lump step η/B; too big a lump knocks it out. This is the Suk & Duan type of failure | Deterministic, set by the geometry. The step size only sets how fast it happens |

**What route 2 really is.**
- The review rebuilt the `c09` tasks and found that **every absorbed prompt had a minority answer**: 0 majority-answer prompts were absorbed across 10 tasks.
- With correlated features, the hidden argmax rule gives the answer classes unequal sizes. The ~0.197 plateau in absorbed fraction is exactly one task's minority share.
- So the drift in route 2 is not interference in general. It is the majority answer class pulling minority-answer prompts onto its answer.

### What we ran to follow this up

1. **P1 (v3a, pre-registered):** class imbalance controlled directly.
   - Setup: the share m of the majority answer was set while the feature geometry was held fixed. Linear and MLP, ρ ∈ {0.5, 0.9}, SGD and Adam, exact and sampled updates, 10 task seeds.
   - Result: persistent absorption rises with m in **16 of 16 cells** (task-level p ≤ 0.0098). It takes 54–100% of minority prompts at m = 0.9 and about 0 when the classes are balanced.
   - Supervised training on the same tasks absorbs nothing.
2. **P2 (v3a):** route 1 checked at equal mean drift.
   - Absorption falls as B grows (bigger B means smaller kicks) in 3 of 4 levels.
   - Response sampling adds absorption only at B = 8.
3. **T11 (v3d), who pushes whom.**
   - Setup: from one state, one step along only the majority's part of the update, only the minority's part, or both.
   - Result: the majority's part lowers the minority, and the minority's own part raises it. The majority wins by 3.6–4.8× at m = 0.7 and 10.7–17.1× at m = 0.9, in every task.
   - So it is **active pushing, then starvation**. Once p is near 0, the minority's own signal vanishes, since ∇p ∝ p.
4. **The odds law** (`analysis/theory/odds_law.md`).
   - At the first step, that push ratio is exactly R = c·m/(1 − m). c comes from the feature geometry: how similar prompts are across vs. within classes.
   - The minority starts falling once m > 1/(1 + c).
   - Verified with nothing fitted on 1,080 runs, and proved in Lean.
   - It describes the first step only, not the outcome.
5. **T12 and T18 (v3d, v3g): plateau, not a second resting point.**
   - Remove the majority's reward and the minority comes back. The deeper it fell, the longer it takes.
   - Under SGD at ρ = 0.9 it stays down for over 10× the training horizon. At ρ = 0.5 it often comes back.

### What this means for H3

| Part | Verdict |
|---|---|
| (a) One effective step sets the outcome | **Fails.** It works within route 1 (the two MLP widths line up under s̄) but not across routes |
| (b) No collapse without sharing | **Holds** (by construction; 2,268 runs) |
| (c) Initialization quantities carry L and p₀ | **Not supported**; the average gradient overlap does not tell the routes apart |
| "Interference causes absorption" (the spirit of H3) | **Holds, in a sharper form.** What matters is not the average overlap between prompts but its **class structure**: whether prompts that look alike want the same answer. A majority answer class pushes a correct minority class down, more completely as its share grows |

**For the report.**
- H3 is stated as planned and reported as failed.
- The two routes come next.
- Route 2 becomes the main result: P1, T11, the odds law and the plateau.
- Suk & Duan's step-size picture stays relevant to route 1 only.
- Prior work to credit:
  - Schaul et al. 2019 (ray interference: loss of correct contexts through shared parameters, plateaus);
  - Li, Pandey & Chen 2026 (the first-step decomposition and its sign at m = 1/K).
- What is ours:
  - class share as a controlled driver for a correct minority under GRPO;
  - the kernel correction c;
  - the levers compared at matched progress.

### Original H3 and restated H3 (kept apart in the report)

| | Original H3 (pre-registered, milestone) | Restated H3 (after the base batch) |
|---|---|---|
| Claim | Stability is set by one effective step, s = ηW_G(p)p(1−p)/B, plus interference through shared parameters | With shared parameters, a prompt's progress is set by the drift that other prompts' updates cause, (η/B)Σ_x W_G(p_x)⟨∇p_y, ∇p_x⟩, not by its own step. When prompts form answer classes, a majority class pushes a correct minority down with strength c·m/(1 − m) and absorbs it |
| Status | **Failed** (base batch, 13,608 runs). Reported as failed | Exploratory evidence: H2 cross-share (> 90% of drift from other prompts; full drift R² 0.999), T11, the odds law, P1. **Confirmatory test: A1** (P1 on 20 fresh tasks, pre-registered in `PREREG_v3d_addendum9.md`) |
| Runs still needed | None. Its runs are done | A1 (required). A2, A3 and A5 confirm the levers and warm start (wanted, not required for H3 itself) |

### Compute for the runs still needed

Per-run times measured on this laptop (single thread, 2026-11-25), on jobs from the `v3o` job list:

| Job | Steps | Seconds |
|---|---|---|
| linear, exact, SGD | 1,000 | 28 |
| linear, exact, SGD | 10,000 | 94 |
| linear, sampled, Adam | 5,000 | 64 |
| MLP, sampled, SGD | 3,333 | 58 |
| MLP, sampled, Adam | 5,000 | 79 |
| MLP, exact, Adam | 5,000 | 87 |
| A2: linear, exact, SGD, forward KL | 20,000 | 184 |
| A2: MLP, sampled, SGD, forward KL | 20,000 | 104 |
| A5: linear, exact, SGD | 10,000 | 32 |

A1 jobs work out to about 20 s of setup plus 7 ms per step (linear) or 12–15 ms per step (MLP). Some longer jobs finish faster than that, apparently because of the runs' early-stop rule.

| Batch | Runs | Steps per run | Estimated CPU-hours |
|---|---|---|---|
| A1 (required) | 9,280 | 1,000–20,000 (mean 3,471) | about 140 (6 jobs timed; range 120–160) |
| A2 (forward KL) | 640 | 20,000 | about 25 |
| A3 (pass@4) | 360 | 20,000 | about 15 (not timed; taken from A2) |
| A5 (warm start) | 160 | 10,000 | about 2 |
| **All of `v3o`** | 10,440 | | **about 180** |

- **This laptop** (4 logical cores): A1 alone is about 35–50 hours of wall clock with every core busy. Not practical.
- **A cluster:** the jobs are independent and single-threaded, so they fit a CPU job array. On 64 cores, A1 is about 2–3 hours and all of `v3o` about 3 hours. Follow the DRAC rules in `CLAUDE.md`: a virtualenv, absolute paths, the `_cpu` account suffix, flushed logs. `experiments/run.py --jobs <list> --index <i>` already runs one job per index and skips finished outputs, so it maps directly onto `--array`.
- **Check first:** the `v3o` batch was pre-registered and its job list committed on Nov 24 "before launch". Ankush may already have run it on the lab machines (`~/rlvr-data`). If so, the compute needed is zero, and only the analysis remains.

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
