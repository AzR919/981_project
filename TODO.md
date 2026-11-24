# TODO

What the project still needs, from now to the final report.
The plan follows `docs/project_guide.html`; this file is the checklist version of it.
Items marked **Decide** need a team decision. Where there is a suggestion, it is only a starting point.

---

## 1. Project direction

The instructor's main comment on the proposal was:

> *"trying to do too much ... at the moment, it is a list of seemingly unrelated experiments. fix the setting - bandit or token-level MDP, test the effect of the algorithm hyper-params, do the group-size scaling law separately. relate everything."*

They also suggested a framing:

> *"test the theoretical predictions on a toy task where we can control all the variables and perform proper ablations."*

### Setting: one synthetic task, no LLM

- Everything runs on the synthetic task family in `synthetic_rl/`. The small-LLM experiment (old E6) is dropped.
- The bandit is the synthetic task with L = 1 and **tabular** parameters (a one-hot prompt index into a linear layer). So "bandit vs. token-level" becomes two knobs of one task:
  - response length L;
  - parameter sharing: tabular, or a shared MLP of some width.
- In the write-up, call it "a factorized token-level policy: L tokens, each drawn independently given x, with no autoregressive structure". The instructor asked about autoregressive structure, so say this plainly.

### Storyline: one equation, three readings

For binary rewards and any advantage rule computed from the group's rewards, the expected update on a prompt is

> E[ĝₓ] = W_G(pₓ) ∇pₓ,  where W_G(p) = E[A | r=1] − E[A | r=0] is the **advantage gap**.

Each core paper reads one property off this equation, and the hyperparameters act through a few effective quantities:

| Role | Governed by | Paper it generalizes | Hypothesis |
|---|---|---|---|
| **What** is optimized | the shape of W_G(p): F_G = ∫W_G | Thrampoulidis et al. | H2 |
| **Where** training stops | W_G(p*) balancing β × the KL pull | Mroueh | H4 |
| **Whether** it is stable | the effective step s = (η/B)·W_G·p(1−p), times interference between prompts | Suk & Duan | H3 |
| **How noisy** it is | G, B and the informative fraction φ = 1 − p^G − (1−p)^G | Bay & Yearick, Nie et al. | H5 |

Key pilot finding: the main failure is **absorption**. Prompts get pushed to a confident wrong answer (p ≈ 0), after which no group contains a correct sample. With tabular parameters and β = 0, p never decreases, so collapse needs parameter sharing.

Credit: the scalar-times-∇p form is in Davis & Recht (2025), Mroueh (2025) and Thrampoulidis et al. Do not write that we "introduce" the identity. See section 1.5 of the guide for safe phrasing.

### Hypotheses

| | Hypothesis | Status |
|---|---|---|
| H1 | The identity holds in every configuration, and the exact equivalences hold along whole runs (RLOO at η = mean-only at ηG/(G−1); GRPO = RLOO at G = 2) | nearly done (checked on one MLP) |
| H2 | Group size changes what GRPO optimizes: progress per prompt follows W_G(p) | needs warm-start |
| H3 | Stability is set by one effective step and by sharing (**main study**) | main study |
| H4 | KL sets where training stops (finite-G p*) and raises the stability threshold, more so for forward KL | tabular ready |
| H5 | Group size, budget and escape from p ≈ 0 (separate study, as the instructor asked) | pilot exists |

### Decisions

- [ ] **Decide** which of H2 and H5 to keep if time is short. H3 and H4 are the core.
- [ ] **Decide** the default task size. Suggestion: K = 3 or 4, L ∈ {1, 2, 4}, d = 16, so K^L stays small enough for exact gradients.
- [ ] **Decide** one symbol for response length: L, or T as in Suk & Duan.
- [ ] **Decide** the sweep ranges per hypothesis (guide section 8). Use at least 3 seeds and plain SGD; Adam would hide the step-size effects.

### Points to define precisely in the write-up

Most of these are now worked out in the guide (sections 3 and 6). They still need to go into the milestone and report in our own words.

- [ ] **The predicted p\*.** Reverse KL: logit p\* = logit p_ref + W_G(p\*)/β. Forward KL: (p\* − p_ref)/(p\*(1 − p\*)) = W_G(p\*)/β. Say that these are per prompt and tabular; with sharing, the prompts are coupled.
- [ ] **The "inner optimization problem".** Mroueh solves the KL-regularized problem exactly at each iteration; we take one SGD step. Guide section 3.6 has the comparison table.
- [ ] **The gradient estimator.** ĝ = (1/G) Σᵢ Aᵢ ∇log π_θ(yᵢ|x) − β∇KL, with each advantage rule. Pass@k is both a metric and an advantage rule; keep the two uses apart.
- [ ] **p₀ and the "base model".** A supervised warm-start produces the base and reference policy. In the tabular case, set the logits directly.
- [x] **Reference for forward vs. reverse KL.** The instructor suggested arXiv:2505.17508 (Zhang et al. 2025, *On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning*). It fits.
- [ ] Add Zhang et al. (arXiv:2505.17508) to `latex/ref.bib`.

---

## 2. Milestone: literature review (10%)

**Requirements**
- Maximum 3 pages, excluding references. Use the NeurIPS template, as the proposal did.
- It must include:
  - title and team members;
  - the project topic;
  - summaries of 3 papers, each covering the problem it studies, its main idea or result, and the assumptions or limitations that matter for our project;
  - how the three papers relate to each other;
  - how they inform our project.
- The goal is **synthesis**, not three independent abstract summaries.

**File:** `latex/milestone.tex`, which builds to `docs/milestone.pdf`.

### Structure changes to `milestone.tex`

- [ ] **Add a synthesis section** after the three reviews, for example "How the three papers relate and inform our project". It is the part the instructor grades on.
  - Base it on the "one equation, three readings" table above: one update, three questions (what is optimized, where it stops, whether it is stable).
  - Shared assumptions: binary rewards, exact expectations or G → ∞, and tabular or idealized updates.
  - Where they differ: Mroueh solves the inner problem exactly; Suk & Duan analyze step sizes with exact gradients; Thrampoulidis et al. analyze the population objective.
  - The gap all three leave: finite G, stochastic steps and shared parameters. Our project tests exactly this.
- [ ] **Update the introduction comments.** They still map the papers to E2, E3 and E4. Map them to H4 (Mroueh), H3 (Suk & Duan) and H2 (Thrampoulidis et al.).
- [ ] **Keep Progress, Updated Plan and Contributions inside `\iffalse`** unless the page budget allows. If there is room, one short paragraph on project status could go in the introduction.
- [ ] **Map the skeleton's four paragraphs per paper to the required elements:**
  - "Setting and assumptions" covers the problem and the assumptions;
  - "Main results" covers the main idea or result;
  - "Limitations" covers the limitations;
  - "Connection" covers how the paper informs our project.

  Keep each review to about half a page, so the three reviews plus synthesis fit in 3 pages.
- [ ] **Write the shared notation block** once: x, π_θ, r ∈ {0,1}, p, G, Aᵢ, β, π_ref, W_G. The reviews should not redefine these.

### Writing (per the policy: we write, Claude fact-checks and proofreads)

- [ ] Mroueh review (Azaan).
- [ ] Suk & Duan review (Ankush). Pin down exactly how their threshold depends on p and on T (the 1/T scaling).
- [x] Thrampoulidis et al. review (Simon): drafted in c89ba51.
  - [ ] Wording pass: "The gradient of the P_k(x), pass@k objective", "more weight to", "use it to compare them".
  - [ ] Check two claims against the paper: the "infinite group limit with k fixed", and that GRPO_k is biased.
- [ ] Synthesis section (all; one person drafts it).
- [ ] Introduction and project topic (one paragraph, in the new framing).
- [ ] Abstract (written last).

### Checks before submitting

- [ ] Fact-check each review against its paper. Check especially every statement about assumptions, and every claimed scaling (1/T, (1−p), p\* > p_ref).
- [ ] Confirm the PDF is at most 3 pages before the references.
- [ ] Confirm all citations resolve and no references are unused.
- [ ] Do a final spelling and grammar pass.

---

## 3. Code and experiments

### Code to add to `synthetic_rl/`

- [ ] **Tabular parameterization** (one-hot prompt index into a linear layer). Needed for H1, H3, H4 and H5. Check that it reproduces `main.py` at L = 1.
- [ ] **Warm-start** to a target p₀. Needed for H2, H3 and H4. It must run before the `Trainer` is created, because the reference policy is frozen at that point.
- [ ] **Exact W_G(p) and diagnostics**:
  - per-prompt p;
  - ‖∇z_x‖² and the gradient-overlap matrix between prompts (the interference);
  - absorbed fraction (p < 0.01) and informative fraction (measured vs. φ).

  `omega()` in `main.py` already computes W_G for GRPO.
- [ ] **Pass@k rules** (RLOO_k, GRPO_k), ported from `main.py` into `calculate_advantages`. Needed for H2.
- [ ] **Forward KL** as an option alongside reverse KL. Needed for H4.
- [ ] **Sweep runner**: configuration in, per-prompt logs out (CSV or JSON), with seeds and SGD. Save results under `results/`, gitignored if large.
- [ ] **Plotting script** for the collapse plots in guide section 8.
- [ ] **Cluster job scripts**, if the sweeps outgrow a laptop. Follow the DRAC rules in `CLAUDE.md`: a virtualenv, absolute paths, and flushed logs.

### Experiments, in order

1. [ ] **H1 + tabular option.**
   - Sampled vs. predicted mean gradient across L, sharing, rule and G.
   - Same-seed trajectories: RLOO at η vs. mean-only at ηG/(G−1); GRPO vs. RLOO at G = 2.
   - Small, and it validates the code.
2. [ ] **H4(a), tabular.** Measured vs. predicted p\* on the diagonal.
   - Example predictions for p_ref = 0.05, β = 2, GRPO, reverse KL: 0.109 at G = 4, 0.157 at G = 16, about 0.17 as G → ∞.
   - Mean-only as G → ∞: 0.0798 for reverse KL, 0.0916 for forward KL.
3. [ ] **H3, the main study.**
   - Sweep η on a log grid; B and G, including fixed-budget pairs; the rule; L ∈ {1, 2, 4}; sharing (tabular, MLP widths 16/64/256, correlated vs. independent features).
   - Collapse plot: absorbed fraction against s̄ × interference.
   - Start with the fixed-budget pairs and the tabular-vs-shared contrast; they are the sharpest tests.
4. [ ] **H4(b, c).** KL as a stabilizer: reuse H3's grid with β > 0, for both KL directions.
5. [ ] **H5 and H2, as time allows.**
   - H5: p₀ × G under a fixed budget N = BG, tabular then shared. Extend `run_pilot.py`.
   - H2: warm-started MLP with mixed-difficulty prompts; per-prompt progress against W_G(p).

### Housekeeping

- [ ] Pin the `torch` version in `requirements.txt`.
- [ ] Fix the run command in the `synthetic_task.py` docstring to `python -m synthetic_rl.synthetic_task`.
- [ ] Delete the commented-out `logits` and `log_ref` lines in `main.py`.
- [ ] Add tests for the tabular option, the warm-start, the pass@k rules and forward KL once they exist.
- [ ] Move the pilot scripts into the repo (for example `pilots/`) so the numbers in guide section 7 can be reproduced.

---

## 4. Final report (later)

- [ ] Present H1–H5 under the one-equation storyline of Section 1.
- [ ] For each hypothesis, show its collapse plot, and state where it held and where it broke.
- [ ] Use the "What we could conclude" section of the guide as the outline for the discussion.

---

## Side note: status after the experiment batches (2026-11-24)

A summary of the work merged in #4–#9, compared with the plan above. Not yet folded into the checklist.

### Who did what

- **Simon:**
  - the shared framework (`synthetic_rl/policy.py`, `trainer.py`, `synthetic_task.py`);
  - the tabular option of the policy;
  - `tests/test_tabular_bandit.py`, which checks the tabular trainer against the NumPy bandit in `main.py` (passes on current `main`);
  - `experiments/h1.py`.
    - Setup: one prompt, K = 4, L = 1, G = 4.
    - Result: the sampled ‖ĝ‖² tracks ‖W_G ∇p‖² over the whole run for mean-only, RLOO and GRPO (`results/h1_plots/`).
    - Caveats: it compares norms only, not directions, and the sampled side carries a small variance bias. It covers one cell of the planned grid. Best used as the H1 figure, with Ankush's vector test as the statistics.
- **Ankush:**
  - the experiment runner and lab launcher;
  - the base and stretch batches;
  - four-way AI review, then plan v3 with pre-registered P1–P5;
  - the v3a–v3c batches (about 24k runs) and Phase A re-analyses;
  - the `llm/` track (C0 pilot, model probes; C1 trained but not analysed).
  - Details: `analysis/review/SUMMARY.md`, `analysis/review/NEXT_PLAN.md`, `analysis/v3*/SUMMARY.md`.

### Policies: plan vs. code

- **One policy class**, `architecture="linear"` (logits = W·x, no bias) or `"mlp"` (two ReLU layers).
- **Tabular = linear + one-hot prompts.** Each prompt owns a column of W, so nothing is shared. Simon's tabular policy is this case; his H1 script uses a single prompt.
- **The levels in `experiments/grids.py` are the planned sharing ladder**, with a `lin` rung added:

  | Level | Policy | Prompt features | Sharing |
  |---|---|---|---|
  | `tab` | linear | one-hot | none |
  | `lin` | linear | Gaussian, d = 16 | shared W |
  | `c05` / `c09` | linear | correlated, ρ = 0.5 / 0.9 | shared, prompts alike |
  | `m64` / `m256` | MLP | Gaussian | shared, nonlinear |

- **New knob (v3):** answer imbalance m, the share of prompts whose correct answer is the majority class. The v3 batches mostly use L = 1, K = 3.

### Results against the hypotheses

| Hypothesis | Result |
|---|---|
| H1 | ✅ 414 runs, vector test, cosine ≥ 0.989; calibrated tighter test (T7) |
| H2 | ✅ Measured W within 1% of exact W_G. GRPO moves toward 2 arcsin√p as G grows: λ = 0, 0.015, 0.22, 0.63 at G = 2, 4, 16, 64 |
| H3 | ❌ **No single effective-step law:** the best candidate is off by up to 4.7 decades on held-out levels |
| H4 | ◐ (a) p\* matches to about 1e-4. (b) The KL-stabilizer slope failed. New: KL fixed points are bistable at G ≥ 16 under SGD (extends Mroueh) |
| H5 | ◐ The first table was a tie-break artifact. Rerun: best G = 2 (tabular), often at the η-grid edge, so undetermined |
| New P1 | ✅ With shared features, GRPO absorbs the **minority-answer class**, more completely as m grows. It happens at small η under the deterministic flow, under SGD and Adam. Absorption is ≈ 0 when classes are balanced, and supervised training does not absorb |
| New P2 | ◐ Per-prompt kick size η/B matters. Response sampling adds absorption only at B = 8 |
| Heckel (T9) | ◐ Helps at small η; doubles absorption with large steps |
| Pass@k rules (v3c) | ✅ Prevent minority collapse; k = 2 with no speed cost |
| Huang | Dropped: the collapse needs graded rewards; we use binary |

### What changed in the story

- **Still holds:** tabular never absorbs (p never decreases, 2,268 runs), and collapse needs parameter sharing.
- **Changed:** how sharing causes collapse.
  - Mainly mean drift pulling minority-answer prompts to the majority answer. There is no step-size threshold, so this is not Suk & Duan's instability.
  - Plus a second route: large per-prompt kicks at small batches.

### Decisions and open items

- **Keep the report toy-only, on the tabular → linear → MLP ladder.**
  - The instructor asked for a fixed setting, and every confirmed result is a toy result.
  - The LLM track brings its own confounds: sampler/trainer mismatch, full fine-tuning vs. LoRA, and GSM8K contamination.
  - At most, report C1 as a short transfer note or as future work.
- **Possible report outline, up the ladder:**
  1. Tabular: exact theory holds (H1, H2, H4) and nothing collapses.
  2. `lin`: large kicks can knock prompts out.
  3. Correlated linear and MLP: minority-answer collapse.
  4. Fixes: pass@k rules prevent it; Heckel weights only partly help.
- **Credit:** Davis & Recht already prove the finite-G identity and the finite-G surrogate. The milestone said "infinite groups … we compute W_G for finite G"; fix this in the final report. Our additions are the calibrated test and the λ(G) measurement.
- **Related work:** read Schaul et al. 2019 ("ray interference") before claiming the interference result is new.
- **Bibliography:** add Zhang et al. (2505.17508) and the missing citations listed in `analysis/review/c.md`.
- **Team understanding:** make sure everyone understands the AI-assisted analysis before writing. Address AI use at the end.
