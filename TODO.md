# TODO

What the project still needs, from now to the final report.
It builds on the proposal (`latex/proposal_2_azaan.tex`) and the instructor's feedback on it.
Items marked **Decide** need a team decision. Where there is a suggestion, it is only a starting point.

---

## 1. Project direction

The instructor's main comment on the proposal was:

> *"trying to do too much ... at the moment, it is a list of seemingly unrelated experiments. fix the setting - bandit or token-level MDP, test the effect of the algorithm hyper-params, do the group-size scaling law separately. relate everything."*

They also suggested a framing:

> *"test the theoretical predictions on a toy task where we can control all the variables and perform proper ablations."*

So the project becomes two studies with one storyline, in place of E1–E6.

### Storyline (suggested)

All three core papers analyze the same update: a group of G responses, an advantage computed within the group, and one policy-gradient step. Each paper predicts a different property of that update:

| Paper | What it predicts | Hyperparameter it is about |
|---|---|---|
| Mroueh (2025) | **Where** training ends up: the fixed point p* when a KL penalty is used | β, KL direction |
| Suk & Duan (2025) | **Whether** training is stable: the step-size threshold, which scales with response length | η, L |
| Thrampoulidis et al. (2026) | **What** the advantage rule actually optimizes: a surrogate reward | advantage type |

Each prediction is derived with exact expectations (G → ∞) or exact optimization.
The question that ties the project together:

> **Which of these predictions survive finite groups and stochastic gradient steps, and where does each one break?**

### The two studies

- **Study A: algorithm hyperparameters (main study).**
  - One fixed setting.
  - Sweep β and the KL direction, η, and the advantage type.
  - Compare each result against the matching paper's prediction.
- **Study B: group-size scaling (separate study).**
  - Set the success rate p and the group size G independently.
  - Measure how learning speed and the fraction of informative groups depend on G·p(1−p), which is the Bay & Yearick law.
  - The plateau under sparse rewards (old E5) becomes the low-p end of this study, not a separate experiment.

### Decisions

- [ ] **Decide the setting.** Suggestion: the token-level synthetic task (`synthetic_rl/`), used throughout.
  - With L = 1 it reduces to a bandit (one token out of K), but with a shared network. That is the regime for the Mroueh check.
  - With L > 1 we can study the step-size threshold. Note that a tabular bandit can't show a threshold, because zero-sum advantages mean it never loses success.
  - Describe it in the report as "a factorized token-level policy: L tokens, each drawn independently given x, with no autoregressive structure". The instructor asked whether there is autoregressive structure, so this should be stated plainly.
- [ ] **Decide what to cut or make optional.**
  - E5 is folded into Study B.
  - E6 (small LLM) becomes optional or a stretch goal.
  - E4's multi-solution diversity study becomes optional. The advantage-rule comparison stays in Study A.
- [ ] **Decide the sweep ranges.** Suggestion: β, η, G, p₀, L, and seeds.
  - Use at least 3 seeds per configuration.
  - Run a coarse grid first, then a finer grid near where each prediction breaks.
- [ ] Write a one-page **experiment spec** (`docs/experiment_plan.md`). For each sweep, record:
  - the prediction being tested and its formula;
  - the variables swept;
  - the metric measured;
  - the plot to produce.

  This spec is also the "how the papers inform the project" part of the milestone.

### Points from the feedback to define precisely (milestone and final report)

- [ ] **The predicted p\*.**
  - Mroueh's fixed point: logit p\* = logit p_ref + Ω/β.
  - Our finite-G version, with Ω_G for each advantage type.
  - Say whether it holds for each prompt or only on average.
- [ ] **The "inner optimization problem".** Mroueh solves the KL-regularized problem exactly at each iteration, whereas we take one SGD step on θ. Make that difference explicit, since it is one of the assumptions we test.
- [ ] **The gradient estimator.** ∇J ≈ (1/G) Σᵢ Aᵢ ∇log π_θ(yᵢ|x) − β∇KL, with the formula for each advantage.
  - Mean-only is REINFORCE with a group-mean baseline.
  - Pass@k is both an evaluation metric and an advantage rule; keep the two uses apart.
- [ ] **Response length symbol.** Use one symbol, T or L, throughout. Suk & Duan use T.
- [ ] **p₀ and the "base model".** In the synthetic task, a supervised warm-start produces the base and reference policy. In the bandit, set the logits directly.
- [ ] **Reference for the forward-vs-reverse KL comparison.** The instructor suggested arXiv:2505.17508; read it, and if it fits, add it to `ref.bib`.

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

**File:** `latex/milestone.tex`, which builds to `docs/milestone.pdf`. Today the skeleton contains only TODO placeholders.

### Structure changes to `milestone.tex`

- [ ] **Add a synthesis section** after the three reviews, for example "How the three papers relate and inform our project". The skeleton has none, and it is the part the instructor grades on.
  - Base it on the table in Section 1 of this file: same update, three questions (where, whether stable, what is optimized).
  - Point out shared assumptions: exact expectations or G → ∞, and binary rewards.
  - Point out where the papers differ: Mroueh solves the inner problem exactly, Suk & Duan analyze step sizes, Thrampoulidis analyzes the population objective.
  - Point out the gap that all three leave: finite-G, stochastic training. Our project tests exactly this.
- [ ] **Update the introduction comments.** They still map the papers to E2, E3 and E4. Rewrite them in terms of Study A and Study B.
- [ ] **Keep Progress, Updated Plan and Contributions inside `\iffalse`** unless the page budget allows. If there is room, one short paragraph on project status could go in the introduction.
- [ ] **Map the skeleton's four paragraphs per paper to the required elements:**
  - "Setting and assumptions" covers the problem and the assumptions;
  - "Main results" covers the main idea or result;
  - "Limitations" covers the limitations;
  - "Connection" covers how the paper informs our project.

  Keep each review to about half a page, so the three reviews plus synthesis fit in 3 pages.
- [ ] **Write the shared notation block** once: x, π_θ, r ∈ {0,1}, p, G, Aᵢ, β, π_ref. The reviews should not redefine these.

### Writing (per the policy: we write, Claude fact-checks and proofreads)

- [ ] Mroueh review (Azaan).
- [ ] Suk & Duan review (Ankush).
- [ ] Thrampoulidis et al. review (Simon).
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

### Infrastructure (needed by both studies)

- [ ] **Warm-start** in `synthetic_rl/`: a short supervised pretraining so p₀ can be set independently of K^(−L). It must run before the `Trainer` is created, because the reference policy is frozen at that point.
- [ ] **Sweep runner**:
  - [ ] configuration in, results out (CSV or JSON per run), with seeds;
  - [ ] SGD as an option alongside Adam, since the step-size study needs plain SGD;
  - [ ] results saved under `results/`, which can be gitignored if large.
- [ ] **Plotting script**: learning curves, final p versus predicted p\*, and success-vs-η curves.
- [ ] **Cluster job scripts**, if the sweeps outgrow a laptop. Follow the DRAC rules in `CLAUDE.md`: a virtualenv, absolute paths, and flushed logs.

### Study A: hyperparameters

- [ ] **β and KL direction.**
  - Run with L = 1.
  - Compare the measured fixed point with logit p\* = logit p_ref + Ω_G/β, across G and advantage types.
  - Repeat with the forward KL. Our derived prediction for it is (1−p_ref)/(1−p) − p_ref/p = Ω_G/β; verify that numerically first.
- [ ] **η and L.**
  - Sweep η on a log grid with SGD.
  - Find the threshold where success stops increasing or collapses.
  - Check how the threshold scales with L and with p₀.
  - Optional: a token-mean log-probability option and mixed-length prompts.
- [ ] **Advantage type.**
  - Compare GRPO, mean-only, RLOO and pass@k at fixed η and β.
  - Port the pass@k advantage from `main.py` into `calculate_advantages`.
  - Optional: a multi-solution task variant. This also needs `success_probability` to support more than one correct sequence.

### Study B: group size

- [ ] Extend `run_pilot.py` (bandit, p₀ × G × seeds) into the full sweep, saving results to files.
- [ ] Measure steps to reach a target success rate, and the informative-group fraction. Compare both with 1 − p^G − (1−p)^G and with the G·p(1−p) law.
- [ ] Check whether the results hold in the token-level setting with L = 1.
- [ ] Optional: use the per-prompt success-rate spread to test Nie et al.'s heterogeneous-p effect.

### Housekeeping

- [ ] Pin the `torch` version in `requirements.txt`.
- [ ] Fix the run command in the `synthetic_task.py` docstring to `python -m synthetic_rl.synthetic_task`.
- [ ] Delete the commented-out `logits` and `log_ref` lines in `main.py`.
- [ ] Add tests for the warm-start and the pass@k advantage once they exist.

---

## 4. Final report (later)

- [ ] Present Study A and Study B under the one storyline from Section 1.
- [ ] For each prediction, state where it held and where it broke, with plots.
- [ ] Optional: a small-LLM check (Qwen2.5-0.5B on GSM8K), only if time allows.
