# Ankush: experiments

Every experiment in the project's experiment index (`docs/EXPERIMENTS.md`), written up in the format of
`docs/Simon.md`: why it was run, what I did, the fixed settings, and the result with its status and a link to the full
write-up. 82 entries, in the order they were run. Three pilots by Azaan and Simon are included for completeness and
marked as theirs. The battery A tests (v3o) get one entry each, because each has its own grid and statistic.

**Promised vs extra:** every entry carries a scope line. Search this page for `Scope: Promised`, `Scope: Core follow-up` or `Scope: Extra` to filter.
- **Promised** (21): in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).
- **Core follow-up** (31): not in the milestone; grew out of a promised result, and the report's toy story relies on it.
- **Extra** (30): beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Shared conventions:**
- **Toy simulator defaults** (`DEFAULTS` in `experiments/run.py`): $K = 3$, $d = 16$ Gaussian features, answers set
  by a fixed linear rule, 512 training prompts, plain SGD, $\beta = 0$. The MLP has two ReLU layers. A setting not
  listed in an entry's table is the default.
- **Absorbed** (toy): a prompt's success stays below $10^-3$ over the last 20% of steps, among prompts that start
  at or above $10^-2$.
- **LLM runs:** GRPO on GSM8K, with vLLM sampling in the same process (temperature 1, at most 512 new tokens), one
  update per batch, one RTX 4090 per run. Tags `q_NN` are line NN+1 of `round1_queue.txt`; tags `r2_NN` are line NN+1
  of `queue.txt`. Code commits come from each run's `COMMIT` file.
- **Commit hashes** from before 2026-11-22 are the old ones: the history was rewritten that day, and the map is in
  PR #8.
- **Raw results:** see "Where the raw results are" in `docs/EXPERIMENTS.md` (Google Drive, access by request).
- **Status labels:** confirmatory, exploratory, refuted, in-sample or known, as in `analysis/FINAL_RESULTS.md`.

## Contents

**Part 1: pilots, the milestone batches and the first pre-registered tests** (31 entries)
- Guide §7 pilots: does the framework point at something real? — *Promised*
- p₀ pilot: group size against starting success — *Promised*
- H1 gradient-norm check: a visual test of the H1 identity — *Promised*
- H1: the advantage-gap identity — *Promised*
- H2: does sharing distort the finite-G surrogate? — *Promised*
- H3: one effective step, sharing and absorption — *Promised*
- H4: KL fixed points, fluctuations and the threshold — *Promised*
- H5: group size at a fixed budget and escape from low p₀ — *Promised*
- S_H1: H1 with a calibrated null and the projected bias — *Promised*
- S_DRIFT: drift or noise? — *Promised*
- S_PRED: predicting which prompts are absorbed — *Promised*
- S_ABL: which condition of the tabular proposition matters? — *Promised*
- S_EQ: two stable KL fixed points — *Promised*
- A1: GRPO's weight between the mean rule and the arcsin surrogate — *Promised*
- A4: lost problems and coverage — *Extra*
- BAY: Bay and Yearick's group-size law — *Promised*
- ADAM: absorption time against η under Adam — *Extra*
- A2, A3, A6: re-scored ablations, H4 slopes, Barakat write-up — *Promised*
- T1 / P1: imbalance at fixed geometry — *Core follow-up*
- T2 / P2: kick size against response noise — *Core follow-up*
- T4: H5 rerun with a deterministic tie-break — *Promised*
- T6: tabular ablations under realistic training — *Core follow-up*
- T8: KL bistability without noise — *Promised*
- T5 / P3: does initial drift predict absorption on balanced tasks? — *Core follow-up*
- T7: bootstrap null for the H1 medium tier — *Promised*
- T9: all-wrong-group fixes — *Promised*
- T1 seed 1: P1 with a second initialisation — *Core follow-up*
- T2 groups: does the B = 8 noise excess shrink with G? — *Core follow-up*
- T6 KL: declines toward or below the reference — *Core follow-up*
- T4 edge: wider η for the T4 grid-edge cells — *Promised*
- T10: pass@k training rules — *Core follow-up*

**Part 2: mechanism, theory, levers, the transformer environment and the final phase** (34 entries)
- T11 (v3d): who pushes whom — *Core follow-up*
- T12 (v3d): plateau or permanent — *Core follow-up*
- T13 (v3d): projection control — *Core follow-up*
- M2 (v3d addendum): is the minority still sinking at 10× the horizon? — *Core follow-up*
- M4 (v3d addendum): imbalance or margin? — *Core follow-up*
- T15 (v3e): threshold in class share — *Core follow-up*
- T16 (v3f): pass@k at matched progress — *Core follow-up*
- T17 (v3g): KL anchor on the minority task — *Core follow-up*
- T18 (v3g): release test — *Core follow-up*
- T19 (v3g): group size and the threshold — *Core follow-up*
- T20 (v3h): correct minority in Li, Pandey and Chen's environment — *Extra*
- T21 (v3i): levers in the transformer environment — *Extra*
- T22 (v3j): junk answers — *Core follow-up*
- v3j2: rerun of lost T22 runs — *Core follow-up*
- T23 (v3k): class-balanced warm start — *Core follow-up*
- T24 (v3l): the late phase, prompt by prompt — *Core follow-up*
- Odds law (theory check) — *Core follow-up*
- Lean proofs (theory check) — *Extra*
- SymPy checks (theory check) — *Extra*
- Late-time probe (theory check) — *Core follow-up*
- Optimizer geometry (theory check) — *Extra*
- Fall model (theory check) — *Core follow-up*
- wsw: replication of Li, Pandey and Chen — *Extra*
- wsw2: replication, seeds 5–9 — *Extra*
- T25 (v3m): exact long-time flow — *Core follow-up*
- T26 (v3n): shared skills — *Extra*
- A1 (v3o): P1 on fresh tasks — *Core follow-up*
- A2 (v3o): forward-KL floor on fresh tasks — *Core follow-up*
- A3 (v3o): pass@4 at matched progress on fresh tasks — *Core follow-up*
- A5 (v3o): warm start on fresh tasks — *Core follow-up*
- G (v3p, T27): forward KL toward a collapsed reference — *Core follow-up*
- H (v3p, T28): Adam's epsilon — *Extra*
- A4 (v3q): T20's share effect on fresh seeds — *Extra*
- J (v3q, T29): transformer environment under small-step SGD — *Extra*

**Part 3: small language models** (17 entries)
- C0: pilot runs before the census — *Extra*
- Model probes: which models fit one GPU — *Extra*
- Census calibration: what an unchanged model looks like — *Extra*
- Llama calibration: learning rate and pool for the second family — *Extra*
- C1 Qwen / P4: does GRPO lose problems beyond the null? — *Extra*
- C1 Llama: the second family — *Extra*
- C1 control: are the held-out declines specific to GRPO? — *Extra*
- C5: the toy's loss drivers in the LLM — *Extra*
- C2 / P5: does a majority procedure collapse a minority procedure? — *Extra*
- C2 matched minority: majority pressure or minority sample count? — *Extra*
- C2 seed replication of the degraded run — *Extra*
- C2 extras: samples per step, a long run, and a supervised control — *Extra*
- SmolLM2 diagnostic: why the SmolLM2 pilots collapsed — *Extra*
- C10: who pushes whom in the LLM — *Extra*
- C9: token-level loss aggregation — *Extra*
- C7 / C8: the KL penalty in the LLM — *Extra*
- C3 and C11: planned, not run — *Extra*

---

# Part 1: pilots, the milestone batches and the first pre-registered tests

## Guide §7 pilots: does the framework point at something real?

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

*Run by Azaan (`docs/project_guide.html` §7); listed here for completeness.*

**Why:** Before writing the hypotheses, Azaan checked that a step-size threshold and the advantage gap $W_G$ show up in real training.

### What I did
1. (7.1) Trained a 64-wide MLP policy with GRPO from a uniform start for 300 steps at each $\eta \in \{0.3, 1, 3, 10, 30\}$ and $L \in \{1, 2, 4\}$, and recorded final training success.
2. (7.2) At $L = 1$ with a fixed budget of 256 samples per step ($B = 256/G$), trained GRPO and the mean rule at a few $(\eta, G)$ pairs and recorded each prompt's end state ($p < 0.01$, $p > 0.99$, entropy).
3. (7.3) For one tabular prompt, computed the exact per-sample signal-to-noise of the correct-logit update for mean-only and GRPO at $p \in \{0.05, 0.2\}$ and $G \in \{2, 8, 32, 128\}$.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | 64-wide MLP (7.1, 7.2); one tabular prompt (7.3) |
| Answers $K$ | $3$ |
| Response length $L$ | $1, 2, 4$ (7.1); $1$ (7.2) |
| Group size and batch | $G = 8$, $B = 32$ (7.1); $B = 256/G$, $G \in \{4, 64\}$ (7.2) |
| Rules | GRPO (7.1); GRPO and mean-only (7.2, 7.3) |
| Optimizer and $\eta$ | Plain SGD; $\eta \in \{0.3, 1, 3, 10, 30\}$ (7.1), $\{1, 10\}$ (7.2) |
| Steps | $300$ |
| Seeds | One seed per configuration |
| KL coefficient $\beta$ | $0$ |
| Absorbed / solved | $p < 0.01$ / $p > 0.99$ |
| Runs | About 20 configurations |
| Pre-registration | None |
| Code | Scripts not in the repo (open item in `TODO.md`); no raw data kept |

### Result
A step-size threshold exists. Failure is absorption: end states are bimodal, with prompts stuck near $p = 0$. GRPO fails earlier than mean-only, e.g. $G = 64$, $\eta = 1$ ends with 49% of prompts below 0.01. Exploratory, single seed. This motivated H3 and H5. See `docs/project_guide.html` §7.

## p₀ pilot: group size against starting success

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

*Run by Simon (`run_pilot.py`); listed here for completeness.*

**Why:** Simon's first look at how $G$ and the starting success $p_0$ affect GRPO, the question that became H5.

### What I did
1. Started a one-question bandit, a softmax over $K = 8$ logits, with the correct answer at $p_0$ and the rest equal.
2. Each step, sampled one group of $G$ answers, scored them with the binary verifier and computed GRPO advantages.
3. Took one exact SGD step on the logits for 2,000 steps, printing $p$, the share of informative groups and the entropy every 200 steps.
4. Repeated for every $(p_0, G, \text{seed})$ in the grid via `run_pilot.py`.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | NumPy softmax bandit over $K = 8$ logits, one question (`main.py`) |
| Starting success $p_0$ | $0.01, 0.1, 0.5$ |
| Group size $G$ | $2, 8, 32$ (one group per step) |
| Advantage rule | GRPO, $\epsilon = 10^{-5}$ |
| Optimizer and $\eta$ | Plain SGD, learning rate $0.1$ |
| Steps | $2{,}000$ |
| Seeds | $0, 1, 2$ |
| KL coefficient $\beta$ | $0$ |
| Runs | $27$ |
| Pre-registration | None |
| Code | `run_pilot.py` on `main` (`5f4cf89`) |

### Result
No write-up found; see `run_pilot.py`. Exploratory.

## H1 gradient-norm check: a visual test of the H1 identity

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

*Run by Simon (`experiments/h1.py`, written up in `docs/Simon.md`); listed here for completeness.*

**Why:** Simon wanted a direct visual check that the sampled gradient's size matches $W_G(p)\nabla p$.

### What I did
1. Started each rule and seed from a fresh one-prompt policy with four answers, so $p_0 = 1/4$.
2. At checkpoints, computed $p$ and $\nabla p$ with autograd and formed the predicted squared norm $\|W_G(p)\nabla p\|^2$ with the exact finite-$G$ weight.
3. Each step sampled 512 groups of 4 and took one SGD step. The tabular run used that step's gradient as the sampled side; the MLP run estimated it at each checkpoint from 500,000 groups.
4. Averaged both squared norms over 10 seeds and plotted them on one log axis.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Bias-free linear tabular policy; MLP option (two ReLU layers, width 128, the `Policy` default) |
| $K$, $L$ | $4$, $1$; correct answer ID `0` |
| Group size $G$ | $4$ |
| Groups per step | $512$ |
| Rules | Mean-only, RLOO, GRPO ($\epsilon = 10^{-5}$) |
| Optimizer and $\eta$ | Plain SGD, $0.4$ |
| Steps | $250$ (checkpoints every 10) |
| Seeds | $0$–$9$ per rule |
| KL coefficient $\beta$ | $0$ |
| Runs | 3 rules × 10 seeds per policy |
| Pre-registration | None |
| Code | `experiments/h1.py` on `main` (`b1c822e`; MLP option `0130bca`); setup in `docs/Simon.md` |

### Result
The sampled norm tracks the predicted norm over the whole run for all three rules. This checks norms only, not directions, in one cell of the grid. Exploratory. Plots go to `results/h1_*_plots/` (not in git).

## H1: the advantage-gap identity

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** The milestone's identity $\mathbb{E}[\hat g] = W_G(p)\nabla p$ is the foundation of every later test, so we checked it on every rule, sharing level and group size.

### What I did
1. Built a policy at fixed parameters (uniform or warm start) and took the first 32 training prompts.
2. Computed the predicted mean update $-\frac{1}{B}\sum_x W_G(p_x)\nabla p_x$ from the autograd Jacobian and the exact $W_G$.
3. Drew $R = 4{,}000$ independent sampled updates (G responses per prompt, binary reward, rule advantage) at the same parameters and averaged them.
4. Compared the two: relative error over Monte Carlo SE, mean squared z-score, share of $|z| > 3$ and cosine.

### Fixed settings
| Setting | Value |
| --- | --- |
| Mode | `gradcheck` |
| Levels | `tab` (one-hot), `lin` (linear, Gaussian features), `m64` (MLP width 64) |
| Start | Uniform; or warm: tabular `spread` ($p$ over logits $[-4, 3]$) for `tab`, supervised `sft_up` to $p_0 = 0.3$ otherwise |
| $K$, $L$ | $3$; $L \in \{1, 2, 4\}$ |
| Group size $G$ | $2, 4, 8, 32$ |
| Rules | mean, RLOO, GRPO, MaxRL, pass@k ($k \in \{2, 4\}$, $k \le G$) |
| Prompts probed ($B$) | $32$ |
| Samples $R$ | $4{,}000$ |
| Pass band | Ratio $\le 1.2$, mean $z^2 \in [0.8, 1.25]$, $|z| > 3$ share $< 1\%$ |
| Runs | $414$ (one per configuration) |
| Pre-registration | Milestone (`docs/milestone.pdf`) |
| Grid and commit | `h1`, batch `base`, `experiments` @ `fd8d511` |

### Result
PASS on group medians (median ratio 0.97–0.99, cosine $\ge 0.989$). The literal per-run band fails in 174 of 414 runs because it judges single noisy statistics. Known (replicates Davis and Recht). See `analysis/base/FINDINGS.md` and `analysis/base/sections/h1.md`.

## H2: does sharing distort the finite-G surrogate?

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** Group size changes what GRPO optimises; we asked whether the measured change in $p$ still follows $W_G$ once prompts share parameters.

### What I did
1. Warm-started each level, then measured the mean one-step change $\Delta p_x$ over $R$ independent full-batch updates from the same parameters.
2. Regressed it on the first-order predicted drift $d_x = \eta\nabla p_x\cdot\text{mean}_j W_G(p_j)\nabla p_j$, and computed the cross share, the part of $d_x$ from other prompts.
3. Paired each rule with the mean rule at the same warm start and compared the binned ratio $\Delta p(\text{rule})/\Delta p(\text{mean})$ with the no-sharing $W_G$ prediction.
4. Ran 200 steps of ordinary training at the same settings to see whether success depends on $G$.

### Fixed settings
| Setting | Value |
| --- | --- |
| Levels | `tab`, `lin`, `m64` |
| Start | `tab`: tabular `spread`; `lin`, `m64`: `sft_up` to $p_0 = 0.4$ |
| $K$, $L$ | $3$, $1$ |
| Group size $G$ | $2, 4, 16, 64$ |
| Rules | mean, RLOO, GRPO, MaxRL, pass@k ($k \in \{2, 4\}$, $k \le G$) |
| One-step part | Full batch (512 prompts), $\eta = 0.5$, $R = 2{,}000$ ($1{,}000$ at $G = 64$) |
| Training part | $B = 32$, 200 steps, $\eta$ = 3 (`tab`), 0.3 (`lin`), 0.03 (`m64`) |
| Seeds | $0, 1, 2$ |
| Pass rule | Tabular on the line ($R^2 \ge 0.95$, slope 0.9–1.1) and RMSE rising with the cross share |
| Runs | $414$ ($207$ one-step + $207$ training) |
| Pre-registration | Milestone |
| Grid and commit | `h2`, batch `base`, `experiments` @ `fd8d511` |

### Result
PASS. Tabular runs sit on the line ($R^2 = 1.000$); with sharing 93% (`lin`) and 97% (`m64`) of each prompt's drift comes from other prompts. The review noted H2 was run as a different question from the promised arcsin shift (done later in A1). See `analysis/base/sections/h2.md`.

## H3: one effective step, sharing and absorption

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** The main milestone claim was that stability is set by one effective step $\bar s = (\eta/B)\,W_G\,p(1-p)$, and that only shared parameters lose prompts.

### What I did
1. Trained every level from a uniform start (all prompts at $p = K^{-L}$) for 5,000 SGD steps over the full $\eta \times (B, G) \times$ rule grid.
2. Logged each prompt's exact $p$ and counted absorbed prompts: started at $p \ge 0.01$, ended below $0.01$.
3. Checked the tabular proposition: no tabular run should ever lower any $p$.
4. Fitted absorbed fraction against four candidate collapse variables ($\bar s$ and interference-weighted versions), with held-out levels, and measured each level's shift in decades.
5. Read off $\eta^*$, the smallest $\eta$ with seed-median absorbed fraction $\ge 0.1$, and its slope in $B$.

### Fixed settings
| Setting | Value |
| --- | --- |
| Levels | `tab`, `lin`, `c05`, `c09` (correlated features, $\rho = 0.5, 0.9$), `m64`, `m256` |
| Start | Uniform |
| $K$, $L$ | $3$; $L \in \{1, 2, 4\}$ |
| $(B, G)$ pairs | $(128, 2), (64, 4), (32, 8), (16, 16), (4, 64), (8, 8), (128, 8)$ |
| Rules | mean, RLOO, GRPO |
| Update | Sampled |
| Optimizer and $\eta$ | Plain SGD; 12 values $0.003, 0.01, 0.03, \dots, 300, 1000$ |
| Steps | $5{,}000$ |
| Seeds | $0, 1, 2$ |
| KL coefficient $\beta$ | $0$ |
| Absorbed | Start $p \ge 0.01$, end $p < 0.01$ |
| Pass rule (collapse) | An interference-weighted candidate beats $\bar s$ on held-out levels and no level shifts by more than one decade |
| Runs | $13{,}608$ |
| Pre-registration | Milestone; verdict rule marked pre-registered in FINDINGS |
| Grid and commit | `h3`, batch `base`, `experiments` @ `fd8d511` |

### Result
Tabular runs never lose a prompt (0 of 2,268 runs with any decrease; PASS). One effective step does not collapse the levels: FAIL, off by up to 4.69 decades; the $\eta^*$ slope in $B$ is 0.83. Refuted as stated. The review later showed `c05`/`c09` absorption is minority-answer collapse. See `analysis/base/sections/h3.md`.

## H4: KL fixed points, fluctuations and the threshold

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** To test where GRPO with a KL penalty settles (Mroueh; Vojnovic and Yun), how much it fluctuates, and whether KL moves the absorption threshold.

### What I did
1. (a) Trained 64 identical tabular prompts started at $p_{\text{ref}}$ for 6,000 steps, and compared the second-half mean $p$ with the stable root of the finite-$G$ fixed-point equation (`theory.fixed_points`).
2. (b) In the same setting at $G = 8$, measured $\mathrm{Var}(\operatorname{logit} p)$ across the 64 replicates and fitted its log-log slope against $\eta/\beta$ (prediction 1).
3. (c) Trained `c05` and `m64` over the full $\eta$ grid at three $\beta$ values and read off $\eta^*$ against $\beta$ ($\beta = 0$ from H3).

### Fixed settings
| Setting | Value |
| --- | --- |
| (a) | Tabular, 64 prompts, $p_{\text{ref}} \in \{0.01, 0.05, 0.2, 0.5\}$, $\beta \in \{0.5, 1, 2, 4\}$, $G \in \{2, 4, 8, 16, 64\}$, $B = 64$, $\eta = 1$, mean and GRPO, 6,000 steps |
| (b) | Tabular, 64 prompts, $p_{\text{ref}} = 0.05$, GRPO, $G = 8$, $B = 64$, $\beta \in \{0.5, 1, 2, 4\}$, $\eta \in \{0.1, 0.3, 1, 3, 10, 30\}$, 6,000 steps |
| (c) | `c05`, `m64`, uniform start, $L = 2$, $G = 8$, $B = 32$, $\beta \in \{0.1, 0.5, 2\}$, 12-value $\eta$ grid, mean and GRPO, 5,000 steps |
| KL direction | Reverse and forward |
| $K$, $L$ | $3$; $L = 1$ in (a), (b) |
| Optimizer | Plain SGD |
| Seeds | $0, 1, 2$ |
| Pass rules | (a) median $|$error$| \le 0.01$; (b) slope against $\eta/\beta$ in 0.8–1.2 for $\eta/\beta \le 2$ |
| Runs | $1{,}968$ ($960$ + $144$ + $864$ in the grid) |
| Pre-registration | Milestone; (b) rule marked pre-registered |
| Grid and commit | `h4`, batch `base`, `experiments` @ `fd8d511` |

### Result
(a) PASS: fixed points match to median error $10^{-4}$, and the guide value 0.109 is reproduced. (b) FAIL: slope 0.40 (forward) and 0.79 (reverse); at fixed $\beta$ the slope in $\eta$ is about 1. Known (Vojnovic and Yun). See `analysis/base/sections/h4.md`.

## H5: group size at a fixed budget and escape from low p₀

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** The instructor asked for a separate group-size study: at a fixed sample budget, which $G$ escapes fastest from a low start?

### What I did
1. Set the starting success $p_0$: exactly for tabular prompts; for `lin` and `m64` by supervised training toward decoy answers (`sft_down`).
2. Trained each $(G, \eta)$ at a fixed budget of $BG = 256$ samples per step for 5,000 steps.
3. Measured samples to target, $256 \times$ the first logged step with mean $p \ge 0.5$, and took the best $\eta$ per configuration.
4. Reported the best $G$ per $p_0$ and the spread of samples to target over $G$.

### Fixed settings
| Setting | Value |
| --- | --- |
| Levels | `tab` (64 prompts), `lin`, `m64` |
| Starting success $p_0$ | $0.01, 0.03, 0.1, 0.3$ |
| $K$, $L$ | $3$, $1$ |
| Group size $G$ | $2, 4, 8, 16, 32, 64, 128$, with $B = 256/G$ |
| Rules | mean, GRPO |
| Optimizer and $\eta$ | Plain SGD; $0.01, 0.03, 0.1, 0.3, 1, 3, 10$ |
| Steps | $5{,}000$ (logged every 25) |
| Seeds | $0, 1, 2$ |
| Pass rule | GRPO's best $G$ falls as $p_0$ rises (tabular) and the mean rule varies by at most $2\times$ over $G$ |
| Runs | $3{,}528$ |
| Pre-registration | Milestone; rule marked pre-registered |
| Grid and commit | `h5`, batch `base`, `experiments` @ `fd8d511` |

### Result
PASS as computed (Spearman $-0.95$), but the review found the best-$G$ table was a tie-break artefact. With the smallest tied $G$ the best $G$ still falls with $p_0$ (32, 16, 4, 2), and the best $\eta$ sat at the grid edge. Rerun as T4. See `analysis/base/sections/h5.md` and `extra/analysis/review/SUMMARY.md`.

## S_H1: H1 with a calibrated null and the projected bias

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** The base H1 per-run band was miscalibrated, so we reran H1 with more samples and tests that know their own noise.

### What I did
1. Reran all 414 H1 configurations with $R = 20{,}000$ samples split into 20 chunks.
2. Loose tier: the base per-rule medians.
3. Medium tier: for each run, built a null for mean $z^2$ from 50 half-against-half splits of its own chunks, counted runs outside the central 95%, and tested the miss rates across splits with $\chi^2$.
4. Tight tier: computed the projected relative bias $b_\parallel$ along the predicted direction with an exact 1-D SE.

### Fixed settings
| Setting | Value |
| --- | --- |
| Configurations | As H1 (levels `tab`, `lin`, `m64`; uniform or warm start; $L \in \{1, 2, 4\}$; $G \in \{2, 4, 8, 32\}$; six rules) |
| Prompts probed | $32$ |
| Samples $R$ | $20{,}000$ in 20 chunks |
| Pass rules | Loose: medians as H1. Medium: miss rate $\le 0.08$ and no split with $\chi^2$ $p \le 0.01$. Tight: every 95% CI inside $\pm 2\%$, no $|b_\parallel/\text{SE}| > 3.9$, pooled bias within $\pm 0.5\%$ |
| Runs | $414$ |
| Pre-registration | Criteria in `experiments/analyze.py`, committed `ab9c270` before results |
| Grid and commit | `s_h1`, batch `stretch`, `fa7f182` |

### Result
Loose PASS. Medium FAIL (miss rate 0.034, but the split by $L$ has $\chi^2$ $p = 0.004$). Tight FAIL because 36 runs have a 95% CI that leaves $\pm 2\%$; no run has $|b_\parallel/\text{SE}| > 3.9$. No bias detected. See `analysis/stretch/sections/s_h1.md`.

## S_DRIFT: drift or noise?

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** To find whether shared-parameter absorption is driven by the mean drift or by sampling noise.

### What I did
1. Trained each configuration three ways from the same start: `sampled` (ordinary GRPO), `expected` (exact expected step on the sampled prompt batch, no response noise) and `expected_full` (exact expected step on all 512 prompts, a deterministic flow).
2. Centred 13 log-spaced $\eta$ values on each level's base $\eta^*$ and measured the absorbed fraction and the median absorption time $t_{50}$.
3. Repeated `sampled` and `expected` at $B \in \{8, 128\}$ (batch block) to fit $\eta^*$ against $B$.
4. Repeated `sampled` and `expected_full` under Adam (Adam block).

### Fixed settings
| Setting | Value |
| --- | --- |
| Levels | `lin`, `c05`, `c09`, `m64` |
| Start | Uniform |
| $K$, $L$ | $3$, $2$ |
| Group size, batch | $G = 8$; $B = 32$ (main), $8, 128$ (batch block) |
| Rule | GRPO |
| $\eta$ grid | 13 values over $\pm 1.5$ decades of base $\eta^*$ (SGD); 13 values around $10^{-2.5}$ (Adam) |
| Steps | $5{,}000$ |
| Seeds | $0$–$4$ |
| Pass rules | `c05`/`c09`: flow absorbs $\ge 0.8\times$ sampled and $t_{50} \propto \eta^s$, $s \in [-1.2, -0.8]$. `m64`: some $\eta$ with sampled $\ge 10\%$ and flow $< 2\%$, and $\eta^*$ slope in $B$ within 0.6–1.2 |
| Runs | $2{,}340$ |
| Pre-registration | Criteria committed `ab9c270` before results |
| Grid and commit | `s_drift`, batch `stretch`, `fa7f182` |

### Result
`c05`/`c09` FAIL as coded; `m64` PASS ($\eta^*$ slope in $B$ = 0.83). The "batch noise" reading was withdrawn by the review: `expected` gives each sampled prompt a 16× larger kick, so the gap measures per-kick size, not noise. See `analysis/stretch/sections/s_drift.md`.

## S_PRED: predicting which prompts are absorbed

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** H3 stretch goal: can the step-0 predicted drift tell, within a run, which prompts will be absorbed?

### What I did
1. Trained each level from a spread start (`sft_mixed`) or a uniform start at four multiples of the base $\eta^*$.
2. Logged each prompt's predicted first-order drift $d_x$ at step 0 and every 500 steps.
3. Scored each prompt by $-d_x$, $-d_x/(p(1-p))$, $-p_x$, $-\|\nabla p_x\|$ and a cross-validated logistic model, and computed the within-run AUC for "absorbed later" (step 0) and "absorbed in the next 500 steps" (checkpoints).

### Fixed settings
| Setting | Value |
| --- | --- |
| Levels | `lin`, `c05`, `c09`, `m64`, `m256` |
| Start | `sft_mixed`, uniform |
| $K$, $L$ | $3$; $L \in \{1, 2\}$ |
| $G$, $B$ | $8$, $32$ |
| Rules | GRPO, mean |
| $\eta$ | Base $\eta^*$ (`BASE_ETA_STAR`) $\times \{0.3, 1, 3, 10\}$ |
| Steps | $5{,}000$; drift every 500 |
| Seeds | $0$–$4$ |
| Pass rule | Drift beats the best single baseline by $\ge 0.05$ AUC with bootstrap CI above 0 |
| Runs | $800$ |
| Pre-registration | Criteria committed `ab9c270` before results |
| Grid and commit | `s_pred`, batch `stretch`, `fa7f182` |

### Result
PASS on `c05`/`c09` (step-0 drift AUC 0.897/0.999), FAIL on the MLPs. The review found drift no better than an answer-rarity baseline (rarity AUC 0.913/1.000). See `analysis/stretch/sections/s_pred.md`.

## S_ABL: which condition of the tabular proposition matters?

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** The tabular proposition says $p$ never falls under a deterministic verifier, a zero-sum rule and SGD; we broke one condition at a time.

### What I did
1. Started tabular prompts spread over logits $[-4, 3]$ and trained GRPO for 5,000 steps.
2. Applied one change per run: 5% or 20% verifier flips, momentum 0.9, KL $\beta \in \{0.1, 1\}$, the sign rule, or Adam.
3. Counted per-prompt decreases, the largest drop $p_{\text{init}} - \min p$, and the absorbed fraction.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Tabular (one-hot, 512 prompts), start `spread` |
| $K$, $L$ | $3$; $L \in \{1, 2, 4\}$ |
| $G$, $B$ | $G \in \{4, 8\}$, $B = 32$ |
| Rule | GRPO (sign rule in its ablation) |
| Conditions | control, flip05, flip20, momentum, kl01, kl1, sign, adam |
| $\eta$ | $0.3, 3, 30, 300$ (SGD); $10^{-3}, 10^{-2}, 10^{-1}$ (Adam) |
| Steps | $5{,}000$ |
| Seeds | $0, 1, 2$ |
| Pass rule | Control never lowers $p$, and every broken condition lowers $p$ in at least one run |
| Runs | $558$ |
| Pre-registration | Criteria committed `ab9c270` before results |
| Grid and commit | `s_abl`, batch `stretch`, `fa7f182` |

### Result
PASS: the control never lowers $p$; a noisy verifier and the sign rule cause absorption (up to 35% and 27%). The review found the PASS over-credited: momentum and Adam "decreases" are round-off (max drop 0). See `analysis/stretch/sections/s_abl.md`.

## S_EQ: two stable KL fixed points

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** H4 stretch goal: at $G \ge 16$ the KL fixed-point equation has two stable roots; does training land on the side it starts?

### What I did
1. For three $(p_{\text{ref}}, \beta)$ pairs, found the unstable root $p_u$ of the GRPO fixed-point equation.
2. Started 64 identical tabular prompts at logit offsets from $p_u$ while the reference stayed at $p_{\text{ref}}$.
3. Trained 10,000 steps and recorded the share ending in the upper state, the end states, and crossings of $p_u$ between snapshots.
4. Ran the mean rule (single root) as a control.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Tabular, 64 identical prompts |
| $K$, $L$ | $3$, $1$ |
| $(p_{\text{ref}}, \beta)$ | $(0.01, 0.5), (0.03, 0.5), (0.2, 1.0)$ |
| Start offsets (logit) | $-3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3$ |
| $G$, $B$ | $64$, $64$ |
| KL | Reverse, $\mathrm{KL}(\pi\|\pi_{\text{ref}})$ |
| Rules and $\eta$ | GRPO at $\eta \in \{0.1, 0.3, 1, 3, 10\}$; mean at $\eta = 1$ |
| Steps | $10{,}000$ (snapshot every 100) |
| Seeds | $0, 1, 2$ |
| Pass rules | At $\eta \le 1$: upper share $< 0.1$ for offsets $\le -0.5$ and $> 0.9$ for $\ge +0.5$; end states within 0.01; crossings rise with $\eta$ |
| Runs | $594$ |
| Pre-registration | Criteria committed `ab9c270` before results |
| Grid and commit | `s_eq`, batch `stretch`, `fa7f182` |

### Result
Bistability confirmed: the basin boundary sits at the predicted root and end states are within 0.01. The "crossings increase with $\eta$" criterion FAILs as coded. Extends Mroueh to finite $G$ under SGD. See `analysis/stretch/sections/s_eq.md`.

## A1: GRPO's weight between the mean rule and the arcsin surrogate

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** The milestone promised that GRPO's weight moves toward the $2\arcsin\sqrt p$ surrogate as $G$ grows; the review showed this was testable from existing data.

### What I did
1. Took the tabular GRPO bins of the base H2 one-step runs.
2. Fitted $W_G(p) = c\,[(1-\lambda) + \lambda/\sqrt{p(1-p)}]$, with $c$ free and in a scale-free form, per $G$.
3. Bootstrapped bins (2,000 draws) for CIs and fitted the same model to the exact $W_G$.

### Fixed settings
| Setting | Value |
| --- | --- |
| Data | Base H2, tabular GRPO, 30 bins per $G$ |
| Group size $G$ | $2, 4, 16, 64$ (exact $\lambda$ also at 8, 32, 128) |
| Bootstrap | 2,000 draws over bins |
| New runs | None |
| Pre-registration | Plan only (`NEXT_PLAN.md`, Phase A) |
| Code and commit | `experiments/analyze_phase_a.py`, `a762c7b` |

### Result
$\lambda$ rises with $G$: 0, 0.015, 0.215, 0.627 at $G$ = 2, 4, 16, 64, matching the exact $\lambda$. Exploratory. See `analysis/phase_a/A1.md`.

## A4: lost problems and coverage

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** To test coverage narrowing (Yue et al.) on existing toy data: does pass@1 rise while pass@64 falls?

### What I did
1. Took every stretch `s_pred` run.
2. Computed exact pass@k $= 1 - (1-p)^k$ averaged over the 512 prompts, at the start and the end.
3. Counted lost prompts ($p_{\text{init}} \ge 0.01$, final $p < 10^{-3}$) and the share of runs where pass@1 rises and pass@64 falls.

### Fixed settings
| Setting | Value |
| --- | --- |
| Data | Stretch `s_pred` (800 runs; $\eta$ from 0.3× to 10× base $\eta^*$) |
| Measures | Exact pass@1, pass@64; lost: $p_{\text{init}} \ge 0.01$, final $p < 10^{-3}$ |
| New runs | None |
| Pre-registration | Plan only (`NEXT_PLAN.md`) |
| Code and commit | `experiments/analyze_phase_a.py`, `a762c7b` |

### Result
In every cell pass@1 rises (0.06 to 0.65) and pass@64 falls (−0.08 to −0.52). GRPO lowers pass@64 more than the mean rule in 10 of 10 pairs. Exploratory; some losses are overshoot. See `analysis/phase_a/A4.md`.

## BAY: Bay and Yearick's group-size law

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** To check a published group-size law against our exact $W_G$, our learned policies and our speed results.

### What I did
1. Computed the exact fidelity $\varphi(G, p) = W_G(p)/W_\infty(p)$ and compared it with their interior law $\varphi \approx 1 - 1/(8Gp(1-p))$.
2. Measured $\varphi$ on the base H2 one-step bins for `tab`, `lin` and `m64`.
3. Compared their budget $G^* = 1/(8\varepsilon p(1-p))$ at $\varepsilon = 0.1$ with the fastest $G$ per sample in T4 + T4e.

### Fixed settings
| Setting | Value |
| --- | --- |
| Data | Base H2 bins (GRPO, SGD); v3a T4 and v3b T4e |
| Grid (exact part) | $G \in \{2, 4, 8, 16, 32, 64\}$, $p \in \{0.02, 0.05, 0.1, 0.2, 0.5\}$ |
| Budget fidelity | $\varepsilon = 0.1$ |
| New runs | None |
| Pre-registration | None |
| Code and commit | `experiments/analyze_phase_a.py`, `a762c7b` |

### Result
Our identity gives their fidelity exactly; the law is within 0.01–0.05 of measurement from $G = 16$ but off by up to 0.25 where $8Gp(1-p) < 4$. Their $G^*$ is not the fastest $G$ per sample (up to 64.5× slower). Exploratory. See `analysis/phase_a/BAY.md`.

## ADAM: absorption time against η under Adam

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** The review listed the Adam timing fit as never run: is Adam absorption drift-driven (time $\propto 1/\eta$)?

### What I did
1. Took the stretch `s_drift` runs, Adam and SGD blocks.
2. For each $\eta$ whose runs absorb $\ge 5\%$ of prompts (seed median), took $t_{50}$, the median step at which absorbed prompts were absorbed.
3. Fitted the slope of $\log t_{50}$ on $\log\eta$ (drift predicts $-1$, noise $-2$) and flagged points at the 5-step floor.

### Fixed settings
| Setting | Value |
| --- | --- |
| Data | Stretch `s_drift` (Adam block: 13 $\eta$ around $10^{-2.5}$, $B = 32$, 5 seeds) |
| Inclusion | Seed-median absorbed $\ge 5\%$ |
| New runs | None |
| Pre-registration | None |
| Code and commit | `experiments/analyze_phase_a.py`, `a762c7b` |

### Result
Slope about $-1$ for `c05` and `c09` under Adam ($-0.95$ to $-0.99$, 13 points each). Exploratory. See `analysis/phase_a/ADAM.md`.

## A2, A3, A6: re-scored ablations, H4 slopes, Barakat write-up

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** Planned in `NEXT_PLAN.md` Phase A: re-score the ablations with a material-drop threshold, report H4 fixed-$\beta$ slopes, and write up the Barakat test from H2.

### What I did
1. Planned only; no write-up exists in `analysis/phase_a/`.
2. The material-drop rule (max drop $> 10^{-3}$) planned for A2 was used in v3a T6.

### Fixed settings
| Setting | Value |
| --- | --- |
| Data | Base and stretch caches |
| New runs | None |
| All settings | See `extra/analysis/review/NEXT_PLAN.md` and `HANDOFF.md` |

### Result
No write-up; `HANDOFF.md` notes "A3 needs one sentence in the report".

## T1 / P1: imbalance at fixed geometry

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** The review found `c05`/`c09` absorption was minority-answer collapse. P1 asks whether the majority share $m$ itself causes it when the feature geometry is held fixed.

### What I did
1. Built tasks with correlated features and a fixed answer rule, solving the class offset so answer 0 has share $m$ (the other two equal).
2. Trained each task from a uniform start with GRPO under SGD (a flow-time $\eta$ grid, $\eta t = 100$) and Adam, with both `expected_full` and `sampled` updates; plus a supervised cross-entropy control.
3. Measured persistent absorption at $\tau = 10^{-3}$ (max $p$ over the last 20% of steps $< \tau$, among prompts with $p_{\text{init}} \ge 10\tau$).
4. Per cell, kept the no-overshoot $\eta$ (balanced arm median $\le 0.02$) and ran a paired one-sided Wilcoxon ($m = 0.9$ vs $1/3$) and a Spearman test of $m$ against absorption.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, MLP width 64 |
| Features | Correlated, $\rho \in \{0.5, 0.9\}$, plus a constant feature |
| Majority share $m$ | $1/3, 0.5, 0.7, 0.9$ |
| $K$, $L$ | $3$, $1$ |
| $G$, $B$ | $8$, $32$ |
| Rule | GRPO |
| Update | `expected_full`, `sampled`; supervised control |
| SGD $\eta$ and steps | $0.01$–$30$ (8 values); steps $= 100/\eta$ clipped to $[1{,}000, 20{,}000]$ |
| Adam $\eta$ and steps | $10^{-4}$–$3\times10^{-2}$ (6 values); 5,000 steps |
| Control | Cross-entropy, $\eta = 1$, 2,000 steps |
| Seeds | 10 task seeds, init seed 0 |
| Thresholds | $\tau = 10^{-3}$ (primary), $10^{-2}$; tail 20% |
| Pass rule | (a) and (b) $p < 0.05$ in every included SGD and Adam cell |
| Runs | $4{,}640$ |
| Pre-registration | `extra/analysis/review/PREREG_v3a.md` (`dc5736e`), confirmatory |
| Grid and commit | `v3_t1`, batch `v3a`, `toy-v3` @ `108055d` |

### Result
Confirmatory PASS in 16/16 cells. Absorption takes the whole minority set, at small $\eta$ and under the deterministic flow; the supervised control absorbs nothing. Corrected task-level $p = 0.0098$ (Holm over P1–P5: 0.049). See `extra/analysis/v3a/SUMMARY.md` and `extra/analysis/v3a/P1_TASK_LEVEL.md`.

## T2 / P2: kick size against response noise

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** The review said the "batch noise" result was really per-kick size $\eta/B$. P2 tests this at equal mean drift.

### What I did
1. Fixed $\eta$ per level so the mean drift is equal across $B$.
2. Trained with `expected` and `sampled` updates at $B \in \{8, 32, 128, 512\}$ (512 without replacement), plus `expected_full` at $B = 32$.
3. (a) Tested Spearman($B$, absorption) $< 0$ for `expected` per level. (b) Tested that `sampled` − `expected` has a 95% bootstrap CI inside $\pm 0.05$ in every level.

### Fixed settings
| Setting | Value |
| --- | --- |
| Levels and $\eta$ | `lin` (300, 1000), `c05` (10, 30), `m64` (0.3, 1), each $L = 2$; `bal` (linear, $\rho = 0.9$, $m = 1/3$, $L = 1$; 3, 10) |
| $K$ | $3$ |
| $G$ | $8$ |
| Batch $B$ | $8, 32, 128, 512$ |
| Rule | GRPO |
| Update | `expected`, `sampled`, `expected_full` |
| Steps | $3{,}000$ |
| Seeds | 10 task seeds |
| Threshold | Persistent absorption, $\tau = 10^{-3}$ |
| Pass rule | (a) in $\ge 3$ of 4 levels and (b) in every level |
| Runs | $720$ |
| Pre-registration | `PREREG_v3a.md`, confirmatory |
| Grid and commit | `v3_t2`, batch `v3a`, `toy-v3` @ `108055d` |

### Result
Confirmatory FAIL. Kick size matters in 3 of 4 levels (Spearman −0.87, −0.72, −0.31; `c05` flat). Response noise adds +0.11 absorption at $B = 8$, but only +0.008 at $B \ge 32$. See `extra/analysis/v3a/SUMMARY.md`.

## T4: H5 rerun with a deterministic tie-break

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** The review found the H5 best-$G$ table was a tie-break artefact with $\eta$ at the grid edge.

### What I did
1. Reran the H5 design with $\eta$ extended to 1000, 5 seeds and $p$ tracked every step.
2. Measured samples to reach mean $p \ge 0.5$ and took the best $\eta$ per $G$.
3. Took the best $G$ as the smallest $G$ within 256 samples of the minimum, with a 1,000-draw seed bootstrap CI, and flagged grid-edge optima.

### Fixed settings
| Setting | Value |
| --- | --- |
| Levels | `tab` (64 prompts), `lin`, `m64` (`sft_down`) |
| $p_0$ | $0.01, 0.03, 0.1, 0.3$ |
| $K$, $L$ | $3$, $1$ |
| $G$, $B$ | $2$–$128$, $B = 256/G$ |
| Rules | mean, GRPO |
| $\eta$ | $0.01, 0.03, \dots, 300, 1000$ (11 values) |
| Steps | $5{,}000$ |
| Seeds | $0$–$4$ |
| Runs | $9{,}240$ |
| Pre-registration | `PREREG_v3a.md`, exploratory |
| Grid and commit | `v3_t4`, batch `v3a`, `toy-v3` @ `108055d` |

### Result
Tabular GRPO's best $G$ is 2 at every $p_0$ (CIs within {2, 4}), but often at the $\eta$-grid edge, so a saturation point. Shared levels: 2–8, wide CIs. Exploratory. See `extra/analysis/v3a/SUMMARY.md`.

## T6: tabular ablations under realistic training

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** To learn which realistic training conditions (momentum, clipping, Adam, KL, PPO epochs) break the tabular "no decrease" result.

### What I did
1. Started tabular prompts spread over logits $[-4, 3]$ and trained GRPO for 5,000 steps under one condition each.
2. Counted runs with a material drop (max over prompts of $p_{\text{init}} - \min p > 10^{-3}$) and the persistent absorption at $\tau = 10^{-3}$.
3. Compared each condition with its prediction.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Tabular, start `spread` |
| $K$, $L$ | $3$; $L \in \{1, 2, 4\}$ |
| $G$, $B$ | $G \in \{4, 8\}$, $B = 32$ |
| Conditions | control, flip05, flip20, momentum 0.9, kl01, kl1, sign, ppo4 (4 epochs, clip 0.2), multi4 (4 epochs), gradclip 1.0, adam |
| $\eta$ | $0.3, 3, 30, 300$ (SGD); $10^{-3}, 10^{-2}, 10^{-1}$ (Adam) |
| Steps | $5{,}000$ |
| Seeds | $0$–$9$ |
| Thresholds | Material drop $> 10^{-3}$; $\tau = 10^{-3}$ |
| Runs | $2{,}580$ |
| Pre-registration | `PREREG_v3a.md`, exploratory |
| Grid and commit | `v3_t6`, batch `v3a`, `toy-v3` @ `108055d` |

### Result
No drops with PPO clipping or batch reuse (both predicted to drop), so the prediction FAILs for those two. Drops with a noisy verifier (74–90% of runs) and the sign rule (68%). Exploratory. See `extra/analysis/v3a/SUMMARY.md`.

## T8: KL bistability without noise

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** To see whether the deterministic expected update can switch KL basins on its own, or whether S_EQ's switching needs sampling noise.

### What I did
1. Used the three bistable S_EQ configurations and the same logit offsets from the unstable root.
2. Trained with `expected_full` plus the analytic KL gradient for 10,000 steps.
3. Checked whether the final $p$ is on the same side of the unstable root as the start.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Tabular, 64 identical prompts |
| $(p_{\text{ref}}, \beta)$ | $(0.01, 0.5), (0.03, 0.5), (0.2, 1.0)$ |
| Offsets | 11 values, $-3$ to $+3$ |
| $G$, $B$ | $64$, $64$ |
| Rule, KL | GRPO, reverse |
| Update | `expected_full` |
| $\eta$ | $1, 3, 10$ |
| Steps | $10{,}000$ |
| Seeds | $0$ |
| Runs | $99$ |
| Pre-registration | `PREREG_v3a.md`, exploratory |
| Grid and commit | `v3_t8`, batch `v3a`, `toy-v3` @ `108055d` |

### Result
PASS: 60/60 runs at $\eta \le 3$ end in the predicted basin, and 99/99 overall stay on their starting side, even at $\eta = 10$. Switching needs sampling noise. See `extra/analysis/v3a/SUMMARY.md`.

## T5 / P3: does initial drift predict absorption on balanced tasks?

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** The review found the S_PRED drift predictor no better than answer rarity. P3 tests it on balanced tasks, beyond rarity and $p_{\text{init}}$.

### What I did
1. Warm-started balanced tasks ($m = 1/3$) with `sft_mixed`, tuned before launch so $p_{\text{init}}$ has sd $\ge 0.25$.
2. Trained GRPO for 5,000 steps at five multiples of the base $\eta^*$ and labelled persistently absorbed prompts.
3. Fitted logistic models with a run fixed effect: base = rarity + logit($p_{\text{init}}$), full = base + step-0 drift, with leave-one-task-out predictions.
4. Computed $\Delta$AUC = AUC(full) − AUC(base) with a 10,000-draw task bootstrap.

### Fixed settings
| Setting | Value |
| --- | --- |
| Levels | `lin`, `c05`, `c09`, `m64` |
| Warm start | `sft_mixed`: `lin` 1,000 steps lr 8; `c05` 1,000, lr 2; `c09` 500, lr 1; `m64` 200, lr 0.5 |
| Majority share $m$ | $1/3$ |
| $K$, $L$ | $3$, $1$ |
| $G$, $B$ | $8$, $32$ |
| Rule, update | GRPO, `sampled` |
| $\eta$ | Base $\eta^*$ ($L = 1$) $\times \{0.3, 1, 3, 10, 30\}$ |
| Steps | $5{,}000$; drift every 500 |
| Seeds | 10 task seeds |
| Rarity | Share of the 10 nearest prompts with a different class |
| Pass rule | $\Delta$AUC $\ge 0.03$ and CI lower bound $> 0$; runs need $\ge 20$ absorbed and $\ge 20$ not |
| Runs | $200$ |
| Pre-registration | `extra/analysis/review/PREREG_v3b.md` (`aa144dc`), confirmatory |
| Grid and commit | `v3_t5`, batch `v3b`, `toy-v3` @ `e61626f` |

### Result
Confirmatory FAIL: $\Delta$AUC 0.009 (CI 0.004–0.014), below the 0.03 bar. Post hoc, with predictors standardised within run, 0.079. See `extra/analysis/v3b/SUMMARY.md`.

## T7: bootstrap null for the H1 medium tier

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** The S_H1 medium tier failed; T7 checks whether a bootstrap null is calibrated before using it.

### What I did
1. Reran the GRPO H1 configurations at $L \in \{1, 2\}$ and $G \in \{4, 8\}$ with 200 bootstrap draws of the mean $z^2$ null.
2. Ranked each run's observed mean $z^2$ within its draws (uniform under the exact identity).
3. Ran a KS test on the tabular ranks; if calibrated, applied the tier to `lin` and `m64`.

### Fixed settings
| Setting | Value |
| --- | --- |
| Configurations | H1 GRPO subset: `tab`, `lin`, `m64`; uniform or warm; $L \in \{1, 2\}$; $G \in \{4, 8\}$ |
| Samples $R$ | $4{,}000$ |
| Bootstrap | 200 draws |
| Seeds | $0$–$9$ |
| Pass rule | Tabular KS $p \ge 0.05$ |
| Runs | $240$ |
| Pre-registration | `PREREG_v3b.md`, exploratory |
| Grid and commit | `v3_t7`, batch `v3b`, `toy-v3` @ `e61626f` |

### Result
PASS: the null is calibrated (tabular KS $p = 0.73$); `lin` KS $p = 0.84$, `m64` 0.055. Exploratory. See `extra/analysis/v3b/SUMMARY.md`.

## T9: all-wrong-group fixes

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** A milestone promise and the only causal test of the absorption mechanism: do rules that give all-wrong groups a gradient protect minority prompts?

### What I did
1. Derived each rule's $W_8(p)$ from `theory.advantage_gap` before any run (Heckel's low-to-high-$p$ weight ratio 6.96 against GRPO's 1.12).
2. Ran each rule on three absorbing setups: T1 minority, T2 big kicks, T6 20% verifier flips.
3. Compared persistent absorption with GRPO (paired one-sided Wilcoxon over tasks).

### Fixed settings
| Setting | Value |
| --- | --- |
| Rules | grpo, mean, heckel, virtual, pm_alpha ($\alpha = 0.1$), nsr ($\lambda = 0.1$) |
| t1-minority | Linear, $\rho = 0.9$, $m = 0.9$, $L = 1$, $G = 8$, $B = 32$, SGD, $\eta \in \{0.1, 1\}$, `expected_full` and `sampled`, steps $= 100/\eta$ clipped |
| t2-kick | `lin`, $L = 2$, $G = 8$, $B = 8$, $\eta = 300$, `expected` and `sampled`, 3,000 steps |
| t6-flip20 | Tabular `spread`, $L = 1$, $G = 8$, $B = 32$, flip 20%, $\eta \in \{30, 300\}$, 5,000 steps |
| $K$ | $3$ |
| Seeds | 10 (task seeds; plain seeds for tabular) |
| Threshold | $\tau = 10^{-3}$ |
| Runs | $480$ |
| Pre-registration | `PREREG_v3b.md`, exploratory |
| Grid and commit | `v3_t9`, batch `v3b`, `toy-v3` @ `e61626f` |

### Result
Prediction too strong. Heckel cuts minority collapse 0.102 → 0.059 at $\eta = 0.1$ ($p = 0.002$), but nothing helps at $\eta = 1$, and Heckel doubles absorption with big kicks (0.21 → 0.42). See `extra/analysis/v3b/SUMMARY.md`.

## T1 seed 1: P1 with a second initialisation

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** Follow-up (a) to P1: does the MLP result depend on the init seed?

### What I did
1. Reran every T1 MLP job with init seed 1.
2. Applied the P1 test as pre-registered to the 8 MLP cells.

### Fixed settings
| Setting | Value |
| --- | --- |
| Configurations | T1 MLP cells (width 64; $\rho \in \{0.5, 0.9\}$; $m \in \{1/3, 0.5, 0.7, 0.9\}$; SGD and Adam; `expected_full`, `sampled`; supervised control) |
| Seeds | 10 task seeds, init seed 1 |
| Test | P1 rule from `PREREG_v3a.md` |
| Runs | $2{,}320$ |
| Pre-registration | `PREREG_v3b.md`, exploratory |
| Grid and commit | `v3_t1s1`, batch `v3b`, `toy-v3` @ `e61626f` |

### Result
P1 replicates in 8/8 MLP cells. Exploratory. See `extra/analysis/v3b/SUMMARY.md`.

## T2 groups: does the B = 8 noise excess shrink with G?

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** Follow-up (b) to P2: the response-noise excess appeared only at $B = 8$; does a larger group remove it?

### What I did
1. Reran the T2 `expected` and `sampled` jobs at $B = 8$ with $G \in \{4, 16, 64\}$ ($G = 8$ is in T2).
2. Computed `sampled` − `expected` per level and $G$, and Spearman($G$, excess) pooled over levels.

### Fixed settings
| Setting | Value |
| --- | --- |
| Configurations | T2 levels and $\eta$, $B = 8$ |
| $G$ | $4, 16, 64$ |
| Update | `expected`, `sampled` |
| Steps | $3{,}000$ |
| Seeds | 10 task seeds |
| Runs | $480$ |
| Pre-registration | `PREREG_v3b.md`, exploratory |
| Grid and commit | `v3_t2g`, batch `v3b`, `toy-v3` @ `e61626f` |

### Result
Yes: the excess shrinks as $G$ grows (Spearman −0.33, $p = 5\times10^{-10}$). Exploratory. See `extra/analysis/v3b/SUMMARY.md`.

## T6 KL: declines toward or below the reference

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** Follow-up (c): T6's KL prediction could not be tested because the running maximum of $p$ was not logged.

### What I did
1. Reran the T6 kl01, kl1 and control jobs with $p$ tracked every step.
2. Split declines into drops below $p_{\text{ref}} = p_{\text{init}}$ and declines from a running maximum above it (`drop_above`).

### Fixed settings
| Setting | Value |
| --- | --- |
| Configurations | T6 tabular `spread`, $L \in \{1, 2, 4\}$, $G \in \{4, 8\}$, $B = 32$ |
| Conditions | control, kl01 ($\beta = 0.1$), kl1 ($\beta = 1$) |
| $\eta$ | $0.3, 3, 30, 300$ |
| Steps | $5{,}000$, tracked every step |
| Seeds | $0$–$9$ |
| Runs | $720$ |
| Pre-registration | `PREREG_v3b.md`, exploratory |
| Grid and commit | `v3_t6kl`, batch `v3b`, `toy-v3` @ `e61626f` |

### Result
KL runs drop below $p_{\text{ref}}$ only at $\eta \ge 30$; the control never falls from a peak. Exploratory. See `extra/analysis/v3b/SUMMARY.md`.

## T4 edge: wider η for the T4 grid-edge cells

**Scope: Promised** — in the milestone (H1–H5, their planned stretch goals, or a test the milestone named).

**Why:** Follow-up (d): many T4 optima sat at the top of the $\eta$ grid.

### What I did
1. Reran T4 for `tab` and `lin` at $p_0 \le 0.1$ with $\eta \in \{3000, 10000\}$, stopping at mean $p = 0.5$.
2. Recomputed the tie-broken best $G$ and checked whether each edge cell's best $\eta$ became interior.

### Fixed settings
| Setting | Value |
| --- | --- |
| Levels | `tab`, `lin` |
| $p_0$ | $0.01, 0.03, 0.1$ |
| $G$, $B$ | $2$–$128$, $B = 256/G$ |
| Rules | mean, GRPO |
| $\eta$ | $3000, 10000$ |
| Steps | Up to $5{,}000$, stop at mean $p \ge 0.5$ |
| Seeds | $0$–$4$ |
| Runs | $840$ |
| Pre-registration | `PREREG_v3b.md`, exploratory (no direction predicted) |
| Grid and commit | `v3_t4e`, batch `v3b`, `toy-v3` @ `e61626f` |

### Result
Best $G$ stays 2 (`lin` at $p_0 = 0.01$: 4); most former edge cells now have an interior optimum. Descriptive. See `extra/analysis/v3b/SUMMARY.md`.

## T10: pass@k training rules

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** Rules that weight hard prompts by about $k(1-p)^{k-1}$ have a low-to-high-$p$ weight ratio far above GRPO's, so they should resist minority collapse.

### What I did
1. Trained GRPO, the mean rule, biased GRPO$_k$ and unbiased RLOO$_k$ ($k \in \{2, 4, 8\}$) at a fixed budget $BG = 256$.
2. Used four tasks: tabular, balanced `lin` and `m64`, and the T1 minority task.
3. Measured exact pass@1, pass@8, pass@64, persistent absorption and minority absorption at the end.
4. Compared each rule with GRPO at the same (level, $G$, $\eta$) with a one-sided Wilcoxon paired over seeds.

### Fixed settings
| Setting | Value |
| --- | --- |
| Levels and $\eta$ | `tab` (`spread`; 3, 30, 300); `lin` (`sft_mixed`, $m = 1/3$; 100, 300, 1000); `m64` (`sft_mixed`, $m = 1/3$; 0.3, 1, 3); minority (linear, $\rho = 0.9$, $m = 0.9$; 0.03, 0.3, 1) |
| $K$, $L$ | $3$, $1$ |
| $G$, $B$ | $G \in \{8, 32\}$, $B = 256/G$ |
| Rules | grpo, mean, passk and rloo_k at $k \in \{2, 4, 8\}$ |
| Steps | $5{,}000$ |
| Seeds | 10 (task seeds; plain seeds for `tab`) |
| Threshold | $\tau = 10^{-3}$ |
| Runs | $1{,}920$ |
| Pre-registration | `extra/analysis/review/PREREG_v3c.md`, exploratory |
| Grid and commit | `v3_t10`, batch `v3c`, `toy-v3` @ `7759008` |

### Result
GRPO absorbs the whole minority set; both pass@k rules absorb 0–5% at every $k$. rloo_k raises pass@64 in 12/12 shared cells. "No speed cost at $k = 2$" was later revised by v3f. Exploratory. See `extra/analysis/v3c/SUMMARY.md`.

---

# Part 2: mechanism, theory, levers, the transformer environment and the final phase

## T11 (v3d): who pushes whom

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** v3a showed the minority collapses at $m \ge 0.7$, but end states cannot tell whether the minority gets too little signal or is actively pushed down by the majority.

### What I did
1. Trained the T1 class-share task from a uniform start (every $p = 1/3$) with GRPO, for linear and MLP policies.
2. At 0, 10%, 50% and 90% of training, copied the policy and optimizer state and took one step along each of: nothing, the majority's part $g_A$ of the exact expected update, the minority's part $g_B$, and the full step $g_A + g_B$.
3. Recorded the exact change in each class's mean success, $\Delta_B(A)$, $\Delta_B(B)$ and $\Delta_B(\text{mixed})$, and the push ratio $R = -\Delta_B(A)/\Delta_B(B)$.
4. Checked the mixed branch against the simulator's own step, and repeated the branch test with 16 sampled batches per branch point for the sampled arm.
5. Scored five predictions per cell (a cell agrees if at least 9 of 10 tasks agree).

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 (`t1_base`) |
| Features | Correlated, pairwise cosine about $\rho$, $\rho \in \{0.5, 0.9\}$; $d = 16$ plus a constant feature; 512 training prompts |
| Answer classes $K$, length $L$ | $K = 3$, $L = 1$ |
| Majority share $m$ | $\{1/3, 0.7, 0.9\}$ |
| Group size $G$, batch $B$ | $G = 8$, $B = 32$ |
| Advantage rule | GRPO (`rule="grpo"`) |
| Arms (optimizer, $\eta$, update) | SGD 0.01 exact (`expected_full`); SGD 0.01 sampled with 16 batches per branch point; Adam $10^{-3}$ exact; Adam $10^{-4}$ exact |
| Steps | SGD 10,000 (flow time $\eta t = 100$, `flow_steps`); Adam 5,000 |
| Branch points | `branch_at = [0, 0.1, 0.5, 0.9]` |
| Start | Uniform |
| Tasks and seeds | Task seeds 0–9 (`TASK_SEEDS`), sampling seed 0 |
| KL $\beta$ | 0 |
| Absorption threshold | Success below $10^{-3}$ over the last 20% of steps |
| Runs | 480 |
| Pre-registration | `extra/analysis/review/PREREG_v3d.md` (`review` `f22540c`), exploratory |
| Grid and launch commit | `v3_t11`; `toy-v3` `7a49348`, job digest `d497e591db3a` |

### Result
All five predictions hold. Before collapse the majority's step lowers the minority, the minority's own step raises it, and the mixed step lowers it (16/16 cells). The push is $R$ = 3.6–4.8 at $m = 0.7$ and 10.7–17.1 at $m = 0.9$ (0.78–0.93 at $m = 1/3$). After collapse the minority's own step is below 1% of its start value (125/125 branch points). Collapse is active reversal, then starvation. Exploratory. See `extra/analysis/v3d/SUMMARY.md`.

## T12 (v3d): plateau or permanent

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** the T1 collapse lasted the whole run, and we needed to know whether it is permanent or a long plateau.

### What I did
1. Reran the T1 minority tasks ($m \in \{0.7, 0.9\}$) for 10× the T1 horizon, with exact and sampled updates.
2. Used three SGD step sizes and two Adam step sizes, with linear Adam at the smallest step to test whether its weak collapse was a short-horizon effect.
3. Called a run "recovered" if its minority class mean was above 0.01 at the last step after falling below $10^{-3}$.
4. Also compared the minority mean at matched progress: the first logged step where the majority mean is 0.10 above its start.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 |
| Features | Correlated, $\rho \in \{0.5, 0.9\}$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | $\{0.7, 0.9\}$ |
| $G$, $B$ | 8, 32 |
| Advantage rule | GRPO |
| Update | Exact (`expected_full`) and sampled |
| Optimizer, $\eta$, steps | SGD 0.01 for 100,000; SGD 0.1 for 10,000; SGD 1 for 10,000; Adam $10^{-4}$ for 50,000; Adam $10^{-3}$ for 50,000 |
| Logging | Every `steps // 200`; per-prompt tracking every 50 steps |
| Start | Uniform |
| Tasks and seeds | Task seeds 0–9, seed 0 |
| KL $\beta$ | 0 |
| Absorption threshold | $10^{-3}$, last 20% of steps |
| Runs | 800 |
| Pre-registration | `PREREG_v3d.md` (`review` `f22540c`), exploratory |
| Grid and launch commit | `v3_t12`; `toy-v3` `7a49348` |

### Result
Prediction 1 (no recovery under SGD) fails: 34 of 409 runs that fell below $10^{-3}$ recovered (8.3%; the limit was 5%), all at $\rho = 0.5$. At $\rho = 0.9$ no SGD run recovered in 10× the horizon (0 of 240). A later correction found that 24 of 80 exact Adam runs at $\rho = 0.9$ do recover. Linear Adam at $10^{-4}$ does collapse at 50,000 steps (3 of 4 cells), and the matched-progress test holds in 16/16 cells. Exploratory. See `extra/analysis/v3d/SUMMARY.md`.

## T13 (v3d): projection control

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** if the majority's step hurts the minority through one direction, removing that direction should prevent the collapse.

### What I did
1. Trained the T1 minority tasks with three versions of the exact update.
2. `projected`: removed the component of the majority's part along the minority's mean success gradient $\nabla \bar p_B$ whenever it was negative, and kept the minority's part.
3. `shortened`: scaled the majority's part down to the projected norm, without changing its direction.
4. `expected_full`: the unmodified update, as the control.
5. Compared the minority absorption between arms with paired one-sided Wilcoxon tests over tasks, per cell.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 |
| Features | Correlated, $\rho \in \{0.5, 0.9\}$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | $\{0.7, 0.9\}$ |
| $G$ | 8 (enters the GRPO weight) |
| Advantage rule | GRPO |
| Update | `expected_full`, `projected`, `shortened` |
| Optimizer, $\eta$, steps | SGD; $\eta = 0.01$ for 10,000 steps, $\eta = 0.1$ for 1,000 steps (flow time 100) |
| Start | Uniform |
| Tasks and seeds | Task seeds 0–9, seed 0 |
| KL $\beta$ | 0 |
| Absorption threshold | $10^{-3}$, last 20% of steps |
| Runs | 480 |
| Pre-registration | `PREREG_v3d.md` (`review` `f22540c`), exploratory |
| Grid and launch commit | `v3_t13`; `toy-v3` `7a49348` |

### Result
Projection removes the collapse in 16/16 cells (minority absorbed 0.00–0.01 at $m = 0.9$, against 0.81–1.00 unmodified). Prediction 2 fails: the shortened step also removes the collapse at $m = 0.7$ and reduces it at $m = 0.9$ (0.47–0.89), so down-weighting the majority is a second lever. Prediction 3 fails for linear policies (majority falls to 0.45–0.86). Exploratory. See `extra/analysis/v3d/SUMMARY.md`.

## M2 (v3d addendum): is the minority still sinking at 10× the horizon?

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** T12 could not say whether runs at $\rho = 0.9$ were still sinking or about to recover.

### What I did
1. Took every T12 run, with no new training.
2. Over the last decade of training (from $T/10$ to $T$), computed $g = \Delta[-\log \bar p_B] / \Delta[-\log(1 - \bar p_A)]$, the growth of the minority's wrong-side margin relative to the majority's margin.
3. Read $g \ge 1$ as still sinking, $0 < g < 1$ as sinking more slowly, and $g \le 0$ as recovering.
4. Reported the median $g$ per cell and the number of runs recovering.

### Fixed settings
| Setting | Value |
| --- | --- |
| Data | The 800 T12 runs (see T12 for all training settings) |
| Measure | $g$ from class-mean success over $[T/10, T]$ |
| Status | Descriptive; written after T12 was seen |
| Pre-registration | `PREREG_v3d_addendum1.md` (`review` `2401287`) |
| Code | Analysis only, `extra/experiments/analyze_v3d.py` |

### Result
At $\rho = 0.9$ under SGD the median $g$ is 0.99–1.01 in all 12 cells, and no run is recovering (0 of 120). At $\rho = 0.5$, $m = 0.7$ it is −0.5 to 0.4, with 4–8 of 10 runs recovering. At strong correlation $g$ cannot separate "permanent" from "a plateau far longer than the run". Descriptive. See `extra/analysis/v3d/SUMMARY.md`.

## M4 (v3d addendum): imbalance or margin?

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** a prompt might be absorbed because of its class or because it has a small score margin, and the two had not been separated.

### What I did
1. Took the T13 `expected_full` runs (the T1 configuration), with no new training.
2. Split all prompts of each task into terciles of score margin.
3. Computed the absorption rate of minority prompts and of majority prompts in each tercile.
4. Checked the prediction: minority terciles within 0.15 of each other, and majority prompts in the lowest tercile absorbed below 0.05.

### Fixed settings
| Setting | Value |
| --- | --- |
| Data | The 160 T13 runs with `update="expected_full"` (see T13 for training settings) |
| Grouping | Terciles of score margin over all prompts of the task |
| Absorption threshold | $10^{-3}$, last 20% of steps |
| Status | Descriptive analysis with one stated prediction |
| Pre-registration | `PREREG_v3d_addendum1.md` (`review` `2401287`) |
| Code | Analysis only, `extra/experiments/analyze_v3d.py` |

### Result
The prediction holds. Majority prompts are absorbed at 0.000 in every tercile of every cell. Minority terciles are within 0.15 of each other in 7 of 8 cells (for example 0.96, 0.95, 0.95 for linear at $\rho = 0.9$, $m = 0.7$). The class decides, not the margin. See `extra/analysis/v3d/SUMMARY.md`.

## T15 (v3e): threshold in class share

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** T11 showed $R$ crosses 1 between $m = 1/3$ and $m = 0.7$, and we wanted to find where, for several $K$.

### What I did
1. Built a grid of majority shares just above $1/K$ for $K = 2$, 3 and 5.
2. Trained each task with SGD and the exact update for 10,000 steps from a uniform start.
3. At step 0, measured the branch changes and $R$, and found $m^*$, where the median of $\Delta_B(\text{mixed})$ crosses zero (linear interpolation in logit $m$).
4. Found the absorption threshold: the smallest grid $m$ with median minority absorption above 0.5.
5. Fitted the slope of $\log R$ on logit $m$ per cell.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 |
| Features | Correlated, $\rho \in \{0.5, 0.9\}$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | $K \in \{2, 3, 5\}$, $L = 1$ |
| Majority share $m$ | $1/K + \{-0.05, 0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.2, 0.35\}$ |
| $G$ | 8 |
| Advantage rule | GRPO |
| Update | Exact (`expected_full`) |
| Optimizer, $\eta$, steps | SGD, 0.01, 10,000 |
| Branch points | `branch_at = [0]` |
| Start | Uniform |
| Tasks and seeds | Task seeds 0–9, seed 0 |
| KL $\beta$ | 0 |
| Absorption threshold | $10^{-3}$, last 20%; threshold in $m$ = median absorption above 0.5 |
| Runs | 1,080 |
| Pre-registration | `PREREG_v3d_addendum1.md` (`review` `2401287`), exploratory |
| Grid and launch commit | `v3_t15`; `toy-v3` `779297f`, job digest `1dcf342238dd` |

### Result
The odds law: $R \propto m/(1-m)$ in 12/12 cells, with slope 0.98–1.05. $m^*$ is just above $1/K$ for $K = 3$ and 5 (0.02–0.11 above), but not for $K = 2$ (0.498–0.529), so prediction 1 holds in 8 of 12 cells. Persistent absorption needs clearly more imbalance than $m^*$ (12/12 cells). Exploratory. See `extra/analysis/v3e/SUMMARY.md`.

## T16 (v3f): pass@k at matched progress

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** in v3c pass@k rules protected the minority, but they also learned the majority more slowly, so protection could not be told apart from slow learning.

### What I did
1. Trained the minority task with eight rules (GRPO, mean, pass@k and its unbiased form `rloo_k` at $k \in \{2, 4, 8\}$), at 4× the v3c horizon.
2. Kept the sample budget fixed at $B \cdot G = 256$.
3. Defined the matched point as the first logged step where the majority mean reaches 0.80.
4. Measured the minority mean at the matched point, and the final class means and absorbed share.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear |
| Features | Correlated, $\rho = 0.9$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | 0.9 |
| $G$, $B$ | $G \in \{8, 32\}$, $B = 256/G$ |
| Advantage rule | `grpo`, `mean`, `passk` and `rloo_k` with $k \in \{2, 4, 8\}$ |
| Update | Sampled |
| Optimizer, $\eta$ | SGD, $\eta \in \{0.03, 0.3, 1\}$ |
| Steps | 20,000; class means logged every 100 steps |
| Start | Uniform |
| Tasks and seeds | Task seeds 0–9, seed 0 |
| KL $\beta$ | 0 |
| Absorption threshold | $10^{-3}$, last 20% |
| Runs | 480 |
| Pre-registration | `PREREG_v3d_addendum2.md` (`review` `a9c0980`), exploratory |
| Grid and launch commit | `v3_t16`; `toy-v3` `e45b533`, job digest `879c78be6482` |

### Result
A trade-off. Under pass@2 the minority dips to 0.03–0.08 at the matched point, then recovers (0.79–0.87 at the end, $\eta = 1$). Under pass@4 it stays at 0.31–0.42, but the majority takes 4–40× longer to reach 0.80. pass@8 protects by hardly moving. GRPO and mean absorb 100%. Predictions 1–3 hold only in part (12 of 36, 4 of 8, 2 of 12 cells). Exploratory. See `extra/analysis/v3f/SUMMARY.md`.

## T17 (v3g): KL anchor on the minority task

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** H4 (KL penalty) had never been tested as a way to protect the minority.

### What I did
1. Trained the minority task with a KL term to the start policy ($p_{\text{ref}} = 1/3$ for every prompt), in both directions and at six strengths.
2. Computed the theory's per-prompt resting points beforehand.
3. Measured the absorbed minority share, final class means, and the minority mean at the matched point (majority at 0.80).
4. Scored five predictions per cell (architecture × update × $\eta$).

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 |
| Features | Correlated, $\rho = 0.9$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | 0.9 |
| $G$, $B$ | 8, 32 |
| Advantage rule | GRPO |
| Update | Exact and sampled |
| Optimizer, $\eta$, steps | SGD, $\eta \in \{0.1, 1\}$, 20,000 steps |
| KL | $\beta \in \{0, 0.01, 0.03, 0.1, 0.3, 1, 3\}$; reverse and forward; reference = start policy |
| Start | Uniform |
| Tasks and seeds | Task seeds 0–9, seed 0 |
| Absorption threshold | $10^{-3}$, last 20% (sensitivity at $10^{-2}$ added in the corrections) |
| Runs | 1,040 |
| Pre-registration | `PREREG_v3d_addendum3.md` (`review` `2df02ca`), exploratory |
| Grid and launch commit | `v3_t17`; `toy-v3` `b059c73`, job digest `5ab4c97ca34d` |

### Result
Direction matters. Forward KL keeps the minority above a floor: at most 2% absorbed at $10^{-3}$ from $\beta = 0.01$, with the majority still at 0.99. At $10^{-2}$ it absorbs 34% at $\beta = 0.01$ and 11% at 0.03. Reverse KL absorbs 100% in every run up to $\beta = 0.1$. Forward is at least as good as reverse in 48/48 comparisons. Later caveat: the reference still solves the minority (see G). Exploratory. See `extra/analysis/v3g/SUMMARY.md`.

## T18 (v3g): release test

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** collapsed prompts stay down, and the plan's guess was a second resting point.

### What I did
1. Collapsed the minority with plain GRPO at $m = 0.9$.
2. At a release step (100, 1,000 or 5,000), dropped the majority's reward term and kept the minority's.
3. After the release, used no KL, reverse KL or forward KL to the start policy; the control had no release.
4. All arms of a task shared the same state at the release. "Returned" means a minority mean of at least 0.2 at step 20,000.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 |
| Features | Correlated, $\rho = 0.9$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | 0.9 |
| $G$ | 8 |
| Advantage rule | GRPO |
| Update | Exact |
| Optimizer, $\eta$, steps | SGD, 0.1, 20,000 |
| Release | `release_at` $\in \{0.005, 0.05, 0.25\}$ (steps 100, 1,000, 5,000) |
| KL after release | none; reverse $\beta \in \{0.01, 0.1\}$; forward $\beta \in \{0.01, 0.1\}$ (`release_beta`, `release_kl`) |
| Start | Uniform, $\beta = 0$ before the release |
| Tasks and seeds | Task seeds 0–9, seed 0 |
| Runs | 320 |
| Pre-registration | `PREREG_v3d_addendum3.md` (`review` `2df02ca`), exploratory |
| Grid and launch commit | `v3_t18`; `toy-v3` `b059c73` |

### Result
There is no second resting point. The control never returns (0 of 20). Without KL the minority returns from step 100 (20/20) and step 1,000 (15/20), but not from step 5,000 (0/20). Prediction 3 fails: reverse KL does bring it back, slowly. Forward KL returns it in 100–600 steps from any depth. Exploratory. See `extra/analysis/v3g/SUMMARY.md`.

## T19 (v3g): group size and the threshold

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** H5 (group size) had never been tested as a way to protect the minority.

### What I did
1. Reran the T15 share grid at $K = 3$, extended to $m = 0.883$, for five group sizes.
2. Used GRPO and the mean rule with the exact update, and sampled GRPO at a fixed budget $B \cdot G = 256$ ($\rho = 0.9$ only).
3. Measured $m^*$ at step 0 and the absorption threshold (smallest grid $m$ with mean minority absorption at least 0.05).
4. Compared both thresholds across $G$.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 |
| Features | Correlated, $\rho \in \{0.5, 0.9\}$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | $1/3 + \{-0.05, 0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.2, 0.35, 0.45, 0.55\}$ (0.283 to 0.883) |
| $G$, $B$ | $G \in \{2, 4, 8, 16, 64\}$; sampled arm $B = 256/G$ |
| Advantage rule | GRPO at every $G$; mean rule at $G \in \{2, 64\}$ |
| Update | Exact (branch at step 0); sampled GRPO at $\rho = 0.9$ |
| Optimizer, $\eta$, steps | SGD, 0.01, 10,000 |
| Start | Uniform |
| Tasks and seeds | Task seeds 0–9, seed 0 |
| KL $\beta$ | 0 |
| Absorption threshold | $10^{-3}$, last 20%; threshold in $m$ = mean absorption at least 0.05 |
| Runs | 4,180 |
| Pre-registration | `PREREG_v3d_addendum3.md` (`review` `2df02ca`), exploratory |
| Grid and launch commit | `v3_t19`; `toy-v3` `b059c73` |

### Result
Group size is not a lever. $m^*$ does not depend on $G$ (largest range 0.00002, true by construction at a uniform start). The absorption threshold moves by at most 0.15 in $m$, with no consistent direction (predictions 2 and 3 fail, 2 of 4 cells each). Exploratory. See `extra/analysis/v3g/SUMMARY.md`.

## T20 (v3h): correct minority in Li, Pandey and Chen's environment

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** we needed to know whether the mechanism holds beyond feature-vector prompts and linear or MLP policies, with token prompts and a transformer.

### What I did
1. Reused their environment unchanged: 64 prompts $(a, b)$, 100 two-digit responses, a 2-layer transformer shared by all prompts, exact success probabilities.
2. Replaced their verifier: split the prompts into $K$ classes (class 0 holds `n_major`), each with one random correct string.
3. Built the base model by SFT to the same target for every prompt (0.9 shared by the class strings, 0.1 spread over all 100), so every prompt starts near $0.9/K$.
4. Trained the shared transformer (exact policy gradient and sampled GRPO), a tabular policy, and per-prompt copies of the transformer.
5. At step 0 measured the push ratio $R$ with branch steps of size 0.001. Over training, measured per-prompt absorption.

### Fixed settings
| Setting | Value |
| --- | --- |
| Environment | Li, Pandey and Chen's code at `05ef005`, torch 2.15; `extra/experiments/enum_env.py` (`mode="enum"`) |
| Policies | Shared 2-layer transformer; tabular; per-prompt copies (`enum_policy`) |
| Tasks ($K$, `n_major`, partition) | $K = 3$: 22, 32, 40, 48, 56, 60 (random); $K = 2$: 32, 40, 48, 56, 60 (random); $K = 2$, 48, parity |
| Base model | SFT target with `floor` 0.1, 600 steps, lr $3 \times 10^{-3}$ (`ENUM_DEFAULTS`) |
| Advantage rule | Standardised advantage: exact (`algo="pg"`) or sampled GRPO (`algo="grpo"`) |
| $G$ | 8 (sampled GRPO) |
| Optimizer and lr | SGD 0.3, 1, 3; Adam $10^{-3}$, $3 \times 10^{-3}$ |
| Steps | 2,000, logged every 20 |
| Branch step | `branch_eta` = 0.001 (shared policy, step 0) |
| Seeds | 0–9; copies: Adam $10^{-3}$, seeds 0–2, three tasks |
| KL | none |
| Absorption threshold | Per-prompt maximum over the last 20% of steps below $10^{-3}$ |
| Runs | 1,809 |
| Pre-registration | `extra/analysis/review/PREREG_T20.md` (`review` `edff353`), exploratory |
| Grid and launch commit | `v3_t20`; `enum-env` `2f701de`, job digest `f69b99f753f9` |

### Result
The first step follows the odds law exactly (slope 1.00 for both $K$, $c = K - 1$ within 5%). There is no decline without sharing (609 runs). Only Adam at lr $10^{-3}$ isolates the share effect: the per-prompt minority absorption rises from about 0.2 at balance to 0.94 at high share, with class 0 untouched. The registered SGD cells (lr 1, 3) break the majority too, and 1b (no absorption at balance) fails in every cell. Exploratory. See `extra/analysis/v3h/SUMMARY.md`.

## T21 (v3i): levers in the transformer environment

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** we needed to know whether the toy's levers (forward KL, reverse KL, pass@k) behave the same way with a transformer on token prompts.

### What I did
1. Used two T20 tasks that collapse: $K = 3$ with 56 of 64 prompts in class 0, and the parity task.
2. Trained the shared transformer under the two step rules that did not break the whole policy in T20.
3. Added one lever per arm: reverse KL $KL(\pi \| \text{base})$, forward KL $KL(\text{base} \| \pi)$, or pass@k weighting $k(1-p)^{k-1}$ of each prompt's loss.
4. Measured the absorbed minority share, the final class means, and the minority's lowest mean.

### Fixed settings
| Setting | Value |
| --- | --- |
| Environment | As T20 (`enum_env.py`, their code at `05ef005`) |
| Policy | Shared transformer |
| Tasks | ($K = 3$, 56, random) and ($K = 2$, 48, parity) |
| Advantage rule | Exact `pg` and sampled `grpo`, $G = 8$ |
| Optimizer and lr | Adam $10^{-3}$; SGD 0.3 |
| Arms | none; reverse KL $\beta \in \{0.03, 0.3\}$; forward KL $\beta \in \{0.03, 0.3\}$; pass@k $k \in \{2, 4\}$ (`enum_beta`, `enum_kl`, `enum_passk`) |
| KL reference | The base model (each class string about $0.9/K$; the v3i corrections quote 0.3 for $K = 3$) |
| Steps | 2,000 |
| Seeds | 0–9 |
| Absorption threshold | $10^{-3}$, last 20% |
| Runs | 560 |
| Pre-registration | `PREREG_T20_addendum1.md` (`review` `f1f97f0`), exploratory |
| Grid and launch commit | `v3_t21`; `enum-env` `351d722` |

### Result
Forward KL transfers: at $\beta = 0.3$ no minority prompt is absorbed in 8/8 cells, and at 0.03 in 6 of 8. Reverse KL at 0.03 does not help (7/7 collapsing cells). pass@k works only with Adam (prediction 4 fails, 3 of 7). Exploratory. See `extra/analysis/v3i/SUMMARY.md`.

## T22 (v3j): junk answers

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** an LLM's wrong answers are spread out and rarely the other class's answer, so the review asked whether that stops the competition.

### What I did
1. Gave two answer classes prompts (class 0 with share $m$) and added $K - 2$ "junk" tokens that are never correct (`used_classes = 2`).
2. Computed in advance the first-step threshold $m^* = (K-1)/(K-1+\kappa)$ from each task's kernel sums.
3. Trained with SGD from a uniform start, measuring step-0 branches (exact arm) and persistent absorption.
4. Compared measured and predicted $m^*$, the shrinkage of $R$ with $1/(K-1)$, and absorption against $m^*$.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 |
| Features | Correlated, $\rho = 0.9$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | $K \in \{2, 3, 6, 20\}$ ($K - 2$ junk tokens), $L = 1$ |
| Majority share $m$ | $\{0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98\}$ |
| $G$, $B$ | 8, 32 |
| Advantage rule | GRPO |
| Update | Exact (branch at step 0) and sampled |
| Optimizer, $\eta$, steps | SGD, 0.03, 10,000 |
| Start | Uniform |
| Tasks and seeds | Task seeds 0–9, seed 0 |
| KL $\beta$ | 0 |
| Absorption threshold | $10^{-3}$ (and $10^{-2}$ as sensitivity), last 20% |
| Runs | 1,280 |
| Pre-registration | `PREREG_v3d_addendum4.md` (`review` `9458401`), exploratory |
| Grid and launch commit | `v3_t22`; `toy-v3` `f0bf112` |

### Result
The first step moves exactly as the formula says: measured $m^*$ within 0.02 of the prediction in 8/8 cells, and $R$ shrinks by 0.054 and 0.053 against $1/19 = 0.053$. But junk does not protect: absorption starts at $m = 0.6$ for every $K$, and with 18 junk answers the minority is fully absorbed from $m = 0.6$ (predictions 3 and 4 fail). Exploratory. See `extra/analysis/v3j/SUMMARY.md`.

## v3j2: rerun of lost T22 runs

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** host `secb1013-d01` rebooted four minutes after the v3j launch and lost its 106 runs.

### What I did
1. Took the 106 jobs of the lost host from the T22 job list.
2. Reran them from the same commit with the same configurations.
3. Merged the results into the v3j folder (`*_rerun` host folders) and analysed them with the rest of T22.

### Fixed settings
| Setting | Value |
| --- | --- |
| Configurations | Identical to T22 (see above) |
| Runs | 106 |
| Pre-registration | `PREREG_v3d_addendum4.md` (same as T22) |
| Grid and launch commit | `v3_t22`, same job list; `toy-v3` `f0bf112` |

### Result
Complete; included in T22's 1,280 runs. Raw data in `v3j2_raw.tar.gz`. See `extra/analysis/v3j/SUMMARY.md`.

## T23 (v3k): class-balanced warm start

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** every toy run started from a policy that cannot tell the classes apart, while the LLM already solves minority problems about half the time.

### What I did
1. Warm-started the policy by supervised training with equal total weight per answer class, until the mean of the class means reached $p_0$ (`init="sft_balanced"`).
2. Ran GRPO from the uniform start (control) and from $p_0 \in \{0.5, 0.7, 0.9\}$.
3. Recorded the start values: each class's success and the minority's mass on the majority's answer.
4. Measured step-0 branches (exact arm), the absorbed minority share at $10^{-3}$ and $10^{-2}$, and the minority's lowest and final means.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 |
| Features | Correlated, $\rho = 0.9$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | $\{0.7, 0.9\}$ |
| $G$, $B$ | 8, 32 |
| Advantage rule | GRPO |
| Update | Exact (branch at step 0) and sampled |
| Optimizer, $\eta$, steps | SGD, 0.1, 10,000 |
| Start | Uniform, or `sft_balanced` to $p_0 \in \{0.5, 0.7, 0.9\}$ (warm-start lr `sft_lr` = 0.5) |
| Tasks and seeds | Task seeds 0–9, seed 0 |
| KL $\beta$ | 0 |
| Absorption threshold | $10^{-3}$ and $10^{-2}$, last 20% |
| Runs | 320 |
| Pre-registration | `PREREG_v3d_addendum5.md` (`review` `2a24e93`), exploratory |
| Grid and launch commit | `v3_t23`; `toy-v3` `a717aab`, job digest `04c06c4b14a9` |

### Result
A head start protects, and a larger majority needs a larger one. $p_0 = 0.5$ removes or greatly reduces the collapse at $m = 0.7$ (0–0.14 absorbed against 1.00), but not at $m = 0.9$ (0.99 linear, 0.69–0.70 MLP). The first step still lowers the minority at every start (6/6 cells). Predictions 2 and 3 hold for the linear policy only (4 of 8 cells each). Exploratory. See `extra/analysis/v3k/SUMMARY.md`.

## T24 (v3l): the late phase, prompt by prompt

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** to test the per-prompt late-phase model from `extra/analysis/theory/late_time.md` on fresh runs, with the prediction rule fixed before launch.

### What I did
1. Trained the linear policy for 100,000 steps, saving every prompt's success at log-spaced steps.
2. Took $\tau_0$ as the first stored step where the majority mean is at least 0.99, and each minority prompt's log-odds $L_y$ there.
3. Integrated the registered per-prompt model from $\tau_0$: the push $\gamma_y \cdot dM/d\tau$ ($\gamma_y = \hat u \cdot \phi_y$, majority as a clock) minus a collective restoring term from classmates.
4. Compared it with the simulation and four baselines (persistence, push only, self-only restoring, one $\gamma$ for all).
5. Counted runs with at least 20 minority prompts below 0.5 at $\tau_0$.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear |
| Features | Correlated, $\rho \in \{0.5, 0.7, 0.9\}$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | $\{0.7, 0.9\}$ |
| $G$ | 8 ($W_0 = W_G(0)$ for GRPO) |
| Advantage rule | GRPO |
| Update | Exact |
| Optimizer, $\eta$, steps | SGD, $\eta \in \{0.1, 1\}$, 100,000 |
| Snapshots | `snapshot_at` = 10, 20, 50, 100, 200, 500, 1,000, 2,000, 5,000, 10,000, 20,000, 50,000, plus 100,000 |
| Start | Uniform |
| Tasks and seeds | Task seeds 0–9, seed 0 |
| KL $\beta$ | 0 |
| Recovery | Simulated $p_y > 0.1$ at the last step |
| Runs | 120 (118 qualify) |
| Pre-registration | `PREREG_v3d_addendum6.md` (`review` `93743f0`), exploratory with fixed rule |
| Grid and launch commit | `v3_t24`; `toy-v3` `5487480` |

### Result
The per-prompt model fails (median Spearman 0.28, needed 0.8), and it does not beat the baselines (69%, 67%, 53% of runs). Recovery counts hold (72%). The run's smallest $\gamma$ ranks its late margin growth (within-cell 0.86), which replicates the T12 probe. The late phase barely reorders prompts (Spearman 0.97–1.00 from $\tau_0$ to the end), so each prompt's outcome is set during the fall. Exploratory. See `extra/analysis/v3l/SUMMARY.md`.

## Odds law (theory check)

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** T11 and T15 measured that $R$ scales with the odds $m/(1-m)$ and that $m^*$ sits just above $1/K$, and we wanted an explanation.

### What I did
1. Derived the first-step change of each prompt's success for a softmax output layer at a uniform start: same-class prompts help each other with weight $1 - 1/K$, cross-class prompts hurt with weight $1/K$.
2. Summed over classes to get $R = S_{BA} / [(K-1) S_{\text{same}} - S_{\text{diff}}]$, and the odds form $R = c \cdot m/(1-m)$ with $c$ set by the kernel means.
3. Computed $R$, $c$ and $m^* = 1/(1 + c(m))$ from each task's features alone, with nothing fitted.
4. Compared them with the step-0 changes the simulator measured in the 1,080 v3e runs.

### Fixed settings
| Setting | Value |
| --- | --- |
| Scope | First step from a uniform start; linear policy, or MLP whose output layer starts at zero |
| Data | 1,080 v3e runs ($K \in \{2, 3, 5\}$, linear and MLP, $\rho \in \{0.5, 0.9\}$, nine $m$, ten tasks), step 0, $\eta = 0.01$ |
| Fitted parameters | None |
| Code | `extra/experiments/check_odds_law.py` |
| Outputs | `extra/analysis/theory/odds_law.md`, `odds_law_check.md`, `odds_law_*.csv` |

### Result
The relative error of $R$ has median $1.43 \times 10^{-3}$ and maximum $1.7 \times 10^{-2}$ (the second-order term of the finite step). Predicted and measured $m^*$ agree within 0.0002 in all 12 cells. This is a first-step identity only: it verifies the algebra and code, and says nothing about the outcome (T22, T23). Its leading order is in Li, Pandey and Chen. See `extra/analysis/theory/odds_law.md`.

## Lean proofs (theory check)

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** to check the algebra behind the odds law formally, not only numerically.

### What I did
1. Stated the facts in Lean 4 with Mathlib in `extra/formal/Formal/RLVR.lean`.
2. Proved tabular monotonicity: with a softmax on one prompt, the group-centred update does not lower the correct answer to first order (`tabular_first_order`, `tabular_first_order_nonneg`).
3. Proved the cross-class factor and its value $\mathbb{1}[a = b] - 1/K$ at a uniform start (`cross_class_factor`, `cross_class_factor_uniform`).
4. Proved the push ratio in block form and in odds form $c \cdot m/(1-m)$ (`push_ratio_block`, `push_ratio_odds`).
5. Built with `lake exe cache get && lake build`.

### Fixed settings
| Setting | Value |
| --- | --- |
| Toolchain | `leanprover/lean4:v4.37.0-rc1` (`extra/formal/lean-toolchain`) |
| Mathlib | `rev = "v4.37.0-rc1"` (`extra/formal/lakefile.toml`); manifest commit `31ce06e` |
| Theorems | 6, listed above |
| `sorry` | None in `RLVR.lean` |
| Scope | First-step algebra only; late-phase results are in the SymPy tests |

### Result
`lake build` passes with no `sorry` (`HANDOFF.md`, 11-23 21:40). Known algebra checked formally. See `extra/formal/README.md`.

## SymPy checks (theory check)

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** to check the first-step algebra and the late-phase algebra symbolically, including the scalar late model.

### What I did
1. Wrote unit tests in SymPy for the cross-class factor, the push ratio in block and odds form, the junk-answer case, and tabular monotonicity.
2. Solved the scalar late model $dL/d\tau = \gamma/\tau - c\,e^{-L}$ exactly (with $y = e^L$ it is linear).
3. Tested the permanence criterion, that pass@k divides the late exponent, and the KL restoring terms as the minority vanishes.

### Fixed settings
| Setting | Value |
| --- | --- |
| File | `extra/tests/test_theory_symbolic.py` |
| Tests | 8 |
| Library | SymPy (`unittest`) |

### Result
All 8 tests pass (`HANDOFF.md`, 11-23 21:40; rerun here: 8 passed). Permanent absorption needs $\gamma > 1$ and a deep enough fall. See `extra/tests/test_theory_symbolic.py` and `extra/analysis/theory/late_time.md`.

## Late-time probe (theory check)

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** to see whether implicit bias (Soudry et al.) explains the late phase: late in training the majority's update tends to its max-margin direction $\hat u$.

### What I did
1. For each linear T12 run, rebuilt the task and solved the majority's hard-margin problem (dual with nonnegative L-BFGS-B) for $\hat u$, scaled so the closest majority prompts have margin 1.
2. Computed each minority prompt's rate $\gamma_y = \hat u \cdot \phi_y$.
3. Run level: ranked M2's margin growth $g$ by the run's smallest $\gamma_y$.
4. Prompt level: tested "$\gamma_y < 1$ recovers" on prompts that fell below $10^{-3}$ (recovered = final success above 0.1).

### Fixed settings
| Setting | Value |
| --- | --- |
| Data | 400 linear T12 runs ($\rho \in \{0.5, 0.9\}$, $m \in \{0.7, 0.9\}$, 5 optimizer arms, exact and sampled, 10 tasks) |
| Max-margin solver | L-BFGS-B on the dual, `maxiter` 20,000, `ftol` $10^{-15}$, `gtol` $10^{-12}$ |
| Code | `extra/experiments/probe_late_time.py` |
| Outputs | `extra/analysis/theory/late_time.md`, `late_time_probe_rows.csv` |

### Result
Run level supported: Spearman 0.89 (SGD, exact), 0.85 (SGD, sampled), 0.76 and 0.84 (Adam). Prompt level refuted: under exact SGD 0% of 147 prompts with $\gamma_y < 1$ recovered. The run-level link replicated on pre-registered runs in T24 (0.86). Exploratory. See `extra/analysis/theory/late_time.md`.

## Optimizer geometry (theory check)

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** to see whether Adam's $\ell_\infty$ max-margin geometry explains why it recovers more often than SGD.

### What I did
1. For the same T12 linear runs, computed both the $\ell_2$ max-margin direction (L-BFGS-B dual) and the $\ell_\infty$ one (linear program: minimise $\|u\|_\infty$ subject to $\phi_A u \ge 1$, HiGHS).
2. Computed $\gamma_{\min}$ under each geometry.
3. Compared which one ranks the late sinking rate better within cells, under SGD and under Adam.

### Fixed settings
| Setting | Value |
| --- | --- |
| Data | Linear T12 runs (as the late-time probe) |
| Solvers | $\ell_2$: L-BFGS-B dual; $\ell_\infty$: `scipy.optimize.linprog`, method `highs` |
| Code | `extra/experiments/probe_optimizer_geometry.py` |
| Outputs | `extra/analysis/theory/late_time.md` (update), `optimizer_geometry_runs.csv` |

### Result
Not supported. Under SGD, $\ell_2$ beats $\ell_\infty$ as expected (within-cell 0.81 against 0.72). Under Adam, $\ell_\infty$ adds nothing (0.60 against 0.62, exact update). The two geometries correlate at 0.75 across tasks, so the test had power. See `extra/analysis/theory/late_time.md`.

## Fall model (theory check)

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** T24 showed that each prompt's outcome is decided during the fall (phase 1), and there was no theory of it.

### What I did
1. Reduced the exact SGD flow of the linear policy to class means (the "class model"), then pooled the minority classes into five log-odds (the class-symmetric model, primary). Per-prompt "test particles" move in the class model's field.
2. Fixed six predictions P1–P6 (depth rank, depth level, within-cell rank, absorbed or not, threshold in $m$, per-prompt order) before any comparison.
3. In-sample: predicted all 6,980 runs of v3e, v3g T19, v3j, v3k and v3l from configuration and starting success alone, committed, then compared.
4. Held-out: froze the model, predicted the 1,360 in-scope v3o runs, committed before reading any v3o outcome, then compared.

### Fixed settings
| Setting | Value |
| --- | --- |
| Scope | Linear policy, exact update, SGD (flow limit, flow time $\tau = \eta t$); sampled and MLP runs reported separately |
| Models | Class-symmetric (`sym`, primary), class model (`cls`), test particles (`tp`) |
| Fitted parameters | None (block kernel means, shares, GRPO weight $W_G$, start, horizon $T = \eta \cdot$ steps) |
| Solver | `solve_ivp` LSODA in log time, `rtol` $10^{-8}$, `atol` $10^{-10}$, $\tau_0 = 10^{-6}$, 400 points |
| Measured depth | Median over minority prompts of $\log_{10}$ of each prompt's lowest success |
| Absorption threshold | $10^{-3}$ over the last 20% (run absorbed if at least half the minority is) |
| In-sample data | v3e, v3g T19, v3j, v3k, v3l (6,980 runs; predictions committed `fb7c0d2`) |
| Held-out data | v3o, fresh seeds 10–29: `v3_a1` (1,280), `v3_a2` $\beta = 0$ (40), `v3_a5` (40); predictions committed `9421c44` |
| Model frozen at | `analysis-v3` `afef9a4` |
| Pre-registration | `fall.md` section 5; held-out rule `extra/analysis/review/PREREG_fall_heldout.md` (`review` `5275c11`) |
| Code | `extra/experiments/fall_ode.py`, `extra/experiments/check_fall.py`, `extra/tests/test_fall_ode.py` |

### Result
In-sample (fit only): passes every rule in every batch except P1 in v3l (0.799). Held-out: passes all rules on the KL and warm-start grids. On the P1 grid it passes P2–P6 and the threshold check (15/16 cells) but fails P1 (0.755) because of large-step runs ($\eta \ge 3$). For $\eta \le 1$ the depth rank is 0.988. Supported out of sample in the flow regime (exploratory). See `extra/analysis/theory/fall.md` and `extra/analysis/theory/fall_results.md`.

## wsw: replication of Li, Pandey and Chen

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** their environment is the base of T20, so we checked it before building on it.

### What I did
1. Ran their code at the pinned commit with their own scripts: all 11 experiments, plus `theory.py` and `exp11_kernel.py`.
2. Ran each job single-threaded in a fresh copy of the code, split into the shards their scripts support (`wsw_jobs.py`, `launch_wsw.sh`).
3. Control: ran their `figures.py` on their own published `results/` and compared with their `numbers.json`.
4. Ran `figures.py` on our results and compared number by number (`compare_numbers.py`), and compared the headline drift table.

### Fixed settings
| Setting | Value |
| --- | --- |
| Code | github.com/Rohan-Pandey1729/RLVR-Reward-Hacking at `05ef005` (MIT) |
| Experiments | E1 and E8 (universality, Adam and SGD, 35 shards each), E2, E3, E4, E5, E7, E9, E9b (20 shards each), E10 (40 shards), E6, E11 (seeds 0–2), `theory.py`, `exp11_kernel.py` |
| Seeds | Script defaults (E1, E8: seeds 0–4) |
| Environment | SECB lab CPUs, torch 2.15.0, single-threaded; matplotlib 3.9.2, numpy 1.26.4 |
| Jobs | 256 |
| Pre-registration | None |
| Code (ours) | `extra/tools/literature/replication/` (branch `literature`); launch commit not in the index (see `wsw_MANIFEST.txt`) |

### Result
Their figure code on their results reproduces 494/494 numbers. Our rerun: of 506 numbers, 130 are identical, 193 within 2%, 183 different (the base models already differ across CPUs). The headline claim replicates: with shared parameters the drift rises with the number of prompts sharing a leak, and without sharing it is flat. See `extra/tools/literature/replication/RESULT.md`.

## wsw2: replication, seeds 5–9

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** their README runs E1 and E8 with ten seeds, but the script default is five.

### What I did
1. Ran E1 and E8 again for seeds 5–9, with the same scripts and shards.
2. Merged the result files under shard numbers offset by 100, so they do not collide with part 1.
3. Included them in the comparison above.

### Fixed settings
| Setting | Value |
| --- | --- |
| Experiments | `exp2_universality.py --opt adam` and `--opt sgd`, `--seeds 5,6,7,8,9`, 35 shards each |
| Environment | As wsw |
| Jobs | 70 |
| Pre-registration | None |
| Code | `wsw_jobs.py --part 2` (branch `literature`) |

### Result
Every job exited cleanly; included in the 506-number comparison. See `extra/tools/literature/replication/RESULT.md`.

## T25 (v3m): exact long-time flow

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** T12 could not tell at $\rho = 0.9$ whether the collapse is permanent or a long plateau.

### What I did
1. Integrated the exact gradient flow of the linear policy under the exact expected GRPO update, in log time, to flow time $10^{14}$ (checked against discrete SGD to within 0.015 decades).
2. Stored every prompt's log-success at 80 log-spaced times from $10^{-1}$ to $10^{14}$.
3. From each minority prompt's state at $\tau_0$ (majority mean at least 0.99), predicted its recovery time with the exact solution of $dL/d\tau = \gamma/\tau - c\,e^{-L}$.
4. Compared with the observed recovery time (first time $p_y \ge 0.5$), and counted the tasks whose minority mean is back above 0.5 at $10^{14}$.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear |
| Features | Correlated, $\rho \in \{0.3, 0.5, 0.7, 0.9, 0.95\}$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | $\{0.7, 0.9\}$ |
| $G$ | 8 ($W_G(0) = 2.6456$) |
| Advantage rule | GRPO, exact expected update (gradient flow, no step size) |
| Flow time | $10^{-1}$ to $10^{14}$, 80 log-spaced points |
| Solver | LSODA, `rtol` $10^{-8}$, `atol` $10^{-10}$; two stiff tasks finished with Radau (`rtol` $10^{-7}$, `atol` $10^{-9}$) and BDF (`rtol` $10^{-6}$, `atol` $10^{-8}$) |
| Start | Uniform |
| Tasks and seeds | Task seeds 0–9 (100 tasks), seed 0 |
| KL $\beta$ | 0 |
| Pre-registration | `PREREG_v3d_addendum7.md` (`review` `15a86b0`), exploratory with fixed rule |
| Code and commit | `extra/experiments/exact_flow.py` (`toy-v3` `3bf676f`), `extra/experiments/check_permanence.py`, `extra/tests/test_exact_flow.py`; computed on the laptop |

### Result
All 100 tasks were integrated (98 with LSODA). The per-prompt predictions fail (48% of "never" prompts recover; timing Spearman 0.34). Mostly a plateau whose length grows with $\rho$: recovery at $10^2$–$10^5$ for $\rho \le 0.5$, up to $10^{12}$ for $\rho \ge 0.9$. Nothing recovers at $\rho = 0.95$, $m = 0.9$. Post hoc, $\gamma_{\min} \ge 1$ marks the tasks that stay collapsed (27 of 30). Exploratory. See `extra/analysis/v3m/SUMMARY.md`.

## T26 (v3n): shared skills

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** in the LLM (C10), training only on the majority procedure raises the minority, and shared steps between procedures were the leading explanation.

### What I did
1. Used four-token answers, where success needs all four tokens.
2. Made positions $0..k-1$ carry the answer class (the "procedure"), and positions $k..3$ follow one balanced rule shared by every prompt (`differing_positions = k`).
3. Computed the push ratio per task from the multi-token overlap sums, with nothing fitted.
4. Measured step-0 branches, branches at 10% and 50% of training, and persistent absorption.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 |
| Features | Correlated, $\rho \in \{0.5, 0.9\}$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | $K = 3$, $L = 4$ |
| Procedure positions $k$ | $\{1, 2, 3, 4\}$ ($k = 4$: no shared skill) |
| Majority share $m$ | $\{0.7, 0.9\}$ |
| $G$ | 8 |
| Advantage rule | GRPO |
| Update | Exact |
| Optimizer, $\eta$, steps | SGD, 0.1, 10,000 |
| Branch points | `branch_at = [0, 0.1, 0.5]` |
| Start | Uniform |
| Tasks and seeds | Task seeds 0–9, seed 0 |
| KL $\beta$ | 0 |
| Absorption threshold | $10^{-3}$, last 20% |
| Runs | 320 |
| Pre-registration | `PREREG_v3d_addendum8.md` (`review` `2334a5d`), exploratory |
| Grid and launch commit | `v3_t26`; `toy-v3` `c991964` |

### Result
The first-step identity holds with multi-token answers (320/320, median error 0.2%). Sharing dilutes the push (median $R$ falls from 3.5–17 to 2–9) but does not prevent collapse (98–100% absorbed in 6 of 8 cells at $k = 1$). Design limit: shared positions agree only at chance, so the positive-transfer test (prediction 2) never applied. Exploratory. See `extra/analysis/v3n/SUMMARY.md`.

## A1 (v3o): P1 on fresh tasks

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** to turn P1 (absorption rises with the majority share, v3a) from exploratory into confirmatory.

### What I did
1. Reran the full `v3_t1` grid unchanged on fresh task seeds 10–29 (`fresh`).
2. Per cell (architecture, $\rho$, optimizer, update; 16 cells), averaged each task over the cell's no-overshoot $\eta$ set.
3. Ran a one-sided sign test that the per-minority-prompt absorption rate at $m = 0.9$ exceeds that at $m = 1/3$, over 20 tasks.
4. Took A1's p-value as the largest of the 16 cell p-values, and entered it in one Holm family of five (with A2–A5).

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 |
| Features | Correlated, $\rho \in \{0.5, 0.9\}$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | $\{1/3, 0.5, 0.7, 0.9\}$ |
| $G$, $B$ | 8, 32 |
| Advantage rule | GRPO |
| Arms | SGD $\eta \in \{0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30\}$, steps `flow_steps` (flow time 100, clipped to 1,000–20,000); Adam $\eta \in \{10^{-4}, 3 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}, 10^{-2}, 3 \times 10^{-2}\}$, 5,000 steps; each exact and sampled; plus a supervised arm (SGD $\eta = 1$, 2,000 steps) |
| Start | Uniform |
| Tasks and seeds | Task seeds 10–29 (`REPLICATION_SEEDS`), seed 0 |
| KL $\beta$ | 0 |
| Absorption threshold | $10^{-3}$, last 20%, among prompts starting at or above $10^{-2}$ |
| Test | Sign test per cell; max over 16 cells; Holm over 5 at $\alpha = 0.05$ |
| Runs | 9,280 |
| Pre-registration | `PREREG_v3d_addendum9.md` (`review` `e9510e4`), confirmatory |
| Grid and launch commit | `v3_a1` (= `fresh(v3_t1)`); `toy-v3` `4391e1a`, job digest `8b81b8a2e7c5` |

### Result
Confirmed: 20 of 20 tasks go up in every one of the 16 cells, $p = 9.5 \times 10^{-7}$, Holm $p = 4.8 \times 10^{-6}$. At $m = 0.9$ the rate is 0.37–1.00 (originally 0.54–1.00). It is a little smaller in three $\rho = 0.5$ cells. Confirmatory. See `extra/analysis/v3o/SUMMARY.md`.

## A2 (v3o): forward-KL floor on fresh tasks

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** to make T17's forward-KL result confirmatory.

### What I did
1. Reran the T17 arms with no anchor and with forward KL at $\beta \in \{0.03, 0.1, 0.3\}$ on fresh tasks.
2. Per (architecture, update, $\eta$) cell and task, computed minority absorption at $\beta = 0$ minus the largest share among the three forward-KL arms.
3. Took each task's median over the 8 cells and ran a one-sided sign test over 20 tasks.
4. Reported as secondary the forward-KL runs with no minority prompt absorbed at $10^{-3}$ and at $10^{-2}$.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 |
| Features | Correlated, $\rho = 0.9$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | 0.9 |
| $G$, $B$ | 8, 32 |
| Advantage rule | GRPO |
| Update | Exact and sampled |
| Optimizer, $\eta$, steps | SGD, $\eta \in \{0.1, 1\}$, 20,000 |
| KL | $\beta = 0$, or forward $\beta \in \{0.03, 0.1, 0.3\}$; reference = start policy |
| Start | Uniform |
| Tasks and seeds | Task seeds 10–29, seed 0 |
| Absorption threshold | $10^{-3}$, last 20%, among prompts starting at or above $10^{-2}$ (secondary: $10^{-2}$ among prompts starting at or above $10^{-1}$) |
| Test | Sign test over 20 tasks; Holm over 5 |
| Runs | 640 |
| Pre-registration | `PREREG_v3d_addendum9.md` (`review` `e9510e4`), confirmatory |
| Grid and launch commit | `v3_a2` (subset of `v3_t17`); `toy-v3` `4391e1a` |

### Result
Confirmed: 20/20 tasks, median difference 1.00, Holm $p = 4.8 \times 10^{-6}$. No minority prompt absorbed at $10^{-3}$ in 480/480 forward-KL runs. At $10^{-2}$ protection is partial at $\beta = 0.03$ (123/160 runs clean). The minority still falls to about 0.02 while the majority is learned. Confirmatory. See `extra/analysis/v3o/SUMMARY.md`.

## A3 (v3o): pass@4 at matched progress on fresh tasks

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** to make T16's pass@4 result confirmatory.

### What I did
1. Reran the T16 arms with GRPO, pass@4 and unbiased pass@4 on fresh tasks.
2. Per task, took the minority mean at the matched point (majority at 0.80) under pass@4 (best of the two forms per $(G, \eta)$ cell, median over cells) minus under GRPO.
3. Counted runs where the majority never reaches 0.80 as 0 for pass@4 (against it).
4. Ran a one-sided sign test over 20 tasks.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear |
| Features | Correlated, $\rho = 0.9$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | 0.9 |
| $G$, $B$ | $G \in \{8, 32\}$, $B = 256/G$ |
| Advantage rule | `grpo`, `passk` with $k = 4$, `rloo_k` with $k = 4$ |
| Update | Sampled |
| Optimizer, $\eta$, steps | SGD, $\eta \in \{0.03, 0.3, 1\}$, 20,000 |
| Start | Uniform |
| Tasks and seeds | Task seeds 10–29, seed 0 |
| KL $\beta$ | 0 |
| Test | Sign test over 20 tasks; Holm over 5 |
| Runs | 360 |
| Pre-registration | `PREREG_v3d_addendum9.md` (`review` `e9510e4`), confirmatory |
| Grid and launch commit | `v3_a3` (subset of `v3_t16`); `toy-v3` `4391e1a` |

### Result
Confirmed: 20/20 tasks, median difference 0.39 (range 0.27–0.45), Holm $p = 4.8 \times 10^{-6}$. The minority mean at the matched point is 0.28–0.44 under pass@4, against $5 \times 10^{-5}$ to $5 \times 10^{-3}$ under GRPO. The cost is unchanged: at $\eta = 0.03$ the majority often never reaches 0.80. Confirmatory. See `extra/analysis/v3o/SUMMARY.md`.

## A5 (v3o): warm start on fresh tasks

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** to make T23's result (a warm start at $p_0 = 0.5$ protects at $m = 0.7$) confirmatory.

### What I did
1. Reran the T23 arms at $m = 0.7$ with the uniform start and with the class-balanced warm start at $p_0 = 0.5$, on fresh tasks.
2. Per (architecture, update) cell and task, computed minority absorption at the uniform start minus at $p_0 = 0.5$.
3. Took each task's median over the 4 cells and ran a one-sided sign test over 20 tasks.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 |
| Features | Correlated, $\rho = 0.9$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | 0.7 |
| $G$, $B$ | 8, 32 |
| Advantage rule | GRPO |
| Update | Exact and sampled |
| Optimizer, $\eta$, steps | SGD, 0.1, 10,000 |
| Start | Uniform, or `sft_balanced` with $p_0 = 0.5$ |
| Tasks and seeds | Task seeds 10–29, seed 0 |
| KL $\beta$ | 0 |
| Absorption threshold | $10^{-3}$, last 20%, among prompts starting at or above $10^{-2}$ |
| Test | Sign test over 20 tasks; Holm over 5 |
| Runs | 160 |
| Pre-registration | `PREREG_v3d_addendum9.md` (`review` `e9510e4`), confirmatory |
| Grid and launch commit | `v3_a5` (subset of `v3_t23`); `toy-v3` `4391e1a` |

### Result
Confirmed: 20/20 tasks, median difference 1.00 (lowest 0.79), Holm $p = 4.8 \times 10^{-6}$. The uniform start absorbs every minority prompt in all 80 runs. At $p_0 = 0.5$ the absorbed share is 0.022 (linear) and 0.0003–0.003 (MLP). Confirmatory. See `extra/analysis/v3o/SUMMARY.md`.

## G (v3p, T27): forward KL toward a collapsed reference

**Scope: Core follow-up** — not in the milestone; grew out of a promised result, and the report's toy story relies on it.

**Why:** a review caveat: T17's reference (the start policy) still solved the minority, while a real reference may already be collapsed.

### What I did
1. Trained the minority task with plain GRPO.
2. At 0.5%, 5% or 25% of training (steps 100, 1,000, 5,000), switched on a forward KL term toward the policy as it was at that step (`anchor_at`), with all classes keeping their reward.
3. Controls: no anchor, and forward KL toward the start policy from step 0 (as T17).
4. Tested whether late anchors restore the minority (minority at the end at most 10× its value at the anchor), and whether the start anchor prevents absorption.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 |
| Features | Correlated, $\rho = 0.9$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | 0.9 |
| $G$ | 8 |
| Advantage rule | GRPO |
| Update | Exact |
| Optimizer, $\eta$, steps | SGD, 0.1, 20,000 |
| Anchor | `anchor_at` $\in \{0.005, 0.05, 0.25\}$, `anchor_kl = "forward"`, `anchor_beta` $\in \{0.01, 0.1, 1\}$ |
| Controls | No anchor; forward KL to the start policy at $\beta \in \{0.01, 0.1, 1\}$ |
| Start | Uniform |
| Tasks and seeds | Task seeds 0–9, seed 0 |
| Absorption threshold | $10^{-3}$, last 20%, among prompts starting at or above $10^{-2}$ |
| Runs | 260 |
| Pre-registration | `PREREG_v3d_addendum10.md` (`review` `e9510e4`), exploratory |
| Grid and launch commit | `v3_t27`; `toy-v3` `4391e1a` |

### Result
Holds. Anchors taken after the fall never restore the minority (180/180), and the start anchor at $\beta \ge 0.1$ prevents absorption (40/40). The fall ends by step 100, so even the 0.5% anchor comes after it. Forward KL protects only while the reference still solves the minority. Exploratory. See `extra/analysis/v3p/SUMMARY.md`.

## H (v3p, T28): Adam's epsilon

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** Adam recovers the minority more often than SGD (T12), and per-coordinate normalisation was the suspected cause.

### What I did
1. Trained the T1 task with Adam at six values of $\epsilon$. A large $\epsilon$ turns Adam into momentum SGD with step $\eta/\epsilon$, so the lr grew with $\epsilon$.
2. Ran an SGD control.
3. Called a run "recovered" if its minority mean fell below $10^{-3}$ at some logged step and was above 0.1 at the end.
4. Reported the late exponent $g$ and $\gamma_{\min}$ under $\ell_2$ and $\ell_\infty$ geometry.

### Fixed settings
| Setting | Value |
| --- | --- |
| Policy | Linear, and MLP with width 64 |
| Features | Correlated, $\rho \in \{0.5, 0.9\}$, $d = 16$ plus a constant, 512 prompts |
| $K$, $L$ | 3, 1 |
| Majority share $m$ | $\{0.7, 0.9\}$ |
| $G$ | 8 |
| Advantage rule | GRPO |
| Update | Exact |
| Optimizer | Adam with `adam_eps` $\in \{10^{-8}, 10^{-5}, 10^{-3}, 10^{-2}, 0.1, 1\}$ at lr $= \max(10^{-3}, 0.1\epsilon)$; SGD control at $\eta = 0.1$ |
| Steps | 10,000 |
| Start | Uniform |
| Tasks and seeds | Task seeds 0–9, seed 0 |
| KL $\beta$ | 0 |
| Runs | 560 |
| Pre-registration | `PREREG_v3d_addendum10.md` (`review` `e9510e4`), exploratory |
| Grid and launch commit | `v3_t28`; `toy-v3` `4391e1a` |

### Result
Fails as registered (4 of 8 cells): half the cells have no Adam recovery to remove. Post hoc: where Adam recovers (MLP, $\rho = 0.5$, $m = 0.7$), recovery drops from 5 of 10 to 0 of 10 between $\epsilon = 10^{-3}$ and $10^{-2}$ at the same lr. This points to per-coordinate normalisation. Exploratory. See `extra/analysis/v3p/SUMMARY.md`.

## A4 (v3q): T20's share effect on fresh seeds

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** to make the transformer-environment share effect under Adam confirmatory, as part of the battery.

### What I did
1. Reran T20's shared-transformer cells under Adam at lr $10^{-3}$ on fresh partition seeds 10–29, with both algorithms.
2. Per seed, computed the fraction of high-share tasks (share at least 0.875) whose minority is absorbed (at least half of its prompts below $10^{-3}$ while class 0 is not), minus that fraction for balanced tasks.
3. Ran a one-sided sign test over 20 seeds, in the same Holm family as A1–A3 and A5.

### Fixed settings
| Setting | Value |
| --- | --- |
| Environment | As T20 (`enum_env.py`, their code at `05ef005`) |
| Policy | Shared transformer |
| Tasks ($K$, `n_major`) | $K = 3$: 22, 32, 40, 48, 56, 60; $K = 2$: 32, 40, 48, 56, 60 (random partition) |
| Advantage rule | Exact `pg` and sampled `grpo`, $G = 8$ |
| Optimizer and lr | Adam, $10^{-3}$ |
| Steps | 2,000, logged every 20 |
| Seeds | Partition seeds 10–29 |
| KL | none |
| Absorption threshold | $10^{-3}$, last 20%; class-level rule: at least half of a class's prompts |
| Test | Sign test over 20 seeds; Holm over 5 |
| Runs | 440 (the addendum's 480 was an arithmetic error) |
| Pre-registration | `PREREG_v3d_addendum9.md` (`review` `e9510e4`), confirmatory |
| Grid and launch commit | `v3_a4`; `enum-env` `4b271a0`, job digest `299bb326e002` |

### Result
Confirmed: 20/20 seeds, mean difference 0.89 (original 0.86), Holm $p = 4.8 \times 10^{-6}$. The minority is absorbed while class 0 is not in 152 of 160 high-share runs, and in 5 of 80 balanced runs. Per prompt, about a fifth of each class is still absorbed at balance. Confirmatory. See `extra/analysis/v3o/SUMMARY.md` (A4 is part of the battery).

## J (v3q, T29): transformer environment under small-step SGD

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** a review caveat: in T20 large SGD steps absorbed both classes, so the share effect looked clean only under Adam.

### What I did
1. Reran T20's $K = 3$ share grid under SGD at two small step sizes.
2. Matched the flow time to SGD $0.3 \times 2{,}000$ steps.
3. Used the exact update and sampled GRPO.
4. Tested three rules: class 0 never absorbed at high share, minority absorbed in at least 80% of high-share runs, and in at most 20% of balanced runs.

### Fixed settings
| Setting | Value |
| --- | --- |
| Environment | As T20 (`enum_env.py`, their code at `05ef005`) |
| Policy | Shared transformer |
| Tasks | $K = 3$, `n_major` $\in \{22, 32, 40, 48, 56, 60\}$, random partition |
| Advantage rule | Exact `pg` and sampled `grpo`, $G = 8$ |
| Optimizer, $\eta$, steps | SGD $0.03 \times 20{,}000$ and $0.1 \times 6{,}000$; logged every `steps // 100` |
| Seeds | 0–9 |
| KL | none |
| Absorption threshold | $10^{-3}$, last 20%; class-level rule: at least half of a class's prompts |
| Runs | 240 |
| Pre-registration | `PREREG_v3d_addendum10.md` (`review` `e9510e4`), exploratory |
| Grid and launch commit | `v3_t29`; `enum-env` `4b271a0` |

### Result
Holds. At small SGD steps class 0 is never absorbed at high share (0/80), the minority always is (80/80), and rarely at balance (4/40). T20's SGD failure was a large-step effect. Per prompt, 8–28% of prompts are still lost at balance. Exploratory. See `extra/analysis/v3q/SUMMARY_J.md`.

---

# Part 3: small language models

## C0: pilot runs before the census

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** Before C1 we needed to know whether the GPU pipeline works, how fast it is, which learning rate to use, and whether our per-problem decline test gives false alarms.

### What I did
1. Evaluated base Qwen2.5-0.5B-Instruct and SmolLM2-1.7B-Instruct on the first 3,000 GSM8K training problems (32 samples each, seeds 0 and 1) and on the 1,319 test problems.
2. Compared the two base evaluations of each model with each other, so every flagged problem is a false alarm. The test is a one-sided Fisher exact test of $p_{after} < p_{before}$ with Benjamini–Hochberg (BH) at $q = 0.10$. A problem is "lost" if it declines significantly and ends at $\hat p \le 0.05$.
3. Trained Qwen for 150 steps on training problems 0–999 at lr $10^{-6}$, $3\cdot10^{-6}$ and $10^{-5}$. Then I evaluated each model on the same 1,000 problems (32 samples each) and counted declines, losses and gains.
4. Built the problem pool: the problems whose pooled base $\hat p$ (64 samples) lies in $[0.1, 0.9]$.
5. Ran the SmolLM2 LoRA pilots, the 256-sample base evaluations for C1, and the C2 difficulty pilot and candidate evaluation.

### Fixed settings
| Setting | Value |
| --- | --- |
| Models | Qwen2.5-0.5B-Instruct (full fine-tuning, fp32 master weights, 8-bit AdamW); SmolLM2-1.7B-Instruct (LoRA $r = 64$, $\alpha = 128$, AdamW) |
| Base evaluations | 3,000 train problems × 32 samples, seeds 0 and 1; test set 1,319 × 32 |
| Qwen lr pilots | $10^{-6}$, $3\cdot10^{-6}$, $10^{-5}$; 150 steps on train problems 0–999 |
| SmolLM2 pilots | LoRA, lr $10^{-5}$, $3\cdot10^{-5}$, $10^{-4}$ (tag `c0b`, invalid: weight-sync bug); rerun at $10^{-5}$, $3\cdot10^{-5}$, $10^{-4}$, $3\cdot10^{-4}$ on `extra/llm/pools/smol_c1_train.json` with `--check-sync` (`c0d`, `c0e`) |
| $G$, prompts per step | 8, 64 (`train.py` defaults) |
| Loss aggregation | Sequence level: $-\sum A \log\pi(\text{completion}) / (64 \cdot 8)$ |
| KL | $\beta = 0$ |
| Evaluation after the pilot | 32 samples on the 1,000 training problems |
| Decline test | One-sided Fisher exact, BH at $q = 0.10$ (`extra/llm/declines.py`); lost = decline ending at $\hat p \le 0.05$ |
| Pool rule | Pooled base $\hat p \in [0.1, 0.9]$: 2,239 of 3,000 (`extra/llm/pools/qwen.json`) |
| 256-sample base evaluations | Qwen and SmolLM2 on the 1,300 C1 problems, seeds 0 and 1 (`c0d`) |
| C2 difficulty pilot | Ranges small, medium, large, xl; 200 problems per procedure; 32 samples (`c0d`). Candidates: range `ml`, 3,000 per procedure, seed 1, 32 samples (`c0e`) |
| Job lists | `extra/llm/jobs/c0_pilot.txt`, `c0d.txt`, `c0e.txt` |
| Run tags | `c0`, `c0b` (SmolLM2, invalid), `c0c` (Qwen lr $10^{-5}$ rerun after a parser overflow), `c0d`, `c0e` |
| Pre-registration | None; the lr rule was then written into `PREREG_C1.md` |
| Code commit | `c0` `5ef76c1`, `c0b` `cd5f9ab`, `c0c` `815ed5b`, `c0d` `a175b73`, `c0e` `4f637df` (old hashes); analysis `extra/llm/analyze_c0.py` at `80ddc4a` |

### Result
The base model against itself gives 0 declines on 3,000 problems. At 32 samples, though, a drop to 0 is detectable only from $\hat p \ge 0.375$; at 256 samples the threshold falls to 0.055, so C1 uses 256. lr $10^{-6}$ was chosen (mean $\hat p$ 0.41 → 0.75, 0 declines). lr $10^{-5}$ is a collapse, not selective loss (247 declines, 168 lost, reward falling from the start). Exploratory. See `extra/analysis/llm/c0_qwen.md`.

## Model probes: which models fit one GPU

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** We had to choose models and fine-tuning modes that train with vLLM on one 24 GB RTX 4090, and check that the sampler stays in step with the trainer.

### What I did
1. For each model and mode, evaluated the base model on the first 64 GSM8K training problems (8 samples each), while logging GPU memory every 500 ms.
2. Ran 5 GRPO steps on `extra/llm/pools/first1000.json` with `--check-sync`, saving at step 3. The sync check logs the vLLM and policy log-probs of one sampled answer after each of the first three weight syncs.
3. Evaluated the step-3 model on the same 64 × 8.
4. Collected peak memory, seconds per step, base $\hat p$ and the sync gap with `extra/llm/analyze_probe.py`.

### Fixed settings
| Setting | Value |
| --- | --- |
| Models | Qwen2.5-0.5B, 1.5B, 3B, 7B-Instruct; SmolLM2-360M-Instruct; SmolLM3-3B; Llama-3.2-1B and 3B-Instruct |
| Modes | Full fine-tuning; LoRA $r = 64$; int8 base with LoRA (`--quant int8 --sync adapter`); nf4 |
| lr | $10^{-5}$ (full), $10^{-4}$ (LoRA) |
| Steps | 5 (checkpoint at step 3) |
| $G$, prompts per step | 8, 64 (defaults) |
| Evaluation | First 64 train problems × 8 samples, before and after |
| Sync gap | Largest $\lvert$vLLM − policy$\rvert$ log-prob of one 64-token answer over the first syncs |
| Job lists | `extra/llm/jobs/probe_open.txt`, `probe_retry.txt`, `probe_llama.txt` |
| Run tags | `q_01`–`q_20` |
| Pre-registration | None |
| Code commit | `q_01`–`q_09` `b69182b`, `q_10`–`q_20` `c5230d1` (old hashes) |

### Result
Six set-ups work: Qwen2.5-0.5B int8, Qwen2.5-1.5B LoRA, Qwen2.5-3B int8, SmolLM2-360M full, and Llama-3.2-1B full and LoRA. The other set-ups failed (see the table). Llama-3.2-1B full fine-tuning (base $\hat p$ 0.30, sync gap 0.14 nats, about 21 GPU-hours per arm) later became the second family for C1. See `extra/analysis/llm/model_probe.md`.

## Census calibration: what an unchanged model looks like

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** Other work counts "locked" problems (some correct before, none after). We wanted to know how many problems such naive rules flag when nothing has changed, compared with our decline and lost tests.

### What I did
1. Took two base evaluations of the same model with different sampling seeds and treated them as "before" and "after". Every flagged problem is a false alarm.
2. For $N' \in \{8, 16, 32, 64, 128, 256\}$, drew $N'$ of the samples per problem without replacement (200 draws), and counted the problems flagged by four rules. Locked: correct before > 0 and correct after = 0. Halved: $p_{after} \le p_{before}/2$ with $p_{before} \ge 0.1$. Decline: Fisher with BH at 10%. Lost: a decline ending at $\le 0.05$.
3. Repeated step 2 on unfiltered problems from another model (at most 32 samples).
4. Measured power: planted a true drop of 50%, 75% or 100% in a random 5% of problems (50 draws) and counted how many of them each rule finds.

### Fixed settings
| Setting | Value |
| --- | --- |
| Filtered pool | Qwen2.5-0.5B base, 1,300 C1 problems (base $\hat p \in [0.1, 0.9]$), 256 samples, seeds 0 and 1 (`base_qwen_n256_s0/s1`, tag `c0d`) |
| Unfiltered set | Llama-3.2-1B base, 3,000 GSM8K train problems, 32 samples, seeds 2 and 3 (tags `r2_00`, `r2_01`) |
| Subsample sizes | 8, 16, 32, 64, 128, 256 (filtered); 8, 16, 32 (unfiltered) |
| Draws | 200 per size (`--draws` default); power table $\min(200, 50) = 50$ |
| RNG seed | 0 |
| Decline test and FDR | One-sided Fisher exact, BH at $q = 0.10$ |
| Training | None |
| Pre-registration | None |
| Code commit | `extra/llm/census_calibration.py` (`llm-transfer` `b4a7dee`); write-up `analysis-v3` `8db6862`, scope fixed in `cdfa10c` / `e83fe10` |

The input files are not named in the write-up. I identified them by rerunning the script with these inputs, which reproduces `census_calibration.md` exactly.

### Result
Our decline and lost tests flag 0 problems at every sample count. The naive rules flag many at small counts. On the filtered pool, "locked" flags 59 at 8 samples and 3 at 32; "halved" flags 200 at 8 samples and 48 at 32. On 3,000 unfiltered problems at 32 samples, "locked" flags 83. Both naive rules reach 0 by 256 samples. The result is scoped to our models and sample counts. It is not a re-analysis of other papers. See `extra/analysis/llm/census_calibration.md`.

## Llama calibration: learning rate and pool for the second family

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** P4 needs a second model family. SmolLM2 failed its lr rule, so Llama-3.2-1B replaced it (amendment 1), and its lr had to be chosen by the same pre-registered rule.

### What I did
1. Evaluated base Llama on the first 3,000 train problems (32 samples, seeds 2 and 3) and on the test set. Seeds 2 and 3 keep the pool independent of the later "before" (seed 0) and null (seed 1) draws.
2. Trained for 150 steps on train problems 0–999 at lr $10^{-6}$, $3\cdot10^{-6}$ and $10^{-5}$, then evaluated each model on those problems (32 samples).
3. Applied the rule with `extra/llm/pick_lr.py`: the smallest lr at which mean $\hat p$ rises by $\ge 0.05$ with $\le 2$ BH-significant declines at $N = 32$.
4. Built the pool (pooled $\hat p \in [0.1, 0.9]$), drew 1,000 training and 300 held-out problems with seed 0 (`extra/llm/make_sets.py`), and ran the two 256-sample base evaluations of those 1,300 problems (seeds 0 and 1).

### Fixed settings
| Setting | Value |
| --- | --- |
| Model | Llama-3.2-1B-Instruct, full fine-tuning, fp32 master weights, 8-bit AdamW, micro-batch 1 |
| lr pilots | $10^{-6}$, $3\cdot10^{-6}$, $10^{-5}$; 150 steps on `extra/llm/pools/first1000.json` |
| $G$, prompts per step | 8, 64 (defaults) |
| KL | $\beta = 0$ |
| Pool | 2,157 of 3,000 (`extra/llm/pools/llama.json`); sets `llama_c1_{train,heldout,eval}.json` (1,000 / 300 / 1,300) |
| Evaluation | 32 samples (pilots); 256 samples, seeds 0 and 1, on 1,300 problems (C1 base) |
| lr rule | Rise $\ge 0.05$ and $\le 2$ declines (BH $q = 0.10$, $N = 32$) |
| Job lists | `extra/llm/jobs/llama_c0.txt`; `extra/llm/jobs/llama_arm.sh` (pool, lr choice, C1 queueing) |
| Run tags | `r2_00`–`r2_05` (calibration), `r2_18`, `r2_19` (256-sample base) |
| Pre-registration | `extra/analysis/review/PREREG_C1_amendment1.md` |
| Code commit | `r2_00`–`r2_05` `8002345`; `r2_18`, `r2_19` `d8b43f1` |

### Result
lr $10^{-6}$ meets the rule: a rise of +0.490 with 0 declines. $3\cdot10^{-6}$ also meets it (+0.478, 1 decline), and $10^{-5}$ fails it (+0.105, 72 declines). The base $\hat p$ is 0.33 on train and 0.26 on test. See `~/rlvr-queue/llama/pick_lr.log` and `c0_llama.md` (not in git; that file labels the base seeds "0 and 1", but the job list uses seeds 2 and 3).

## C1 Qwen / P4: does GRPO lose problems beyond the null?

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** This is the confirmatory test of P4. The toy loses minority prompts under GRPO. Does small-LLM GRPO lose GSM8K training problems beyond what sampling noise alone produces?

### What I did
1. Drew 1,000 training and 300 held-out problems with seed 0 from the Qwen pool (`extra/llm/pools/qwen_c1_*.json`).
2. Trained Qwen with GRPO for 500 steps at lr $10^{-6}$ (primary arm) and $3\cdot10^{-6}$ (secondary arm), 3 seeds each.
3. Evaluated the final model on all 1,300 problems (256 samples, seed 0), plus a 32-sample trajectory point at step 250. "Before" is base seed 0 and the null is base seed 1, both at 256 samples.
4. Per problem: one-sided Fisher exact test, with BH at 10% over the 1,000 training problems (held-out problems tested separately). Per run: lost count $L$ against Binomial(1000, $r_0$), where $r_0 = (L_0 + 1)/1001$. Per family: partial-conjunction $p = \min(1, 2p_{(2)})$, meaning "at least 2 of 3 seeds". P4 needs $p < 0.05$ in both families.

### Fixed settings
| Setting | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct, full fine-tuning, fp32 master weights, 8-bit AdamW |
| Problems | Pool of 2,239; 1,000 train, 300 held-out (seed 0) |
| lr | $10^{-6}$ (primary), $3\cdot10^{-6}$ (secondary) |
| Steps | 500 |
| $G$, prompts per step | 8, 64 |
| Loss aggregation | Sequence level, divided by $64 \cdot 8$ |
| KL | $\beta = 0$ |
| Sampler | vLLM, temperature 1, ≤ 512 tokens |
| Evaluation | 256 samples per problem (final, seed 0); 32 at step 250 |
| Seeds | 0, 1, 2 per arm |
| Decline / loss test | One-sided Fisher, BH $q = 0.10$; lost = decline to $\hat p \le 0.05$; binomial test against $r_0$; family $p = \min(1, 2p_{(2)})$ |
| Run tags | lr $10^{-6}$: `c1q` (3 hosts); lr $3\cdot10^{-6}$: `q_35` (seed 0; reruns `q_00`, which failed with a CUDA fault), `q_21` (seed 1), `q_36` (seed 2; final evaluation relaunched) |
| Base evaluations | `base_qwen_n256_s0/s1` (tag `c0d`) |
| Pre-registration | `extra/analysis/review/PREREG_C1.md` and `PREREG_C1_amendment1.md` (`review` `2bdc507`) |
| Code commit | `c1q` `a175b73`; `q_21` `c5230d1`; `q_35` `7efbe5b`; `q_36` `dd69acc` (old hashes); analysis `extra/llm/analyze_c1.py` |

### Result
**P4 fails (refuted).** At lr $10^{-6}$, 0 of 1,000 training problems are lost in every seed (partial-conjunction $p = 1$), and mean $\hat p$ rises from 0.49 to 0.98. On held-out problems, 8–12 of 300 decline and 1–5 are lost (null: 0). At $3\cdot10^{-6}$, 15, 12 and 1 training problems are lost. At $3\cdot10^{-6}$, lost problems end on one repeated wrong answer, and no lost problem is shared between seeds. See `extra/analysis/llm/C1_C5_SUMMARY.md`, `c1_qwen.md` and `c1_qwen_runs.csv`.

## C1 Llama: the second family

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** P4 is registered over two model families. This run tests whether the Qwen result holds for Llama-3.2-1B.

### What I did
1. Used the Llama pool and sets from the calibration step (1,000 train and 300 held-out problems).
2. Trained for 500 steps at lr $10^{-6}$ (chosen by the rule), 3 seeds.
3. Evaluated the final model on the 1,300 problems at 256 samples, with seed 0 and a second "after" draw with seed 1. Also evaluated at step 250 (32 samples).
4. Ran the same tests as for Qwen with `extra/llm/analyze_c1.py`.

### Fixed settings
| Setting | Value |
| --- | --- |
| Model | Llama-3.2-1B-Instruct, full fine-tuning, 8-bit AdamW, micro-batch 1 |
| Problems | Pool of 2,157; 1,000 train, 300 held-out |
| lr | $10^{-6}$ |
| Steps | 500 |
| $G$, prompts per step | 8, 64 |
| Loss aggregation | Sequence level |
| KL | $\beta = 0$ |
| Evaluation | 256 samples (after, seeds 0 and 1); 32 at step 250 |
| Seeds | 0, 1, 2 |
| Decline / loss test | As C1 Qwen |
| Run tags | `r2_20`, `r2_21`, `r2_22` (base: `r2_18`, `r2_19`) |
| Pre-registration | `PREREG_C1.md`, `PREREG_C1_amendment1.md` |
| Code commit | `d8b43f1`; job lines written by `extra/llm/jobs/llama_arm.sh` |

### Result
Fails for this family too ($p = 1$): 0 training problems lost in all 3 seeds (0.42 → 0.99). Held-out declines are 10, 12 and 13 of 300, and lost problems 4, 5 and 9 (null 0), which replicates Qwen. See `extra/analysis/llm/c1_llama.md`, `c1_llama_runs.csv` and `C1_C5_SUMMARY.md`.

## C1 control: are the held-out declines specific to GRPO?

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** C1 showed a few held-out declines in every run. This control asks whether any fine-tuning on the same 1,000 problems produces them.

### What I did
1. Sampled 16 completions per training problem from base Qwen (seeds 10–12) and kept up to 4 correct ones.
2. Trained with cross-entropy on those completions (rejection-sampling fine-tuning) at the C1 optimizer and lr, for 8 epochs with batches of 64 (about 500 steps), 3 seeds.
3. Evaluated on the 1,300 C1 problems at 256 samples and analysed them as in C1.
4. Applied the rule fixed in advance. "Specific to GRPO" if all three runs have $D_{RFT} \le 4$ held-out declines; "generic" if at least two have $\ge 8$. The comparison is marked confounded if training success ends below 0.75.

### Fixed settings
| Setting | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct, full fine-tuning, 8-bit AdamW (as C1) |
| Problems | The 1,000 C1 training problems; 300 held-out |
| Targets | Up to 4 correct of 16 base samples per problem; sampling seeds 10, 11, 12 |
| lr | $10^{-6}$ |
| Batch, epochs | 64, 8 (`extra/llm/sft.py`) |
| Evaluation | 256 samples per problem, seed 0 |
| Seeds | 0, 1, 2 |
| Decline test | As C1 (Fisher, BH $q = 0.10$) |
| Run tags | `r2_23`, `r2_59`, `r2_60` |
| Job list | `extra/llm/jobs/c1_rft.txt` |
| Pre-registration | `extra/analysis/review/PREREG_C1_amendment2.md` (`review` `e024dcc`) |
| Code commit | `r2_23` `d8b43f1`; `r2_59`, `r2_60` `c7a1d7e`; analysis `extra/llm/analyze_control.py` |

### Result
By the registered count rule the declines are specific to GRPO: the control has 0, 0 and 0 held-out declines, against 12, 8 and 9 for GRPO. **But the comparison is confounded:** the control reached training success of only 0.65 (bar 0.75; GRPO 0.98). Exploratory. See `extra/analysis/llm/c1_control.md`, `c1_control_runs.csv` and `C1_C5_SUMMARY.md`.

## C5: the toy's loss drivers in the LLM

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** In the toy (T6), a noisy verifier and the sign rule cause absorption. We tested whether they cause lost problems in the LLM, together with a 10× learning rate.

### What I did
1. Started from the C1 primary configuration and changed one thing per arm. Verifier flip: each reward flipped with probability 0.2. Sign rule: `--rule sign`. lr × 10: $10^{-5}$.
2. Trained for 500 steps, 3 seeds per arm.
3. Evaluated on the 1,300 C1 problems at 256 samples (32 at step 250).
4. Counted declines and losses as in C1 and compared them with the C1 primary arm (Mann–Whitney over seeds, descriptive only at $n = 3$ against 3).

### Fixed settings
| Setting | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct, full fine-tuning, 8-bit AdamW |
| Problems | C1 sets (1,000 train, 300 held-out) |
| Arms | `--reward-flip 0.2`; `--rule sign`; `--lr 1e-5` |
| lr | $10^{-6}$ (flip, sign), $10^{-5}$ (lr arm) |
| Steps | 500 |
| $G$, prompts per step | 8, 64 |
| Loss aggregation, KL | Sequence level; $\beta = 0$ |
| Evaluation | 256 samples (final), 32 at step 250 |
| Seeds | 0, 1, 2 per arm |
| Decline test | As C1 |
| Run tags | Flip `q_23`–`q_25`; sign `q_26`–`q_28`; lr $10^{-5}$ `q_29`–`q_31` |
| Pre-registration | Exploratory section of `PREREG_C1.md` (plan: `NEXT_PLAN.md`) |
| Code commit | `c5230d1` (old hash); analysis `extra/llm/analyze_c1.py` |

### Result
The toy's predictions are not supported. Flip and sign learn less ($\hat p$ 0.90–0.93) but lose nothing extra (training lost: 1, 0, 0 and 0, 0, 0). lr $10^{-5}$ collapses (1,000, 511 and 1,000 lost): the output becomes unparsable at the 512-token limit, which is instability, not selective loss. Exploratory. See `extra/analysis/llm/C1_C5_SUMMARY.md`, `c5_qwen.md` and `c5_qwen_runs.csv`.

## C2 / P5: does a majority procedure collapse a minority procedure?

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** This is the confirmatory test of P5. A toy answer class corresponds to a solution procedure in the LLM. Does training mostly on one procedure push down a minority procedure that shares the same template?

### What I did
1. Generated two procedures that share each problem template and differ in one cue sentence: "each" $a(b - c)$ and "total" $ab - c$. Kept candidates with base $\hat p \in [0.15, 0.85]$, stratified into 7 bins, giving a universe of 1,000 plus 300 held-out problems per procedure (2,600 in all).
2. Built 1,000-problem training mixes with majority share $m \in \{0.5, 0.8, 0.95\}$, for both roles (majority "each" or "total"). Mix seeds were 0 and 1, plus seed 2 at $m \in \{0.5, 0.95\}$: 16 runs, with the run seed equal to the mix seed.
3. Trained Qwen for 500 steps with the C1 primary settings. Evaluated the run's 1,000 training problems and all 600 held-out problems at 256 samples (32 at step 250). I also ran hint evaluations on the 600 held-out problems at 64 samples (neutral hint, own first step, other procedure's first step).
4. Primary test (amendment 1): the per-run decline rate on the 300 held-out minority problems (Fisher against base seed 0, BH at 10% over the panel). Within each role, an exact permutation test of the Spearman correlation between $m$ and that rate (all 560 assignments, one-sided). P5 needs $p < 0.05$ in both roles.

### Fixed settings
| Setting | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct, full fine-tuning, 8-bit AdamW |
| Problems | `extra/llm/pools/c2/all.json` (2,600): per procedure 1,000 universe + 300 held-out; candidates range `ml`, seed 1, base $\hat p \in [0.15, 0.85]$ |
| Mixes | 1,000 train problems; $m \in \{0.5, 0.8, 0.95\}$ × 2 roles; mix seeds 0, 1 (+2 at $m = 0.5, 0.95$) |
| lr, steps | $10^{-6}$, 500 |
| $G$, prompts per step | 8, 64 |
| Loss aggregation, KL | Sequence level; $\beta = 0$ |
| Evaluation | 256 samples on 1,600 problems (`mix_*_eval_ids.json`); 32 at step 250; hints at 64 on 600 held-out |
| Base | 256 samples on all 2,600, seeds 0 (before) and 1 (null) |
| Primary test | Exact permutation Spearman ($m$, decline rate), 560 assignments per role, one-sided; "untestable" if a role has < 10 declines |
| Run tags | Main: `r2_07`–`r2_17`, `r2_24`–`r2_28` (16); base: `q_32`, `q_33`; base hints: `r2_06`; excluded: `q_34` (old code) |
| Job lists | `extra/llm/jobs/c2.txt`, `c2_v2.txt` |
| Pre-registration | `extra/analysis/review/PREREG_C2.md`, `PREREG_C2_amendment1.md` |
| Code commit | `8002345` (summary); `r2_24`, `r2_25`, `r2_28` record `36380cb`, `r2_26`, `r2_27` record `b9747d1` (restarted after a reboot); base `q_32`, `q_33` `c5230d1`; analysis `extra/llm/analyze_c2.py` |

### Result
**P5 fails (refuted).** The role "each" gives $p = 0.0536$ (Spearman 0.64), the role "total" gives $p = 0.0071$; P5 needs both below 0.05. Exploratory evidence against the toy's mechanism: the minority procedure improves by +0.36 to +0.45 in 15 of 16 runs. Only 2–9% of problems decline, and none switches to the other procedure's answer. After Holm over P1–P5, only P1 holds. See `extra/analysis/llm/C2_SUMMARY.md`, `c2.md` and `c2_runs.csv`.

## C2 matched minority: majority pressure or minority sample count?

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** In C2, a larger $m$ also means fewer minority samples per step. This arm holds the minority's exposure fixed and changes only the amount of majority data.

### What I did
1. Every step drew exactly 4 minority prompts and $n_A \in \{4, 16, 76\}$ majority prompts (matching $m = 0.5, 0.8, 0.95$) from the balanced 500 + 500 training pool.
2. Divided the loss by $80 \times G$ in all arms, so a minority sample has the same weight everywhere.
3. Trained for 500 steps, both roles, seeds 0 and 1 (12 runs). Evaluated at 256 samples on the 1,600-problem set.
4. Ran the permutation test of Spearman($n_A$, held-out minority decline rate) on 6 runs per role (90 assignments).

### Fixed settings
| Setting | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct, full fine-tuning, 8-bit AdamW |
| Problems | `mix_m0.5_each_s0` train (500 per procedure); eval `mix_m0.5_each_s0_eval_ids.json` |
| Prompts per step | `--strata` 4 minority + 4, 16 or 76 majority; `--denominator 80` |
| lr, steps | $10^{-6}$, 500 |
| $G$ | 8 |
| KL | $\beta = 0$ |
| Evaluation | 256 samples (final), 32 at step 250 |
| Seeds | 0, 1 per arm and role |
| Test | Exact permutation Spearman, 90 assignments per role (smallest $p = 0.011$) |
| Run tags | `r2_29`–`r2_40` |
| Job list | `extra/llm/jobs/c2_extras.txt` (first 12 lines) |
| Pre-registration | `PREREG_C2_amendment1.md` (control 1b) |
| Code commit | `r2_29` `36380cb`; `r2_30`–`r2_40` `b9747d1` |

### Result
The decline rate does not depend on the number of majority prompts (Spearman −0.36 and −0.72; $p = 0.80$ and 0.97). The mean gain is the same at 4 and at 76 majority prompts. The weak trend in C2 follows the minority's own sample count, not pressure from the majority. See `extra/analysis/llm/C2_SUMMARY.md` ("The controls").

## C2 seed replication of the degraded run

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** One C2 run (majority "each", $m = 0.95$, seed 1) lost 45 of 300 held-out minority problems. Three seeds cannot tell a 1-in-3 event from a 1-in-20 one.

### What I did
1. Reran that run's problem draw (`mix_m0.95_each_s1`) with five new training seeds (3–7).
2. Ran the same draw at $m = 0.5$ (`mix_m0.5_each_s1`) with the same seeds, as the reference.
3. Evaluated as in the C2 main runs, without the hint evaluations.
4. Counted runs that were "broadly degraded", fixed in advance as at least 30 of 300 held-out minority problems lost.

### Fixed settings
| Setting | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct, full fine-tuning, 8-bit AdamW |
| Problems | Draw 1, majority "each", $m = 0.95$ and $m = 0.5$ |
| lr, steps | $10^{-6}$, 500 |
| $G$, prompts per step | 8, 64 |
| KL | $\beta = 0$ |
| Evaluation | 256 samples (final), 32 at step 250 |
| Seeds | 3, 4, 5, 6, 7 per share |
| Degradation rule | ≥ 30 held-out minority problems lost |
| Run tags | `r2_61`–`r2_70` (odd: $m = 0.95$; even: $m = 0.5$) |
| Job list | `extra/llm/jobs/c2_seeds.txt` |
| Pre-registration | `PREREG_C2_amendment2.md` (`review` `2425a0b`) |
| Code commit | `c7a1d7e`; analysis `extra/llm/analyze_c2.py` |

### Result
0 of 10 new runs degrade, so the original run is one outlier in eight at $m = 0.95$. A moderate share effect does show on this draw: held-out minority declines are 17–23 at $m = 0.95$ against 6–11 at $m = 0.5$. That comparison was not registered. Exploratory; P5 is unchanged. See `extra/analysis/llm/c2_seeds.md`, `c2_seeds_runs.csv` and `C2_SUMMARY.md`.

## C2 extras: samples per step, a long run, and a supervised control

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** These runs test three rival explanations for the C2 declines: too few samples per step, a loss that appears only later, and declines from any fine-tuning rather than from GRPO.

### What I did
1. Ran $m = 0.95$, majority "each", with 16 and 256 prompts per step (3 seeds each, draws $s0$–$s2$). The 64-prompt main runs are the middle point.
2. Ran a long arm: $m = 0.95$, both roles, seed 0, 1,500 steps, with 256-sample evaluations at steps 500, 1,000 and 1,500.
3. Ran a rejection-sampling control: up to 4 correct base completions (of 16) per training problem, cross-entropy at lr $10^{-6}$, batch 64, 8 epochs, at $m \in \{0.5, 0.95\}$, majority "each", seeds 0–2.
4. Compared held-out minority declines and losses with the GRPO main runs.

### Fixed settings
| Setting | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct, full fine-tuning, 8-bit AdamW |
| Prompts per step | 16 or 256 (sample-count arm); 64 otherwise |
| Steps | 500; long arm 1,500 |
| lr | $10^{-6}$ (GRPO and control) |
| $G$, KL | 8; $\beta = 0$ |
| Control | `extra/llm/sft.py`, targets from 16 samples (seeds 10–12), up to 4 correct; batch 64, 8 epochs |
| Evaluation | 256 samples on the 1,600-problem set |
| Seeds | 0–2 (sample-count, control); 0 (long arm) |
| Run tags | 16 / 256 prompts `r2_71`–`r2_76`; long arm `r2_77`, `r2_78`; control `r2_79`–`r2_84` |
| Job list | `extra/llm/jobs/c2_extras.txt` |
| Pre-registration | Designed in `PREREG_C2_amendment1.md` (controls 2, delay arm, supervised control); reported as exploratory |
| Code commit | `c7a1d7e`; analysis script: see `extra/analysis/llm/EXTRAS_2026-11-24.md` |

### Result
Going from 64 to 256 prompts changes nothing (16–22 declines). At 16 prompts one seed degenerated as a whole, and the other two are noisier for both procedures. There is no delayed loss: the "each" arm has 34, 33 and 28 declines at steps 500, 1,000 and 1,500. Declines occur under both the control and GRPO, but problems driven to $\le 0.05$ occur mostly under GRPO. The control's gain is borderline-weak (about half of GRPO's). Exploratory, first look. See `extra/analysis/llm/EXTRAS_2026-11-24.md`.

## SmolLM2 diagnostic: why the SmolLM2 pilots collapsed

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** The SmolLM2 LoRA pilots rose and then collapsed after about 100 steps. We tested whether a mismatch between the vLLM sampler and the trainer causes this.

### What I did
1. Trained SmolLM2-1.7B with LoRA at lr $10^{-4}$ for 200 steps on its 1,000 C1 training problems.
2. Ran it once without correction and once with truncated importance weights: $w = \min(\pi/\pi_{sampler}, C)$ with $C = 2$.
3. Compared the training reward and the answer length from the training logs.

### Fixed settings
| Setting | Value |
| --- | --- |
| Model | SmolLM2-1.7B-Instruct, LoRA $r = 64$, $\alpha = 128$, AdamW (not 8-bit) |
| Problems | `extra/llm/pools/smol_c1_train.json` (1,000) |
| lr, steps | $10^{-4}$, 200 |
| $G$, prompts per step | 8, 64 |
| Importance weights | None, or `--tis 2.0` |
| KL | $\beta = 0$ |
| Evaluation | None (training logs only) |
| Seeds | 0 |
| Run tags | `r2_85` (no correction), `r2_86` (`--tis 2.0`) |
| Job list | `extra/llm/jobs/smol_diag.txt` |
| Pre-registration | None |
| Code commit | `c7a1d7e` |

### Result
Without the correction, the reward collapses to 0.02 by step 50 and answers run to the length limit. With it, training is normal (reward 0.81–0.84). The earlier SmolLM2 failures were a sampler–trainer mismatch, not a property of the model. Exploratory. See `extra/analysis/llm/EXTRAS_2026-11-24.md`.

## C10: who pushes whom in the LLM

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** C2 measured the majority's effect only through the outcome of full runs. C10 measures the sign directly, as the toy's T11 does: does training on the majority alone lower the minority?

### What I did
1. Started from the base weights and drew problems by procedure from the balanced C2 pool (`mix_m0.5_each_s0`).
2. Ran three arms with the same loss denominator of 80 prompts: mixed (76 majority + 4 minority prompts per step), majority-only (76) and minority-only (4). Both roles, seeds 0–2: 18 runs of 100 steps.
3. Evaluated the 600 held-out problems at 128 samples at steps 50 and 100.
4. Computed the change in mean success on the 300 held-out minority problems against base seed 0 (256 samples), with a 2,000-draw problem bootstrap per run. The null is base seed 1 minus seed 0.

### Fixed settings
| Setting | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct, full fine-tuning, 8-bit AdamW |
| Problems | `mix_m0.5_each_s0` train (500 per procedure); held-out 600 (`heldout_ids.json`) |
| Arms | `--strata` 76 + 4, 76 + 0, 0 + 4; `--denominator 80` |
| lr, steps | $10^{-6}$, 100 (checkpoint at 50) |
| $G$ | 8 |
| KL | $\beta = 0$ |
| Evaluation | 128 samples per held-out problem at steps 50 and 100 |
| Seeds | 0, 1, 2 per arm and role |
| Statistic | Change in mean held-out minority success; bootstrap 95% interval (2,000 draws) |
| Run tags | `r2_41`–`r2_58` |
| Job list | `extra/llm/jobs/c10.txt` |
| Pre-registration | `extra/analysis/review/PREREG_C10.md` (`review` `75e0436`) |
| Code commit | `b9747d1` (summary); `r2_56`–`r2_58` record `c7a1d7e`; analysis `extra/llm/analyze_branch.py` |

### Result
Prediction 1 fails in all 6 runs: majority-only training **raises** the minority by +0.11 to +0.34. Minority-only training raises it by +0.30 to +0.35. Additivity fails (the mixed change is 62% and 75% of the sum). This is positive transfer, the opposite of the toy. Exploratory. With Adam, the size of $R = -0.79$ cannot be compared with the toy's, but the sign stands. See `extra/analysis/llm/C10_SUMMARY.md`, `c10.md` and `c10_runs.csv`.

## C9: token-level loss aggregation

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** Many GRPO implementations average the loss over tokens rather than sequences. We checked whether this changes the C1 census.

### What I did
1. Repeated the C1 primary configuration with `--agg token`: the loss is divided by the total number of completion tokens, rescaled by the mean length at step 1.
2. Trained for 500 steps, 3 seeds.
3. Evaluated on the 1,300 C1 problems at 256 samples (32 at step 250).
4. Ran the C1 census with `extra/llm/analyze_c1.py` and compared it with C1 seeds 0–2.

### Fixed settings
| Setting | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct, full fine-tuning, 8-bit AdamW |
| Problems | C1 sets (1,000 train, 300 held-out) |
| lr, steps | $10^{-6}$, 500 |
| $G$, prompts per step | 8, 64 |
| Loss aggregation | Token level (`--agg token`) |
| KL | $\beta = 0$ |
| Evaluation | 256 samples (final), 32 at step 250 |
| Seeds | 0, 1, 2 |
| Decline test | As C1 |
| Run tags | `r2_91`–`r2_93` |
| Job list | `extra/llm/jobs/c8_c9.txt` |
| Pre-registration | `extra/analysis/review/PREREG_v3c.md` (C9, exploratory) |
| Code commit | `c7a1d7e` |

### Result
The same as sequence aggregation: 0 training problems lost, and 5, 13 and 12 held-out declines (C1: 12, 8, 9). Mean answer length over the last 50 steps is also similar (250–264 tokens against 251–262); the full length trajectory is not compared in the write-up. Exploratory. See `extra/analysis/llm/c9.md` and `c9_runs.csv`.

## C7 / C8: the KL penalty in the LLM

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** This is the LLM stretch goal for H4. Does a KL penalty prevent the held-out losses? Does each problem settle at the toy's finite-$G$ fixed point? Does forward KL keep more problems near base than reverse KL (C8)?

### What I did
1. Trained with reverse KL against the frozen base model at $\beta \in \{0, 0.02, 0.1, 0.5\}$ and $G \in \{4, 16\}$, with 512 samples per step (C7). Trained with forward KL at $\beta \in \{0.1, 0.5\}$ and $G = 16$ (C8). Each run was 800 steps with 2 seeds.
2. Evaluated on the 1,300 C1 problems at 256 samples (32 at step 400). Then I ran the C1 census and turnover with `extra/llm/analyze_c1.py`.
3. For question 2, I computed each problem's stable fixed point $p^*$ nearest $p_{ref}$ at $\beta_{eff} = s\beta$, where $p_{ref}$ is base success over 512 samples. I fitted one scale $s$ per run on a log grid from 0.01 to 100. Then I regressed $\mathrm{logit}\,\hat p_{after}$ on $\mathrm{logit}\,p^*$ and compared the fit with a baseline that uses $p_{ref}$ alone.
4. Replaced the runs that broke numerically. I fixed the KL code (bounded log-ratio, non-finite updates skipped) and reran the affected runs.

### Fixed settings
| Setting | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct, full fine-tuning, 8-bit AdamW |
| Problems | C1 sets (1,000 train, 300 held-out) |
| lr, steps | $10^{-6}$, 800 (checkpoint at 400) |
| $G$, prompts per step | $G = 4$ × 128 or $G = 16$ × 32 (512 samples per step) |
| KL (C7) | Reverse, to the frozen base; $\beta \in \{0, 0.02, 0.1, 0.5\}$; gradient of $k_2 = \tfrac12(\log\pi - \log\pi_{ref})^2$ per token, summed over the completion |
| KL (C8) | Forward (`--kl forward`), gradient of $k_3$; $\beta \in \{0.1, 0.5\}$, $G = 16$ |
| Log-ratio bound | $[-20, 20]$ from `a7bd0aa` on; no gradient clipping (by design) |
| Loss aggregation | Sequence level |
| Evaluation | 256 samples (final, seed 0); 32 at step 400 |
| Seeds | 0, 1 per setting |
| Decline test | As C1 (Fisher, BH $q = 0.10$) |
| Run tags | C8: `r2_88`, `r2_89`, `r2_90`, `r2_117` (replaces `r2_87`). C7: `r2_94`–`r2_97`, `r2_100`, `r2_101` (`r2_98` resumed), `r2_102`–`r2_104`, `r2_111`–`r2_116` (replace `r2_105`–`r2_110`), `r2_118` (replaces `r2_99`). `r2_114` broke and is left out |
| Job lists | `extra/llm/jobs/c7.txt`, `c8_c9.txt` |
| Pre-registration | C7: `extra/analysis/review/NEXT_PLAN.md` (exploratory); C8: `PREREG_v3c.md` |
| Code commit | `r2_88`–`r2_100` `c7a1d7e`; `r2_101`–`r2_104` `30d2b4b`; `r2_111`–`r2_118` `a7bd0aa` (KL fix); analysis `extra/llm/analyze_c1.py`, `extra/llm/analyze_kl_fixed_points.py` |

### Result
Any $\beta \ge 0.02$ removes the held-out declines (0–4 declines, 0 lost, against 14–26 and 3–10 without KL). The cost grows with $\beta$. The fixed point tracks the end state ($R^2$ 0.71–0.96) but adds nothing beyond base success alone, and $\beta_{eff}$ grows roughly like $\sqrt\beta$. $G = 16$ settles above $G = 4$ at every $\beta$. The C8 prediction is not supported: forward and reverse KL behave the same. Exploratory. See `extra/analysis/llm/C7_C8_SUMMARY.md`, `c78.md`, `c78_runs.csv` and `c78_fixed_points.md`.

## C3 and C11: planned, not run

**Scope: Extra** — beyond what was promised (a second environment, the LLMs, formal verification or a side question); not needed to follow the report.

**Why:** C3 would have tested natural-data rarity without training. C11 would have tested a fix in the LLM, gated on a P5 pass or a push-down in C10.

### What I did
1. Planned C3 in `NEXT_PLAN.md`: pre-specified rarity measures (answer format, strategy cluster, operation sequence) and a logistic regression of decline on rarity.
2. Ran neither. C11's gate was never met: P5 failed, and C10 showed positive transfer.

### Fixed settings
| Setting | Value |
| --- | --- |
| Status | Not run |
| Plan | `extra/analysis/review/NEXT_PLAN.md` (C3); gate stated in `extra/analysis/llm/C10_SUMMARY.md` (C11) |

### Result
Not run. There was no collapse in the LLM to fix. See `docs/EXPERIMENTS.md` and `extra/analysis/llm/C10_SUMMARY.md`.
