# Review C: where our results sit in the literature

Sources: `papers/notes`, `papers/reviews/{deep,triage,TIERS.md}`, `papers.bib`, `latex/ref.bib`. PDFs opened to
check a claim: Mroueh 2503.06639 (§3), Vojnovic & Yun 2502.18548 (App., uniqueness), Huang et al. 2611.14467
(§2.1), Zhou et al. 2603.01162 (Thm 7). Keys are `ref.bib` keys where they exist, otherwise `papers.bib` keys.

## 1. Result by result (ranked by importance to the final report)

| # | Closest prior work | What it shows | Status | Risk | Safe phrasing |
|---|---|---|---|---|---|
| 4 Absorption needs sharing; 3 mechanisms; no single effective step | `barakat2026prompt` Def. 3.1, Prop. 4.1, Prop. E.1; `suk2025gradientgap` Thm 4, Eqs. 13–14; `li2026two` Prop. 2; Song et al. 2607.01083 (not in any bib); `huang2026coverage` Thm 2.1; `baudry2025does` Thm 2 | Barakat: interference through shared parameters can lower pass@1, one step, exact gradients; no interference means no conflict. Suk–Duan: a constructed oversized step drives p to 0. Two Clocks and Song: collapse time ∝ 1/η (LLM, Adam), from a postulated drift threshold. Huang: exact-gradient tabular collapse, but only with graded rewards. Baudry: tabular absorption from reward noise | (i) extends (exact-flow test of the 1/η law and B-independence, with interference as the cause); (ii) confirms Suk–Duan in a multi-prompt model; (iii) new: the noise that matters is batch composition, not response sampling, η* ∝ B^0.83; "no single variable" is new and refutes our own H3 | (i) medium, (ii) high if claimed new, (iii) low | "With a deterministic binary verifier, absorption required parameter sharing in every run. We separate three routes: deterministic interference drift (time ∝ η^−0.89, independent of B, reproduced by the exact flow), one-step overshoot as in Suk & Duan, and noise from prompt-batch selection. No single step variable aligns them across architectures." |
| 3 Tabular monotonicity and ablations | `yu2026implicit` Thm 1; `baudry2025does` Lemma 1; `mei2024small` Thm 2; `rad2026rate`; `nie2026starvation`; `zhang2026ema` Eq. 9 | Yu: with zero-sum advantages, unsampled responses are not updated (tabular, one response-level softmax). Baudry: with deterministic rewards the best arm stays the mode at any η. Mei: no permanent collapse, tabular, any η. Rad: with J > 0 noise changes the rate, not the outcome (expected flow). Nie: Sign (A = 2r − 1) helps LLMs. Zhang: KL step ceiling ηβλ_max < 2 | Extends: a per-step proposition for L-token factorised policies with groups and baselines, which none of these states. Qualifies Rad (noisy verifier with J > 0 still absorbs at large η) and Nie (Sign breaks monotonicity, up to 27% absorbed). β = 1 at η = 300 confirms Zhang's ceiling | medium (an easy lemma; the L = 1 case is near Baudry and Yu) | "Under the stated conditions p never fell (0 of 2,268 runs), and each condition matters. Momentum and Adam do produce small decreases (44/72 and 49/54 runs) but never absorb." Do not say "momentum and Adam preserve monotonicity". |
| 2 KL fixed points and finite-G bistability | `vojnovic2025alignment` Eq. 11; `mroueh2025rlvr` Thm 2, Prop. 1, Fig. 2; `tang2025pitfalls`; `gxchen2026kl` | Vojnovic: finite-G stationary equations, both KL directions; closed forms only for G = 2 and G → ∞, each with a unique root. **Mroueh, Prop. 1 and p. 7: "the fixed points are not unique in general, and p* = 1 is almost always a fixed point"** (G → ∞ with ε-smoothing, exact iteration map) | Fixed points: confirms. Bistability: extends. The new parts are a finite G (W_G ≤ √(G−1) acts like Mroueh's ε), stability under SGD rather than under the iteration map, a basin boundary at the predicted unstable root, and barrier-limited switching that grows with η | **medium-high** if worded as "first multiple equilibria" | "Mroueh already notes that the G → ∞ map can have several fixed points. We show that at finite G ≥ 16, SGD has two stable states whose basin boundary is the predicted unstable root, and that noise switches between them only when the barrier is small." |
| 5 Per-prompt drift predictor | `barakat2026prompt` Eq. 7 (agreement score a_θ(x)) | a_θ(x) = ⟨∇p_x, ∇J_1⟩ marks prompts pushed down, for one step. Barakat warns that it does not establish persistent collapse | Extends: our d_x is a_θ with W_G weights. Testing it against multi-step absorption is new. Checkpoint drift is at chance except c09 | low–medium | "The step-0 interference score of Barakat et al., weighted by W_G, ranks absorbed prompts (AUC 0.61–1.0) better than p or ‖∇p‖, but it loses this power after the initial transient." |
| 7 H2: sharing dominates drift; RLOO G-invariant | `davis2025objective` §4; `ghosh2026predictable` Prop. 10; `barakat2026prompt` κ | Davis: RLOO has W = 1 at every G. Ghosh: the mean trajectory and p* are G-invariant for GRPO | Confirms Davis over multi-step training. **Contradicts Ghosh** for GRPO (200-step success varies up to 1.3× over G for m64). The cross-share above 0.9 is a new measurement in Barakat's terms | low | "Without sharing, each rule moves prompts as W_G predicts. With sharing, over 90% of a prompt's first-order drift comes from other prompts. RLOO's trajectory is flat in G, as W = 1 implies, while GRPO's is not, contrary to Ghosh et al." Say "consistent with depending on G only through W_G", not "shown". |
| 8 KL threshold not monotone in β | `zhang2026ema` Eq. 9; `huang2026coverage` §2.1; `cesani2026log` | Zhang: KL adds its own step ceiling. Huang: KL to the base does not prevent collapse. Cesani: a forward-KL-like barrier keeps p off the boundary (tabular) | New as a measurement and consistent with all three. Forward KL lowered η* at c05 (30 → 1) | medium (3 seeds, and the threshold may mix KL fixed points below 0.01 with KL overshoot) | "In our runs a reference KL did not reliably protect against absorption, and forward KL made it easier at c05." |
| 6 H5: best G falls as p₀ rises | `zhou2026grpo` Thm 7, Eq. 12; `nguyen2026vip` Prop. 4.2–4.3; `heckel2026asymmetric`; `khatri2026art` Table 1; `he2026advantage` App. F; `plyusov2026fgrpo` | Zhou: G* = √(c3/c1) is independent of the budget but depends on the data and the policy. Nguyen: allocation depends on p. Khatri: G in 8–32 is second-order at LLM scale | Confirms the intuition behind φ (rare successes need G ~ 1/p) and resolves Zhou's "depends on the data" in p₀. Tabular only: with sharing, best G is not monotone (p₀ = 0.3 gives 128/64). Review A rates the verdict fragile (ties at 6,400 samples) | medium | "In the tabular task, the best G at BG = 256 fell as p₀ rose (Spearman −0.95)." Do not generalise to shared models. |
| 1 Identity E[ĝ] = W_G∇p and surrogate | `davis2025objective` §4 Eqs. 4.1–4.2, §6 Fig. 3; `zheng2026rl2ml` Thm 2.1; `pang2026tic`; `yang2026group`; `okonkwo2026arcus` | All published at finite G. ARCUS also gives the arc-length form with boundary ramps of width 1/√(2G) | Known; our contribution is validation only (bias ≤ 1e-4 per rule against a calibrated null, 6 rules, 3 architectures) | **high** if claimed | "We verify Davis & Recht's finite-G identity numerically…" Fix `milestone.tex:258–260` and `:322`. |

## 2. Gap map

**Gaps the literature leaves open that our results address:**

- How prompt interference plays out over many steps. Barakat is one step and population-level; we follow it to absorption and lost pass@64 (m64 drops from 0.81 to 0.26). This gives Wu et al. (`wu2025invisible`) and `yue2025doesrl` one concrete mechanism, not the only one; `yuan2026diversity` and Barakat propose others.
- Telling drift from noise with exact gradients. Two Clocks could only postulate the drift threshold, and Song et al. only fit it. We can switch noise off: the exact flow, the batch-expected step and the sampled step separate the mechanisms. Huang, Sheng & Zheng (`huang2025variance`) suggest splitting rollout noise from batch noise but never test it.
- The conditions behind "collapse needs sharing". Baudry, Mei and Cesani cover tabular bandits with G = 1, no baseline and noisy rewards. We give the boundary conditions for groups, zero-sum baselines, a deterministic verifier and L tokens.
- Stability of the finite-G KL equilibria under SGD. Vojnovic gives stationarity only; Mroueh analyses the exact iteration at G → ∞.

**What we cannot claim:**

- Relevance at LLM scale. There is no autoregressive structure, the MLPs are tiny, there is no clipping, and the main grid uses plain SGD; the Adam blocks show a different picture (m64 absorbs under sampled Adam at η ≈ 0.01, while the exact flow does not).
- Non-binary or graded rewards. Huang's coverage collapse lives there, and our monotonicity proposition fails there.
- Off-policy updates, staleness and several updates per rollout (Song, Ghosh, Yao).
- Any universal threshold law. The exponents (0.83, −0.89) are specific to each architecture.
- That absorption is permanent. In lin, the final absorbed fraction is only 0.15 of its peak (Review A).

## 3. Missing citations (the milestone cites only 13 keys)

**Must add** (the `papers.bib` key is listed; none is in `ref.bib`):

- `vojnovic2025alignment`: the H4 fixed points.
- `barakat2026prompt`: the interference kernel and the predictor.
- `he2026advantage`: φ.
- `baudry2025does` and `mei2024small`: the tabular baselines for "needs sharing".
- `li2026two`, plus Song et al. 2607.01083, which is not in any bib and has to be added: the 1/η law.
- `yu2026implicit`: zero-sum, and the empirical collapse.
- `rad2026rate`: the noisy verifier.
- `zhang2026ema`: the KL step ceiling.
- `ghosh2026predictable`: G-invariance, which we contradict.
- `zhou2026grpo`, `nguyen2026vip` and `khatri2026art`: H5.
- `zheng2026rl2ml`, `pang2026tic` and `okonkwo2026arcus`: the identity.

**Should add:**

- `cesani2026log`, `gxchen2026kl`, `tang2025pitfalls`, `huang2025variance`, `wu2025invisible`, `tajwar2026maxrl` and `yang2026group`.
- Also cite `yue2025doesrl` and `plyusov2026fgrpo`, which are already in `ref.bib` but not cited.

**Key mismatch:** `ref.bib` and `papers.bib` use different keys for the same papers (for example `suk2025gradientgap` and `suk2025rlvr`). Merge them into one file.

**Possibly missing, unverified:** Schaul, Borsa, Modayil & Pascanu (2019), *Ray interference: a source of plateaus in deep RL*. Huang et al. cite it (p. 9) as prior work on failure from "parameter interference". From its title it may overlap mechanism (i), plateaus from interference under on-policy coupling. Check it before calling mechanism (i) new.

## 4. Verdict

The theory the project began with (H1, H2 and the H4 stationary equations) is published, mostly by Davis & Recht and by Vojnovic & Yun. Mroueh already observed multiple KL fixed points in the infinite-group limit. What is new is empirical and diagnostic. In a setting where p and ∇p are exact and noise can be switched off, we show:

- with a deterministic verifier, absorption at p ≈ 0 requires parameter sharing;
- it happens by three separable routes: interference drift that survives the exact flow, Suk–Duan overshoot, and noise from prompt-batch selection rather than response sampling;
- no single effective step variable predicts it across architectures, which refutes our own H3;
- the W_G-weighted interference score of Barakat et al. picks out the absorbed prompts at initialisation;
- at finite G the KL equilibria are bistable, with the predicted basin boundary, and noise switches between them only when the barrier is small.

The final report should present this as a controlled, exact-gradient dissection of a failure that the LLM papers (Two Clocks, Song, Wu, Yue) only observe. It should attribute all the formulas and keep every claim to binary rewards, small models and plain SGD.
