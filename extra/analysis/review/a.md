# Review A: what the milestone promised and what the experiments delivered

Scope: `latex/milestone.tex` (promises at lines 306–336), the two proposals (E1–E6), the base and stretch
batches (`experiments/grids.py`, `analysis/{base,stretch}/FINDINGS.md`). Paths are relative to `981_project/`.
Numbers marked (R) were recomputed from the caches for this review; the snippets are in the appendix.

## 1. Promise-by-promise table

| # | Promise (milestone line) | What we ran | Result / verdict | Status | Quality |
|---|---|---|---|---|---|
| H1a | "the identity (1) holds at finite G" (306–307) | `h1()` grids.py:36, 414 gradchecks R=4000; `s_h1()` :114, R=20,000 | Per-rule median ratio 0.95–1.02, \|z\|>3 share 0.25% vs 0.27% Gaussian (base FINDINGS:21–39); tight test: no \|b∥/SE\|>3.9, pooled bias ≤1e-4 per rule (stretch FINDINGS:74–83) | delivered | 5: two independent precisions, calibrated null. The per-run criterion was relaxed after the fact (base FINDINGS:41), and that is stated. |
| H1b | "rules whose W_G differ by a constant match after rescaling the step" (307–308) | No dedicated grid. Unit tests `tests/test_training.py:25–38`. Indirectly: G=2 rows of the Ghosh table and the H3 G=2 runs | GRPO≈RLOO≈MaxRL at G=2: identical 200-step means (base FINDINGS:76–93). H3 at G=2: median \|final_mean(grpo)−final_mean(rloo)\| = 6e-7, but 5.1% of 648 pairs differ by >0.01, all at η ≥ 0.1 (R). Mean vs RLOO at η·G/(G−1) was never run, but it is an algebraic identity (same per-sample update) | partial | 3: only the trivial constant (≈1 at G=2) was checked empirically. The large-η divergence is chaos and should be mentioned. |
| H2 | "group size alone moves GRPO's surrogate from p at G=2 toward 2 arcsin √p" (308–309) | `h2()` :45 tested a different question: does sharing distort the per-prompt W_G ratio (base FINDINGS:48–70), plus the Ghosh G table | As run: PASS for "tabular on the W_G line, sharing breaks it". The promised claim has no verdict in FINDINGS. **It can be delivered from the existing cache:** fitting W_meas = (1−λ)·1 + λ/√(p(1−p)) on tabular GRPO bins gives λ = 0.000, 0.229, 0.626, 0.886 at G = 2, 4, 16, 64, and measured W matches exact W_G to a median 0.01–0.03% (R) | delivered-changed (promise testable but not reported) | 3 as reported, 5 once the λ table is added |
| H3 | "stability is set by one effective step ηW_G(p)p(1−p)/B … together with how strongly prompts interfere" (309–311) | `h3()` :56, 13,608 runs; candidates (i)–(iv) analyze.py:543–551; stretch `s_drift` :120, `s_pred` :137, `s_abl` :158 | Pre-registered collapse **FAIL**: best candidate is s̄ alone, max level shift 4.69 decades (2.38 with L ≤ 2) (base FINDINGS:139–172). c09 absorbs at the smallest η, 0.003, in 42 configurations (:135), so for c09 there is no step-size threshold at all. Post hoc there are three mechanisms (stretch FINDINGS:245–256): drift (c09), one-shot overshoot (c05), batch noise (m64). The tabular proposition PASSes (2,268 runs, 0 decreases; base :102) | delivered-changed (the claim is refuted; a new story replaces it) | 4: honest pre-registration and strong follow-up controls (deterministic flow, B-scaling, Adam). The stretch drift criterion also failed for c05 (stretch :254). |
| H3-link | Suk–Duan failure "shows up as prompts pushed to p≈0 … recovery is slow" (316–319) | H3 absorbed fraction; s_pred | Shared levels absorb, tabular never does. Recovery differs by level: the median ratio of final to peak absorbed fraction is 1.00 for c09, m64 and m256, 0.58 for c05 and 0.15 for lin (R). So for lin, recovery is not slow | delivered-changed | 4: the link holds, but for c09 it is not Suk–Duan's large-step failure |
| H4 | "SGD with a KL penalty settles at the finite-G fixed point … not Mroueh's limit (0.11 at G=4 vs 0.17)" (311–315) | `h4()` (a) :66, 960 runs; `s_eq()` :170 | Median \|error\| 1e-4; guide value 0.1090 measured 0.1090 (base FINDINGS:295–307). Against the G→∞ root, converged GRPO reverse-KL runs miss by a median 0.076 / 0.047 / 0.031 / 0.013 / 0.0026 at G = 2/4/8/16/64, versus 0.0003–0.00003 for the finite-G root; 12 of the configurations have no stable G→∞ root at all (R). Bistability at G=64 confirmed (stretch :443–445) | delivered | 5 for the fixed points. Tabular only (no sharing); 104/960 runs from p_ref 0.01 had not converged (base :293) |
| H4(b) | not in the milestone (plan only): Var ∝ η/β | `h4()` (b) :73 | FAIL against η/β; at fixed β the slope in η is 0.93–1.17 (base :334–343) | n/a | 3: the post-hoc reading is plausible |
| H5 | "Group-size scaling is a separate study (H5)" (315) | `h5()` :85, 3,528 runs, BG=256 | Pre-registered PASS (base :433), but see finding 4: the best-G table is decided by ties at the 6,400-sample resolution | partial | 2: the milestone promise was too vague to test, and the verdict is fragile |
| S1 | Heckel weights for all-incorrect groups, "a natural test of the mechanism" (325–327) | none (no rule in `trainer.py`; grep finds nothing) | — | not done | — |
| S2 | Huang reference resets "an extra KL setting for H4" (327–331) | none | — | not done | — |
| S3 | Bay & Yearick / Nie group statistics "bear on H5" (331–333) | H5 informative-group and heterogeneity tables | Measured/predicted φ = 1.000–1.001 (base :412–416); Nie excess of all-same groups up to 0.378 (:418). Bay & Yearick's G·p(1−p) law was never fitted. Nie's fix A = 2r−1 (`sign`) appears only as an ablation, where it *causes* absorption (up to 27%, stretch :382) | partial | 3: φ is clean; the law named in the proposal (E1) is untested |
| S4 | Ge: standardization as adaptive step, "consistent with the W p(1−p) view" (334–335) | s̄ (`diagnostics.py:91`) | This is a reading, not a test. s̄ fails as a single collapse variable | delivered-changed | 2 |
| P | Proposal E3 (threshold ∝ 1/L and with p₀), E4 (pass@k and diversity), E5 (escaping sparse rewards), E6 (small LLM), forward KL (E2) | Forward KL in H4(a); pass@k only in H1/H2; E5 partly in H5 | Forward KL delivered (converged median error 8e-5, base :297). The 1/L scaling is not quantified (the L=4 thresholds are an artefact, base :133). No diversity metric. No LLM | partial / not done | — |

## 2. Findings ranked, with the cheapest fix

1. **[high] The central claim (H3) is refuted.** The final report must present the three-mechanism account as
   post-hoc and say that the pre-registered collapse failed. No new run is needed. To strengthen it at zero
   cost, report the c09 η = 0.003 result (base FINDINGS:135, 273) as direct evidence that absorption can
   happen without a step-size threshold.
2. **[high] Prior art.** See section 3. This is a wording change only.
3. **[medium] H2 as promised was never reported, but the existing data suffices.** From
   `analysis/base/tables/h2_binned_ratio.csv`, W_G^grpo = ratio·(G−1)/G. Plot it against p for each G, with
   the curves 1 and 1/√(p(1−p)), and give λ = 0, 0.23, 0.63, 0.89. That needs no CPU time.
   Caveat: Davis & Recht §6, Fig. 3 already show this curve, so it is a validation, not a contribution.
4. **[medium] The H5 verdict hangs on ties.** In tabular GRPO, samples-to-target tie at 32,000 for
   G = 32–128 (p₀ = 0.01), at 12,800 for G ≥ 4 (p₀ = 0.1), and at 6,400 for every G at p₀ = 0.3 (R).
   `analyze.py:1007` breaks ties with an unstable `sort_values(...).first()`, which is why
   "best G = 128" is reported at p₀ = 0.01 and 0.03 (base :396). Taking the smallest G that reaches
   the minimum gives 32, 16, 4, 2, which still falls with p₀, so the conclusion survives, but the table
   must change. The mean-rule criterion (spread ≤ 2×, got exactly 2.00) is one logging interval:
   12,800 vs 6,400. Cheapest fix: rerun tabular H5 at p₀ ∈ {0.1, 0.3} with `log_every=1`:
   2 × 7 G × 2 rules × 7 η × 3 seeds = 588 jobs, at the measured 20.8 s/job ≈ **3.4 CPU-h**.
5. **[medium] The Bay & Yearick law is untested.** As a first look from the existing cache, log
   samples-to-target correlates with log(G·p̄) at −0.81/−0.76 (tab grpo/mean), but with log p̄ alone at
   −0.93/−0.98, and the m64 correlations are +0.11/+0.13 (R). So G·p does not collapse the data better
   than p does. Fit their closed form on `h5_best_eta.csv`. That needs no CPU time.
6. **[medium] Heckel weights were not done.** This needs a new rule in `trainer.py` and in theory
   `gap_coefficients` (picked up automatically). Grid: tab/lin/m64 × p₀ {0.01, 0.03} × G {4, 8, 16} ×
   {grpo, grpo+heckel} × 5 η × 3 seeds = 540 jobs × ~21 s ≈ **3.2 CPU-h**. Measure samples-to-target and
   the absorbed fraction (m64 absorbs 20–28% in H5, base :426–427).
7. **[low] Huang resets were not done.** With binary rewards, the coverage failure that Huang et al. fix
   cannot occur, and resetting the reference to the current policy reproduces Mroueh's "Mirror GRPO"
   (p→1, milestone:153–156). Either drop the sentence or run a tiny s_eq-style grid: reset period
   {∞, 500, 2000} × β {0.5, 2} × G {4, 64} × 3 seeds × 64 replicates = 36 jobs × ~32 s ≈ **0.3 CPU-h**.
8. **[low] H1b covers only the trivial case.** Say that rescaling equivalence holds by construction
   for mean/RLOO (tests:25–31) and was observed for GRPO = RLOO at G = 2, with 5% of pairs diverging
   chaotically at η ≥ 0.1.
9. **[low] H4 does not test sharing.** H4(a) and s_eq are tabular. If the report claims "SGD with
   shared parameters", either restrict the claim or check the non-absorbed prompts in the H4(c)
   runs from the existing cache.

## 3. Milestone wording that must change

- **258–260**: "Davis & Recht and Heckel use this form for infinite groups. We compute W_G exactly for
  finite G". This is wrong: Davis & Recht Eq. 4.1–4.2 give it at finite G (HANDOFF.md §1, and our own
  `synthetic_rl/theory.py:6` cites them). Rewrite as "Davis & Recht derive this at finite G; we verify it".
- **322–324**: "the infinite-group version of (1)". The same error.
- **306–309 (H1, H2)**: present these as replications of Davis & Recht §4 and §6 / Fig. 3, not as tests of
  new claims.
- **311–315 (H4)**: cite Vojnovic & Yun Eq. 11 for the finite-G stationary equations (both KL
  directions). Change "the finite-G fixed point" to "a stable root", because at G = 64 there are two
  (stretch :443).
- **309–311 (H3)**: this can no longer be claimed. Replace it with: no single effective step collapses the
  data; there are three mechanisms (drift, overshoot, batch noise).
- **316–319**: "Suk and Duan's failure shows up as prompts pushed to p≈0 … recovery is slow". Qualify
  it: in c09 absorption occurs at η = 0.003, so it is not a step-size failure, and lin largely
  recovers (final/peak absorbed 0.15).
- **289–292**: "the weight that sets the objective also sets … a larger step". The support is weak: η*
  differs between GRPO and mean only at large G (m64 at G = 64: 0.1 vs 0.3, base :209–211) and is equal
  at G = 8 (:120–131). Soften it.
- **325–331**: remove the Heckel and Huang promises, or deliver them (items 6 and 7).
- **331–333**: if φ is reported, cite He et al. App. F for it (HANDOFF.md §1). Define "advantage gap"
  explicitly, because Jia et al. use the term differently.

## 4. Overall verdict

The milestone's quantitative predictions that come straight from the identity (H1, H4) were delivered
convincingly. H4 is the strongest result: the finite-G root is hit to 1e-4 and rejects the G→∞ root
by up to 0.076. The milestone's own contribution, H3 (one effective step plus interference sets
stability), was pre-registered and failed. It was replaced by a well-controlled but post-hoc account
with three mechanisms. That account is the most interesting finding, but it must be presented as
exploratory. H2 was run as a different question from the one promised; the promised one can be answered
from data already on disk. H5 was promised only vaguely and its pass rests on tie-breaking at coarse
resolution. Two of the side promises (Heckel, Huang) were not attempted. Prior art (Davis & Recht,
Vojnovic & Yun, He et al.) means H1, H2 and the H4 equations are replications, and the milestone
currently misstates this at lines 258–260 and 322. Net: about 60% of the promises were delivered as
stated; the rest are fixable mostly with analysis plus about 7 CPU-hours.

## Appendix: recomputation snippets (run from `981_project/`)

```python
# H2 arcsin interpolation (tabular GRPO bins)
d=pd.read_csv("analysis/base/tables/h2_binned_ratio.csv"); t=d[(d.level=="tab")&(d.rule=="grpo")]
W=t.measured*(t.G-1)/t.G; Winf=1/np.sqrt(t.p*(1-t.p)+1e-5)
lam = sum((W-1)*(Winf-1))/sum((Winf-1)**2) per G  -> 0.000, 0.229, 0.626, 0.886
# H4 vs G->inf: h4_a_runs.csv, grpo reverse converged; fixed_points(p0,beta,math.inf,"grpo") stable root
#   median |late_mean - p_inf| by G: 0.076, 0.047, 0.031, 0.013, 0.0026 ; finite-G error 3e-4..3e-5
# H1b: runs_h3.csv G==2, pivot final_mean on rule: |grpo-rloo| median 6.4e-7, share>0.01 = 0.051 (eta>=0.1)
# Recovery: runs_h3.csv non-tab, max_absorbed>0.05: median absorbed/max_absorbed by level
#   c05 0.58, c09 1.00, lin 0.15, m64 1.00, m256 1.00
# H5 ties and Bay-Yearick: h5_best_eta.csv pivot samples by (rule,p0) x G (tab);
#   corr(log samples, log G*p_init) vs corr(log samples, log p_init) per level/rule
# Costs: runs_h5.csv mean seconds tab 20.8, all 21.1; runs_s_eq.csv 32.1
```
