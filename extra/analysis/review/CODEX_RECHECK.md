# Independent recheck of the Codex audit (2026-11-21)

**Sources.** Codex reviewed a live local snapshot in which 96 of 320 T1 result files were iCloud
placeholders. This recheck uses the complete caches:
- `results/v3a/cache/v3_t1.csv` (17,279 runs);
- `results/v3b/cache/v3_t1s1.csv`;
- `results/v3a/cache/v3_t2.csv`.

Code: `analysis-v3@606353e`, `llm-transfer@0508082`.

**Verdicts:** ✅ agree · ◐ agree in part · ✗ disagree.

| Codex claim | Verdict | Our numbers |
|---|---|---|
| P1 at the smallest step: positive for all MLP cells and linear SGD; weak for linear Adam | ✅ Confirmed on complete data | MLP: 10/10 tasks positive in all 8 cells, +8.3 to +10.3 points. Linear SGD: 10/10 tasks, +8.3 to +10.2. Linear Adam: 2–7/10 tasks, +0.04 to +0.74 |
| Our reading of linear Adam | ◐ (Codex calls it a boundary; we think it is mostly horizon) | At η = 1e-4 × 5,000 steps the flow time is 0.5. Absorption time ∝ 1/η under Adam (`phase_a/ADAM.md`), and at η ≥ 1e-3 linear Adam absorbs the whole minority set (`v3a` figure). Codex's matched-progress test decides this; adopted |
| MLP init seed 1: all 10 tasks positive in 8 cells, +9.3 to +10.3 | ✅ | Reproduced: 9.30–10.27, 10/10 in every cell |
| The pooled P1 p (3e-8) is not seed-level | ✅ (already corrected, `fa77a29`) | Task-level exact Wilcoxon: max p = 0.0098, 16/16 cells |
| P2 per-B table (B = 8: +11.22 [8.65, 13.85]; …) | ✅ Reproduced exactly | Same means and intervals to ±0.05 points |
| P3: registered result fails; ΔAUC +0.009; scaling is fitted before folds | ✅ | The post-hoc sensitivity will refit scaling within folds |
| C1 analyser accepts partial inputs, and a family verdict from 2 rows | ✅ | Fixed in `0508082`: refuses missing ids, n ≠ 256, steps ≠ 500 or repeated seeds; "untestable" below 3 seeds |
| C2: date the fixed-panel sensitivity before outcomes; 8 clusters per role | ✅ | Goes into the PREREG_C2 amendment, pending the user |
| The launcher records HEAD but copies the working tree | ✅ | v3a–v3c were launched from committed trees (untracked files only). Next step: record `git status --porcelain` and a digest of the job list in DONE |
| Ten new papers | ◐ Not verifiable from our notes; none is in `papers/graph` | Triage them (an AI triage agent, abstracts only) before citing, as the project's working rules require |
| Exploration memo, §1 analytic control: Δlogit_B ∝ (1 − f) + f·cos θ | ✅ Derivation checked: Δθ ∝ f·x_A + (1 − f)·x_B at p = 0.5, so x_B·Δθ follows | At 120°: f = 0.5 → +0.25, f = 2/3 → 0, f = 0.9 → −0.35, as stated |
| Exploration memo, four-branch replay (A-only, B-only, mixed from one state) | ✅ Sound and cheap (CPU) | Replay on the lab machines: run h4/01500 diverged across platforms, so exact replay must use the same platform |
