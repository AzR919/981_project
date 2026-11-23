# v3g: H4 and H5 as prevention levers (T17 KL anchor, T18 release test, T19 group size)

All 5,540 runs returned and none failed. They were launched from a clean commit (`toy-v3` `b059c73`;
job digest `5ab4c97ca34d`). Pre-registered in `PREREG_v3d_addendum3.md` (`review` `2df02ca`);
exploratory. `experiments/analyze_v3g.py` regenerates `FINDINGS.md` and the tables.

Task: K = 3 answer classes, uniform start, majority share m; the KL reference is the start policy.

## T17: does a KL anchor protect the minority? (H4)

1,040 runs at m = 0.9, ρ = 0.9: linear and MLP, exact and sampled updates, η ∈ {0.1, 1}.

| Anchor | Minority prompts absorbed | Majority (final) | Minority (final) | Minority when the majority reaches 0.80 |
|---|---|---|---|---|
| none | 100% | 1.00 | 0.00 | 0.000 |
| reverse KL, β ≤ 0.1 | 100% in every run | 1.00 | 0.00 | 0.000 |
| reverse KL, β = 0.3 | 16% (linear), 33% (MLP) | 0.99–1.00 | 0.54–0.58 | 0.001–0.002 |
| reverse KL, β = 1 | 0–4% | 0.79–0.80 | 0.40–0.81 | 0.18–0.27 |
| forward KL, β = 0.01 | 0.2–2% | 0.99–1.00 | 0.30 (linear), 0.55 (MLP) | 0.002 |
| forward KL, β = 0.03 | ≤ 0.1% | 0.98–0.99 | 0.57 (linear), 0.98 (MLP) | 0.005 |
| forward KL, β = 0.1 | 0% | 0.95–0.97 | 0.36 (linear), 0.97 (MLP) | 0.02 |
| forward KL, β = 0.3 | 0% | 0.89–0.92 | 0.21 (linear), 0.92 (MLP) | 0.06 |

(Medians over tasks and over the four update × η settings of each architecture.)

| Prediction | Result |
|---|---|
| 1. Protection does not get worse as β grows | 14 of 16 series |
| 2. The majority reaches 0.80 at β ≤ 0.3; never at β = 3 | 64 of 64 cells; 12 of 16 (the linear task at η = 1 crosses 0.80 briefly at β = 3) |
| 3. Forward KL at β ∈ {0.03, 0.1, 0.3}: none absorbed, majority reaches 0.80, minority below 0.2 at that point | **23 of 24 cells** |
| 4. Reverse KL at β ≤ 0.3 does not prevent absorption | **24 of 32.** Holds at β ≤ 0.1 (every run fully absorbed); fails at β = 0.3, where most minority prompts recover |
| 5. Forward protects at least as well as reverse at equal β | 48 of 48 |

**Reading.**
- **The direction of the KL term decides.** Forward KL prevents absorption at every strength tried,
  including β = 0.01, where the majority still ends at 0.99. Reverse KL does nothing up to β = 0.1
  and works only in a narrow window (β = 0.3 to 1) before it caps the majority.
- **Why:** the forward-KL pull on a prompt does not shrink as its success goes to zero; the
  reverse-KL pull does.
- **It keeps the minority alive, it does not shield it.** With any anchor the minority still falls
  to 0.002–0.06 while the majority is learned, and recovers later. With the MLP under forward KL
  every prompt, minority included, ends at the single-prompt resting point from theory
  (0.997, 0.992, 0.974, 0.920 for β = 0.01 … 0.3). The linear policy's minority ends lower
  (0.2–0.6).
- **Cost.** Forward KL does not delay the majority (0.80 by step 100, as without an anchor). The
  cost is the ceiling the anchor puts on final success, which grows with β.

## T18: why do collapsed prompts stay down? (H4)

320 runs. Phase 1 is plain GRPO at m = 0.9. At the release step the majority stops being rewarded.
"Returned": minority mean ≥ 0.2 at step 20,000.

| After the release | Release at step 100 (minority at 1e-3) | Step 1,000 (1e-4 to 1e-5) | Step 5,000 (3e-5 or lower) |
|---|---|---|---|
| no release (control) | 0 of 20 return | | |
| no KL | 20 of 20 (about 1,600 steps) | 15 of 20 (about 15,000 steps) | 0 of 20 |
| reverse KL, β = 0.01 | 20 of 20 (1,200) | 20 of 20 (10,300) | 0 of 20 |
| reverse KL, β = 0.1 | 20 of 20 (300–400) | 20 of 20 (2,100–2,500) | 20 of 20 (9,000–10,500) |
| forward KL, β = 0.01 | 20 of 20 (200–300) | 20 of 20 (200–500) | 20 of 20 (200–600) |
| forward KL, β = 0.1 | 20 of 20 (100) | 20 of 20 (100) | 20 of 20 (100) |

| Prediction | Result |
|---|---|
| 1. Control: no return | 2 of 2 architectures |
| 2. No KL: return depends on depth; none at step 5,000; time to return rises with 1/p at release | holds (Spearman 0.81 over 35 returning runs) |
| 3. Reverse KL: no return at release steps 1,000 and 5,000 | **2 of 8: fails.** Reverse KL does bring the minority back, slowly; only β = 0.01 at step 5,000 does not return in time |
| 4. Forward KL at β = 0.1 returns at every release step | 6 of 6 |

**Reading.**
- **There is no second resting point here.** Once the majority's pressure is removed, the minority
  comes back whenever some restoring force is left and there is enough time. This matches the
  fixed-point calculation in the pre-registration, and it corrects the plan's first idea that the
  bistability of T8 explains the collapse. T8's two resting points need a low reference success.
- **Collapsed prompts stay down for two reasons:** the majority keeps pushing (the control never
  returns), and the deeper the collapse, the slower every restoring force acts, except forward KL.
  Without KL the time to return grows with 1/p; at p ≈ 3e-5 it exceeds the horizon.
- **Forward KL returns in 100–600 steps from any depth.** Reverse KL needs 9,000–10,500 steps from
  the deepest release at β = 0.1.

## T19: does group size move the collapse threshold? (H5)

4,180 runs: the T15 share grid (m from 0.283 to 0.883), G ∈ {2, 4, 8, 16, 64}.

| Prediction | Result |
|---|---|
| 1. The step-0 threshold m\* does not depend on G | 4 of 4 cells (largest range 0.00002) |
| 2. The absorption threshold rises with G under grpo | **2 of 4 cells: fails.** No consistent direction |
| 3. It does not move under the mean rule | **2 of 4: fails** |
| 4. Any shift is at most 0.2 in m | holds (largest 0.15) |

Smallest share with absorption (mean minority absorption ≥ 0.05), exact update, grpo:

| | G = 2 | 4 | 8 | 16 | 64 |
|---|---|---|---|---|---|
| linear, ρ = 0.5 | 0.683 | 0.683 | 0.683 | 0.683 | 0.683 |
| linear, ρ = 0.9 | 0.533 | 0.453 | 0.453 | 0.453 | 0.453 |
| MLP, ρ = 0.5 | 0.533 | 0.533 | 0.533 | 0.683 | 0.683 |
| MLP, ρ = 0.9 | 0.413 | 0.373 | 0.413 | 0.413 | 0.453 |

**Reading.**
- **Group size is not a lever.** It cannot change the first step (prediction 1, exact), and later
  shifts are one or two grid points with no consistent sign.
- **The weight-ratio argument was incomplete.** G also changes the size of the reward weight, and
  so the effective step: the mean rule's threshold moves between G = 2 and G = 64 as well, although
  its weight does not depend on p. At a fixed number of steps this confounds any effect of G.
- **Sampling does not matter here.** At a budget of B·G = 256 the sampled thresholds equal the
  exact ones in 9 of 10 cases.

## The four levers side by side (linear task, ρ = 0.9, m = 0.9)

| Lever | Minority absorbed | Majority reaches 0.80 | Minority at that point | Final majority / minority |
|---|---|---|---|---|
| plain GRPO | 100% | step 100 | 0.0001 | 1.00 / 0.00 |
| group size (H5) | 100% at every G | | | no protection |
| reverse KL, β ≤ 0.1 (H4) | 100% | step 100 | 0.000 | 1.00 / 0.00 |
| reverse KL, β = 0.3 | 3–29% | step 100 | 0.001 | 0.99 / 0.5–0.9 |
| forward KL, β = 0.03 (H4) | ≤ 0.2% | step 100 | 0.005 | 0.98 / 0.57 |
| pass@2 (v3f) | 0–12% | step 100 | 0.03–0.08 | 0.98 / 0.79–0.87 |
| pass@4 (v3f) | 0% | steps 350–3,900 | 0.31–0.42 | 0.92–0.95 / 0.67–0.72 |
| projecting out the harmful direction (v3d) | 0% in 16 of 16 cells | not measured at the matched point | | see `analysis/v3d/SUMMARY.md` |

Only pass@4 and the projection keep the minority up *while* the majority is learned. Forward KL and
pass@2 let it fall and bring it back at almost no cost to the majority. Reverse KL at usual
strengths and group size do not help.

## What this means for H4 and H5

- **H4** now has a result of its own: the KL term matters for the minority, and its direction
  matters more than its strength. The milestone's fixed-point result stays as a verification; the
  failed η/β scaling stays reported as failed.
- **H5** is a clean negative: group size does not protect the minority. The milestone's "best G"
  pattern stays reported as an artefact of the learning-rate grid.
- A note for the LLM runs: the reverse arm of our LLM trainer and the forward arm (C8) differ in
  exactly this respect, so C7 and C8 are now on the storyline.

## Deviations

1. **`absorption_from`.** The pre-registration defined it as "mean minority absorption ≥ 0.05" and
   also said "as in addendum 1", but v3e used "median above 0.5". The registered wording is used for
   the predictions; `FINDINGS.md` shows both. The conclusions are the same under either.
2. **T18 predictions** were changed from the plan before launch (one resting point, not two); this
   is recorded in the pre-registration itself.
