# v3h: a correct minority class in Li, Pandey and Chen's environment (T20)

All 1,809 runs returned and none failed. They were launched from a clean commit (`enum-env`
`2f701de`; job digest `f69b99f753f9`), with their code at pinned commit `05ef005` on the torch
version they pin (2.15). Pre-registered in `PREREG_T20.md` (`review` `edff353`); exploratory.
`experiments/analyze_v3h.py` regenerates `FINDINGS.md`, `runs.csv` and `cells.csv`.

Setup: their 64 token prompts, 2-layer transformer shared by all prompts, 100 possible responses,
exact success probabilities. Ours: K answer classes, each with one correct string; class 0 holds
n_major of the 64 prompts; every prompt starts at success about 0.9/K. 2,000 steps, 10 seeds.

## The first step: the odds law holds exactly

Push ratio R at step 0 (median over seeds), random class assignment:

| Share of class 0 | 0.5 | 0.625 | 0.75 | 0.875 | 0.938 |
|---|---|---|---|---|---|
| K = 2, measured | 0.97 | 1.62 | 2.93 | 6.84 | 14.6 |
| K = 2, (K − 1) × odds | 1.00 | 1.67 | 3.00 | 7.00 | 15.0 |
| K = 3, measured | 1.90 | 3.24 | 5.81 | 13.3 | 28.5 |
| K = 3, (K − 1) × odds | 2.00 | 3.33 | 6.00 | 14.0 | 30.0 |

(K = 3 at share 0.344: R = 1.01.) The slope of log R on logit m is 1.00 for both K, and c is K − 1
within 5%. That is the value the odds law gives when prompts carry no information about their
class, which is the case here by construction.

## What happens over training

Shared transformer, exact update. "Others absorbed": at least half of the minority prompts stay
below 1e-3 while class 0 does not.

| Step rule | Share ≥ 0.875 (40 runs each) | Balanced classes (20 runs each) |
|---|---|---|
| Adam, lr 1e-3 | minority absorbed in 39; class 0 never | no class loses half its prompts in 18, but every run absorbs 18–20% of each class's prompts (see the corrections) |
| Adam, lr 3e-3 | minority absorbed in 40; class 0 never | neither in 9; one class absorbed in 9 |
| SGD, lr 0.3 | minority absorbed in 39; class 0 in 2 | one class absorbed in 15, either one |
| SGD, lr 1 | **both classes absorbed in 27** | both or one absorbed in all 20 |
| SGD, lr 3 | **both classes absorbed in 37** | both or one absorbed in all 20 |

Sampled GRPO (G = 8) gives the same picture, a little stronger: with Adam the minority is absorbed
in 80 of 80 high-share runs and class 0 in 1.

Without sharing: in 600 tabular runs and 9 runs with one copy of the transformer per prompt, no
prompt ever falls below its start (largest fall 1.8e-7), no prompt is absorbed, and both classes
end near 1.

## Predictions

| Prediction | Result |
|---|---|
| 1a. Collapse at share ≥ 0.875 (SGD at lr 1 and 3) | **8 of 8 cells by the registered statistic, but these cells do not show the mechanism:** at lr 1 and 3 the majority class is absorbed too in 67 of 80 runs, so the policy as a whole breaks down. The clean evidence is in the cells I did not name: Adam (both rates) and SGD at lr 0.3, where the minority is absorbed in 118 of 120 runs and class 0 in 2 |
| 1b. No absorption at balanced classes (every SGD rate) | **0 of 6 cells: fails.** With SGD one class takes over and the other is absorbed, in either direction. Only Adam at lr 1e-3 keeps both (18 of 20) |
| 2. No sharing, no decline | **holds** (609 runs) |
| 3. Odds law: slope within 0.8–1.2 | **holds**: 1.00 and 1.00 |
| 4. R = 1 within 0.1 of 1/K | K = 2: 0.508, **holds**. K = 3: not computable as registered, because R is already 1.01 at the smallest share on the grid (0.344); that is consistent with a crossing at 1/3 |

## The parity task (class 0 = "a·b is even", 48 of 64 prompts)

Here the class is a function of the prompt, so the minority can in principle be told apart.

| | Minority absorbed | Minority's lowest mean | Minority at the end |
|---|---|---|---|
| exact update, Adam (20 runs) | 0% | 0.000–0.003 | **1.00 in all 20: full recovery** |
| sampled GRPO, Adam (20 runs) | 100% | 0.000 | 0.00: no recovery |
| exact or sampled, SGD lr 0.3 (20 runs) | 93% | 0.000 | 0.00 (1 run of 10 recovers in each) |
| tabular; per-prompt copies | 0% | never below the start | 0.96–1.00 |

R at step 0 is 2.93, the same as for a random assignment at that share: the base model does not yet
use the parity.

## Reading

- **The mechanism carries over** from feature-vector prompts and a linear or MLP policy to token
  prompts and a transformer: a majority class drives a correct minority class to near zero, only
  when parameters are shared, and the first step follows the odds law with no fitted quantity.
- **Balance is a knife edge when prompts carry no class information** (not new: see the corrections)**.** The law gives R = 1 at
  m = 1/K in that case, so neither class is protected, and with plain SGD one class takes over at
  random. In our toy the features carry class information (c < K − 1), R is below 1 at balance, and
  nothing is absorbed there. So "no collapse at balance" depends on c, as the law says.
- **Recovery depends on having a signal at low success, and on the optimizer** (see the corrections)**.** On the parity task the exact update
  with Adam brings the minority back from 0.003 to 1.00 in every run. Sampled GRPO does not: once
  every sampled group is all wrong, there is nothing to learn from. This is the cleanest
  demonstration so far of the second ingredient of the collapse: the push starts it, and the loss
  of signal at low success keeps it.
- **A caution on my own prediction 1a.** I named the two largest SGD rates. They are in a regime
  where everything breaks, the majority included, which is the large-step route and not the
  class-share mechanism. The report should cite the Adam and SGD lr 0.3 cells for the mechanism and
  say that these were not the registered cells.

## Deviations

1. Prediction 4 for K = 3 cannot be computed as registered (no crossing on the grid).
2. The interpretation of prediction 1a departs from its registered statistic, as explained above.

## Corrections after the review of 2026-11-23 (`analysis/review/ADVERSARIAL_v5.md`)

1. **Only Adam at lr 1e-3 isolates the effect of the class share.** Per prompt (the registered
   measure), the share of the non-zero classes' prompts absorbed under the exact update is:

   | Step rule | Balanced classes | Share ≥ 0.875 | Class 0 prompts absorbed at share ≥ 0.875 |
   |---|---|---|---|
   | Adam, lr 1e-3 | 0.18 (and 0.20 for class 0) | 0.94 | 0.00 |
   | Adam, lr 3e-3 | 0.42 | 0.99 | 0.00 |
   | SGD, lr 0.3 | 0.48 | 0.98 | 0.08 (in 11 of 40 runs) |
   | SGD, lr 1 | 0.50 | 0.91 | 0.76 |
   | SGD, lr 3 | 0.76 | 0.99 | 0.93 |

   SGD at lr 0.3 loses a class at balance too, so its high-share collapse cannot be attributed to
   the share. "118 of 120 clean runs" should not be cited. The statement the data support: under
   Adam at lr 1e-3 the per-prompt rate rises from about 0.2 at balance to 0.94 at high share, with
   class 0 untouched.
2. **Prediction 1a:** passed as registered, but the registered cells (SGD lr 1 and 3) break the
   majority too; the cell that isolates the effect was not named in advance.
3. **"Neither class absorbed"** in the table uses a class-level rule (at least half of a class's
   prompts). Prediction 1b is per prompt and fails in every cell, Adam included.
4. **Not new.** Zero first-order drift at m = 1/K and its dependence on the step size are in Li,
   Pandey and Chen (Prop. 2, section 6.1); the "knife edge" reading was also made after prediction
   1b failed, and the cleanest cell does not fit it neatly (both classes lose a fifth). That
   recovery needs a signal at low success is Schaul et al.'s f(0) = 0.
5. **Recovery on the parity task needs Adam as well as the exact update:** with Adam the exact
   update recovers (20 of 20) and sampling blocks it; with SGD at lr 0.3 neither recovers (1 of 10
   each).
6. **By construction:** c = K − 1 here because classes are assigned at random, so this environment
   checks the odds factor, not the correction c. The tabular control cannot fail (a tabular correct
   logit cannot be lowered by its own prompt's update); the per-prompt-copies control ran in 9 runs.
