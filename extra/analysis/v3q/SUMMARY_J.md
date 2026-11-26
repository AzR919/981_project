# v3q, batch J: the transformer environment under small-step SGD (T29)

All 240 J runs returned and none failed. Launched from `enum-env` `4b271a0`. Pre-registered in
`PREREG_v3d_addendum10.md` (`review` `e9510e4`); exploratory. `extra/experiments/analyze_v3q_j.py` regenerates
`FINDINGS_J.md` and `runs_J.csv`. `v3_a4` in the same batch belongs to battery A and is not analysed here.

**Setup:**
- T20's environment: 64 token prompts, a 2-layer transformer shared by all prompts, exact success;
- K = 3, n_major ∈ {22, 32, 40, 48, 56, 60} (share 0.34 to 0.94);
- SGD at η = 0.03 × 20,000 steps and η = 0.1 × 6,000 steps, which matches the flow time of SGD 0.3 × 2,000;
- exact update (pg) and sampled GRPO (G = 8), 10 seeds.

**Definitions (following T20):**
- **Absorbed prompt:** stays below 1e-3 over the last 20% of steps.
- **Minority absorbed:** at least half of the non-zero classes' prompts are absorbed while class 0 is not.
- **Class 0 absorbed:** at least half of class 0's prompts are.

| | Class 0 absorbed | Minority absorbed | Minority prompts absorbed (per prompt) |
|---|---|---|---|
| Share ≥ 0.875 (80 runs) | 0 of 80 | 80 of 80 | 0.99–1.00 |
| Share 0.5 to 0.75 (120 runs) | 0 of 120 | 115 of 120 (5 of 10 for exact, η = 0.03, at share 0.5) | 0.52–1.00 |
| Balanced, n_major = 22 (40 runs) | 1 of 40 | 4 of 40 | 0.08–0.28 (class 0: 0.09–0.31) |

| Prediction | Pass rule | Result |
|---|---|---|
| J1a. Class 0 never absorbed at share ≥ 0.875 | 0 runs | **Holds: 0 of 80** |
| J1b. Minority absorbed in high-share runs | ≥ 80% | **Holds: 80 of 80 (100%)** |
| J1c. Minority absorbed in balanced runs | ≤ 20% | **Holds: 4 of 40 (10%)** |

Each (algorithm, η) arm passes on its own as well. The worst balanced arm is GRPO at η = 0.1 with 2 of 10, exactly at
the limit.

## Reading

- **T20's SGD failure was a large-step effect.** At η ≤ 0.1, SGD shows the same clean share effect as Adam: the
  minority is absorbed in every high-share run and class 0 is never touched.
- **For comparison, in T20:** SGD at lr 1 and 3 absorbed both classes in 64 of 80 runs, and SGD at lr 0.3 lost a
  class at balance in 15 of 20.
- **Here, balance is mostly safe:** 4 of 40 runs by the class rule, and per prompt 8% (exact, η = 0.03) to 28%.
  That is roughly T20's Adam lr 1e-3 rate (0.18), against 0.48 for SGD 0.3. The class means at balance never drop
  below their start (0.30).
- **The effect is not specific to high shares.** At share 0.5, already well above 1/K, the minority is absorbed in 35
  of 40 runs. That is consistent with the odds law putting the turning point at m = 1/K.

## Deviations

1. **Pooling.** The three parts are pooled over both η and both algorithms ("both algorithms" in the setup). Each arm
   also passes separately.
2. **Class rule at balance.** With K = 3 at n_major = 22, the "minority" is classes 1 and 2 together (42 of 64
   prompts). A run in which one of them is fully absorbed and the other untouched therefore counts as "minority
   absorbed". Per-class shares are in `FINDINGS_J.md`; "any class absorbed" happens in 7 of 40 balanced runs.
3. **Per prompt, balance is not absorption-free** (8–28% of prompts), as in T20's correction 3. The prediction uses
   the class rule, so it passes. A per-prompt "no absorption at balance" rule would fail, as T20's 1b did.

## What it means for the project

- **The transformer environment now supports the class-share mechanism under plain SGD as well as Adam,** as long as
  steps are small.
- T20's registered SGD cells, which broke the majority too, can now be replaced by cells that isolate the share
  effect under both optimizers. This removes the caveat that the effect was clean "only under Adam at lr 1e-3".
- **Two caveats remain:**
  - per prompt, balance still loses a minority of prompts;
  - collapse at share 0.5 shows the relevant boundary is 1/K, not a "large majority". That is the expected
    consequence of classes assigned at random (c = K − 1).
