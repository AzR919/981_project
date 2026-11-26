# 981_project

Optimization Project for SFU CMPT981

## Start here: the promised work, and what is extra

This repo is organised so the work promised in the milestone comes first.

- **The milestone tested five hypotheses (H1–H5)** about GRPO with right/wrong rewards, in a synthetic task where
  every success probability is exact.
- **Everything beyond the milestone is in `extra/`.** Read it if you want to: the promised work does not depend on it.

| Where | What |
|---|---|
| `docs/milestone.pdf`, `docs/project_guide.html`, `docs/rlvr_primer.html` | The milestone, the theory and definitions, and background |
| `synthetic_rl/`, `main.py`, `run_pilot.py`, `experiments/h1.py`, `docs/Simon.md` | The shared framework and the pilots |
| `experiments/run.py`, `grids.py`, `analyze.py`, `analyze_phase_a.py` | The simulator, the job grids and the analysis of the promised batches |
| `analysis/base/`, `analysis/stretch/`, `analysis/phase_a/` | Results for H1–H5, the stretch goals, and the re-analyses of H2, H4 and H5 |
| `analysis/FINAL_RESULTS.md` | Every claim with its status (one table per topic; each section labelled Promised or beyond) |
| `docs/EXPERIMENTS.md`, `docs/Ankush.md` | An index of every experiment, and a write-up of each (labelled Promised, Core follow-up or Extra) |
| `extra/` | Everything beyond the milestone: see `extra/README.md` |

**Four promised re-tests were run later** and live in `extra/analysis/v3a/` and `v3b/`, next to the post-milestone
P1–P3: T4 and T4 edge (H5), T7 (H1) and T8 (H4). T9, the all-wrong-group test the milestone named, is there too.

---

## Shared synthetic experiment code

Use the components in `synthetic_rl/` when building a synthetic experiment:

- **Use `synthetic_task.py` to generate data and verify answers.**
- **Use `policy.py` to create the model that learns to answer.**
- **Use `trainer.py` to train and evaluate the model.** Create a `Trainer` with the policy, optimizer, and training settings, then call `train()`. It handles sampling, verification, advantage calculation, and policy updates. Use `evaluate()` to measure performance without updating the model.
