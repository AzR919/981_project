# Presentation figures

Code for the figures in `docs/presentation.md`. Images are written to `figures/out/`, which git ignores.

```bash
python figures/make_figures.py         # figures from committed tables (seconds)
python figures/compute_figures.py      # figures that need new runs (minutes)
```

Pass words to build only some figures, for example `python figures/make_figures.py h4 push`.

## Figures

| File in `out/` | Slide | Source | Needs runs? |
|---|---|---|---|
| `s03_advantage_gap.png` | 3: W_G(p) for each rule and G | `synthetic_rl/theory.py` (analytic) | no |
| `s05_ladder.png` | 5: the sharing ladder | schematic | no |
| `s08_h1_identity.png` | 8: H1, error against sampling noise; cosine per rule | `analysis/base/tables/h1_runs.csv` | no |
| `s08_h1_norms_tabular/`, `s08_h1_norms_mlp/` | 8: Simon's ‖g‖² along training | `experiments/h1.py` | **yes** (`compute_figures.py h1`) |
| `s09_h2_sharing.png` | 9: H2, tabular on the W_G line; drift from other prompts by level | `analysis/base/tables/h2_*.csv` | no |
| `s10_h4_fixed_points.png` | 10: H4, measured against predicted p* | `analysis/base/tables/h4_a_runs.csv` | no |
| `s12_h3_thresholds.png` | 12: H3, thresholds by level; the shift one s̄ would need | `analysis/base/tables/h3_thresholds.csv`, `h3_collapse.csv` | no |
| `s15_push.png` | 15: T11 push ratio against m; the odds law, measured against predicted | `analysis/v3d/tables/t11_cells.csv`, `analysis/theory/odds_law_runs.csv` | no |
| `s16_dose_response.png` | 16: share of minority prompts absorbed against m (P1, task level) | `analysis/v3a/P1_TASK_LEVEL.md` | no |
| `s16_trajectory.png` | 16: class means over training at m = 1/3, 0.7, 0.9 | 3 new runs via `experiments/run.py` | **yes** (`compute_figures.py trajectory`, about 3 minutes) |
| `s17_levers.png` | 17: forward vs. reverse KL; cost of forward KL | `analysis/v3g/t17_runs.csv` | no |

The tables come from the lab-machine runs (`results/`, also ignored). The figures here only re-plot them.
