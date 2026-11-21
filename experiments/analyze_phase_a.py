"""
Phase A of plan v3: analyses of the existing base and stretch caches (no new runs).

usage:
    python experiments/analyze_phase_a.py a4 [--stretch results/stretch] [--out analysis/phase_a]

a4: lost problems and coverage. Toy p is exact, so pass@k = 1 - (1 - p)^k exactly;
per run, the change in mean pass@1 and pass@64 from p_init to the final p of the
stretch predictor runs (s_pred), and the number of prompts lost (p_init >= 0.01,
final p < 1e-3).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze import markdown  # noqa: E402


def a4(stretch, out):
    cache = os.path.join(stretch, "cache")
    runs = pd.read_csv(os.path.join(cache, "runs_s_pred.csv"))
    arrays = dict(np.load(os.path.join(cache, "prompts_s_pred.npz")))
    r = runs.iloc[arrays["row_train"]].reset_index(drop=True)
    p0, p1 = arrays["p_init"], arrays["p_final"]
    for k in (1, 64):
        r[f"d_pass{k}"] = np.nanmean(1 - (1 - p1) ** k, axis=1) - np.nanmean(1 - (1 - p0) ** k, axis=1)
    r["narrowing"] = (r.d_pass1 > 0) & (r.d_pass64 < 0)
    r["lost"] = np.nansum((p0 >= 0.01) & (p1 < 1e-3), axis=1)
    cells = r.groupby(["level", "init", "rule"]).agg(
        runs=("lost", "size"), d_pass1=("d_pass1", "mean"), d_pass64=("d_pass64", "mean"),
        narrowing_share=("narrowing", "mean"), lost_mean=("lost", "mean")).reset_index()
    wide = cells.pivot_table(index=["level", "init"], columns="rule", values="d_pass64").reset_index()
    more = int((wide.grpo < wide["mean"]).sum())
    lines = ["# A4: lost problems and coverage in the toy (stretch `s_pred`, existing data)", "",
             "pass@k is exact (toy p is known): pass@k = 1 − (1 − p)^k, averaged over the 512 prompts, "
             "from p_init to the final p. Lost: p_init ≥ 0.01 and final p < 1e-3. `narrowing_share`: runs "
             "where pass@1 rises and pass@64 falls. Runs span η from 0.3× to 10× the base-batch absorption "
             "threshold, so some losses are overshoot.", "",
             markdown(cells), "",
             f"- Every cell: mean pass@1 rises ({cells.d_pass1.min():.2f} to {cells.d_pass1.max():.2f}) and "
             f"mean pass@64 falls ({cells.d_pass64.max():.2f} to {cells.d_pass64.min():.2f}).",
             f"- GRPO lowers pass@64 more than the mean rule in {more} of {len(wide)} (level, init) pairs.", ""]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "A4.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[-3:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("a4",))
    parser.add_argument("--stretch", default="results/stretch")
    parser.add_argument("--out", default="analysis/phase_a")
    args = parser.parse_args()
    {"a4": a4}[args.what](args.stretch, args.out)


if __name__ == "__main__":
    main()
