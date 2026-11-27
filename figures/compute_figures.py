"""Presentation figures that need new (small) runs. Both run on a laptop.

    python figures/compute_figures.py h1          # Simon's H1 norm plots, tabular and MLP
    python figures/compute_figures.py trajectory  # minority success over training (3 runs)
    python figures/compute_figures.py             # both

Outputs go to figures/out/ (git-ignored); raw run files to figures/out/raw/.
"""

import gzip
import json
import subprocess
import sys

from common import OUT, ROOT, plt, save


def h1_norms():
    """Slide 8: run Simon's experiments/h1.py for both policies (a few minutes for tabular; longer for MLP)."""
    for policy in ("tabular", "mlp"):
        out = OUT / f"s08_h1_norms_{policy}"
        print(f"running experiments/h1.py --policy {policy} -> {out}", flush=True)
        subprocess.run([sys.executable, "-m", "experiments.h1", "--policy", policy, "--output", str(out)],
                       cwd=ROOT, check=True)


def trajectory():
    """Slide 16: class means over training at m = 1/3, 0.7 and 0.9 (linear, ρ = 0.9, exact update, SGD)."""
    raw = OUT / "raw"
    configs = []
    for m in (1 / 3, 0.7, 0.9):
        configs.append(dict(
            mode="train", K=3, L=1, d=16, features="correlated", rho=0.9, answers="rule", n_train=512,
            n_test=256, arch="linear", init="uniform", rule="grpo", G=8, B=32, eta=0.1, optimizer="sgd",
            update="expected_full", steps=5000, log_every=25, snapshot_every=5000, majority=m, task_seed=0,
            seed=0, track_every=5, name=f"traj/m{m:.2f}"))
    for cfg in configs:
        path = raw / f"{cfg['name']}.json.gz"
        if not path.exists():
            print(f"running {cfg['name']}", flush=True)
            subprocess.run([sys.executable, "experiments/run.py", "--config", json.dumps(cfg), "--out", str(raw)],
                           cwd=ROOT, check=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = {"0.33": "#55A868", "0.70": "#DD8452", "0.90": "#C44E52"}
    for cfg in configs:
        d = json.load(gzip.open(raw / f"{cfg['name']}.json.gz"))
        steps = [h["step"] for h in d["history"]]
        major = [h["class_mean"][0] for h in d["history"]]
        minor = [h["class_mean"][1] for h in d["history"]]
        key = cfg["name"].split("m")[-1]
        ax.plot(steps, major, color=colors[key], lw=2, label=f"m = {key}: majority")
        ax.plot(steps, minor, color=colors[key], lw=2, ls="--", label=f"m = {key}: minority")
    ax.set(yscale="log", ylim=(1e-6, 1.5), xlabel="training step", ylabel="class mean success",
           title="The minority falls and stays down as m grows")
    ax.legend(fontsize=8, ncol=1, loc="lower left")
    save(fig, "s16_trajectory")


JOBS = {"h1": h1_norms, "trajectory": trajectory}


def main():
    for name in sys.argv[1:] or JOBS:
        JOBS[name]()


if __name__ == "__main__":
    main()
