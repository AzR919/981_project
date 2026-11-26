"""Shared paths and plot style for the presentation figures."""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
OUT = Path(__file__).resolve().parent / "out"  # git-ignored

sys.path.insert(0, str(ROOT))

# One colour per sharing level, from no sharing to most sharing, and per rule.
LEVEL_COLORS = {
    "tab": "#4C72B0", "lin": "#55A868", "c05": "#C4AD66", "c09": "#DD8452",
    "m64": "#C44E52", "m256": "#8172B3",
}
LEVEL_NAMES = {
    "tab": "tabular", "lin": "linear", "c05": "linear, ρ=0.5", "c09": "linear, ρ=0.9",
    "m64": "MLP-64", "m256": "MLP-256",
}
RULE_COLORS = {"mean": "#55A868", "rloo": "#4C72B0", "grpo": "#C44E52"}
RULE_NAMES = {"mean": "Mean-only", "rloo": "RLOO", "grpo": "GRPO"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#E3E6EA", "grid.linewidth": 0.8, "legend.frameon": False,
})


def save(fig, name):
    """Save a figure as out/<name>.png and print where it went."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {os.path.relpath(path, ROOT)}")
    return path
