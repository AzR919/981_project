"""Presentation figures that need no new runs.

Each figure reads a table already committed under analysis/ (or is computed
analytically) and is written to figures/out/<name>.png.

    python figures/make_figures.py            # all figures
    python figures/make_figures.py h4 odds    # only those whose name contains a word
"""

import math
import re
import sys

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from common import ANALYSIS, LEVEL_COLORS, LEVEL_NAMES, RULE_COLORS, RULE_NAMES, plt, save

from synthetic_rl.theory import advantage_gap  # noqa: E402  (common puts the repo root on the path)

BASE = ANALYSIS / "base" / "tables"


def s03_advantage_gap():
    """Slide 3: W_G(p) for the three rules at several group sizes."""
    p = np.linspace(0.005, 0.995, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for G, style in ((2, ":"), (4, "-."), (16, "--"), (math.inf, "-")):
        label = "∞" if G == math.inf else str(G)
        ax.plot(p, advantage_gap(p, G, "grpo"), style, color=RULE_COLORS["grpo"], label=f"GRPO, G={label}")
    ax.plot(p, advantage_gap(p, 4, "rloo"), color=RULE_COLORS["rloo"], label="RLOO (any G)")
    ax.plot(p, advantage_gap(p, 4, "mean"), color=RULE_COLORS["mean"], label="Mean-only, G=4: (G−1)/G")
    ax.set(xlabel="success probability p", ylabel="advantage gap $W_G(p)$", ylim=(0, 6),
           title="The weight each rule puts on ∇p")
    ax.legend(fontsize=9, loc="upper center", ncol=2)
    save(fig, "s03_advantage_gap")


def s05_ladder():
    """Slide 5: the ladder of parameter sharing (schematic)."""
    rungs = [
        ("tabular", "one-hot prompts\neach prompt owns its logits", "no sharing", "tab"),
        ("linear", "logits = W·x\nGaussian features", "shared W", "lin"),
        ("linear, correlated", "features with ρ = 0.5, 0.9\nprompts look alike", "more", "c09"),
        ("MLP", "two ReLU layers\nwidth 64 / 256", "most", "m64"),
    ]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.set_axis_off()
    for i, (name, body, share, key) in enumerate(rungs):
        x = i * 2.35
        ax.add_patch(FancyBboxPatch((x, 0), 2.0, 1.6, boxstyle="round,pad=0.05",
                                    fc=LEVEL_COLORS[key], alpha=0.18, ec=LEVEL_COLORS[key], lw=2))
        ax.text(x + 1.0, 1.25, name, ha="center", fontsize=12, weight="bold")
        ax.text(x + 1.0, 0.6, body, ha="center", va="center", fontsize=9)
        ax.text(x + 1.0, -0.35, share, ha="center", fontsize=9, color="#555")
        if i:
            ax.annotate("", (x - 0.05, 0.8), (x - 0.3, 0.8), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set(xlim=(-0.2, 9.3), ylim=(-0.6, 1.8))
    ax.set_title("One task, increasing parameter sharing", fontsize=13)
    save(fig, "s05_ladder")


def s08_h1_identity():
    """Slide 8: H1. Measured error of the averaged update against its Monte Carlo noise."""
    df = pd.read_csv(BASE / "h1_runs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for level, g in df.groupby("level"):
        ax.scatter(g.mc_rel, g.rel_err, s=10, alpha=0.6, color=LEVEL_COLORS.get(level), label=LEVEL_NAMES.get(level, level))
    lim = [df.mc_rel.min() * 0.8, df.mc_rel.max() * 1.2]
    ax.plot(lim, lim, "k--", lw=1, label="error = sampling noise")
    ax.set(xscale="log", yscale="log", xlabel="Monte Carlo standard error (relative)",
           ylabel="‖measured − W_G∇p‖ (relative)", title="No bias beyond sampling noise")
    ax.legend(fontsize=8)
    ax = axes[1]
    order = ["mean", "rloo", "grpo", "passk2", "passk4", "maxrl"]
    data = [df[df.rule == r].cosine.values for r in order if (df.rule == r).any()]
    ax.boxplot(data, tick_labels=[r for r in order if (df.rule == r).any()], showfliers=True)
    ax.set(ylabel="cosine(measured, predicted)", title=f"{len(df)} configurations", ylim=(0.985, 1.0005))
    fig.suptitle("H1: the averaged GRPO update equals $W_G(p)\\,∇p$", y=1.02)
    save(fig, "s08_h1_identity")


def s09_h2_sharing():
    """Slide 9: H2. Per-prompt change relative to the mean rule, against the no-sharing prediction."""
    df = pd.read_csv(BASE / "h2_binned_ratio.csv")
    fo = pd.read_csv(BASE / "h2_first_order_runs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    g = df[(df.level == "tab") & (df.rule == "grpo")]
    ax.scatter(g.predicted, g.measured, s=14, alpha=0.7, color=LEVEL_COLORS["tab"])
    lo, hi = g.predicted.min() * 0.9, g.predicted.max() * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xlabel="predicted from the prompt's own $W_G$", ylabel="measured change, GRPO ÷ mean rule",
           title="Tabular: every prompt on the $W_G$ line")
    ax = axes[1]
    levels = [lv for lv in ("tab", "lin", "m64") if (fo.level == lv).any()]
    shares = [100 * fo[fo.level == lv].cross_share.median() for lv in levels]
    bars = ax.bar(range(len(levels)), shares, color=[LEVEL_COLORS[lv] for lv in levels])
    for bar, s in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 2, f"{s:.0f}%", ha="center")
    ax.set_xticks(range(len(levels)), [LEVEL_NAMES[lv] for lv in levels])
    ax.set(ylim=(0, 110), ylabel="share of a prompt's drift from other prompts (%)",
           title="With sharing, other prompts drive the change")
    fig.suptitle("H2: each prompt follows its own $W_G$ only without sharing", y=1.03)
    save(fig, "s09_h2_sharing")


def s10_h4_fixed_points():
    """Slide 10: H4(a). Measured resting point against the finite-G prediction."""
    df = pd.read_csv(BASE / "h4_a_runs.csv")
    agg = df.groupby(["p0", "beta", "G", "rule", "kl"]).agg(pred=("predicted", "first"), meas=("late_mean", "mean"),
                                                            conv=("converged", "all")).reset_index()
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for kl, marker in (("reverse", "o"), ("forward", "s")):
        g = agg[(agg.kl == kl) & agg.conv]
        ax.scatter(g.pred, g.meas, marker=marker, s=22, alpha=0.7, label=f"{kl} KL")
    moving = agg[~agg.conv]
    ax.scatter(moving.pred, moving.meas, marker="x", s=22, color="#999", label="still moving at the end")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="predicted $p^*$ (finite-G fixed point)", ylabel="measured resting point",
           title=f"H4: median error {df.error.abs().median():.0e}")
    ax.legend(fontsize=9)
    save(fig, "s10_h4_fixed_points")


def s12_h3_thresholds():
    """Slide 12: H3. Step-size thresholds per sharing level, and the shift a single s̄ would need."""
    thr = pd.read_csv(BASE / "h3_thresholds.csv")
    col = pd.read_csv(BASE / "h3_collapse.csv")
    levels = ["lin", "c05", "c09", "m64", "m256"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, level in enumerate(levels):
        g = thr[(thr.level == level) & thr.eta_star.notna()]
        x = i + rng.uniform(-0.25, 0.25, len(g))
        ax.scatter(x, g.eta_star, s=12, alpha=0.6, color=LEVEL_COLORS[level])
    ax.set_xticks(range(len(levels)), [LEVEL_NAMES[lv] for lv in levels], rotation=20)
    ax.set(yscale="log", ylabel="η* (smallest η that loses ≥ 10% of prompts)",
           title="Thresholds span 5 decades (tabular: never)")
    ax.axhline(0.003, color="#999", ls=":", lw=1)
    ax.text(len(levels) - 0.6, 0.0036, "smallest η tried", fontsize=8, color="#777", ha="right")
    ax = axes[1]
    row = col.iloc[0]
    shifts = [row[f"shift_{lv}"] for lv in levels]
    ax.bar(range(len(levels)), shifts, color=[LEVEL_COLORS[lv] for lv in levels])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(levels)), [LEVEL_NAMES[lv] for lv in levels], rotation=20)
    ax.set(ylabel="shift needed to line up (decades of s̄)",
           title="One effective step s̄ does not line them up")
    fig.suptitle("H3 as planned fails", y=1.02)
    save(fig, "s12_h3_thresholds")


def s15_push():
    """Slide 15: the majority's push. T11 branch ratio and the odds law."""
    t11 = pd.read_csv(ANALYSIS / "v3d" / "tables" / "t11_cells.csv")
    odds = pd.read_csv(ANALYSIS / "theory" / "odds_law_runs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    g = t11[t11["at"] == "0T"]
    for (arch, rho), h in g.groupby(["arch", "rho"]):
        h = h.sort_values("m")
        ax.plot(h.m, h.med_ratio, "o-", label=f"{arch}, ρ={rho}")
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set(xlabel="majority share m", ylabel="push ÷ minority's own gain", yscale="log",
           title="Measured (T11): the majority outweighs the minority")
    ax.legend(fontsize=8)
    ax = axes[1]
    x = odds.c * odds.m / (1 - odds.m)
    for K, h in odds.groupby("K"):
        ax.scatter(x[h.index], h.R, s=8, alpha=0.5, label=f"K = {K}")
    xx = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
    ax.plot(xx, xx, "k--", lw=1, label="measured = predicted")
    ax.set(xscale="log", yscale="log", xlabel="predicted c · m/(1−m)  (c from the features, nothing fitted)",
           ylabel="measured push ratio R at step 0",
           title=f"Derived: R = c·m/(1−m); {len(odds)} runs, median error "
                 f"{100 * odds.err_R.abs().median():.2f}%")
    ax.title.set_fontsize(10)
    ax.legend(fontsize=8)
    save(fig, "s15_push")


def s16_dose_response():
    """Slide 16: share of minority prompts absorbed against the majority share (P1, task level)."""
    text = (ANALYSIS / "v3a" / "P1_TASK_LEVEL.md").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if re.match(r"\| (linear|mlp)/", line)]
    ms = [1 / 3, 0.5, 0.7, 0.9]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for line in rows:
        cells = [c.strip() for c in line.strip("|").split("|")]
        arch, rho, opt, update = cells[0].split("/")
        if update != "expected_full":
            continue
        rates = [float(v) for v in cells[3:7]]
        ax.plot(ms, rates, marker="o", color="#C44E52" if arch == "mlp" else "#4C72B0",
                ls="-" if opt == "sgd" else "--", alpha=0.85 if rho == "0.9" else 0.5,
                label=f"{arch}, ρ={rho}, {opt}")
    ax.set(xlabel="majority share m (1/3 = balanced)", ylabel="minority prompts absorbed",
           ylim=(-0.03, 1.03), title="P1: more imbalance, more collapse (16/16 cells)")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "s16_dose_response")


def s17_levers():
    """Slide 17: KL direction as a lever (T17). Minority absorption and majority success against β."""
    df = pd.read_csv(ANALYSIS / "v3g" / "t17_runs.csv")
    none = df[df.kl == "none"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for kl, color in (("forward", "#4C72B0"), ("reverse", "#C44E52")):
        g = df[df.kl == kl].groupby("beta").minority_absorbed.mean()
        axes[0].plot(g.index, g.values, "o-", color=color, label=f"{kl} KL")
    axes[0].axhline(none.minority_absorbed.mean(), color="#777", ls=":", label="no KL")
    fwd = df[df.kl == "forward"]
    for arch, ls in (("linear", "-"), ("mlp", "--")):
        g = fwd[fwd.arch == arch].groupby("beta").agg(maj=("majority_final", "median"),
                                                     mino=("minority_final", "median"))
        axes[1].plot(g.index, g.maj, "o", ls=ls, color="#4C72B0", label=f"{arch}: majority")
        axes[1].plot(g.index, g.mino, "s", ls=ls, color="#DD8452", label=f"{arch}: minority")
    axes[0].set(xscale="log", xlabel="β", ylabel="minority prompts absorbed", title="Forward KL protects at small β")
    axes[1].set(xscale="log", xlabel="β (forward KL)", ylabel="final success (median)",
                title="The cost: a ceiling that falls with β")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("T17: a KL anchor to the start policy (m = 0.9, ρ = 0.9)", y=1.02)
    save(fig, "s17_levers")


FIGURES = [s03_advantage_gap, s05_ladder, s08_h1_identity, s09_h2_sharing, s10_h4_fixed_points,
           s12_h3_thresholds, s15_push, s16_dose_response, s17_levers]


def main():
    words = sys.argv[1:]
    for make in FIGURES:
        if not words or any(w in make.__name__ for w in words):
            make()


if __name__ == "__main__":
    main()
