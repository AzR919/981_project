"""
C2 analysis (minority-procedure collapse), as pre-registered in PREREG_C2.md and its amendment 1.

usage:
    python llm/analyze_c2.py --problems llm/pools/c2/all.json --heldout llm/pools/c2/heldout_ids.json \\
        --base0 c2_base_n256_s0.json.gz --base1 c2_base_n256_s1.json.gz --base-hints <dir> \\
        --main <run dirs...> [--matched <run dirs...>] --out analysis/llm/c2.md

A run dir holds run/config.json, run/groups.jsonl, after_n256.json.gz and (main runs) hint_*.json.gz.
The role, majority share and seed are read from the run's config. Primary test, per role: exact
permutation test of the Spearman correlation between m and the decline rate on the 300 held-out problems
of the minority procedure (the same problems in every run of the role).
"""

import argparse
import gzip
import itertools
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.analyze_c0 import markdown  # noqa: E402
from llm.declines import compare  # noqa: E402

FDR = 0.1
OTHER = {"each": "total", "total": "each"}


def read(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def permutation_trend(x, y):
    """One-sided exact permutation p-value for a positive Spearman correlation between x and y: the share
    of distinct assignments of the x labels to the runs whose correlation is at least the observed one."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.ptp(y) == 0:
        return np.nan, 1.0
    observed = stats.spearmanr(x, y).statistic
    seen = set(itertools.permutations(x.tolist()))
    hits = sum(stats.spearmanr(np.array(p), y).statistic >= observed - 1e-12 for p in seen)
    return float(observed), hits / len(seen)


def describe(path):
    """(role, m, seed, arm) of a run from its config."""
    cfg = json.load(open(os.path.join(path, "run", "config.json")))
    if cfg.get("strata"):
        counts = dict(item.split("=") for item in cfg["strata"].split(":", 1)[1].split(","))
        major = max(counts, key=lambda k: int(counts[k]))
        n_major, n_minor = int(counts[major]), int(counts[OTHER[major]])
        return dict(role=major, m=n_major / (n_major + n_minor), seed=cfg["seed"], arm="matched", n_major=n_major)
    m, major = re.search(r"mix_m([0-9.]+)_(each|total)_s\d+_train", cfg["problems"]).groups()
    return dict(role=major, m=float(m), seed=cfg["seed"], arm="main", n_major=None)


def panel_table(base, after, ids):
    df = compare({i: base[i] for i in ids}, {i: after[i] for i in ids}, FDR)
    return df


def group_stats(path, problems):
    """Per procedure: prompts sampled and informative / all-wrong / all-correct groups, per step."""
    rows = {p: dict(prompts=0, informative=0, all_wrong=0, all_correct=0) for p in OTHER}
    steps = 0
    with open(os.path.join(path, "run", "groups.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            steps += 1
            for pid, rewards in zip(rec["ids"], rec["rewards"]):
                r = rows[problems[pid]["procedure"]]
                r["prompts"] += 1
                total = sum(rewards)
                r["all_wrong" if total == 0 else "all_correct" if total == len(rewards) else "informative"] += 1
    return {p: {k: v / max(steps, 1) for k, v in r.items()} for p, r in rows.items()}


def run_summary(path, problems, held, base, base_answers):
    info = describe(path)
    minority = OTHER[info["role"]]
    after_file = read(os.path.join(path, "after_n256.json.gz"))
    after, answers = after_file["counts"], after_file.get("answers", {})
    panel = [i for i in held if problems[i]["procedure"] == minority]
    major_panel = [i for i in held if problems[i]["procedure"] == info["role"]]
    df = panel_table(base, after, panel)
    dm = panel_table(base, after, major_panel)
    change = (df.p_after - df.p_before).to_numpy()
    rng = np.random.default_rng(0)
    boot = change[rng.integers(len(change), size=(2000, len(change)))].mean(axis=1)
    declined = df[df.decline]
    switch = [max(answers[i], key=answers[i].get) == problems[i]["other_answer"] for i in declined.id] if answers else []
    base_other = np.mean([base_answers[i].get(problems[i]["other_answer"], 0) / base[i][2] for i in panel]) if base_answers else np.nan
    after_other = np.mean([answers[i].get(problems[i]["other_answer"], 0) / after[i][2] for i in panel]) if answers else np.nan
    trained = [i for i in after if i not in set(held) and problems[i]["procedure"] == minority]
    dt = panel_table(base, after, trained) if trained else None
    row = dict(info, run=os.path.basename(os.path.dirname(path.rstrip("/"))), panel=len(df),
               declines=int(df.decline.sum()), decline_rate=float(df.decline.mean()), lost=int(df.lost.sum()),
               mean_change=float(change.mean()), change_lo=float(np.quantile(boot, 0.025)),
               change_hi=float(np.quantile(boot, 0.975)), switch_share=float(np.mean(switch)) if switch else np.nan,
               other_answer_base=float(base_other), other_answer_after=float(after_other),
               majority_decline_rate=float(dm.decline.mean()), majority_change=float((dm.p_after - dm.p_before).mean()),
               trained_minority_decline_rate=float(dt.decline.mean()) if dt is not None else np.nan)
    groups = group_stats(path, problems)
    for proc, label in ((minority, "minority"), (info["role"], "majority")):
        for k, v in groups[proc].items():
            row[f"{label}_{k}_per_step"] = v
    for h in ("neutral", "first_step", "wrong_step"):
        f = os.path.join(path, f"hint_{h}.json.gz")
        if os.path.exists(f):
            c = read(f)["counts"]
            row[f"hint_{h}"] = float(np.mean([c[i][0] / c[i][2] for i in panel]))
    return row


def trend_block(runs, x, title):
    lines, ps = [f"### {title}", ""], {}
    for role, g in runs.groupby("role"):
        total = int(g.declines.sum())
        rho, p = permutation_trend(g[x], g.decline_rate)
        ps[role] = p if total >= 10 else np.nan
        verdict = "untestable (fewer than 10 declines)" if total < 10 else f"Spearman {rho:.2f}, exact one-sided p = {p:.4f}"
        lines.append(f"- majority `{role}` (minority `{OTHER[role]}`), {len(g)} runs, {total} declines on the panel: {verdict}.")
    return lines + [""], ps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--base0", required=True)
    parser.add_argument("--base1", required=True)
    parser.add_argument("--base-hints", default=None)
    parser.add_argument("--main", nargs="+", required=True)
    parser.add_argument("--matched", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    problems = {q["id"]: q for q in json.load(open(args.problems))}
    held = json.load(open(args.heldout))
    b0, b1 = read(args.base0), read(args.base1)
    base, base_answers = b0["counts"], b0.get("answers", {})
    lines = ["# C2: minority-procedure collapse", "",
             "Generated by `llm/analyze_c2.py`; tests as pre-registered in `PREREG_C2.md` and its amendment 1.", ""]
    null = []
    for proc in OTHER:
        panel = [i for i in held if problems[i]["procedure"] == proc]
        df = panel_table(base, b1["counts"], panel)
        null.append(dict(panel_procedure=proc, problems=len(df), declines=int(df.decline.sum()), lost=int(df.lost.sum()),
                         base_p=float(df.p_before.mean()),
                         other_answer_share=float(np.mean([base_answers[i].get(problems[i]["other_answer"], 0) / base[i][2]
                                                           for i in panel])) if base_answers else np.nan))
    lines += ["## Null and base rates (base seed 0 against seed 1, held-out panels)", "", markdown(pd.DataFrame(null)), ""]

    main_runs = pd.DataFrame([run_summary(p, problems, held, base, base_answers) for p in args.main])
    main_runs = main_runs.sort_values(["role", "m", "seed"])
    show = ["role", "m", "seed", "run", "declines", "decline_rate", "lost", "mean_change", "change_lo", "change_hi",
            "switch_share", "other_answer_after", "majority_decline_rate", "trained_minority_decline_rate"]
    lines += ["## Main runs: the 300 held-out minority problems", "", markdown(main_runs[show]), ""]
    block, ps = trend_block(main_runs, "m", "P5: does the minority decline rate rise with m?")
    lines += block
    tested = [p for p in ps.values() if p == p]
    if len(tested) == 2:
        lines += [f"**P5: {'PASS' if max(tested) < 0.05 else 'FAIL'}** (larger p = {max(tested):.4f}; needs p < 0.05 in both roles).", ""]
    else:
        lines += [f"**P5: untestable in {2 - len(tested)} role(s)**: too few declines to test a trend.", ""]

    sig = ["role", "m", "seed", "minority_prompts_per_step", "minority_informative_per_step", "minority_all_wrong_per_step",
           "minority_all_correct_per_step", "majority_informative_per_step"]
    lines += ["## Rival 1: signal share (per step, from the group log)", "", markdown(main_runs[sig]), ""]
    hint_cols = [c for c in ("hint_neutral", "hint_first_step", "hint_wrong_step") if c in main_runs]
    if hint_cols:
        hints = main_runs[["role", "m", "seed"] + hint_cols].copy()
        lines += ["## Rival 3: access against ability (minority panel success with hints, N = 64)", ""]
        if args.base_hints:
            rows = []
            for proc in OTHER:
                panel = [i for i in held if problems[i]["procedure"] == proc]
                row = {"minority procedure": proc, "no hint": float(np.mean([base[i][0] / base[i][2] for i in panel]))}
                for h in ("neutral", "first_step", "wrong_step"):
                    c = read(os.path.join(args.base_hints, f"base_hint_{h}.json.gz"))["counts"]
                    row[h] = float(np.mean([c[i][0] / c[i][2] for i in panel]))
                rows.append(row)
            lines += ["Base model:", "", markdown(pd.DataFrame(rows)), "", "Trained runs:", ""]
        lines += [markdown(hints), ""]
    if args.matched:
        matched = pd.DataFrame([run_summary(p, problems, held, base, base_answers) for p in args.matched])
        matched = matched.sort_values(["role", "n_major", "seed"])
        lines += ["## Matched-minority arm (4 minority prompts per step in every run)", "",
                  markdown(matched[["role", "n_major", "seed", "declines", "decline_rate", "mean_change", "change_lo",
                                    "change_hi", "minority_informative_per_step"]]), ""]
        block, _ = trend_block(matched, "n_major", "Does the minority decline rate rise with the number of majority prompts?")
        lines += block
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    main_runs.to_csv(args.out.replace(".md", "_runs.csv"), index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
