"""
Job list for the replication of Li, Pandey and Chen, "Wrong the Same Way" (NeurIPS 2026 workshop; code
github.com/Rohan-Pandey1729/RLVR-Reward-Hacking, MIT, pinned commit 05ef005).

usage:  python tools/literature/replication/wsw_jobs.py > jobs_wsw.txt

One line per job: the command to run from their src/ folder, in a fresh copy of their code with an empty
results/ and cache/. Their scripts take --shard/--nshards and split the work into disjoint shards; figures.py
collects result files by pattern, so the number of shards does not matter. All runs are single-threaded
(their default), as their README recommends for reproducibility. The experiments and arguments are the ones
in their README ("Reproducing").
"""

SHARDED = [
    ("exp2_universality.py --opt adam", 35),   # E1, headline: 7 universality levels x 5 seeds (base models)
    ("exp2_universality.py --opt sgd", 35),    # E8
    ("exp6_blockperm.py", 20),                 # E2
    ("exp3_factorial.py", 20),                 # E3
    ("exp5_practice.py --part a", 20),         # E4 sampled GRPO
    ("exp5_practice.py --part b", 20),         # E5 KL sweep
    ("exp4_scaling.py", 20),                   # E7
    ("exp9_maskonly.py", 20),                  # E9
    ("exp9_maskonly.py --mode constant_mass", 20),   # E9b
    ("exp10_copies.py", 40),                   # E10, per-prompt copies (the slowest)
]
SINGLE = [
    "exp5_practice.py --part c",               # E6 verifier hardening
    "exp12_threshold.py --seed 0",             # E11
    "exp12_threshold.py --seed 1",
    "exp12_threshold.py --seed 2",
    "theory.py",                               # propositions and mechanism probes
    "exp11_kernel.py",
]


def main():
    lines = []
    # The slowest jobs first, so they do not finish last.
    for script, n in sorted(SHARDED, key=lambda s: "exp10" not in s[0]):
        lines += [f"python {script} --shard {i} --nshards {n}" for i in range(n)]
    lines += [f"python {s}" for s in SINGLE]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
