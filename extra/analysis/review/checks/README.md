# Review checks

Scripts behind the claims in `../SUMMARY.md` and `../plan_critique.md`. Run from `981_project/`
after `experiments/analyze.py collect` has built `results/base/cache` and `results/stretch/cache`.

| Script | What it checks | Origin |
|---|---|---|
| `verify_summary.py` | Every main-session check in SUMMARY.md: minority-answer collapse (c09, c05), fixed flow time, kick size against the threshold ratio, H5 ties, the H2 arcsin share, and the plan-critique probabilities | main session |
| `flush_compare.py` | One training run with and without `torch.set_flush_denormal`. Usage: `flush_compare.py <repo> <0/1> '<config json>' <out.npy>` | reviewer D |
| `absorbed_set_overlap.py` | Jaccard overlap of the prompts absorbed under `sampled` and under `expected_full` | reviewer B / D |
| `forward_kl_binary_check.py` | Forward-KL closed form projected onto a binary reward (GX-Chen) against finite-G GRPO | planning |
| `wg_explore.py`, `wg_explore_sharing.py` | Early exploration of the GRPO W_G and of a two-prompt sharing example | planning |
