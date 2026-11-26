# Experiment index

This file lists every experiment the project ran, from the first pilots to the latest batches. For each one it
says what it was, why we ran it, how big it was, whether it was pre-registered, which code ran it, where its
results are, and what it found. It is an index, not a write-up: the numbers come from the batch summaries, and
each row points to the file that has the details. If a row and its summary disagree, the summary wins.

**Per-experiment write-ups:** `docs/Ankush.md` describes every experiment here in the format of `docs/Simon.md`
(why it was run, what was done step by step, the fixed settings, and the result).

## Promised vs extra

Every experiment carries one of three labels, so the promised work can be read apart from the rest. The same label is
on each entry in `docs/Ankush.md` (search for `Scope: Promised`, `Scope: Core follow-up` or `Scope: Extra`) and on
each section of `analysis/FINAL_RESULTS.md`.

| Label | Meaning | Count |
|---|---|---|
| **Promised** | In the milestone: H1–H5, their planned stretch goals, or a test the milestone named (such as Heckel's weights, T9) | 21 |
| **Core follow-up** | Not in the milestone, but grew out of a promised result (H3's failure led to minority collapse), and the report's toy story relies on it | 31 |
| **Extra** | Beyond what was promised: the transformer environment, the LLMs, formal verification, Adam side-questions; not needed to follow the report | 30 |

- **Promised:** Guide §7 pilots, p₀ pilot, H1 gradient-norm check, H1, H2, H3, H4, H5, S_H1, S_DRIFT, S_PRED, S_ABL, S_EQ, A1, BAY, A2, A3, A6, T4, T8, T7, T9, T4 edge.
- **Core follow-up:** T1 / P1, T2 / P2, T6, T5 / P3, T1 seed 1, T2 groups, T6 KL, T10, T11 (v3d), T12 (v3d), T13 (v3d), M2 (v3d addendum), M4 (v3d addendum), T15 (v3e), T16 (v3f), T17 (v3g), T18 (v3g), T19 (v3g), T22 (v3j), v3j2, T23 (v3k), T24 (v3l), Odds law (theory check), Late-time probe (theory check), Fall model (theory check), T25 (v3m), A1 (v3o), A2 (v3o), A3 (v3o), A5 (v3o), G (v3p, T27).
- **Extra:** A4, ADAM, T20 (v3h), T21 (v3i), Lean proofs (theory check), SymPy checks (theory check), Optimizer geometry (theory check), wsw, wsw2, T26 (v3n), H (v3p, T28), A4 (v3q), J (v3q, T29), C0, Model probes, Census calibration, Llama calibration, C1 Qwen / P4, C1 Llama, C1 control, C5, C2 / P5, C2 matched minority, C2 seed replication of the degraded run, C2 extras, SmolLM2 diagnostic, C10, C9, C7 / C8, C3 and C11.

**To review only what was promised,** read the Promised entries in `docs/Ankush.md`, section 1 of
`analysis/FINAL_RESULTS.md`, and the base and stretch rows below. The report's main body is Promised plus Core
follow-up; Extra appears at most as one short section (LLMs) or not at all.

Written 2026-11-25 from the summaries, `HANDOFF.md`, `experiments/grids.py`, the LLM queue files and the
branches named below. Paths are relative to the repo root on `main`.

## How to get the raw results

The raw results are not in git. Each batch is packed as one archive, `<batch>_raw.tar.gz`, with a
`<batch>_MANIFEST.txt` next to it (file count, result-file count, hosts, launch commit and job digest, archive
size, SHA-256).

- **Google Drive (all raw results, one zip):** [CMPT981_raw_results_2026-11-25.zip](https://drive.google.com/file/d/1_CJhksy4fW8L8Y7cFIobi8Q5NaFyZX9p/view?usp=sharing)
  (1.75 GB). Access is by request: open the link and click "Request access"; Ankush approves.
  - It holds the folder `rlvr-share/`: all 23 per-batch archives (`<batch>_raw.tar.gz`, LLM runs included,
    model checkpoints excluded), one `<batch>_MANIFEST.txt` each, and a `README.txt` explaining the layout.
  - Check the download: `shasum -a 256 CMPT981_raw_results_2026-11-25.zip` should print
    `56f8023713002c67c3e385d247bf43a41c8369ca6fbf3b764b2c10dbff40c01e`. Each inner archive's SHA-256 is in its
    manifest.
  - Unpack: `unzip CMPT981_raw_results_2026-11-25.zip`, then `tar -xzf rlvr-share/<batch>_raw.tar.gz`.
- **Ankush's laptop:** `~/rlvr-data/<batch>/`, and `~/rlvr-data/lab_home/<batch>/` (a copy of the lab results
  for base, stretch, v3a–v3f and llm).
- **SFU lab network home** (user `asa715`): `~/rlvr_results/<batch>/<host>/`, one folder per lab machine
  (for example `secb1010u-a01`). The LLM runs are in `~/rlvr_results/llm/<tag>/<host>/`.

Unpack with `tar -xzf <batch>_raw.tar.gz`. You get `<batch>/<host>/...`. Toy runs are one `.json.gz` per run
(config, history, per-prompt snapshots); each host folder also has a `DONE` file with the launch commit and,
from v3d on, the job digest.

### Raw data per batch

| Batch | Result files | Archive (size) | Laptop | Lab |
|---|---|---|---|---|
| base | 19,932 | `base_raw.tar.gz` (621 MB) | `~/rlvr-data/lab_home/base/` | `~/rlvr_results/base/<host>/` |
| stretch | 4,706 | `stretch_raw.tar.gz` (275 MB) | `~/rlvr-data/lab_home/stretch/` | `~/rlvr_results/stretch/<host>/` |
| v3a | 17,279 | `v3a_raw.tar.gz` (198 MB) | `~/rlvr-data/lab_home/v3a/` | `~/rlvr_results/v3a/<host>/` |
| v3b | 5,280 | `v3b_raw.tar.gz` (68 MB) | `~/rlvr-data/lab_home/v3b/` | `~/rlvr_results/v3b/<host>/` |
| v3c | 1,920 | `v3c_raw.tar.gz` (30 MB) | `~/rlvr-data/lab_home/v3c/` | `~/rlvr_results/v3c/<host>/` |
| v3d | 1,760 | `v3d_raw.tar.gz` (32 MB) | `~/rlvr-data/v3d/` | `~/rlvr_results/v3d/<host>/` |
| v3e | 1,080 | `v3e_raw.tar.gz` (17 MB) | `~/rlvr-data/v3e/` | `~/rlvr_results/v3e/<host>/` |
| v3f | 480 | `v3f_raw.tar.gz` (15 MB) | `~/rlvr-data/v3f/` | `~/rlvr_results/v3f/<host>/` |
| v3g | 5,540 | `v3g_raw.tar.gz` (116 MB) | `~/rlvr-data/v3g/` | `~/rlvr_results/v3g/<host>/` |
| v3h | 1,809 | `v3h_raw.tar.gz` (7.8 MB) | `~/rlvr-data/v3h/` | `~/rlvr_results/v3h/<host>/` |
| v3i | 560 | `v3i_raw.tar.gz` (2.9 MB) | `~/rlvr-data/v3i/` | `~/rlvr_results/v3i/<host>/` |
| v3j | 1,280 | `v3j_raw.tar.gz` (18 MB) | `~/rlvr-data/v3j/` (includes the `*_rerun` host folders) | `~/rlvr_results/v3j/<host>/` |
| v3j2 | 106 | `v3j2_raw.tar.gz` (1.7 MB) | `~/rlvr-data/v3j2/` | `~/rlvr_results/v3j2/<host>/` |
| v3k | 320 | `v3k_raw.tar.gz` (7.9 MB) | `~/rlvr-data/v3k/` | `~/rlvr_results/v3k/<host>/` |
| v3l | 120 | `v3l_raw.tar.gz` (15 MB) | `~/rlvr-data/v3l/` | `~/rlvr_results/v3l/<host>/` |
| v3m | 98 `.npz` (one per task) | `v3m_raw.tar.gz` (9.8 MB) | `~/rlvr-data/v3m/` | none (computed on the laptop) |
| v3n | 320 | `v3n_raw.tar.gz` (7.0 MB) | `~/rlvr-data/v3n/` | `~/rlvr_results/v3n/<host>/` |
| v3o | 10,440 | `v3o_raw.tar.gz` (145 MB) | `~/rlvr-data/v3o/` | `~/rlvr_results/v3o/<host>/` |
| v3p | 820 | `v3p_raw.tar.gz` (15 MB) | `~/rlvr-data/v3p/` | `~/rlvr_results/v3p/<host>/` |
| v3q | 680 | `v3q_raw.tar.gz` (3.2 MB) | `~/rlvr-data/v3q/` | `~/rlvr_results/v3q/<host>/` |
| wsw | 256 jobs (their own `jsonl` format) | `wsw_raw.tar.gz` (14 MB) | `~/rlvr-data/wsw/` | `~/rlvr_results/wsw/<host>/j<index>/results/` |
| wsw2 | 70 jobs | `wsw2_raw.tar.gz` (2.0 MB) | `~/rlvr-data/wsw2/` | `~/rlvr_results/wsw2/<host>/` |
| llm | 358 `.json.gz` evaluations, plus logs | `llm_raw.tar.gz` (50 MB; model checkpoints left out) | `~/rlvr-data/lab_home/llm/<tag>/<host>/`, base evaluations in `~/rlvr-data/llm/` | `~/rlvr_results/llm/<tag>/<host>/` |

## How results are organised in the repo

- `analysis/<batch>/FINDINGS.md` is **generated** by a script (`experiments/analyze*.py`,
  `experiments/check_*.py` or `llm/analyze_*.py`). Every number in it is recomputed from the raw results.
- `analysis/<batch>/SUMMARY.md` is **written by hand**: the verdicts,
  the reading, deviations and later corrections. Start here.
- `runs.csv` and `cells.csv` (or `tables/*.csv`) hold one row per run or per cell, written by the same script.
- Pre-registrations are in `analysis/review/PREREG_*.md` (branch `review`, merged into `main`). Each was
  committed before its batch launched.
- Base and stretch have no SUMMARY; their FINDINGS has a "Verdicts at a glance" table, and the hand-written
  verdicts are in `analysis/review/SUMMARY.md`.
- **Commit hashes.** The hashes quoted below are the ones the summaries record. On 2026-11-22 the history of
  the work branches was rewritten to clean commit messages, so hashes cited from before that date (base to v3c,
  C0, C1, C2 pre-registrations) are the old ones. The old commits are still in the backup tags, and the
  old-to-new map is in PR #8 (`HANDOFF.md`, section 3h).

## Timeline

| Phase | When | What | Batches or tags |
|---|---|---|---|
| 1. Pilots | before 2026-11-16 | Small single-seed runs for the proposal and the project guide; Simon's H1 check | guide §7, `run_pilot.py`, `experiments/h1.py` |
| 2. Milestone tests H1–H5, and stretch goals | 11-20 | The five milestone hypotheses on the synthetic task, then the stretch tests | base, stretch |
| 3. Four-way review | 11-20 evening | Four AI reviews of milestone, stretch, literature and flaws; claims re-checked | `analysis/review/` |
| 4. Plan v3: pre-registered P1–P3 and fixes | 11-20 to 11-21 | Minority-class collapse (P1), kick size (P2), predictor (P3), H5 rerun, ablations, pass@k; Phase A re-analyses | v3a, v3b, v3c, Phase A |
| 5. Mechanism, theory, levers, transformer env | 11-22 to 11-23 | Who pushes whom, threshold in class share, prevention levers (pass@k, KL, group size), Li et al.'s transformer env, junk answers, warm start, late phase; odds law and Lean/SymPy checks | v3d–v3l, theory |
| 6. Small LLMs | 11-20 to 11-25 | C0 pilot, model probes, census calibration, C1 census (Qwen, Llama), C5 loss drivers, C2 minority procedure, C10 branch test, C9, C7/C8 KL | `llm` tags `c0*`, `c1q`, `q_*`, `r2_*` |
| 7. Latest: permanence, shared skills, replication, review caveats | 11-23 to 11-24 | Replication of Li, Pandey and Chen; exact long-time flow (T25); shared skills (T26); confirmatory battery A1–A5; caveats G, H, J; theory of the fall | wsw, wsw2, v3m, v3n, v3o, v3p, v3q |

The plan for phases 4–6 is `analysis/review/NEXT_PLAN.md`. The day-by-day record is `HANDOFF.md` in the parent
folder (not in git).

Column key for the tables below: **Runs** is the number of jobs in the grid (from `experiments/grids.py`) or
the number of runs a summary reports. **Code** gives the grid name in `experiments/grids.py` and the branch and
launch commit (from the `DONE` files or the summary).

---

## 1. Pilots (before the milestone)

| ID | What it was | Why we ran it | Runs | Pre-reg | Code | Results | Outcome |
|---|---|---|---|---|---|---|---|
| Guide §7 pilots (Azaan) | Single-seed runs, 300 steps, K = 3, 64-wide MLP, plain SGD, β = 0: (7.1) final success against η for L = 1, 2, 4; (7.2) per-prompt end state under a fixed budget of 256 samples; (7.3) exact per-sample signal-to-noise by rule and G | To check that the framework (advantage gap W_G, effective step) points at something real before writing the hypotheses | about 20 configurations (one seed each) | no | Scripts not in the repo (open item in `TODO.md`: "Move the pilot scripts into the repo") | `docs/project_guide.html` §7; no raw data kept | A step-size threshold exists; failure is **absorption** (bimodal end states, prompts stuck at p ≈ 0), not uniform degradation; GRPO fails earlier than mean-only. This motivated H3 and H5 |
| p₀ pilot (Simon) | p₀ ∈ {0.01, 0.1, 0.5} × G ∈ {2, 8, 32} × 3 seeds with the NumPy bandit `main.py` (K = 8, GRPO, β = 0, lr 0.1, 2,000 steps) | First look at group size and starting success (later H5) | 27 | no | `run_pilot.py` (`main`, commit `5f4cf89`) | no write-up found; see `run_pilot.py` | see `run_pilot.py` |
| H1 gradient-norm check (Simon) | One prompt, K = 4, L = 1, G = 4, 512 groups per step, SGD lr 0.4, 250 steps, 10 seeds; sampled ‖ĝ‖² against ‖W_G ∇p‖², for mean-only, RLOO and GRPO; tabular and MLP policies | A direct visual check of the H1 identity | 3 rules × 10 seeds per policy | no | `experiments/h1.py` (`main`; `b1c822e`, MLP option `0130bca`); setup in `docs/Simon.md` | Plots go to `results/h1_*_plots/` (ignored by git; regenerate with the commands at the top of `experiments/h1.py`) | Sampled norm tracks the predicted norm over the whole run for all three rules. Norms only, not directions; one cell of the grid (`TODO.md`, side note) |

## 2. Milestone hypotheses H1–H5 and stretch goals (2026-11-20)

The synthetic task: prompts with features, K answers, a linear or MLP softmax policy, binary reward, GRPO and
other group rules. Levels form a sharing ladder: `tab` (one-hot, nothing shared), `lin` (Gaussian features),
`c05`/`c09` (correlated features, ρ = 0.5/0.9), `m64`/`m256` (MLPs).

### base (19,932 runs, 0 failed; code `experiments` `fd8d511`)

Hypotheses from the milestone (`docs/milestone.pdf`); FINDINGS marks the H3, H4 and H5 verdict rules as
pre-registered. Results: `analysis/base/FINDINGS.md`, `analysis/base/sections/`, `analysis/base/tables/`.
Raw: `base_raw.tar.gz`.

| ID | What it was | Why | Runs | Pre-reg | Code | Results | Outcome |
|---|---|---|---|---|---|---|---|
| H1 | Gradcheck: average R = 4,000 sampled updates at fixed parameters against −(1/B) Σ W_G(p_x)∇p_x from autograd | The advantage-gap identity E[ĝ] = W_G(p)∇p is the foundation of every other test | 414 | milestone | grid `h1` | `analysis/base/sections/h1.md`, `analysis/base/tables/h1_*.csv` | PASS on group medians (cosine ≥ 0.989); the literal per-run criterion fails in 174 of 414 runs (miscalibrated band) |
| H2 | One-step and short-training runs: does the measured change follow W_G, with and without sharing? | Group size changes what GRPO optimises; does sharing distort the surrogate? | 414 (207 + 207) | milestone | grid `h2` | `analysis/base/sections/h2.md`, `analysis/base/tables/h2_*.csv` | PASS. Tabular on the line (R² ≥ 0.95); with sharing 93–97% of each prompt's drift comes from other prompts |
| H3 | η × B × G × rule × level grid: absorbed fraction and final success | Main study: is stability set by one effective step s̄ = (η/B)·W_G·p(1−p), and only with sharing? | 13,608 | milestone | grid `h3` | `analysis/base/sections/h3.md`, `analysis/base/tables/h3_*.csv` | Tabular runs never lose a prompt (PASS). **One effective step does not collapse the levels (FAIL;** off by up to 4.69 decades) |
| H4 | KL fixed points (a), fluctuations against η/β (b), KL and the absorption threshold (c) | Where training settles with a KL penalty (Mroueh; Vojnovic & Yun) | 1,968 | milestone | grid `h4` | `analysis/base/sections/h4.md`, `analysis/base/tables/h4_*.csv` | (a) PASS: fixed points match to median error 1e-4. (b) FAIL: slope against η/β |
| H5 | Group size at a fixed budget BG and escape from low p₀ | Separate group-size study, as the instructor asked | 3,528 | milestone | grid `h5` | `analysis/base/sections/h5.md`, `analysis/base/tables/h5_*.csv` | PASS as computed, but the best-G table was a tie-break artefact (review); with the smallest tied G the best G still falls with p₀ (32, 16, 4, 2) |

### stretch (4,706 runs, 0 failed; code `fa7f182`; criteria committed `ab9c270` before the results)

Results: `analysis/stretch/FINDINGS.md`, `analysis/stretch/sections/`, `analysis/stretch/tables/`. Raw: `stretch_raw.tar.gz`.

| ID | What it was | Why | Runs | Pre-reg | Code | Results | Outcome |
|---|---|---|---|---|---|---|---|
| S_H1 | H1 again at R = 20,000, with a calibrated per-run null and the projected bias | The base H1 band was miscalibrated | 414 | criteria in `analyze_s_*` (`experiments/analyze.py`) | grid `s_h1` | `analysis/stretch/sections/s_h1.md` | Loose PASS; medium FAIL (χ² p = 0.004); tight FAIL only for L = 4 uniform runs. No bias detected at any tier with the precision to see it |
| S_DRIFT | Sampled, expected and full expected updates; a B block and an Adam block | Is absorption driven by drift or by noise? | 2,340 | same | grid `s_drift` | `analysis/stretch/FINDINGS.md` "H3 stretch: interference drift" | c05/c09 FAIL, m64 PASS. The "batch noise" reading was later withdrawn by the review (per-kick size, not noise) |
| S_PRED | Per-prompt absorption predictor (step-0 drift) against p and ‖∇p‖ baselines | H3 stretch: predict which prompts are absorbed | 800 | same | grid `s_pred` | `analysis/stretch/FINDINGS.md` "per-prompt absorption predictor" | PASS on c05/c09, FAIL on the MLPs. The review found drift no better than an answer-rarity baseline |
| S_ABL | Break one condition of the tabular "p never decreases" proposition at a time | Which condition matters? | 558 | same | grid `s_abl` | `analysis/stretch/sections/s_abl.md` | PASS: the control never lowers p; noisy verifier and the sign rule cause absorption |
| S_EQ | Start on both sides of the unstable KL root at G ≥ 16 | H4 stretch: two stable KL fixed points | 594 | same | grid `s_eq` | `analysis/stretch/FINDINGS.md` "two stable KL fixed points" | Bistability confirmed (basin boundary at the predicted root, end states within 0.01); the "crossings increase with η" criterion FAILs as coded |

## 3. Four-way review (2026-11-20 evening)

Not an experiment, but it decided everything after it. Four AI reviewers read the milestone and the results;
the main session re-checked each claim against the code and caches.

| ID | What it was | Results | Outcome |
|---|---|---|---|
| Review a–d | (a) milestone against delivery, (b) stretch goals, (c) literature and gaps, (d) adversarial flaw hunt | `analysis/review/SUMMARY.md`, `a.md`, `b.md`, `c.md`, `d.md`, `plan_critique.md`; checks in `analysis/review/checks/` (`verify_summary.py` reproduces them) | c09/c05 "interference drift" is **minority-answer collapse**; "batch-composition noise" withdrawn (per-kick size); H5 best-G table is a tie-break artefact; H3 refuted as stated. Led to plan v3 (`analysis/review/NEXT_PLAN.md`) with five pre-registered hypotheses P1–P5 |
| Later reviews | `ADVERSARIAL_v4.md`, `CODEX_RECHECK.md`, `ADVERSARIAL_v5.md` (2026-11-23) and `ADVERSARIAL_v5_reviewer.md` | `analysis/review/` | v5's corrections are appended to each batch SUMMARY as "Corrections after the review of 2026-11-23" (for example P1's task-level p-value, forward-KL thresholds) |

## 4. Plan v3: v3a, v3b, v3c and Phase A (2026-11-20 to 11-21)

### Phase A: re-analyses of existing data (no new runs; commit `a762c7b`)

| ID | What it was | Why | Data used | Pre-reg | Code | Results | Outcome |
|---|---|---|---|---|---|---|---|
| A1 | Fit GRPO's weight between the mean rule (λ = 0) and the 2 arcsin √p surrogate (λ = 1), by G | The milestone promised the arcsin shift (H2) | base H2 tabular runs | plan only | `experiments/analyze_phase_a.py` | `analysis/phase_a/A1.md` | λ rises with G: 0, 0.015, 0.215, 0.627 at G = 2, 4, 16, 64 (exact λ agrees) |
| A4 | Lost problems and coverage: exact pass@1 against pass@64 | Coverage narrowing (Yue et al.) from existing data | stretch `s_pred` | plan only | same | `analysis/phase_a/A4.md` | In every cell pass@1 rises and pass@64 falls; GRPO lowers pass@64 more than the mean rule in 10 of 10 pairs |
| BAY | Bay & Yearick's group-size law against exact theory, learned policies and our speed results | Check a published law with our exact W_G | base H2 bins, T4 + T4e | no | same | `analysis/phase_a/BAY.md` | Our identity gives their fidelity φ exactly; their budget G* is not the fastest G per sample |
| ADAM | Absorption time against η under Adam | Is Adam absorption drift-driven (slope −1)? | stretch `s_drift` | no | same | `analysis/phase_a/ADAM.md` | Slope about −1 for c05 and c09 under Adam |
| A2, A3, A6 | Re-score ablations, H4 slopes, Barakat write-up | Planned in `NEXT_PLAN.md` | – | – | – | – | No write-up in `analysis/phase_a/`; see `NEXT_PLAN.md` and `HANDOFF.md` ("A3 needs one sentence in the report") |

### v3a (17,279 runs, 0 failed; code `toy-v3` `108055d`; pre-reg `analysis/review/PREREG_v3a.md`)

Results: `analysis/v3a/SUMMARY.md`, `FINDINGS.md` (`experiments/analyze_v3.py all`), `P1_TASK_LEVEL.md`,
`tables/`. Raw: `v3a_raw.tar.gz`.

| ID | What it was | Why | Runs | Pre-reg | Code | Outcome |
|---|---|---|---|---|---|---|
| T1 / **P1** | Absorption against majority share m at fixed feature geometry; SGD and Adam; supervised control | Review: c09/c05 absorption was minority collapse. Is class imbalance the cause? | 4,640 | confirmatory (P1) | grid `v3_t1` | **PASS** in 16/16 cells; whole minority set absorbed, at small η and under the deterministic flow; supervised control absorbs nothing. Task-level p = 0.0098 (Holm over P1–P5: 0.049) |
| T2 / **P2** | Per-kick size η/B against response noise at equal mean drift | Review: "batch noise" was really per-kick size | 720 | confirmatory (P2) | grid `v3_t2` | **FAIL**: kick size matters (3 of 4 levels), but response noise adds +0.11 absorption at B = 8 |
| T4 | H5 rerun: wider η grid, exact step to mean p = 0.5, 5 seeds | Review: H5 table was a tie-break artefact | 9,240 | exploratory | grid `v3_t4` | Tabular best G = 2 at every p₀, often at the η-grid edge |
| T6 | Ablations of the tabular proposition under realistic training (momentum, clipping, Adam, KL, PPO epochs) | Which realistic conditions cause drops? | 2,580 | exploratory | grid `v3_t6` | No drops with PPO clipping or batch reuse; drops with a noisy verifier and the sign rule |
| T8 | KL bistability with the exact expected update (no noise) | Does the deterministic update switch basins? | 99 | exploratory | grid `v3_t8` | PASS 60/60 (99/99 on the side they start); switching needs sampling noise |

### v3b (5,280 runs, 0 failed; code `toy-v3` `e61626f`; pre-reg `analysis/review/PREREG_v3b.md`)

Results: `analysis/v3b/SUMMARY.md`, `FINDINGS.md` (`experiments/analyze_v3b.py all`), `tables/`.
Raw: `v3b_raw.tar.gz`.

| ID | What it was | Why | Runs | Pre-reg | Code | Outcome |
|---|---|---|---|---|---|---|
| T5 / **P3** | Does initial drift predict absorbed prompts on balanced tasks beyond rarity and p_init? | Review: the predictor was no better than rarity | 200 | confirmatory (P3) | grid `v3_t5` | **FAIL**: ΔAUC 0.009 (bar 0.03). Post hoc, standardised within run: 0.079 |
| T7 | Bootstrap null for mean z² (H1 medium tier) | H1 medium tier failed in stretch | 240 | exploratory | grid `v3_t7` | PASS: null calibrated (KS p = 0.73) |
| T9 | All-wrong-group fixes (Heckel weights, virtual correct sample, ±α) on T1, T2 and T6 configurations | Milestone promise; the only causal test of the absorption mechanism | 480 | exploratory | grid `v3_t9` | Prediction too strong: Heckel cuts minority collapse 0.10 → 0.06 at η = 0.1, not at η = 1, and doubles absorption with big steps |
| T1 seed 1 | T1 MLP cells with a second initialisation seed | Follow-up (a) to P1 | 2,320 | exploratory | grid `v3_t1s1` | P1 replicates in 8/8 MLP cells |
| T2 groups | T2 at B = 8 with G ∈ {4, 16, 64} | Follow-up (b): does the B = 8 noise excess shrink with G? | 480 | exploratory | grid `v3_t2g` | Yes (Spearman −0.33, p = 5e-10) |
| T6 KL | T6 KL conditions with the decline above p_ref tracked | Follow-up (c): T6's KL prediction was untestable | 720 | exploratory | grid `v3_t6kl` | KL runs drop below p_ref only at η ≥ 30 |
| T4 edge | T4 with η up to 10,000 | Follow-up (d): T4 optima were at the grid edge | 840 | exploratory | grid `v3_t4e` | Best G stays 2; most cells now have an interior optimum |

### v3c (1,920 runs, 0 failed; code `toy-v3` `7759008`; pre-reg `analysis/review/PREREG_v3c.md`, exploratory)

Results: `analysis/v3c/SUMMARY.md`, `FINDINGS.md` (`experiments/analyze_v3c.py`), `runs.csv`. Raw: `v3c_raw.tar.gz`.

| ID | What it was | Why | Runs | Pre-reg | Code | Outcome |
|---|---|---|---|---|---|---|
| T10 | pass@k training rules (biased GRPO_k, unbiased RLOO_k) against GRPO and mean, on minority, shared and tabular tasks | Rules that weight hard prompts by about k(1−p)^(k−1) should resist minority collapse | 1,920 | exploratory | grid `v3_t10` | GRPO absorbs the whole minority; both pass@k rules absorb 0–5%. "No speed cost at k = 2" was later revised by v3f |

## 5. Mechanism, theory, levers and the transformer environment (2026-11-22 to 11-23)

All toy batches below: launched from a clean commit (`dirty=0`), job digest in the `DONE` files, 0 failed.
Pre-registrations are `analysis/review/PREREG_v3d*.md` and `PREREG_T20*.md`; all are exploratory.

| ID | What it was | Why | Runs | Pre-reg | Code | Results | Outcome |
|---|---|---|---|---|---|---|---|
| **v3d** T11 | Four-branch test: majority-only, minority-only and mixed steps from the same state | Is collapse active reversal or starvation? | 480 | `PREREG_v3d.md` | `v3_t11`; `toy-v3` `7a49348` | `analysis/v3d/SUMMARY.md`, `FINDINGS.md`, `tables/`; `v3d_raw.tar.gz` | All 5 predictions hold: the majority's step lowers the minority, 3.6–4.8× its own gain at m = 0.7, 10.7–17.1× at m = 0.9 |
| v3d T12 | T1 minority tasks at 10× the horizon; linear Adam at the smallest step | Plateau or permanent? | 800 | same | `v3_t12` | same | Prediction 1 fails: 8.3% recover, only at ρ = 0.5. At ρ = 0.9 none recover under SGD in 10× the horizon |
| v3d T13 | Project out the majority step's component against the minority's success gradient | Does removing the harmful direction prevent collapse? | 480 | same | `v3_t13` | same | Projection removes collapse in 16/16 cells; shortening the majority step also helps (prediction 2 fails) |
| v3d M2 | Minority margin growth over the last decade of T12 | Addendum: is the minority still sinking at 10×? | (T12 runs) | descriptive | analysis only | same | g ≈ 1 at ρ = 0.9 (still sinking); recovering at ρ = 0.5 |
| v3d M4 | Absorption by class and by margin tercile | Addendum: imbalance or margin? | (T13 runs) | descriptive | analysis only | same | Class decides, margin does not (majority 0.000 in every tercile) |
| **v3e** T15 | Share grid for K = 2, 3, 5: where does the mixed step start lowering the others? | Locate the break-even ratio found in T11 | 1,080 | `PREREG_v3d_addendum1.md` | `v3_t15`; `toy-v3` `779297f` | `analysis/v3e/SUMMARY.md`, `FINDINGS.md`, `cells.csv`, `per_m.csv`; `v3e_raw.tar.gz` | **Odds law**: R ∝ m/(1−m) in 12/12 cells (slope 0.98–1.05); threshold just above 1/K; persistent absorption needs clearly more imbalance |
| **v3f** T16 | pass@k rules at matched majority progress, 4× the v3c horizon | v3c's pass@k result might be slow learning | 480 | `PREREG_v3d_addendum2.md` | `v3_t16`; `toy-v3` `e45b533` | `analysis/v3f/SUMMARY.md`, `FINDINGS.md`, `runs.csv`, `cells.csv`; `v3f_raw.tar.gz` | A trade-off: pass@2 lets the minority dip to 0.03–0.08 then recover; pass@4 keeps it at 0.31–0.42 but the majority is 4–40× slower |
| **v3g** T17 | KL anchor (reverse and forward, several β) on the minority task | H4 as a prevention lever | 1,040 | `PREREG_v3d_addendum3.md` | `v3_t17`; `toy-v3` `b059c73` | `analysis/v3g/SUMMARY.md`, `FINDINGS.md`, `t17_*.csv`; `v3g_raw.tar.gz` | Direction matters: forward KL keeps the minority above a floor (absorbed ≤ 2% at the 1e-3 threshold; 34% at β = 0.01 at 1e-2); reverse KL does nothing up to β = 0.1 |
| v3g T18 | Collapse, then stop rewarding the majority ("release"), with or without KL | Why do collapsed prompts stay down? | 320 | same | `v3_t18` | `t18_*.csv` | No second resting point: the minority returns if some restoring force is left and there is time; forward KL returns in 100–600 steps from any depth |
| v3g T19 | T15 share grid at K = 3 for G ∈ {2, …, 64} | H5 as a lever: does G move the threshold? | 4,180 | same | `v3_t19` | `t19_*.csv` | Group size is not a lever (no consistent shift, at most 0.15 in m) |
| **v3h** T20 | Correct minority class in Li, Pandey and Chen's environment: 64 token prompts, shared 2-layer transformer, exact success probabilities | Does the mechanism carry over to token prompts and a transformer? | 1,809 | `PREREG_T20.md` | `v3_t20` (`experiments/enum_env.py`); `enum-env` `2f701de`; their code at `05ef005` | `analysis/v3h/SUMMARY.md`, `FINDINGS.md`, `runs.csv`, `cells.csv`; `v3h_raw.tar.gz` | First step follows the odds law exactly (slope 1.00); minority absorbed only with sharing. Only Adam lr 1e-3 isolates the share effect (0.2 at balance → 0.94 at high share) |
| **v3i** T21 | KL anchors and pass@k in the transformer environment | Do the toy levers transfer? | 560 | `PREREG_T20_addendum1.md` | `v3_t21`; `enum-env` `351d722` | `analysis/v3i/SUMMARY.md`, `FINDINGS.md`, `runs.csv`, `cells.csv`; `v3i_raw.tar.gz` | Forward KL transfers (β = 0.3: none absorbed in 8/8 cells); reverse KL at 0.03 does not; pass@k works only with Adam |
| **v3j** T22 | Two classes with prompts plus K − 2 junk answers that are never correct | LLM wrong answers are spread out: does that stop the competition? | 1,280 | `PREREG_v3d_addendum4.md` | `v3_t22`; `toy-v3` `f0bf112` | `analysis/v3j/SUMMARY.md`, `FINDINGS.md`, `runs.csv`, `cells.csv`; `v3j_raw.tar.gz` | First step moves exactly as the formula says, but junk does **not** protect: absorption from m = 0.6 for every K |
| v3j2 | Rerun of the 106 T22 runs lost when host `secb1013-d01` rebooted | Complete T22 | 106 | same | same commit and job list | merged into `analysis/v3j/`; `v3j2_raw.tar.gz` (also copied into `~/rlvr-data/v3j/*_rerun`) | Complete; included in the 1,280 |
| **v3k** T23 | Class-balanced supervised warm start to p0 ∈ {0.5, 0.7, 0.9} before GRPO | The LLM starts already solving the minority about half the time | 320 | `PREREG_v3d_addendum5.md` | `v3_t23`; `toy-v3` `a717aab` | `analysis/v3k/SUMMARY.md`, `FINDINGS.md`, `runs.csv`, `cells.csv`; `v3k_raw.tar.gz` | A head start protects, and higher shares need a bigger one: p0 = 0.5 removes collapse at m = 0.7, not at m = 0.9 (0.99 absorbed, linear) |
| **v3l** T24 | Per-prompt snapshots at log-spaced steps to 100,000 steps; linear policy, SGD, exact update | Test the per-prompt late-phase model from `analysis/theory/late_time.md` | 120 | `PREREG_v3d_addendum6.md` | `v3_t24`; `toy-v3` `5487480` | `analysis/v3l/SUMMARY.md`, `FINDINGS.md` (`experiments/check_late_time.py`), `runs.csv`, `prompts.csv`; `v3l_raw.tar.gz` | Per-prompt model fails (Spearman 0.28); run-level sinking rate follows the majority's max-margin geometry (0.86); the late phase barely reorders prompts |

### Theory checks (no new batches unless stated)

| ID | What it was | Why | Data | Code | Results | Outcome |
|---|---|---|---|---|---|---|
| Odds law | First-order identity for the first step from a uniform start: push ratio R = c·m/(1−m) | Explain T11/T15 | 1,080 v3e runs | `experiments/check_odds_law.py` | `analysis/theory/odds_law.md`, `odds_law_check.md`, `odds_law_*.csv` | Matches the simulator to median relative error 1.4e-3; predicted m* within 0.0002 in 12 cells. A first-step identity only, silent about the outcome (T22, T23) |
| Lean proofs | Tabular monotonicity, cross-class factor, push ratio in block and odds form | Formal check of the algebra | – | `formal/Formal/RLVR.lean` | `formal/README.md` | `lake build` passes with no `sorry` (`HANDOFF.md`, 11-23 21:40) |
| SymPy checks | First-step algebra and the scalar late model's exact solution | Symbolic check of the late-phase algebra | – | `tests/test_theory_symbolic.py` | same file | 8 tests (`HANDOFF.md`, 11-23 21:40) |
| Late-time probe | Majority's max-margin direction û; minority rate γ_y = û·φ_y | Does implicit bias (Soudry et al.) explain the late phase? | 400 T12 runs | `experiments/probe_late_time.py` | `analysis/theory/late_time.md`, `late_time_probe_rows.csv` | Run level supported (Spearman 0.89 under SGD); per-prompt "γ < 1 recovers" refuted |
| Optimizer geometry | ℓ2 against ℓ∞ max-margin geometry under SGD and Adam | Does Adam's ℓ∞ geometry explain its recoveries? | T12 runs | `experiments/probe_optimizer_geometry.py` | `analysis/theory/late_time.md` (update), `optimizer_geometry_runs.csv` | Not supported (Adam: 0.60 against 0.62) |
| Fall model | Reduced class-level ODE for phase 1 (the fall), with six predictions P1–P6 | The outcome is decided during the fall; no theory of it yet | in-sample: v3e, v3g T19, v3j, v3k, v3l; held-out: v3o | `experiments/fall_ode.py`, `experiments/check_fall.py`, `tests/test_fall_ode.py` | `analysis/theory/fall.md`; predictions `analysis/theory/fall_predictions.csv`; held-out rule `analysis/review/PREREG_fall_heldout.md` | `analysis/theory/fall_results.md`, `fall_compare.md`, `fall_compare_heldout.md`, `fall_runs*.csv`. In-sample (fit only): passes every rule in every batch except P1 in v3l (0.799). **Held-out on 1,360 fresh runs:** passes all rules on the KL and warm-start grids; on the P1 grid passes P2–P6 and the threshold check (15/16) but **fails P1 (0.755)** because of large-step runs (η ≥ 3); for η ≤ 1 the depth rank is 0.988 |

## 6. Small LLMs (GPU track, 2026-11-20 to 11-25)

Qwen2.5-0.5B-Instruct unless stated; GRPO on GSM8K with our `calculate_advantages`; 8-bit AdamW; one RTX 4090
per run. Code on `llm-transfer` (merged into `main`): `llm/train.py`, `llm/evaluate.py`, analysis in
`llm/analyze_*.py`. Job lists: `llm/jobs/*.txt`. Round 1 ran under tags `q_NN` (line NN of
`~/rlvr-queue/round1_queue.txt`), round 2 under `r2_NN` (line NN of `~/rlvr-queue/queue.txt`, host in
`queue.log`). Raw: `llm_raw.tar.gz` (one folder per tag), laptop `~/rlvr-data/lab_home/llm/<tag>/`, lab
`~/rlvr_results/llm/<tag>/<host>/`.

**Tag replacements** (`HANDOFF.md`): r2_111–116 replace r2_105–110 (stopped and rerun on the fixed KL code);
r2_117 replaces r2_87; r2_118 replaces r2_99; r2_101 is r2_98 resumed. `q_34` is an old C2 run excluded by the
C2 amendment; `q_00` failed (CUDA fault).

| ID | What it was | Why | Runs (tags) | Pre-reg | Code | Results | Outcome |
|---|---|---|---|---|---|---|---|
| C0 pilot | Base evaluations (Qwen, SmolLM2); Qwen lr pilots 1e-6, 3e-6, 1e-5; SmolLM2 LoRA pilots; C2 difficulty pilot and candidates; N = 256 base evaluations | Feasibility, throughput, learning rate for C1, decline-test calibration | `c0`, `c0b` (SmolLM2, invalid), `c0c` (Qwen 1e-5 rerun), `c0d`, `c0e` | lr rule in `PREREG_C1.md` | `llm/jobs/c0_pilot.txt`, `c0d.txt`, `c0e.txt`; `llm/analyze_c0.py` | `analysis/llm/c0_qwen.md` | lr 1e-6 chosen; lr 1e-5 is a collapse, not selective loss; N = 256 needed to see losses at low p |
| Model probes | Base evaluation, 5 GRPO steps with sync check, GPU memory trace, for Qwen 0.5B–7B, SmolLM2/3, Llama 1B/3B in full, LoRA, int8, nf4 | Choose models and modes that fit one 24 GB GPU | `q_01`–`q_20` | no | `llm/jobs/probe_*.txt`; `llm/analyze_probe.py` | `analysis/llm/model_probe.md` | DONE: Qwen2.5-0.5B int8, Qwen2.5-1.5B LoRA, Qwen2.5-3B int8, SmolLM2-360M full, Llama-3.2-1B full and LoRA; the other set-ups FAILED (see the table) |
| Census calibration | Two base evaluations with different seeds compared as before/after | How many false "lost" problems does a naive rule flag? | no training (base evaluations) | no | `llm/census_calibration.py` | `analysis/llm/census_calibration.md` | Our decline and lost tests flag 0 false problems; naive "locked" and "halved" rules flag many at ≤ 64 samples |
| Llama calibration | Llama base evaluations and lr pilots 1e-6, 3e-6, 1e-5; N = 256 base evaluations for C1 | Pick Llama's learning rate by the pre-registered rule | `r2_00`–`r2_05`, `r2_18`, `r2_19` | `PREREG_C1_amendment1.md` | `llm/jobs/llama_c0.txt`, `llama_arm.sh` | `analysis/llm/c1_llama.md` | lr 1e-6 by the rule |
| C1 Qwen / **P4** | 500 steps on 1,000 training problems; 256 samples per problem before and after on training and 300 held-out problems; lr 1e-6 (primary) and 3e-6 | Does small-LLM GRPO lose problems beyond the calibrated null? | lr 1e-6: `c1q` (3 seeds); lr 3e-6: `q_35`, `q_21`, `q_36` | confirmatory: `PREREG_C1.md`, amendment 1 | `llm/jobs/c1q_lr1e-6.txt`; `llm/analyze_c1.py` | `analysis/llm/C1_C5_SUMMARY.md`, `c1_qwen.md`, `c1_qwen_runs.csv` | **P4 FAIL**: 0 training problems lost at lr 1e-6; 8–12 of 300 held-out problems decline. At lr 3e-6: 15, 12, 1 lost |
| C1 Llama | Same as C1 with Llama-3.2-1B, lr 1e-6, 3 seeds | P4 needs both families | `r2_20`–`r2_22` | same | `llm/analyze_c1.py` | `analysis/llm/c1_llama.md`, `c1_llama_runs.csv` | 0 training problems lost; 10–13 held-out declines (replicates Qwen) |
| C1 control | Rejection-sampling fine-tuning on the model's own correct answers, same problems and lr | Are the held-out declines specific to GRPO? | `r2_23`, `r2_59`, `r2_60` | `PREREG_C1_amendment2.md` | `llm/jobs/c1_rft.txt`; `llm/analyze_control.py` | `analysis/llm/c1_control.md`, `c1_control_runs.csv` | 0 held-out declines, but confounded: the control only reached training success 0.65 (bar 0.75) |
| C5 | Loss drivers from the C1 setting: verifier flips 20%, sign rule, lr × 10 | Toy: noisy verifier and sign rule cause absorption | `q_23`–`q_31` (3 seeds each) | exploratory (`NEXT_PLAN.md`) | `llm/analyze_c1.py` | `analysis/llm/C1_C5_SUMMARY.md`, `c5_qwen.md`, `c5_qwen_runs.csv` | Flip and sign learn less but lose nothing; lr 1e-5 collapses (unparsable output) |
| C2 / **P5** | Two procedures (`each`, `total`) share one template; majority share m ∈ {0.5, 0.8, 0.95}; both roles; 3 seeds; 300 held-out minority problems | Toy answer class ↔ LLM procedure: does the majority collapse the minority? | main: `r2_07`–`r2_17`, `r2_24`–`r2_28` (16); base: `q_32`, `q_33`; hints `r2_06` | confirmatory: `PREREG_C2.md`, amendment 1 | `llm/jobs/c2.txt`, `c2_v2.txt`; `llm/analyze_c2.py` | `analysis/llm/C2_SUMMARY.md`, `c2.md`, `c2_runs.csv` | **P5 FAIL** (p = 0.0536 in one role). The minority **improves** by +0.36 to +0.45 in 15 of 16 runs; no switch to the other procedure |
| C2 matched minority | 4 minority prompts per step with 4, 16 or 76 majority prompts | Separate majority pressure from minority sample count | `r2_29`–`r2_40` (12) | `PREREG_C2_amendment1.md` | `llm/jobs/c2_extras.txt` | `analysis/llm/C2_SUMMARY.md` "The controls" | No dependence on the number of majority prompts |
| C2 seeds | The degraded run's problem draw rerun with 5 new seeds at m = 0.95 and 0.5 | One run (m = 0.95, seed 1) degraded broadly | `r2_61`–`r2_70` (10) | `PREREG_C2_amendment2.md` | `llm/jobs/c2_seeds.txt` | `analysis/llm/c2_seeds.md`, `c2_seeds_runs.csv` | Broad degradation 0 of 10 new runs: a seed outlier. A moderate share effect on held-out declines (17–23 against 6–11) |
| C2 extras | 16 and 256 prompts per step; long arm (1,500 steps); rejection-sampling control | Samples per step, delayed loss, GRPO-specificity | `r2_71`–`r2_76`, `r2_77`–`r2_78`, `r2_79`–`r2_84` | exploratory | `llm/jobs/c2_extras.txt` | `analysis/llm/EXTRAS_2026-11-24.md` | 64 → 256 prompts changes nothing; no delayed loss; declines occur under both RFT and GRPO, losses mostly under GRPO |
| SmolLM2 diagnostic | LoRA lr 1e-4, with and without truncated importance weights | Why the SmolLM2 pilots collapsed | `r2_85`, `r2_86` | no | `llm/jobs/smol_diag.txt` | `analysis/llm/EXTRAS_2026-11-24.md` | A sampler–trainer mismatch: with truncated importance weights it trains normally |
| C10 | Branch test: mixed (76 + 4), majority-only (76), minority-only (4) prompts per step, 100 steps, both roles, 3 seeds | Measure the sign of the majority's effect on the minority directly (the toy's T11) | `r2_41`–`r2_58` (18) | `PREREG_C10.md` | `llm/jobs/c10.txt`; `llm/analyze_branch.py` | `analysis/llm/C10_SUMMARY.md`, `c10.md`, `c10_runs.csv` | Majority-only training **raises** the minority (+0.11 to +0.34) in all 6 runs: positive transfer, the opposite of the toy |
| C9 | Token-level loss aggregation instead of sequence-level | Does aggregation change losses? | `r2_91`–`r2_93` (3) | exploratory | `llm/jobs/c8_c9.txt` | `analysis/llm/c9.md`, `c9_runs.csv` | Same as C1: 0 training problems lost; held-out declines 5, 13, 12 |
| C7 / C8 | Reverse KL β ∈ {0, 0.02, 0.1, 0.5} × G ∈ {4, 16} (C7) and forward KL β ∈ {0.1, 0.5}, G = 16 (C8); 800 steps, 2 seeds | LLM stretch goal: does the KL penalty behave as the toy predicts? | 20: `r2_88`–`r2_90`, `r2_94`–`r2_97`, `r2_100`–`r2_104`, `r2_111`–`r2_118` (r2_114 broke numerically and is left out of the settings table) | C7 exploratory (`NEXT_PLAN.md`); C8 in `PREREG_v3c.md` | `llm/jobs/c7.txt`, `c8_c9.txt`; `llm/analyze_c1.py`, `llm/analyze_kl_fixed_points.py`; KL fix `a7bd0aa` | `analysis/llm/C7_C8_SUMMARY.md`, `c78.md`, `c78_runs.csv`, `c78_fixed_points.md` (first look: `c78_partial.md`) | Any β ≥ 0.02 removes the held-out declines (0–4, 0 lost); forward ≈ reverse (C8 prediction not supported); G = 16 settles above G = 4 |
| C3, C11 | Natural-data rarity (C3); a fix in the LLM (C11) | Planned | – | – | – | – | Not run. C11's gate (P5 pass or a push-down in C10) was never met |

**Holm over P1–P5** (`analysis/llm/C2_SUMMARY.md`): only P1 holds at 0.05 (adjusted 0.049, or 0.039 with P2's
registered input).

## 7. Latest: replication, permanence, shared skills, confirmatory battery, review caveats (2026-11-23 to 11-24)

| ID | What it was | Why | Runs | Pre-reg | Code | Results | Outcome |
|---|---|---|---|---|---|---|---|
| **wsw** | Full replication of Li, Pandey and Chen, "Wrong the Same Way": all 11 experiments plus `theory.py` and `exp11_kernel.py`, their scripts at pinned commit `05ef005`, torch 2.15, single-threaded | Their environment is the base of T20; check it before building on it | 256 jobs | no | `tools/literature/replication/` (`launch_wsw.sh`, `wsw_jobs.py`, `compare_numbers.py`; branch `literature`) | `tools/literature/replication/RESULT.md`, `numbers_comparison.md`, `numbers_control.md`; `wsw_raw.tar.gz` | Their figure code on their results reproduces 494/494 numbers. Our rerun: 130 of 506 identical, 193 within 2%. **The headline claim replicates** (drift rises with shared leaks only with sharing) |
| wsw2 | E1 and E8 seeds 5–9 (their README uses 10 seeds; the script default is 5) | Complete the replication | 70 jobs | no | same | same; `wsw2_raw.tar.gz` | Included in the comparison above |
| **v3m** T25 | Exact gradient flow of the linear policy to flow time 1e14; K = 3, ρ ∈ {0.3, …, 0.95}, m ∈ {0.7, 0.9}, 10 tasks each | Permanent or a long plateau? T12 could not tell at ρ = 0.9 | 100 tasks (98 integrated) | `PREREG_v3d_addendum7.md` | `experiments/exact_flow.py` (`toy-v3` `3bf676f`), `experiments/check_permanence.py`, `tests/test_exact_flow.py` | `analysis/v3m/FINDINGS.md`, `SUMMARY.md` (being written); `v3m_raw.tar.gz` | All 100 tasks integrated (two needed Radau/BDF). Per-prompt predictions fail; mostly a plateau whose length grows with ρ (recovery 1e2–1e5 at ρ ≤ 0.5, up to 1e12 at ρ ≥ 0.9); nothing recovers at ρ = 0.95, m = 0.9; γ_min ≥ 1 marks the tasks that stay collapsed (27 of 30) |
| **v3n** T26 | 4-token answers: k positions carry the answer class, 4 − k follow one shared balanced rule | C10 suggests shared skills cause positive transfer in the LLM | 320 | `PREREG_v3d_addendum8.md` | `v3_t26`; `toy-v3` `c991964` | `analysis/v3n/SUMMARY.md`, `FINDINGS.md`, `runs.csv`, `cells.csv`; `v3n_raw.tar.gz` | First-step identity holds with multi-token answers (320/320). Sharing dilutes the push but does not prevent collapse (98–100% absorbed in 6 of 8 cells). Design limit: shared positions agree only at chance, so the positive-transfer test never applied |
| **v3o** A1, A2, A3, A5 | Confirmatory replication battery on fresh task seeds 10–29: A1 = P1 (v3a), A2 = forward-KL floor (T17), A3 = pass@4 at matched progress (T16), A5 = warm start p0 = 0.5 (T23) | Turn the main exploratory results into confirmatory ones | 10,440 (A1 9,280; A2 640; A3 360; A5 160) | **confirmatory**: `PREREG_v3d_addendum9.md` (one Holm family of 5 with A4) | `v3_a1`, `v3_a2`, `v3_a3`, `v3_a5`; `toy-v3` `4391e1a` | `analysis/v3o/SUMMARY.md`, `FINDINGS.md`, `results.csv`, `runs.csv`, `a1_cells.csv` … `a5_cells.csv`; `v3o_raw.tar.gz` | **All five confirmed** (with A4): 20 of 20 fresh tasks the predicted way in every test, Holm p 4.8e-6 each; effect sizes held (A1 a little smaller in three ρ = 0.5 cells) |
| **v3p** G (T27) | Forward KL toward a collapsed reference (anchor at 0.5%, 5% or 25% of training) | Review caveat: T17's reference still solved the minority | 260 | `PREREG_v3d_addendum10.md` | `v3_t27`; `toy-v3` `4391e1a` | `analysis/v3p/SUMMARY.md`, `FINDINGS.md`, `runs.csv`, `cells.csv`; `v3p_raw.tar.gz` | **Holds:** anchors after the fall never restore the minority (180/180); the start anchor at β ≥ 0.1 prevents absorption (40/40). Forward KL protects only while the reference still solves the minority |
| v3p H (T28) | Adam's ε sweep, with lr scaled with ε; SGD control | Is Adam's per-coordinate normalisation why it recovers more often than SGD? | 560 | same | `v3_t28` | same, plus `h_gamma_corr.csv` | **Fails as registered (4 of 8 cells):** half the cells have no Adam recovery to remove. Post hoc: where Adam recovers, it switches off between ε = 1e-3 and 1e-2 at fixed lr, pointing to the per-coordinate normalisation |
| **v3q** A4 | T20's share effect under Adam (lr 1e-3, shared transformer, exact) on fresh partition seeds 10–29 | Part of the confirmatory battery | 440 (grid count; the addendum says 480) | `PREREG_v3d_addendum9.md` | `v3_a4`; `enum-env` `4b271a0` | `analysis/v3o/SUMMARY.md` (A4 is part of the battery), `a4_seeds.csv`, `a4_cells.csv`; `v3q_raw.tar.gz` | **Confirmed:** 20/20 seeds, mean difference 0.89 (original 0.86) |
| v3q J (T29) | T20 share grid under SGD at small step sizes (η ∈ {0.03, 0.1}), flow time matched | Review caveat: at large SGD steps both classes were absorbed | 240 | `PREREG_v3d_addendum10.md` | `v3_t29`; `enum-env` `4b271a0` | `analysis/v3q/SUMMARY_J.md`, `FINDINGS_J.md`, `runs_J.csv`; `v3q_raw.tar.gz` | **Holds:** at small SGD steps class 0 is never absorbed at high share (0/80), the minority always is (80/80), and rarely at balance (4/40) |

Smoke tests (`~/rlvr-data/smoke`, `smoke_enum`, `lab_home/smoke`, LLM `smoke_r2`, `check`) only checked the code
and are not indexed as experiments.

## Still to do

- ~~Battery A analysis (Holm over 5)~~: done 11-25, all five confirmed.
- ~~Fall model: in-sample comparison, and held-out test on v3o~~: done 11-25.
- ~~G/H/J analyses~~: done 11-25.
- ~~T25: the last stuck task, then the analysis~~: done 11-25.
- `analysis/FINAL_RESULTS.md`.
- Report outline, figures script, LaTeX skeleton.
- Team writes, Claude fact-checks.
- ~~Raw-archive links~~: done 11-25 (Google Drive, access by request).
