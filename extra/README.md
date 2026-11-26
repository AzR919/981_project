# extra/: work beyond the milestone

Everything here goes beyond what the milestone promised. It is kept because the report and the analysis draw on parts
of it, but nobody needs to read it to follow the promised work. Each experiment is labelled in `docs/EXPERIMENTS.md`
and `docs/Ankush.md`:
- **Core follow-up:** grew out of a promised result (H3's failure led to minority collapse); the report's main story
  relies on it.
- **Extra:** a second environment, the LLMs, formal verification, or a side question.

| Folder | What | Label |
|---|---|---|
| `analysis/v3a` … `v3g`, `v3j` … `v3p` | Minority collapse: P1–P3, the mechanism (T11–T13), the threshold and odds law (T15), the levers (T16–T19), junk answers, warm start, the late phase, permanent or plateau (T25), the confirmatory battery (v3o), the review caveats (v3p) | Core follow-up (T26 in `v3n` and H in `v3p` are Extra) |
| `analysis/theory/` | The odds law, the late-time and fall models | Core follow-up |
| `analysis/v3h`, `v3i`, `v3q` | The transformer environment of Li, Pandey and Chen (T20, T21, A4, J) | Extra |
| `analysis/llm/` | Small language models (C0–C10) | Extra |
| `analysis/review/` | Pre-registrations, reviews and the checks behind them | Process record |
| `experiments/` | Analysis and theory scripts for the batches above (they import `experiments/run.py` from the main tree) | — |
| `llm/` | The LLM training, evaluation and queue code | Extra |
| `formal/` | Lean 4 proofs of the first-step algebra | Extra |
| `tools/` | The literature tools and the replication of Li, Pandey and Chen | Extra |
| `cluster/` | Lab launch scripts for the CPU batches | Infrastructure |
| `tests/` | Tests for the code in `extra/` | — |

**Running the code:**
- From the repo root, scripts run as before, with `extra/` in front: `python extra/experiments/analyze_v3n.py`,
  `python extra/llm/train.py …`.
- Tests: `pytest tests extra/tests`.
- The raw results are linked from `docs/EXPERIMENTS.md` (Google Drive, access by request).
