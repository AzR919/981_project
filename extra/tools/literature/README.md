# Literature tools

The code that built the reading list in `../../../papers/` (outside the repo: the PDFs, README.md,
papers.bib, notes and graph). The PDFs and notes are not tracked here; this folder keeps the code
and its small metadata inputs, so the collection can be rebuilt.

| File | Role |
|---|---|
| `tsearch.py`, `verify.py`, `orv.py` | Find and verify papers: arXiv title search, arXiv metadata lookup, OpenReview venue check |
| `snowball.py`, `cite_edges.py` | Citation snowball and citation edges from the Semantic Scholar API (`cite_edges.json` is its output) |
| `manifest.py` | The curated reading list: id, authors, venue, tier, linked hypothesis |
| `build.py` | Download and check the PDFs in `manifest.py`; writes `download_log.json` |
| `write_index.py` | Write `papers/README.md` and `papers/papers.bib` |
| `relations.py`, `build_notes.py` | Curated relations and per-paper notes / knowledge-graph nodes and edges |
| `show.py` | Print candidate metadata rows during triage |
| `*_meta.json`, `download_log.json` | Metadata snapshots the scripts read |
| `checks/` | Numerical checks of claims in the deep reviews (Davis & Recht W_G, Vojnovic & Yun gap, MaxRL rules, finite-G KL fixed points) |

The scripts are kept as they were run. Several take the working folder as an argument (see each
docstring) because they ran from a scratch directory.
