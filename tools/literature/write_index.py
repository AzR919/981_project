"""Write README.md and papers.bib into the papers folder from manifest.py + download_log.json."""
import json
import re
import sys

sp, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, sp)
import manifest as M  # noqa: E402

meta = json.load(open(sp + "/cands_meta.json"))
meta.update(json.load(open(sp + "/seeds_meta.json")))
meta[M.BAUDRY["id"]] = M.BAUDRY
log = json.load(open(sp + "/download_log.json"))

HYP = {"H1": "H1 — the identity and the estimator", "H2": "H2 — what is optimized (surrogates, advantage shaping, pass@k)",
       "H3": "H3 — stability, step size and interference", "H4": "H4 — KL regularization and where training stops",
       "H5": "H5 — group size, uninformative groups and absorption at p ≈ 0", "BG": "Background"}


def authors(aid, n=3):
    au = [a.strip() for a in meta[aid]["authors"]]
    last = [a.split()[-1] for a in au]
    return ", ".join(last) if len(last) <= n else f"{last[0]} et al."


def ref(aid):
    if aid == M.BAUDRY["id"]:
        return "[NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4e683f0ce6b91e7fbdae9d32642d88f-Abstract-Conference.html)"
    return f"[{aid}](https://arxiv.org/abs/{aid})"


def venue(v, src):
    if src.startswith("or:"):
        return f"[{v}]({M.OR}{src[3:]})"
    if src.startswith("http"):
        return f"[{v}]({src})"
    return v


def title(aid):
    return meta[aid]["title"].replace("|", "/").replace("$^\\varepsilon$", "ε")


def table(rows):
    lines = ["| Paper | Venue (verified) | ID | Rel. | Why it matters for the project | File |", "|---|---|---|:-:|---|---|"]
    for aid, _, _, v, src, rel, _, link in sorted(rows, key=lambda r: -r[5]):
        f = log[aid]["file"]
        lines.append(f"| **{title(aid)}** — {authors(aid)} | {venue(v, src)} | {ref(aid)} | {rel} | {link} | [pdf]({f}) |")
    return "\n".join(lines)


new = M.TIER_A + M.TIER_B
by_id = {r[0]: r for r in new}
n_ok = sum(v["ok"] for v in log.values())
md = []
md.append("# Reading list: frontier work close to the GRPO/RLVR project\n")
md.append("Compiled 2026-11-19 for the CMPT 981 project (`981_project/`). Window: November 2023 to November 2026.\n")
md.append(f"- **{len(M.TIER_A)}** new papers verified at a top venue, in `tierA_top_venue/`\n"
          f"- **{len(M.TIER_B)}** new close preprints, in `tierB_preprints/`\n"
          f"- **{len(M.CITED)}** papers the project already cites, in `cited_core/`\n"
          f"- **{len(M.LISTED)}** further papers listed below but not downloaded\n"
          f"- BibTeX for every downloaded paper is in `papers.bib`\n")
md.append("Relevance is scored 1–5 against the project's hypotheses; 5 means the same setting (binary reward, group baseline, "
          "and a result on objectives, fixed points, step sizes or uninformative groups). "
          "Hypothesis labels follow `981_project/TODO.md`.\n")

md.append("## Read these first\n")
md.append("New papers, not yet cited in the project, in suggested reading order.\n")
for i, aid in enumerate(M.READ_FIRST, 1):
    r = by_id[aid]
    md.append(f"{i}. **{title(aid)}** ({authors(aid)}; {r[3]}; {ref(aid)}) — {r[7]} [[pdf]({log[aid]['file']})]")
md.append("")

md.append("## New papers, by hypothesis\n")
for h in ["H1", "H2", "H3", "H4", "H5", "BG"]:
    rows = [r for r in new if r[6] == h]
    if rows:
        md.append(f"### {HYP[h]}\n")
        md.append(table(rows) + "\n")

md.append("## Already cited in the project\n")
md.append("Downloaded for completeness. The venue column is what was verified on 2026-11-19; "
          "see the next section for entries where this differs from `latex/ref.bib`.\n")
md.append("| Paper | Venue (verified) | ID | Used for | File |\n|---|---|---|---|---|")
for aid, _, _, v, src, h in M.CITED:
    md.append(f"| {title(aid)} — {authors(aid)} | {venue(v, src)} | {ref(aid)} | {h} | [pdf]({log[aid]['file']}) |")
md.append("")

md.append("## Venue updates for `latex/ref.bib`\n")
md.append("These cited papers now have a published venue that `ref.bib` or the guide does not record. `ref.bib` was not changed.\n")
md.append("| Paper | ref.bib says | Verified venue |\n|---|---|---|\n"
          f"| Heckel, Soltanolkotabi and Thrampoulidis, *Asymmetric Prompt Weighting* | arXiv preprint | [ICML 2026]({M.OR}nYwP9hcr38) |\n"
          f"| Liu et al., *Understanding R1-Zero-Like Training* (Dr. GRPO) | arXiv preprint | [COLM 2025]({M.OR}5PAF7PAY2Y) |\n"
          f"| Rad et al., *Rate or Fate?* (primer only) | not in ref.bib | [ICML 2026 spotlight]({M.OR}LwB2EacVT6) |\n"
          f"| Shao et al., *Spurious Rewards* (primer only) | not in ref.bib | [ICML 2026]({M.OR}tqTNOpkP5j) |\n"
          f"| Hu et al., *BroRL* (primer only) | not in ref.bib | [ICML 2026]({M.OR}KmS7pdFBEh) |\n"
          "| Tan et al., *Scaling Behaviors of LLM RL Post-Training* (primer only) | not in ref.bib | ACL 2026 (author-stated on arXiv) |\n")
md.append("Suk and Duan, Mroueh, and Davis and Recht are still arXiv-only as far as could be verified "
          "(Suk and Duan appears on OpenReview as an ICLR 2026 submission with no acceptance record).\n")

md.append("## Listed, not downloaded\n")
md.append("Relevant but either further from the project or covered by a downloaded paper. Venues marked "
          "\"per Semantic Scholar\" or \"author-stated\" were not confirmed on a proceedings page.\n")
for h in ["H2", "H3", "H4", "H5"]:
    rows = [r for r in M.LISTED if r[4] == h]
    md.append(f"### {HYP[h]}\n")
    md.append("| Paper | Venue | ID | Rel. | Note |\n|---|---|---|:-:|---|")
    for aid, label, v, rel, _, note in sorted(rows, key=lambda r: -r[3]):
        md.append(f"| {label} | {v} | [{aid}](https://arxiv.org/abs/{aid}) | {rel} | {note} |")
    md.append("")

md.append("## How this list was built\n")
md.append("1. **Seeds.** The 29 arXiv IDs already in `latex/ref.bib`, `docs/rlvr_primer.html`, `docs/project_guide.html` and `TODO.md`.\n"
          "2. **Citation snowball.** Papers citing or cited by 12 core seeds, from the Semantic Scholar Graph API "
          "(2,000 unique papers), ranked by how many theory seeds they connect to. Two seeds (Huang et al. 2026, Rad et al. 2026) "
          "returned no citation data, so papers that cite only those are missing.\n"
          "3. **Topic searches.** Web and arXiv title searches for each hypothesis, plus softmax policy-gradient theory.\n"
          "4. **Verification.** Every ID was resolved through the arXiv API for title, authors and date. "
          "Venues were confirmed by exact-title match on OpenReview (linked in the venue column), or taken from the "
          "authors' arXiv comment or a proceedings page. Anything else is labelled as a preprint.\n"
          "5. **Files.** Each PDF was checked to be a real PDF whose first pages contain the paper's title.\n")
md.append("Caveats: relevance scores and the one-line links are judgements from abstracts, not from reading each paper in full. "
          "OpenReview links for NeurIPS 2026 papers show acceptance; the camera-ready version may differ from the arXiv PDF saved here. "
          "The Baudry et al. paper has no arXiv version and was saved from the NeurIPS proceedings.\n")
open(out + "/README.md", "w").write("\n".join(md))

# ---------------------------------------------------------------- papers.bib
VENUE_BOOK = {"NeurIPS": "Advances in Neural Information Processing Systems", "ICML": "International Conference on Machine Learning",
              "ICLR": "International Conference on Learning Representations", "COLM": "Conference on Language Modeling",
              "ACL": "Proceedings of the Annual Meeting of the Association for Computational Linguistics"}


def brace(t):
    t = t.replace("&", "\\&")
    return re.sub(r"(?<![\w{\\$])([A-Za-z0-9.@\-]*[A-Z][A-Za-z0-9.@\-]*[A-Z][A-Za-z0-9.@\-]*|R1-Zero-Like|Dr\.)", r"{\1}", t)


def key(aid, short, year):
    last = re.sub(r"[^a-z]", "", meta[aid]["authors"][0].split()[-1].lower())
    word = re.sub(r"[^a-z0-9]", "", short.split("-")[0].lower())
    return f"{last}{year}{word}"


bib, seen = ["% BibTeX for every paper downloaded into this folder. Standalone: not merged into 981_project/latex/ref.bib.",
             "% Venues are those verified on 2026-11-19 (see README.md). Author lists are as given on arXiv.", ""], set()
for folder_rows in (M.TIER_A, M.TIER_B, M.CITED):
    for row in folder_rows:
        aid, short, v = row[0], row[2], row[3]
        m = meta[aid]
        conf = next((k for k in VENUE_BOOK if v.startswith(k)), None)
        year = re.search(r"(20\d\d)", v).group(1) if re.search(r"(20\d\d)", v) and not v.startswith("arXiv") else m["published"][:4]
        k = key(aid, short, year)
        while k in seen:
            k += "b"
        seen.add(k)
        au = " and ".join(a.strip() for a in m["authors"][:12]) + (" and others" if len(m["authors"]) > 12 else "")
        note = "" if aid == M.BAUDRY["id"] else f"arXiv:{aid}"
        if conf:
            fields = [("title", brace(m["title"])), ("author", au), ("booktitle", VENUE_BOOK[conf]), ("year", year)]
            kind = "inproceedings"
        elif v.startswith("TMLR") or v.startswith("Nature"):
            fields = [("title", brace(m["title"])), ("author", au),
                      ("journal", "Transactions on Machine Learning Research" if v.startswith("TMLR") else "Nature"), ("year", year)]
            kind = "article"
        else:
            fields = [("title", brace(m["title"])), ("author", au), ("journal", f"arXiv preprint arXiv:{aid}"), ("year", year)]
            kind, note = "article", ""
        if note:
            fields.append(("note", note))
        body = ",\n".join(f"  {a:<9} = {{{b}}}" for a, b in fields)
        bib.append(f"@{kind}{{{k},\n{body}\n}}\n")
open(out + "/papers.bib", "w").write("\n".join(bib))
print("README rows:", len(new), "new +", len(M.CITED), "cited; listed", len(M.LISTED), "| pdf ok:", n_ok, "/", len(log), "| bib entries:", len(seen))
