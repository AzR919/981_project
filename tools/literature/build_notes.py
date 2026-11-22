"""Build one note per paper, a tier table, and a node/edge file for a knowledge graph.

usage (from anywhere):  python3 build_notes.py

Reads manifest.py, relations.py, the metadata JSONs and cite_edges.json beside this file, plus
reviews/triage/*.md and reviews/deep/*.md. Writes notes/*.md, graph/papers.json and
reviews/TIERS.md. Safe to re-run: every output is regenerated.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import manifest as M  # noqa: E402
import relations as R  # noqa: E402

meta = json.load(open(HERE + "/cands_meta.json"))
meta.update(json.load(open(HERE + "/seeds_meta.json")))
meta[M.BAUDRY["id"]] = dict(M.BAUDRY, abstract="", comment="", published="2025-12-01")
log = json.load(open(HERE + "/download_log.json"))
cites = json.load(open(HERE + "/cite_edges.json")) if os.path.exists(HERE + "/cite_edges.json") else {"edges": [], "status": {}}

# ------------------------------------------------------------------ library rows
papers = {}
for folder, rows in (("tierA_top_venue", M.TIER_A), ("tierB_preprints", M.TIER_B)):
    for aid, _, _, venue, src, rel, hyp, link in rows:
        papers[aid] = dict(id=aid, folder=folder, venue=venue, src=src, rel_abstract=rel, hyp=[hyp], oneliner=link, status="new")
for aid, _, _, venue, src, hyp in M.CITED:
    hyps = re.findall(r"H\d", hyp) or ["BG"]
    papers[aid] = dict(id=aid, folder="cited_core", venue=venue, src=src, rel_abstract=None, hyp=hyps, oneliner="", status="cited", used_for=hyp)
for aid, p in papers.items():
    p["slug"] = os.path.basename(log[aid]["file"])[:-4]
    p["pdf"] = log[aid]["file"]
    p["pages"] = log[aid].get("pages", "")

# ------------------------------------------------------------------ triage blocks
FIELD = re.compile(r"^- \*\*(.+?):\*\*\s*(.*)$")
for path in sorted(glob.glob(ROOT + "/reviews/triage/*.md")):
    for block in re.split(r"^## ", open(path).read(), flags=re.M)[1:]:
        head, _, body = block.partition("\n")
        key = head.split(" — ")[0].strip()
        aid = M.BAUDRY["id"] if "Baudry" in key else key
        if aid not in papers:
            continue
        fields, cur = {}, None
        for line in body.split("\n"):
            m = FIELD.match(line)
            if m:
                cur = m.group(1).strip()
                fields[cur] = m.group(2).strip()
            elif cur and line.strip():
                fields[cur] += "\n" + line.rstrip()
        papers[aid]["triage"] = fields
        papers[aid]["triage_file"] = "reviews/triage/" + os.path.basename(path)

for path in glob.glob(ROOT + "/reviews/deep/*.md"):
    name = os.path.basename(path)
    for aid, p in papers.items():
        tag = "Baudry2025" if aid == M.BAUDRY["id"] else aid
        if tag in name:
            p["deep_file"] = "reviews/deep/" + name


def first_word(text, options, default):
    m = re.match(r"\W*([A-Z][A-Z\-]+)", text or "")
    return m.group(1) if m and m.group(1) in options else default


for aid, p in papers.items():
    t = p.get("triage", {})
    p["overlap"] = first_word(t.get("Overlap risk"), {"NONE", "PARTIAL", "HIGH"}, "")
    p["triage_rec"] = first_word(t.get("Recommendation"), {"DEEP", "SKIM", "CITE-ONLY", "DROP"}, "")
    m = re.match(r"\D*(\d)", t.get("Relevance", ""))
    p["rel_triage"] = int(m.group(1)) if m else None
    p["relevance"] = p["rel_triage"] or p["rel_abstract"]
    p["tier"] = "full review" if aid in R.DEEP else {"SKIM": "skim", "CITE-ONLY": "cite only", "DROP": "cite only", "DEEP": "skim"}.get(p["triage_rec"], "already cited" if p["status"] == "cited" else "not triaged")
    text = " ".join([meta[aid]["title"], p["oneliner"], t.get("Main claims", ""), t.get("Setting", "")])
    p["concepts"] = [c for c, rx in R.CONCEPTS.items() if re.search(rx, text, flags=re.I)]


def authors(aid, n=3):
    last = [a.strip().split()[-1] for a in meta[aid]["authors"]]
    return ", ".join(last) if len(last) <= n else f"{last[0]} et al."


def link(aid):
    p = papers[aid]
    return f"[[{p['slug']}]] ({authors(aid)}, {meta[aid]['title'][:70]})"


cite_out, cite_in = {}, {}
for a, b in cites["edges"]:
    if a in papers and b in papers:
        cite_out.setdefault(a, []).append(b)
        cite_in.setdefault(b, []).append(a)

# ------------------------------------------------------------------ notes
os.makedirs(ROOT + "/notes", exist_ok=True)
for old in glob.glob(ROOT + "/notes/*.md"):
    os.remove(old)
q = lambda s: '"' + str(s).replace('"', "'") + '"'
for aid, p in papers.items():
    m, t = meta[aid], p.get("triage", {})
    ident = "NeurIPS 2025 proceedings (no arXiv version)" if aid == M.BAUDRY["id"] else f"arXiv:{aid}"
    url = M.BAUDRY["pdf"] if aid == M.BAUDRY["id"] else f"https://arxiv.org/abs/{aid}"
    fm = ["---", f"id: {q(ident)}", f"title: {q(m['title'])}", "authors: [" + ", ".join(q(a.strip()) for a in m["authors"][:15]) + "]",
          f"date: {m['published']}", f"venue: {q(p['venue'])}", f"library: {p['folder']}", f"status: {p['status']}",
          "hypotheses: [" + ", ".join(p["hyp"]) + "]", f"relevance: {p['relevance'] if p['relevance'] else 'null'}",
          f"overlap_risk: {p['overlap'] or 'null'}", f"review_tier: {q(p['tier'])}",
          "concepts: [" + ", ".join(q(c) for c in p["concepts"]) + "]", f"url: {url}", f"pdf: ../{p['pdf']}", "---", ""]
    body = [f"# {m['title']}", "", f"{', '.join(a.strip() for a in m['authors'][:15])}{' et al.' if len(m['authors']) > 15 else ''}. "
            f"{p['venue']}. {ident}. {p['pages']} pages.", ""]
    if p["status"] == "cited":
        body += [f"Already cited by the project; used for: {p['used_for']}.", ""]
    one = t.get("Check of my one-liner", "")
    if one and not one.startswith("ACCURATE"):
        body += ["## In one sentence", "", re.sub(r"^(Corrected|CORRECTED)[:.]?\s*", "", one).strip().strip('"'), ""]
    elif p["oneliner"]:
        body += ["## In one sentence", "", p["oneliner"], ""]
    for title, key in (("Setting", "Setting"), ("Main claims", "Main claims"), ("In the project's notation", "In our notation"),
                       ("Evidence", "Evidence"), ("Hypotheses touched", "Hypotheses touched"), ("Overlap with the project", "Overlap risk"),
                       ("Practical consequence", "Practical consequence"), ("Baseline rule for the synthetic task", "Baseline rule we could add"),
                       ("Soundness", "Soundness (preprints only)"), ("Triage recommendation", "Recommendation")):
        if t.get(key):
            body += [f"## {title}", "", t[key].strip(), ""]
    rel = [r for r in R.RELATIONS if r[0] == aid or r[2] == aid]
    if rel or cite_out.get(aid) or cite_in.get(aid):
        body += ["## Relations", ""]
        for s, kind, d, note, basis in rel:
            a = "this paper" if s == aid else link(s)
            b = "this paper" if d == aid else (R.PROJECT_NODES[d].split(" — ")[0] + " of the project" if d.startswith("PROJECT") else link(d))
            body.append(f"- {a} **{kind}** {b}. {note} *(basis: {basis})*")
        if cite_out.get(aid):
            body += ["", "Cites, within this library: " + "; ".join(link(x) for x in sorted(cite_out[aid])) + "."]
        if cite_in.get(aid):
            body += ["", "Cited by, within this library: " + "; ".join(link(x) for x in sorted(cite_in[aid])) + "."]
        body.append("")
    if m.get("abstract"):
        body += ["## Abstract", "", m["abstract"], ""]
    src = [f"PDF: `{p['pdf']}`"]
    if p.get("triage_file"):
        src.append(f"triage note: `{p['triage_file']}` (pages read: {t.get('Pages read', 'not recorded')})")
    if p.get("deep_file"):
        src.append(f"full review: `{p['deep_file']}`")
    body += ["## Sources", "", "; ".join(src) + ".",
             "Setting, claims and overlap are from the triage agent's reading unless a full review is listed; the abstract is from arXiv.", ""]
    open(f"{ROOT}/notes/{p['slug']}.md", "w").write("\n".join(fm + body))

# ------------------------------------------------------------------ graph json
nodes, edges = [], []
for aid, p in papers.items():
    m = meta[aid]
    nodes.append(dict(id=p["slug"], type="paper", arxiv=None if aid == M.BAUDRY["id"] else aid, title=m["title"], authors=[a.strip() for a in m["authors"]],
                      date=m["published"], venue=p["venue"], library=p["folder"], status=p["status"], relevance=p["relevance"],
                      overlap_risk=p["overlap"] or None, review_tier=p["tier"], note=f"notes/{p['slug']}.md", pdf=p["pdf"]))
    for h in p["hyp"]:
        if h != "BG":
            edges.append(dict(source=p["slug"], target=f"PROJECT:{h}", type="addresses", basis="CLASSIFIED"))
    for c in p["concepts"]:
        edges.append(dict(source=p["slug"], target="CONCEPT:" + c, type="about", basis="KEYWORD"))
for k, v in R.PROJECT_NODES.items():
    nodes.append(dict(id=k, type="hypothesis", title=v))
for c in R.CONCEPTS:
    nodes.append(dict(id="CONCEPT:" + c, type="concept", title=c))
for a, b in cites["edges"]:
    if a in papers and b in papers:
        edges.append(dict(source=papers[a]["slug"], target=papers[b]["slug"], type="cites", basis="SEMANTIC_SCHOLAR"))
for s, kind, d, note, basis in R.RELATIONS:
    edges.append(dict(source=papers[s]["slug"], target=d if d.startswith("PROJECT") else papers[d]["slug"], type=kind, note=note, basis=basis))
os.makedirs(ROOT + "/graph", exist_ok=True)
json.dump(dict(nodes=nodes, edges=edges), open(ROOT + "/graph/papers.json", "w"), indent=1, ensure_ascii=False)

# ------------------------------------------------------------------ tiers
rows = [p for p in papers.values() if p["status"] == "new" or p["id"] in R.DEEP]
order = {"full review": 0, "skim": 1, "cite only": 2, "not triaged": 3, "already cited": 4}
out = ["# Tiers after triage", "",
       "Built from the six triage batches in `reviews/triage/`. Relevance and overlap risk are the triage agents' "
       "judgements after reading each paper's main text; overlap risk is how much of what the project planned to claim the paper already contains.", ""]
for tier, blurb in (("full review", "Tier 1 — full review in `reviews/deep/`"), ("skim", "Tier 2 — cite with a paragraph; the triage note is enough"),
                    ("cite only", "Tier 3 — cite only")):
    sel = sorted([p for p in rows if p["tier"] == tier], key=lambda p: (-(p["relevance"] or 0), p["id"]))
    out += [f"## {blurb} ({len(sel)})", "", "| Paper | Hyp. | Rel. | Overlap | Triage said | Note |", "|---|---|:-:|---|---|---|"]
    for p in sel:
        rec = (p.get("triage", {}).get("Recommendation", "") or ("not triaged; added after a triage agent flagged it" if p["status"] == "new" else "cited core paper, re-reviewed after triage")).replace("\n", " ")
        rec = re.sub(r"^\W*(DEEP|SKIM|CITE-ONLY|DROP)\W*", "", rec)
        out.append(f"| {authors(p['id'])}, *{meta[p['id']]['title'][:80]}* | {', '.join(p['hyp'])} | {p['relevance'] or '–'} | {p['overlap'] or '–'} | {p['triage_rec'] or '–'} | {rec[:260]} — [note](../notes/{p['slug']}.md) |")
    out.append("")
open(ROOT + "/reviews/TIERS.md", "w").write("\n".join(out))

n_tri = sum("triage" in p for p in papers.values())
print(f"notes: {len(papers)} | with triage: {n_tri} | with full review: {sum('deep_file' in p for p in papers.values())} | "
      f"graph: {len(nodes)} nodes, {len(edges)} edges ({sum(e['type'] == 'cites' for e in edges)} citation) | "
      f"tiers: " + ", ".join(f"{k}={sum(p['tier'] == k for p in rows)}" for k in order if any(p['tier'] == k for p in rows)))
print("new papers without triage:", [p["id"] for p in papers.values() if p["status"] == "new" and "triage" not in p])
