"""Look up arXiv IDs: title, authors, date, comment, journal_ref (arXiv API)
plus venue and citation count (Semantic Scholar batch API).

usage: verify.py out.json id1 id2 ...   (or ids from a file with @path)
"""
import json
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


def fetch(url, data=None, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            time.sleep(5 + 5 * i)
    return None


def arxiv_meta(ids):
    out = {}
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        raw = fetch("https://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(chunk))
        if not raw:
            continue
        for e in ET.fromstring(raw).findall("a:entry", NS):
            eid = e.findtext("a:id", "", NS).rsplit("/", 1)[-1].split("v")[0]
            title = e.findtext("a:title", None, NS)
            if not title:
                continue
            out[eid] = {
                "id": eid,
                "title": " ".join(title.split()),
                "authors": [a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)],
                "published": e.findtext("a:published", "", NS)[:10],
                "comment": " ".join((e.findtext("x:comment", "", NS) or "").split()),
                "journal_ref": e.findtext("x:journal_ref", "", NS) or "",
                "abstract": " ".join(e.findtext("a:summary", "", NS).split()),
                "cat": (e.find("x:primary_category", NS).get("term") if e.find("x:primary_category", NS) is not None else ""),
            }
        time.sleep(3)
    return out


def s2_meta(ids):
    out = {}
    url = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,venue,publicationVenue,citationCount,externalIds"
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        raw = fetch(url, json.dumps({"ids": ["arXiv:" + x for x in chunk]}).encode())
        if not raw:
            continue
        for aid, p in zip(chunk, json.loads(raw)):
            if p:
                out[aid] = {"s2_venue": p.get("venue") or "", "cites": p.get("citationCount"),
                            "doi": (p.get("externalIds") or {}).get("DOI", "")}
        time.sleep(3)
    return out


if __name__ == "__main__":
    out_path, args = sys.argv[1], sys.argv[2:]
    ids = []
    for a in args:
        ids += open(a[1:]).read().split() if a.startswith("@") else [a]
    ids = list(dict.fromkeys(ids))
    meta = arxiv_meta(ids)
    s2 = s2_meta(ids)
    for k, v in s2.items():
        meta.setdefault(k, {"id": k}).update(v)
    missing = [i for i in ids if i not in meta or "title" not in meta[i]]
    json.dump(meta, open(out_path, "w"), indent=1)
    for i in ids:
        m = meta.get(i, {})
        if "title" not in m:
            continue
        au = m["authors"]
        print(f"{i} | {m['published']} | {m['title'][:95]} | {au[0] if au else ''}{' +' + str(len(au) - 1) if len(au) > 1 else ''} | c={m.get('cites')} | S2:{m.get('s2_venue', '')[:30]} | {m['comment'][:110]} | {m['journal_ref'][:50]}")
    print("MISSING:", missing)
