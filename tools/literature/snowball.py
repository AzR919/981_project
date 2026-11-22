"""Citation snowball over the core seeds via the Semantic Scholar Graph API.

Writes snowball.json: every paper that cites or is referenced by a core seed,
with the list of seeds it is connected to.
"""
import json
import sys
import time
import urllib.error
import urllib.request

OUT = sys.argv[1]
CORE = {
    "2503.06639": "Mroueh",
    "2510.08539": "SukDuan",
    "2510.23049": "Thrampoulidis",
    "2510.13651": "DavisRecht",
    "2602.11128": "Heckel",
    "2611.14467": "Huang",
    "2503.20783": "DrGRPO",
    "2505.17508": "ZhangKL",
    "2601.23135": "Ge",
    "2605.07689": "Nie",
    "2603.01162": "ZhouU",
    "2601.04411": "Rad",
}
FIELDS = "title,year,venue,publicationVenue,externalIds,citationCount,publicationDate,authors"
BASE = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{}/{}?fields={}&limit=1000&offset={}"


def get(url):
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(4 + 4 * attempt)
        except Exception:
            time.sleep(4 + 4 * attempt)
    return None


papers = {}
status = {}
for aid, name in CORE.items():
    for kind, key in (("citations", "citingPaper"), ("references", "citedPaper")):
        offset, n = 0, 0
        while True:
            data = get(BASE.format(aid, kind, FIELDS, offset))
            if not data or not data.get("data"):
                break
            for row in data["data"]:
                p = row.get(key) or {}
                pid = p.get("paperId")
                if not pid:
                    continue
                e = papers.setdefault(pid, {**p, "cites": [], "cited_by": []})
                e["cites" if kind == "citations" else "cited_by"].append(name)
                n += 1
            if "next" not in data:
                break
            offset = data["next"]
            time.sleep(1.5)
        status[f"{name}.{kind}"] = n
        time.sleep(1.5)

json.dump(list(papers.values()), open(OUT, "w"))
print(json.dumps(status, indent=0))
print("total unique", len(papers))
