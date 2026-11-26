"""Citation edges among library papers, from Semantic Scholar reference lists."""
import json, sys, time, urllib.request, urllib.error
sp = sys.argv[1]
sys.path.insert(0, sp)
import manifest as M
ids = [r[0] for r in M.TIER_A + M.TIER_B + M.CITED if r[0][0].isdigit() and r[0] != "0000.baudry"] + ["2607.01083"]
lib = set(ids)
edges, status = [], {}
for aid in ids:
    refs, offset, ok = [], 0, False
    while True:
        url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{aid}/references?fields=externalIds,title&limit=1000&offset={offset}"
        data = None
        for attempt in range(7):
            try:
                data = json.load(urllib.request.urlopen(url, timeout=40)); break
            except urllib.error.HTTPError as e:
                if e.code == 404: break
                time.sleep(4 + 4 * attempt)
            except Exception:
                time.sleep(4 + 4 * attempt)
        if not data or data.get("data") is None: break
        ok = True
        for row in data["data"]:
            p = row.get("citedPaper") or {}
            ax = (p.get("externalIds") or {}).get("ArXiv")
            t = (p.get("title") or "").lower()
            if ax in lib: refs.append(ax)
            elif "does stochastic gradient really succeed for bandits" in t: refs.append("0000.baudry")
        if "next" not in data: break
        offset = data["next"]; time.sleep(1.2)
    status[aid] = (ok, len(set(refs)))
    edges += [(aid, r) for r in sorted(set(refs)) if r != aid]
    time.sleep(1.3)
json.dump({"edges": edges, "status": status}, open(sp + "/cite_edges.json", "w"))
print("papers", len(ids), "with data", sum(s[0] for s in status.values()), "edges", len(edges), "no data:", [k for k, s in status.items() if not s[0]])
