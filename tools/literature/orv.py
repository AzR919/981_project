"""Confirm venues on OpenReview by exact-title match. Prints id | venues found."""
import json, sys, time, urllib.parse, urllib.request, re
sp = sys.argv[1]
m = json.load(open(sp + "/cands_meta.json"))
m.update(json.load(open(sp + "/seeds_meta.json")))
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
out = {}
for k in sys.argv[2:]:
    t = m[k]["title"]
    url = "https://api2.openreview.net/notes/search?type=terms&content=title&limit=10&term=" + urllib.parse.quote(t[:120])
    ven = []
    for attempt in range(4):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=40))
            for n in d.get("notes", []):
                c = n.get("content", {})
                if norm(c.get("title", {}).get("value", "")) == norm(t):
                    ven.append(f"{c.get('venue', {}).get('value', '?')} [openreview.net/forum?id={n.get('forum')}]")
            break
        except Exception as e:
            time.sleep(5)
    out[k] = ven
    print(k, "|", t[:60], "|", m[k].get("s2_venue", "")[:40], "|", m[k]["comment"][:90], "|", m[k]["journal_ref"][:60], "\n     OR:", "; ".join(ven) or "-")
    time.sleep(1.2)
json.dump(out, open(sp + "/orv.json", "w"), indent=1)
