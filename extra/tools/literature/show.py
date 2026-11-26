import json, sys
sp, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = json.load(open(sp + "/cands_meta.json"))
ids = open(sp + "/cands.txt").read().split()
for k in ids[lo:hi]:
    v = m.get(k, {})
    if "title" not in v: print(k, "NO ARXIV"); continue
    au = v["authors"]
    ven = (v.get("s2_venue") or "").replace("arXiv.org", "")[:24]
    print(f"{k}|{v['published'][2:7]}|{v['title'][:74]}|{au[0].split()[-1] if au else ''}+{len(au)-1}|c{v.get('cites')}|{ven}|{v['comment'][:48]}|{v['journal_ref'][:24]}\n  ~{v['abstract'][170:360]}")
