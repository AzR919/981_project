"""Download the PDFs in manifest.py, check them, and write download_log.json.

usage: build.py <scratchpad> <papers_dir>
"""
import json
import os
import re
import subprocess
import sys
import time

sp, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, sp)
import manifest as M  # noqa: E402

meta = json.load(open(sp + "/cands_meta.json"))
meta.update(json.load(open(sp + "/seeds_meta.json")))
meta[M.BAUDRY["id"]] = M.BAUDRY

GROUPS = [("tierA_top_venue", M.TIER_A), ("tierB_preprints", M.TIER_B), ("cited_core", M.CITED)]
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).split()
log = {}

for folder, rows in GROUPS:
    os.makedirs(os.path.join(out, folder), exist_ok=True)
    for row in rows:
        aid, author, short = row[0], row[1], row[2]
        m = meta[aid]
        year = m["published"][:4]
        tag = "NeurIPS2025" if aid == M.BAUDRY["id"] else aid
        fname = f"{year}_{author}_{short}_{tag}.pdf"
        path = os.path.join(out, folder, fname)
        url = m.get("pdf") or f"https://arxiv.org/pdf/{aid}"
        ok = os.path.exists(path) and open(path, "rb").read(5) == b"%PDF-"
        for attempt in range(4):
            if ok:
                break
            subprocess.run(["curl", "-sSL", "-m", "180", "-A", "Mozilla/5.0", "-o", path, url])
            ok = os.path.exists(path) and open(path, "rb").read(5) == b"%PDF-"
            time.sleep(3 + 6 * attempt)
        entry = {"file": f"{folder}/{fname}", "ok": ok, "size": os.path.getsize(path) if os.path.exists(path) else 0}
        if ok:
            txt = subprocess.run(["pdftotext", "-l", "2", path, "-"], capture_output=True, text=True, errors="ignore").stdout
            words = [w for w in norm(m["title"]) if len(w) > 3][:6]
            page = " ".join(norm(txt))
            entry["title_hits"] = f"{sum(w in page for w in words)}/{len(words)}"
            entry["pages"] = subprocess.run(["pdfinfo", path], capture_output=True, text=True, errors="ignore").stdout.partition("Pages:")[2].split("\n")[0].strip()
        elif os.path.exists(path):
            os.remove(path)
        log[aid] = entry
        print(aid, entry, flush=True)

json.dump(log, open(sp + "/download_log.json", "w"), indent=1)
bad = [k for k, v in log.items() if not v["ok"]]
print("DONE", len(log), "failed:", bad)
