"""arXiv title search: print id | date | title | first author for each query."""
import sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
NS = {"a": "http://www.w3.org/2005/Atom"}
for q in sys.argv[1:]:
    url = "https://export.arxiv.org/api/query?max_results=4&search_query=" + urllib.parse.quote(q)
    for attempt in range(4):
        try:
            raw = urllib.request.urlopen(url, timeout=40).read(); break
        except Exception:
            raw = None; time.sleep(6)
    print("##", q)
    if raw:
        for e in ET.fromstring(raw).findall("a:entry", NS):
            au = e.find("a:author/a:name", NS)
            print("  ", e.findtext("a:id", "", NS).rsplit("/", 1)[-1], "|", e.findtext("a:published", "", NS)[:10], "|", " ".join(e.findtext("a:title", "", NS).split())[:110], "|", au.text if au is not None else "")
    time.sleep(3.2)
