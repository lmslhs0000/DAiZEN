"""Fetch the Gasgoo CPCA model page for <year>/<month> for every slug in jobs.csv (largest series first).
Public pages only (robots.txt allows /autosales/), >=2.5 s between requests, stop on HTTP 429, cache in cache_pages/.
usage: python fetch_latest.py 2026 8  -> latest_<y>_<m>.csv (slug, name, p1,p2,p3 periods, v1,v2,v3, ytd, status, url)"""
import sys, re, time, urllib.request, urllib.error
from pathlib import Path
import pandas as pd
sys.path.insert(0, r"D:\daizen\experiments\data_audit_v1\china_wholesale\group_0")
import gg  # parser only
H = Path(__file__).resolve().parent; C = H / "cache_pages"; C.mkdir(exist_ok=True)
y, m = int(sys.argv[1]), int(sys.argv[2])
jobs = pd.read_csv(H / "jobs.csv", encoding="utf-8-sig")
out = H / f"latest_{y}_{m}.csv"
last = [0.0]
def get(url, f):
    if f.exists() and f.stat().st_size > 1000:
        return f.read_text(encoding="utf-8"), "cache"
    w = 2.5 - (time.time() - last[0])
    if w > 0: time.sleep(w)
    last[0] = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            h = r.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("HTTP 429 - stop", url, flush=True); raise SystemExit(2)
        return "", f"HTTP{e.code}"
    except Exception as e:
        return "", type(e).__name__
    f.write_text(h, encoding="utf-8")
    return h, "live"
rows = []
for i, r in jobs.iterrows():
    url = f"{gg.BASE}/autosales/model/{r.slug}/{y}/{m}.html"
    h, st = get(url, C / f"m_{r.slug}_{y}_{m}.html")
    p = gg.parse(h) if h else dict(periods=[], rows=[], title_name="")
    row0 = p["rows"][0] if p["rows"] else ("", [None] * 3, None)
    pr = p["periods"] + [""] * (3 - len(p["periods"]))
    rows.append(dict(slug=r.slug, title_name=p["title_name"], row_name=row0[0], p1=pr[0], p2=pr[1], p3=pr[2],
                     v1=row0[1][0], v2=row0[1][1], v3=row0[1][2], ytd=row0[2], status=st if p["rows"] else (st + ";no_data"), url=url))
    if i % 25 == 0:
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig"); print(i, r.slug, st, row0[1], flush=True)
pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig"); print("done", len(rows), flush=True)
