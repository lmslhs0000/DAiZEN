"""Re-parse cached Gasgoo model pages (cache_pages/m_<slug>_<y>_<m>.html) -> latest_<y>_<m>.csv.
The line's own row is present only when the line had a value in the 3-month window; otherwise the first row is the OEM.
Line row = first row if 4 rows (line, OEM, segment, body type) or its name equals the page title name."""
import sys, re
from pathlib import Path
import pandas as pd
sys.path.insert(0, r"D:\daizen\experiments\data_audit_v1\china_wholesale\group_0")
import gg
H = Path(__file__).resolve().parent
y, m = int(sys.argv[1]), int(sys.argv[2])
jobs = pd.read_csv(H / "jobs.csv", encoding="utf-8-sig")
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
rows = []
for s in jobs.slug:
    f = H / "cache_pages" / f"m_{s}_{y}_{m}.html"
    url = f"{gg.BASE}/autosales/model/{s}/{y}/{m}.html"
    if not f.exists() or f.stat().st_size < 1000:
        rows.append(dict(slug=s, status="not_fetched", url=url)); continue
    p = gg.parse(f.read_text(encoding="utf-8"))
    pr = p["periods"] + [""] * (3 - len(p["periods"]))
    if not p["rows"]:
        st, line = "page_no_table", None
    elif len(p["rows"]) >= 4 or norm(p["rows"][0][0]) == norm(p["title_name"]):
        st, line = "ok", p["rows"][0]
    else:
        st, line = "ok_line_no_value_3m", None
    v = line[1] if line else [None] * 3
    rows.append(dict(slug=s, title_name=p["title_name"], row_name=line[0] if line else "", p1=pr[0], p2=pr[1], p3=pr[2],
                     v1=v[0], v2=v[1], v3=v[2], ytd=line[2] if line else None, oem=p["rows"][1 if line else 0][0] if p["rows"] else "",
                     status=st, url=url))
df = pd.DataFrame(rows); df.to_csv(H / f"latest_parsed_{y}_{m}.csv", index=False, encoding="utf-8-sig")
print(df.status.value_counts())
