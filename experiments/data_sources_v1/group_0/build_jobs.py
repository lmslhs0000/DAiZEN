"""Combine china_wholesale group_0..2 mapping.csv with our 677 China series; list CPCA line slugs by volume -> jobs.csv"""
import re, pandas as pd
from pathlib import Path
H = Path(__file__).resolve().parent
B = Path(r"D:\daizen\experiments\data_audit_v1\china_wholesale")
s = pd.read_csv(H / "series.csv", encoding="utf-8-sig")
ms = []
for g in range(3):
    m = pd.read_csv(B / f"group_{g}/mapping.csv", encoding="utf-8-sig"); m["map_group"] = g; ms.append(m)
m = pd.concat(ms)
SL = re.compile(r"([a-z0-9][a-z0-9\-]*-\d{1,5})(?![\d])")
m["slugs"] = m.cpca_lines.fillna("").map(lambda x: " ".join(dict.fromkeys(SL.findall(x))))
j = s.merge(m, on=["Brand", "Model"], how="left")
j.to_csv(H / "map_joined.csv", index=False, encoding="utf-8-sig")
rows = []
for _, r in j.sort_values("sales_last12", ascending=False).iterrows():
    for sl in str(r.slugs).split() if isinstance(r.slugs, str) else []:
        rows.append(dict(slug=sl, Brand=r.Brand, Model=r.Model, sales_last12=r.sales_last12, confidence=r.confidence))
jb = pd.DataFrame(rows).drop_duplicates("slug")
jb.to_csv(H / "jobs.csv", index=False, encoding="utf-8-sig")
print(len(j), j.slugs.notna().sum(), (j.slugs.fillna("") == "").sum(), len(jb))
print(j[(j.slugs.fillna("") == "") & j.cpca_lines.notna()][["Brand", "Model", "cpca_lines", "confidence"]].to_string())
