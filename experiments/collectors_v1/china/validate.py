"""Validation: run the collector for 2026-05..2026-07 (pages up to 2026-08 allowed, cache first, polite live fetch for
missing pages) for all 677 mapped China series and compare with raw/data.csv.
Output: validation.csv (Country, Brand, Model, Month, ours, collected, diff, match, status, flags, confidence, rule,
source_key, sales_last12) and a printed summary (series counts and volume shares)."""
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]   # 묶음 루트(어느 PC 든)
sys.path.insert(0, str(HERE))
import gasgoo
from collect_china import collect

MONTHS = ["2026-05", "2026-06", "2026-07"]
mp = pd.read_csv(HERE / "mapping.csv", encoding="utf-8-sig")
sr = pd.read_csv(HERE / "split_rules.csv", encoding="utf-8-sig")
fetch = "--no-fetch" not in sys.argv
S, Lines = collect(2026, 7, mp, sr, fetch=fetch, upto="2026-08", months=MONTHS)
raw = pd.read_csv(ROOT / "raw" / "data.csv", encoding="utf-8-sig", dtype={"Month": str})
raw = raw[raw.Country == "China"][["Brand", "Model", "Month", "Sales"]].rename(columns={"Sales": "ours"})
cov = pd.read_csv(ROOT / "experiments" / "data_sources_v1" / "group_0" / "coverage.csv", encoding="utf-8-sig")[["Brand", "Model", "sales_last12"]]
V = S.merge(raw, on=["Brand", "Model", "Month"], how="left").merge(cov, on=["Brand", "Model"], how="left")
V = V.rename(columns={"value": "collected"})


def match(r):
    if r.status == "no_source":
        return "no_source"
    if pd.isna(r.collected):
        if r.status == "not_positive":
            return "equal_zero" if r.ours == 0 else "collected_not_positive"
        return "no_value"
    if "allocation" in str(r.status):
        return "allocation"
    d = abs(r.collected - r.ours)
    if d <= 1:
        return "equal"
    if d <= max(1, 0.02 * r.ours):
        return "within_2pct"
    return "differs"


V["diff"] = V.collected - V.ours
V["match"] = V.apply(match, axis=1)
cols = ["Country", "Brand", "Model", "Month", "ours", "collected", "diff", "match", "status", "flags", "confidence", "rule",
        "source_key", "sales_last12", "urls"]
V = V.sort_values(["sales_last12", "Brand", "Model", "Month"], ascending=[False, True, True, True])
V[cols].to_csv(HERE / "validation.csv", index=False, encoding="utf-8-sig")
Lines.to_csv(HERE / "work" / "validation_lines.csv", index=False, encoding="utf-8-sig")

tot = cov.sales_last12.sum()
ser = V.groupby(["Brand", "Model"]).agg(sales=("sales_last12", "first"), m=("match", lambda x: ",".join(sorted(set(x)))))


def cls(m):
    s = set(m.split(","))
    if s <= {"equal", "equal_zero"}:
        return "all 3 months equal"
    if s <= {"equal", "equal_zero", "within_2pct"}:
        return "equal or within 2%"
    if "no_source" in s:
        return "no source"
    if s == {"allocation"}:
        return "allocation (shared line)"
    return "some month differs / no value"


ser["cls"] = ser.m.map(cls)
summ = ser.groupby("cls").agg(series=("sales", "size"), sales=("sales", "sum"))
summ["share_pct"] = (100 * summ.sales / tot).round(2)
print(summ.to_string())
print(V.match.value_counts().to_string())
print("requests:", gasgoo.STATS)
summ.to_csv(HERE / "work" / "validation_summary.csv", encoding="utf-8-sig")
