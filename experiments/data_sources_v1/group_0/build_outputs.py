"""Build coverage.csv for group_0 (China, 677 series) from map_joined.csv + latest_2026_8.csv (Gasgoo model pages)."""
import pandas as pd, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent
j = pd.read_csv(H / "map_joined.csv", encoding="utf-8-sig")
lat = pd.read_csv(H / "latest_parsed_2026_8.csv", encoding="utf-8-sig").set_index("slug")
raw = pd.read_csv(r"D:\daizen\raw\data.csv", encoding="utf-8-sig")
raw = raw[(raw.Country == "China") & (raw.Month == "2026-07")].set_index(["Brand", "Model"]).Sales
B = Path(r"D:\daizen\experiments\data_audit_v1\china_wholesale")
um = pd.concat([pd.read_csv(B / f"group_{g}/unmatched.csv", encoding="utf-8-sig") for g in range(3)]).set_index(["Brand", "Model"]).reason
# JV / foreign brands whose Gasgoo brand article has a text table (alternative route)
out = []
for _, r in j.iterrows():
    key = (r.Brand, r.Model)
    our_jul = raw.get(key, np.nan)
    slugs = r.slugs.split() if isinstance(r.slugs, str) and r.slugs.strip() else []
    conf_map = r.confidence if isinstance(r.confidence, str) else "none"
    d = dict(Country="China", Brand=r.Brand, Model=r.Model, sales_last12=r.sales_last12)
    if not slugs or conf_map in ("low", "none"):
        why = um.get(key, "no CPCA/Gasgoo line in mapping")
        d.update(source_id="none", frequency="", needs_split="", match_name=str(r.cpca_lines) if isinstance(r.cpca_lines, str) else "",
                 confidence="low", automatable="no", note=f"not mapped to a CPCA line: {why}")
        out.append(d); continue
    got = [s for s in slugs if s in lat.index and lat.loc[s, "status"] != "not_fetched"]
    parsed = [s for s in got if str(lat.loc[s, "status"]).startswith("ok")]
    p1 = {s: lat.loc[s, "p1"] for s in parsed}
    aug = [lat.loc[s, "v1"] for s in parsed if lat.loc[s, "status"] == "ok" and lat.loc[s, "p1"] == "2026-08"]
    jul = [lat.loc[s, "v2"] for s in parsed if lat.loc[s, "status"] == "ok" and lat.loc[s, "p2"] == "2026-07"]
    aug_sum = np.nansum(aug) if aug and not all(pd.isna(aug)) else np.nan
    jul_sum = np.nansum(jul) if jul and not all(pd.isna(jul)) else np.nan
    names = " + ".join(f"{lat.loc[s,'title_name'] if s in got and isinstance(lat.loc[s,'title_name'], str) else '?'} [{s}]" for s in slugs)
    notes = []
    if len(got) < len(slugs): notes.append(f"{len(slugs)-len(got)} line(s) not fetched")
    missing = [s for s in got if s not in parsed]
    if missing: notes.append("no table on 2026-08 page: " + ",".join(missing))
    quiet = [s for s in parsed if lat.loc[s, "status"] != "ok"]
    if quiet: notes.append("line(s) with no value Jun-Aug 2026: " + ",".join(quiet))
    jul_match = (not pd.isna(our_jul)) and (not pd.isna(jul_sum)) and abs(our_jul - jul_sum) <= 1
    if pd.isna(our_jul) and pd.isna(jul_sum): jul_txt = "no 2026-07 value in ours or CPCA"
    else: jul_txt = f"2026-07 CPCA {jul_sum if not pd.isna(jul_sum) else '-'} vs ours {our_jul if not pd.isna(our_jul) else '-'}" + (" (equal)" if jul_match else "")
    notes.append(f"2026-08 read = {aug_sum if not pd.isna(aug_sum) else '-'}; {jul_txt}")
    ok_pages = len(parsed) == len(slugs)
    ratio = (jul_sum / our_jul) if (not pd.isna(our_jul) and not pd.isna(jul_sum) and our_jul > 0) else np.nan
    if ok_pages:
        auto = "yes"
        if jul_match:
            conf = "high"
        elif not pd.isna(ratio) and 0.5 <= ratio <= 2:
            conf = "medium"; notes.append("2026-07 not equal to ours (ratio %.2f) - line set to recheck" % ratio)
        elif not pd.isna(ratio):
            conf, auto = "low", "partial"; notes.append("2026-07 level differs >2x from ours (ratio %.2f) - mapping/basis to recheck before use" % ratio)
        elif pd.isna(jul_sum) and not pd.isna(our_jul) and our_jul > 0:
            conf, auto = "low", "partial"; notes.append("CPCA line(s) show no value Jun-Aug 2026 while ours has 2026-07 sales - line probably replaced by a new slug; find it")
        elif not pd.isna(jul_sum) and jul_sum > 0 and (pd.isna(our_jul) or our_jul == 0):
            conf = "low"; notes.append("CPCA has 2026-07 value but ours is 0/blank - check")
        else:
            conf = {"high": "medium", "medium": "low"}.get(conf_map, "low")
    elif parsed:
        auto, conf = "partial", "low"
    else:
        auto, conf = "no", "low"
    if pd.isna(aug_sum) and ok_pages: notes.append("all lines '-' for 2026-08 (no sales or not reported)")
    d.update(source_id="CN_GASGOO_MODEL", frequency="monthly", needs_split="none", match_name=names, confidence=conf,
             automatable=auto, note=f"mapping {conf_map} (data_audit_v1/china_wholesale group_{int(r.map_group)}); " + "; ".join(notes))
    out.append(d)
c = pd.DataFrame(out).sort_values("sales_last12", ascending=False)
# manual split notes: one CPCA line covering two of our series
SPLIT = {("Volkswagen", "Tharu XR"): "tharu-1583 = Tharu + Tharu XR (CPCA has no separate XR line)",
         ("Volkswagen", "Tharu"): "tharu-1583 = Tharu + Tharu XR (CPCA has no separate XR line)",
         ("Volkswagen", "Teramont"): "teramont-1378 = Teramont + Teramont X", ("Volkswagen", "Teramont X"): "teramont-1378 = Teramont + Teramont X",
         ("Volkswagen", "Sagitar S"): "sagitar-582 = Sagitar + Sagitar S from 2026-04", ("Volkswagen", "Sagitar"): "sagitar-582 = Sagitar + Sagitar S from 2026-04"}
for k, v in SPLIT.items():
    m = (c.Brand == k[0]) & (c.Model == k[1])
    if m.any():
        c.loc[m, ["source_id", "frequency", "needs_split", "automatable", "confidence"]] = ["CN_GASGOO_MODEL", "monthly", "none", "partial", "low"]
        c.loc[m, "note"] = "CPCA line shared with another of our series: " + v + "; the line total is automatic but the split between our two series is not published (needs a share rule or a second source); " + c.loc[m, "note"].astype(str)
c.to_csv(H / "coverage.csv", index=False, encoding="utf-8-sig")
tot = c.sales_last12.sum()
print(len(c), tot)
print(c.groupby("automatable").sales_last12.agg(["count", "sum"]).assign(share=lambda x: x["sum"] / tot))
print(c.groupby(["automatable", "confidence"]).sales_last12.agg(["count", "sum"]).assign(share=lambda x: x["sum"] / tot))
