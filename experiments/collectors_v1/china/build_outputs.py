"""Build mapping.csv (all 677 China series), proposed_fixes.csv, proposed_fixes_review.csv, no_source.csv.
Inputs: data_sources_v1/group_0 coverage.csv + map_joined.csv (current mapping), work/decisions.csv (this round's line
decisions), work/series_cpca.csv (from work/analyze.py: our values vs CPCA line sums per month), raw/data.csv (old_sales).
Rules (fixed before looking at the result totals):
  proposed_fixes.csv  (apply-ready): series whose line set is confirmed by numbers (>= 6 months where ours already equals
      the CPCA sum, or a decision with fix_policy=auto), month <= 2026-07, CPCA value read (page/article, status ok),
      no anomaly flag (isolated zero / spike / dip), and |ours - CPCA| > 1.
      For non-decision series only months still holding the pre-unification value (ours == data_before_wholesale) are
      proposed - i.e. the leftovers the 2026-10-10 merge skipped (series-year ratio filter / scope-mismatch years).
  proposed_fixes_review.csv: every other differing month (name-only mappings, flagged values, months where one line
      shows no positive value, Changan-style gappy 2026 reporting, line row absent = 'not positive' -> new_sales 0).
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
W = H / "work"
ROOT = Path(__file__).resolve().parents[3]   # 묶음 루트(어느 PC 든)
G0 = ROOT / "experiments" / "data_sources_v1" / "group_0"
cov = pd.read_csv(G0 / "coverage.csv", encoding="utf-8-sig")
mj = pd.read_csv(G0 / "map_joined.csv", encoding="utf-8-sig")
dec = pd.read_csv(W / "decisions.csv", encoding="utf-8-sig")
cases = pd.read_csv(W / "cases.csv", encoding="utf-8-sig")
S = pd.read_csv(W / "series_cpca.csv", encoding="utf-8-sig", dtype={"Month": str})
RAW = pd.read_csv(ROOT / "raw" / "data.csv", encoding="utf-8-sig", dtype={"Month": str})
RAW = RAW[RAW.Country == "China"].set_index(["Brand", "Model", "Month"]).Sales

NO_REASON = {
    ("Wuling", "Hongguang EV"): "CPCA 'Hongguang BEV' (id 2754) exists only from 2026-07 (1986 vs ours 2442); no line for 2025-01..2026-06",
    ("Wuling", "Sunshine EV"): "no CPCA line (name search + all new ids 2708-2797); Wuling microvan, likely outside CPCA passenger model list",
    ("Wuling", "Rongguang"): "no CPCA line (microvan; outside CPCA passenger model list)",
    ("Wuling", "Sunshine"): "no CPCA line (microvan; outside CPCA passenger model list)",
    ("Wuling", "Aishang"): "no CPCA line found (name + new-id search)",
    ("Avatr", "06"): "no CPCA line (Gasgoo has AVATR 07/07L/12/06T only; 06T BEV/REEV ids 2756/2757 start 2026-07 and are a different model)",
    ("Avatr", "11"): "no CPCA line (Gasgoo has AVATR 07/07L/12/06T only)",
    ("Karry", "Jiang Tun"): "no CPCA line (Karry light commercial)",
    ("Zeekr", "009"): "no CPCA line found (Gasgoo has ZEEKR X / 007GT / MIX / 'ZEEKR')",
    ("SRM", "Jinhaishi"): "no CPCA line (SRM light commercial van)",
    ("Chery", "C-DM"): "ambiguous name (C-DM is Chery's PHEV family, many lines); no single CPCA line matches by numbers",
    ("Ruichi", "EC75"): "no CPCA line (electric logistics van)",
    ("Geely", "A7 EV"): "galaxy-a7-2521 equals our Geely A7 EM exactly; Gasgoo has no separate A7 EV line",
    ("Hongqi", "Tiangong 05"): "no CPCA line (Gasgoo has Tiangong 06 and 08 only)",
    ("Honda", "S7"): "no CPCA line found (name + new-id search)",
    ("Yangwang", "U8 L"): "no separate CPCA line (Yangwang U8 id 2277 = our Yangwang U8)",
    ("Exeed", "RX C-DM"): "no matching CPCA line (Gasgoo has Exeed / Exeed TXL / Exeed VX only)",
    ("Exeed", "LX"): "exeed-1845 'Exeed' does not match (2 of 48 months equal, CPCA = 4x ours); no LX line",
    ("Exeed", "VX C-DM"): "no matching CPCA line",
    ("Exeed", "RX"): "no matching CPCA line",
    ("Exeed", "LX C-DM"): "no matching CPCA line",
    ("Baojun", "Xiangjing"): "no CPCA line found (name + new-id search)",
    ("Baojun", "Yep"): "no CPCA line found (name + new-id search)",
    ("Forthing", "Leiting"): "no CPCA line found",
    ("JAC", "e3"): "e3-1737 is BYD e3 (OEM BYD Auto on the page); no JAC e3 line",
    ("BAW", "Yongshi"): "no CPCA line found",
    ("NIO", "ET9"): "no CPCA line found (Gasgoo NIO lines: ES9 id 2722 is a different model)",
    ("Skyworth", "HT-i"): "no CPCA line found",
    ("Nissan", "Paladin"): "no CPCA line found (pickup/SUV outside list)",
    ("Nissan", "Venucia Star"): "no CPCA line found (Venucia lines on Gasgoo: D60, VX6, Da V PHEV, 'Venucia')",
    ("BYD", "M9"): "no CPCA line found",
    ("BYD", "e9"): "no CPCA line found",
    ("Maxus", "Territory"): "no CPCA line found",
    ("Isuzu", "mu-X"): "no CPCA line (pickup-based SUV, commercial list)",
    ("Changan", "Plus"): "ambiguous name; no matching CPCA line",
    ("JAC", "X8 E"): "only 'Jianghuai X8 PLUS' (ICE, 3-7k/month) exists; no X8 E line",
    ("Genesis", "G90"): "g90-2064 is SAIC-Maxus G90; no Genesis line",
    ("DS", "3 EV"): "no CPCA line; our values are 0",
    ("Hyundai", "ix35"): "CPCA line hyundai-ix35-31 has no value since at least 2024-07 and Hyundai brand article 2026-06 has no ix35 row; our values since then are pre-unification (16888 retail)",
    ("Cowin", "Xuanjie"): "CPCA line showjet-1801 has no value since at least 2024-07 (Showjet EV id 1976 is a different, larger line); our values are 16888 retail",
}


def key_of(slugs):
    if not isinstance(slugs, str) or not slugs.strip():
        return ""
    return "+".join(s for s in re.split(r"[,\s+]+", slugs) if s)


def build_mapping():
    rows = []
    d = dec.set_index(["Brand", "Model"])
    cs = cases.set_index(["Brand", "Model"])
    for _, r in mj.iterrows():
        k = (r.Brand, r.Model)
        c = cov[(cov.Brand == r.Brand) & (cov.Model == r.Model)].iloc[0]
        if k in d.index:
            x = d.loc[k]
            rows.append(dict(Country="China", Brand=r.Brand, Model=r.Model, source_id="CN_GASGOO_MODEL", source_key=x.source_key,
                             rule=x.rule, confidence=x.confidence, note="collectors_v1 decision: " + x.note))
            continue
        key = key_of(r.slugs)
        if k in cs.index:
            case = cs.loc[k, "case"]
            if case == "no":
                rows.append(dict(Country="China", Brand=r.Brand, Model=r.Model, source_id="", source_key="", rule="none",
                                 confidence="none", note="no CPCA source: " + NO_REASON.get(k, "no CPCA line found")))
                continue
            if k in NO_REASON:
                rows.append(dict(Country="China", Brand=r.Brand, Model=r.Model, source_id="", source_key="", rule="none",
                                 confidence="none", note="no current CPCA source: " + NO_REASON[k] + f" (old line {key})"))
                continue
            note = {"level2x": "lines unchanged; 2026 (and other leftover) months of ours are still pre-unification values -> see proposed_fixes",
                    "dash": "lines unchanged; the 2026-08 page only lacked an August value (a page drops the line row when the queried month is 0/-); see validation"}.get(case, "")
            rows.append(dict(Country="China", Brand=r.Brand, Model=r.Model, source_id="CN_GASGOO_MODEL", source_key=key, rule="sum",
                             confidence="medium", note=f"{note}; was {case} in data_sources_v1"))
            continue
        conf = "high" if c.confidence == "high" else ("medium" if c.confidence == "medium" else "low")
        rows.append(dict(Country="China", Brand=r.Brand, Model=r.Model, source_id="CN_GASGOO_MODEL" if key else "", source_key=key,
                         rule="sum" if key else "none", confidence=conf if key else "none",
                         note=("data_sources_v1: " + str(c.note))[:400]))
    return pd.DataFrame(rows)


def classify(M):
    dd = dec.set_index(["Brand", "Model"])
    nosrc = set(map(tuple, M[M.rule == "none"][["Brand", "Model"]].values))
    prop, rev = [], []
    for (b, m), g in S.groupby(["Brand", "Model"]):
        g = g.sort_values("Month")
        g = g[g.Month <= "2026-07"]
        pol = dd.loc[(b, m), "fix_policy"] if (b, m) in dd.index else "default"
        conf = dd.loc[(b, m), "confidence"] if (b, m) in dd.index else "medium"
        dnote = (" - " + dd.loc[(b, m), "note"]) if (b, m) in dd.index else ""
        if pol == "none" or (b, m) in nosrc:
            continue
        eq = ((g.ours - g.cpca).abs() <= 1) & g.cpca.notna()
        confirmed = eq.sum() >= 6 or pol == "auto"
        # Changan-style gappy 2026 reporting: a '-'/not-positive month between positive months while ours > 0
        g26 = g[g.Month >= "2026-01"]
        npm = g26.comp_status.astype(str).str.match(r"^(not_positive|dash)").values
        cpv = g26.cpca.fillna(0).values
        flag26 = g26.flag.notna().any() if "flag" in g26 else False
        gappy = flag26 or any(npm[i] and cpv[:i].max(initial=0) >= 500 and cpv[i + 1:].max(initial=0) >= 500 for i in range(len(npm)))
        for _, r in g.iterrows():
            old = RAW.get((b, m, r.Month), np.nan)
            base = dict(Country="China", Month=r.Month, Brand=b, Model=m, old_sales=old)
            st = str(r.comp_status)
            if pd.isna(r.cpca):
                if st.startswith(("not_positive", "dash")) and r.ours > 0:
                    rev.append(dict(base, new_sales=0, reason=f"CPCA line(s) {r.source_key}: no positive value shown for this month (row absent / '-'); our value is not CPCA-basis. Inferred 0 - REVIEW",
                                    evidence=r.urls, tier="review", why="not_positive"))
                continue
            if abs(r.ours - r.cpca) <= 1:
                continue
            leftover = pd.notna(r.orig) and abs(r.ours - r.orig) <= 0.5
            flag = r.flag if isinstance(r.flag, str) else ""
            row = dict(base, new_sales=int(round(r.cpca)),
                       reason=f"CPCA wholesale (Gasgoo) line sum {r.source_key} replaces non-CPCA value [China basis unified to CPCA wholesale, 2026-10-10; collectors_v1 mapping {conf}]{dnote}",
                       evidence=r.urls)
            ok_status = st in ("ok",)
            if pol != "review" and confirmed and ok_status and not flag and not (gappy and r.Month >= "2026-01") and conf in ("high", "medium") and ((leftover and r.Month >= "2023-01") or pol == "auto"):
                prop.append(dict(row, tier="proposed", why="auto" if pol == "auto" else "leftover"))
            else:
                why = []
                if pol == "review": why.append("decision fix_policy=review")
                if not confirmed: why.append("mapping not confirmed by >=6 equal months")
                if conf == "low": why.append("name-only mapping")
                if not ok_status: why.append("status " + st[:40])
                if flag: why.append("flag " + flag)
                if gappy and r.Month >= "2026-01": why.append("2026 CPCA values of this line are gappy/flagged (a '-' month between positive months, or an isolated zero / spike / dip in 2026)")
                if not leftover and pol != "auto": why.append("month was already converted by the 2026-10-10 merge (CPCA revision or line change)")
                if leftover and r.Month < "2023-01" and pol != "auto": why.append("pre-2023 month (16888 was CPCA then): a difference means a scope/mapping difference, not a basis leftover")
                rev.append(dict(row, tier="review", why="; ".join(why)))
    cols = ["Country", "Month", "Brand", "Model", "old_sales", "new_sales", "reason", "evidence"]
    P = pd.DataFrame(prop)
    R = pd.DataFrame(rev)
    return P, R, cols


if __name__ == "__main__":
    M = build_mapping()
    assert len(M) == 677 and not M.duplicated(["Brand", "Model"]).any()
    M.to_csv(H / "mapping.csv", index=False, encoding="utf-8-sig")
    P, R, cols = classify(M)
    P[cols].to_csv(H / "proposed_fixes.csv", index=False, encoding="utf-8-sig")
    R[cols + ["why"]].to_csv(H / "proposed_fixes_review.csv", index=False, encoding="utf-8-sig")
    print(M.confidence.value_counts().to_string())
    print(M.rule.str.split(":").str[0].value_counts().to_string())
    print("proposed", len(P), "series", P.groupby(["Brand", "Model"]).ngroups if len(P) else 0,
          "net", (P.new_sales - P.old_sales).sum() if len(P) else 0)
    print("review", len(R), "series", R.groupby(["Brand", "Model"]).ngroups if len(R) else 0)
    if len(P):
        print(P.assign(d=P.new_sales - P.old_sales).groupby(["Brand", "Model"]).agg(n=("d", "size"), d=("d", "sum"), why=("why", "first")).sort_values("d").to_string())
