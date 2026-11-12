"""우리 Europe 시계열 ↔ 공식 등록(EEA EU27+NO+IS + 영국 DfT + 스위스 auto-schweiz) 비교, 나라별 몫 표.

출력
  eea_shares_2025.csv : 차종 × 나라 2025 등록 대수와 그 차종 유럽 합계 안의 몫(완전도 보정 포함).
                        월별 자료가 없는 나라 몫을 추정할 때 쓴다(아래 'coverage' 열 = 그 나라 자료 종류).
  validation.csv      : (1) europe_annual 2024·2025, (2) europe_quarter 2025 — 우리 값 / 공식 합계 비율
                        (3) out/validation_*.csv (collect_fr/it/pl 검증) 를 이어 붙임.

완전도 보정(2025): EEA 2025 잠정판은 나라별로 ACEA 연간 합계보다 적다(PL 74%, FI 68% 등).
  n_adj = n_raw × (ACEA 나라 합계 / 그 나라 EEA 합계). 영국은 ACEA/DfT, 스위스는 ACEA/auto-schweiz.
  차종별로 빠진 비율이 같다는 가정이므로 '추정'이다. raw 와 adj 를 둘 다 남긴다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import CACHE, HERE, RAW_DATA
from europe_map import TYPE_SPLIT_MAKES, classify, classify_frame, refine_by_type

DFT = HERE.parents[1] / "data_sources_v1" / "group_2" / "samples" / "dft_df_VEH0160_UK.csv"
ACEA = HERE.parents[1] / "data_sources_v1" / "europe_deep" / "acea_2025_totals.csv"
ISO = {"Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR", "Cyprus": "CY", "Czechia": "CZ",
       "Denmark": "DK", "Estonia": "EE", "Finland": "FI", "France": "FR", "Germany": "DE", "Greece": "GR",
       "Hungary": "HU", "Ireland": "IE", "Italy": "IT", "Latvia": "LV", "Lithuania": "LT", "Luxembourg": "LU",
       "Malta": "MT", "Netherlands": "NL", "Poland": "PL", "Portugal": "PT", "Romania": "RO", "Slovakia": "SK",
       "Slovenia": "SI", "Spain": "ES", "Sweden": "SE", "Iceland": "IS", "Norway": "NO", "Switzerland": "CH",
       "UnitedKingdom": "UK"}
# europe_deep/README.md 분류(2026-10-15): 나라별 차종 자료 종류
COVERAGE = {**{c: "monthly_full" for c in ["DE", "ES", "PL", "NL", "SE", "CH", "DK", "RO", "IE", "FI", "LU"]},
            "UK": "quarterly_full", "FR": "monthly_top100_ytd", "IT": "monthly_top50", "NO": "monthly_top30"}
# 승용(M1) 등록 자료와 정의가 다른 차종: 승합/상용 성격이라 M1 등록만으로는 일부만 잡힘
LCV_LIKE = {("Fiat", "Ducato"), ("Fiat", "Doblo"), ("Fiat", "Scudo"), ("Ford", "Transit"), ("Ford", "Transit Custom"),
            ("Ford", "Transit Courier"), ("Mercedes-Benz", "Sprinter"), ("Mercedes-Benz", "Vito"),
            ("Mercedes-Benz", "Citan"), ("Mercedes-Benz", "EQT"), ("Mercedes-Benz", "EQV"), ("Renault", "Trafic"),
            ("Renault", "Kangoo"), ("Renault", "Master"), ("Citroen", "Berlingo"), ("Citroen", "Jumpy"),
            ("Citroen", "Jumper"), ("Peugeot", "Expert"), ("Peugeot", "Boxer"), ("Opel/Vauxhall", "Combo"),
            ("Toyota", "Proace"), ("Toyota", "Proace City"), ("Toyota", "Proace Max"), ("Volkswagen", "Caddy"),
            ("Volkswagen", "Crafter"), ("Nissan", "Townstar"), ("Nissan", "Primastar"), ("Nissan", "Interstar"),
            ("Kia", "PV5"), ("Hyundai", "Staria"), ("Ford", "Ranger"), ("Toyota", "Hilux")}


# ---------------------------------------------------------------- 자료 읽기

def load_our(year: int) -> pd.DataFrame:
    d = pd.read_csv(RAW_DATA)
    d = d[(d["Country"] == "Europe") & d["Month"].str.startswith(str(year))].copy()
    d["Quarter"] = d["Month"].str[5:7].astype(int).map(lambda m: f"{year}Q{(m - 1) // 3 + 1}")
    return d


def load_eea(year: int) -> pd.DataFrame:
    p = CACHE / f"eea_monthly_{year}_ALL_ft.csv"
    if not p.exists():
        import eea_query
        eea_query.monthly_counts(year, None, with_fuel=True).to_csv(p, index=False)
    e = pd.read_csv(p, keep_default_na=False)
    # 형식 유형(T)으로 갈라야 하는 제조사는 T 까지 받은 행으로 바꿔 끼운다
    pt = CACHE / f"eea_monthly_{year}_typesplit.csv"
    if not pt.exists():
        import eea_query
        eea_query.monthly_counts_with_type(year, TYPE_SPLIT_MAKES).to_csv(pt, index=False)
    t = pd.read_csv(pt, keep_default_na=False)
    e = pd.concat([e[~e["Mk"].isin(TYPE_SPLIT_MAKES)], t], ignore_index=True)
    e = classify_frame(e, "Mk", "Cn", "Ft")
    e = refine_by_type(e, "T")
    e["Quarter"] = e["Month"].str[5:7].map(lambda m: f"{year}Q{(int(m) - 1) // 3 + 1}" if m.isdigit() else "NA")
    return e.rename(columns={"MS": "country"})


def load_uk(year: int) -> pd.DataFrame:
    d = pd.read_csv(DFT)
    c = d[d["BodyType"] == "Cars"].copy()
    qs = [f"{year} Q{i}" for i in range(1, 5)]
    for q in qs:
        c[q] = pd.to_numeric(c[q], errors="coerce").fillna(0)
    g = c.groupby(["Make", "GenModel", "Model", "Fuel"], as_index=False)[qs].sum()
    g = g[g[qs].sum(axis=1) > 0]
    # 세부 Model 열로 먼저 분류(GenModel 은 'BYD SEAL' 아래 Seal U, 'HYUNDAI IONIQ' 아래 5/6 을 묶음),
    # 안 되면(MODEL MISSING 등) GenModel 로.
    g = g.rename(columns={"Model": "SrcModel"})
    g = classify_frame(g, "Make", "SrcModel", "Fuel")
    for i in g.index[g["Model"].isna()]:
        b, m, r, cf = classify(g.at[i, "Make"], g.at[i, "GenModel"], g.at[i, "Fuel"])
        if m is not None:
            g.loc[i, ["Brand", "Model", "rule", "confidence"]] = [b, m, "genmodel:" + r, cf]
    long = g.melt(id_vars=["Make", "GenModel", "SrcModel", "Fuel", "Brand", "Model", "rule", "confidence"],
                  value_vars=qs,
                  var_name="q", value_name="n")
    long["Quarter"] = long["q"].str.replace(" ", "")
    long["country"] = "UK"
    return long


def _read_ch_sheet(x: pd.DataFrame) -> pd.DataFrame:
    hdr = x.index[(x[0] == "Marke") & (x[1] == "Modell")][0]
    t = x.iloc[hdr + 1:, [0, 1, 2]].copy()
    t.columns = ["Marke", "Modell", "n"]
    t["n"] = pd.to_numeric(t["n"], errors="coerce")
    t["Modell"] = t["Modell"].astype(str).str.strip()
    t["Marke"] = t["Marke"].astype(str).str.strip()
    t = t[t["Marke"].notna() & t["n"].notna() & ~t["Marke"].isin(["Total", "Diverse"])]
    return t.groupby(["Marke", "Modell"], as_index=False)["n"].sum()


def load_ch(year: int) -> pd.DataFrame:
    """auto-schweiz ModellePW<연도>.xlsx: 시트 = 1월~해당월 누계 → 월값 = 누계 차이."""
    p = CACHE / f"ch_ModellePW{year}.xlsx"
    sheets = pd.read_excel(p, sheet_name=None, header=None)
    names = list(sheets)[:12]
    prev, rows = None, []
    for i, sn in enumerate(names, start=1):
        cur = _read_ch_sheet(sheets[sn]).set_index(["Marke", "Modell"])["n"]
        mon = cur.sub(prev, fill_value=0) if prev is not None else cur
        prev = cur
        for (mk, md), v in mon.items():
            rows.append({"Marke": mk, "Modell": md, "Month": f"{year}-{i:02d}", "n": v})
    df = pd.DataFrame(rows)
    df = classify_frame(df, "Marke", "Modell")
    df["Quarter"] = df["Month"].str[5:7].map(lambda m: f"{year}Q{(int(m) - 1) // 3 + 1}")
    df["country"] = "CH"
    return df


def completeness_2025(eea: pd.DataFrame, uk: pd.DataFrame, ch: pd.DataFrame) -> pd.Series:
    a = pd.read_csv(ACEA)
    a["country"] = a["country"].map(ISO)
    a = a.dropna(subset=["country"]).set_index("country")["registrations_2025"]
    got = eea.groupby("country")["n"].sum()
    got["UK"] = uk["n"].sum()
    got["CH"] = ch["n"].sum()
    return (a / got).dropna().rename("completeness_factor")


# ---------------------------------------------------------------- 출력

def conf_rank(c):
    return {"high": 3, "medium": 2, "low": 1}.get(c, 0)


def series_conf(*frames) -> pd.Series:
    allc = pd.concat([f[["Brand", "Model", "confidence", "n"]] for f in frames])
    allc = allc[allc["Model"].notna() & (allc["n"] > 0)]
    allc["r"] = allc["confidence"].map(conf_rank)
    return allc.groupby(["Brand", "Model"])["r"].min().map({3: "high", 2: "medium", 1: "low"})


def annual_validation(year: int, our, eea, uk, ch, comp=None) -> pd.DataFrame:
    o = our.groupby(["Brand", "Model"])["Sales"].sum().rename("our")
    parts = {
        "eea": eea[eea["Model"].notna()].groupby(["Brand", "Model"])["n"].sum(),
        "uk": uk[uk["Model"].notna()].groupby(["Brand", "Model"])["n"].sum(),
        "ch": ch[ch["Model"].notna()].groupby(["Brand", "Model"])["n"].sum(),
    }
    v = pd.concat([o] + [s.rename(k) for k, s in parts.items()], axis=1).fillna(0)
    v = v[v["our"] > 0]
    v["official"] = v["eea"] + v["uk"] + v["ch"]
    if comp is not None:
        e2 = eea[eea["Model"].notna()].copy()
        e2["adj"] = e2["n"] * e2["country"].map(comp).fillna(1.0)
        v["official_adj"] = (e2.groupby(["Brand", "Model"])["adj"].sum().reindex(v.index).fillna(0)
                             + v["uk"] * comp.get("UK", 1.0) + v["ch"] * comp.get("CH", 1.0))
    else:
        v["official_adj"] = v["official"]
    v["ratio"] = v["our"] / v["official"].replace(0, np.nan)
    v["ratio_adj"] = v["our"] / v["official_adj"].replace(0, np.nan)
    v["confidence"] = series_conf(eea, uk, ch).reindex(v.index)
    v = v.reset_index()
    v["lcv_like"] = [(b, m) in LCV_LIKE for b, m in zip(v["Brand"], v["Model"])]
    v.insert(0, "period", str(year))
    v.insert(0, "check", "europe_annual")
    return v


# 공식 자료가 갈라 주지 못하는 차종 짝(어느 한 출처라도 묶어서 보고): 합쳐서 비교한다.
PAIR_GROUPS = {
    "Audi Q3+Q3 Sportback": [("Audi", "Q3"), ("Audi", "Q3 Sportback")],
    "Audi Q4+Q4 Sportback": [("Audi", "Q4"), ("Audi", "Q4 Sportback")],
    "Audi Q6+Q6 Sportback": [("Audi", "Q6 e-tron"), ("Audi", "Q6 Sportback e-tron")],
    "BMW 2 Series+Active Tourer": [("BMW", "2 Series"), ("BMW", "2 Series Active Tourer")],
    "Skoda Enyaq+Coupe": [("Skoda", "Enyaq"), ("Skoda", "Enyaq Coupe")],
    "Mercedes EQE+EQE SUV": [("Mercedes-Benz", "EQE"), ("Mercedes-Benz", "EQE SUV")],
    "Mercedes EQS+EQS SUV": [("Mercedes-Benz", "EQS"), ("Mercedes-Benz", "EQS SUV")],
    "Fiat Panda+Grande Panda": [("Fiat", "Panda"), ("Fiat", "Grande Panda")],
    "Toyota Corolla+Corolla Cross": [("Toyota", "Corolla"), ("Toyota", "Corolla Cross")],
    "Land Rover Range Rover+Sport": [("Land Rover", "Range Rover"), ("Land Rover", "Range Rover Sport")],
}


def group_validation(ann: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for per, a in ann.groupby("period"):
        for name, members in PAIR_GROUPS.items():
            g = a[[(b, m) in members for b, m in zip(a["Brand"], a["Model"])]]
            if g.empty:
                continue
            s = g[["our", "eea", "uk", "ch", "official", "official_adj"]].sum()
            rows.append({"check": "europe_annual_pairgroup", "period": per, "Brand": members[0][0], "Model": name,
                         **s.to_dict(), "ratio": s["our"] / s["official"] if s["official"] else np.nan,
                         "ratio_adj": s["our"] / s["official_adj"] if s["official_adj"] else np.nan,
                         "lcv_like": False})
    return pd.DataFrame(rows)


def quarter_validation(year: int, our, eea, uk, ch) -> pd.DataFrame:
    o = our.groupby(["Brand", "Model", "Quarter"])["Sales"].sum().rename("our")
    e = eea[eea["Model"].notna()].groupby(["Brand", "Model", "Quarter"])["n"].sum().rename("eea")
    u = uk[uk["Model"].notna()].groupby(["Brand", "Model", "Quarter"])["n"].sum().rename("uk")
    c = ch[ch["Model"].notna()].groupby(["Brand", "Model", "Quarter"])["n"].sum().rename("ch")
    v = pd.concat([o, e, u, c], axis=1).fillna(0)
    v = v[v["our"] > 0]
    v["official"] = v[["eea", "uk", "ch"]].sum(axis=1)
    v["ratio"] = v["our"] / v["official"].replace(0, np.nan)
    v = v.reset_index().rename(columns={"Quarter": "period"})
    v.insert(0, "check", "europe_quarter")
    v["lcv_like"] = [(b, m) in LCV_LIKE for b, m in zip(v["Brand"], v["Model"])]
    return v


def shares_table(eea, uk, ch, comp) -> pd.DataFrame:
    e = eea[eea["Model"].notna()].groupby(["Brand", "Model", "country"])["n"].sum().reset_index()
    e["source"] = "EEA co2cars_2025Pv31 (provisional)"
    u = uk[uk["Model"].notna()].groupby(["Brand", "Model", "country"])["n"].sum().reset_index()
    u["source"] = "DfT VEH0160 (2025 Q1-Q4)"
    c = ch[ch["Model"].notna()].groupby(["Brand", "Model", "country"])["n"].sum().reset_index()
    c["source"] = "auto-schweiz ModellePW2025 (CH+FL)"
    s = pd.concat([e, u, c], ignore_index=True).rename(columns={"n": "n_2025_raw"})
    s["completeness_factor"] = s["country"].map(comp).fillna(1.0).round(4)
    s["n_2025_adj"] = (s["n_2025_raw"] * s["completeness_factor"]).round(1)
    tot = s.groupby(["Brand", "Model"])["n_2025_adj"].transform("sum")
    s["share_of_series"] = (s["n_2025_adj"] / tot).round(6)
    s["coverage"] = s["country"].map(COVERAGE).fillna("brand_only_or_none")
    return s.sort_values(["Brand", "Model", "n_2025_adj"], ascending=[True, True, False])


def coverage_summary(shares: pd.DataFrame) -> pd.DataFrame:
    p = shares.pivot_table(index=["Brand", "Model"], columns="coverage", values="share_of_series",
                           aggfunc="sum", fill_value=0).reset_index()
    return p


def main():
    out = []
    comp = None
    for year in (2025, 2024):
        our, eea, uk, ch = load_our(year), load_eea(year), load_uk(year), load_ch(year)
        if year == 2025:
            comp = completeness_2025(eea, uk, ch)
            comp.round(4).to_frame().to_csv(HERE / "out" / "completeness_2025.csv")
            sh = shares_table(eea, uk, ch, comp)
            sh.to_csv(HERE / "eea_shares_2025.csv", index=False, encoding="utf-8")
            coverage_summary(sh).to_csv(HERE / "out" / "eea_shares_2025_coverage_summary.csv", index=False)
            out.append(quarter_validation(year, our, eea, uk, ch))
            # 키 수준 대응표(EEA·DfT·CH)
            km = []
            for src, f, mk, md in [("EEA", eea, "Mk", "Cn"), ("UK_DFT", uk, "Make", "SrcModel"),
                                   ("CH_AUTOSCHWEIZ", ch, "Marke", "Modell")]:
                k = f.groupby([mk, md, "Brand", "Model", "rule", "confidence"], dropna=False)["n"].sum().reset_index()
                k["source_id"] = src
                k["source_key"] = k[mk].astype(str) + " | " + k[md].astype(str)
                km.append(k[["source_id", "source_key", "Brand", "Model", "rule", "confidence", "n"]])
            pd.concat(km).to_csv(HERE / "out" / "map_keys_eea_uk_ch_2025.csv", index=False, encoding="utf-8")
        out.append(annual_validation(year, our, eea, uk, ch, comp if year == 2025 else None))
    v = pd.concat(out, ignore_index=True)
    v = pd.concat([v, group_validation(v[v["check"] == "europe_annual"])], ignore_index=True)
    for f in sorted((HERE / "out").glob("validation_*.csv")):
        v = pd.concat([v, pd.read_csv(f)], ignore_index=True)
    v.insert(1, "Country", "Europe")
    v.to_csv(HERE / "validation.csv", index=False, encoding="utf-8")
    # 요약 출력
    for (chk, per), g in v[v["check"].isin(["europe_annual"])].groupby(["check", "period"]):
        for lab, gg in [("all", g), ("passenger(non-LCV)", g[~g["lcv_like"].astype(bool)])]:
            r = gg["ratio_adj" if per == "2025" else "ratio"].dropna()
            w = gg["our"].sum() / gg["official_adj" if per == "2025" else "official"].sum()
            print(f"{chk} {per} {lab}: n={len(r)} median={r.median():.3f} p10={r.quantile(.1):.3f} "
                  f"p90={r.quantile(.9):.3f} share_in_0.9-1.1={((r > .9) & (r < 1.1)).mean():.2f} "
                  f"weighted={w:.3f}")


if __name__ == "__main__":
    main()
