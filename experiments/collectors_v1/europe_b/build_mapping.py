"""mapping_europe_b.csv 만들기: 각 출처의 (제조사|모델) 문자열 → 우리 Europe 시계열.

입력(먼저 실행): europe_validate.py → out/map_keys_eea_uk_ch_2025.csv, collect_fr.py → out/fr_pfa_ytd.csv,
                 collect_it.py → out/it_unrae_top50.csv, collect_pl.py → out/pl_<월>_by_model.csv
열: Country, Brand, Model, source_id, source_key, rule, confidence, note, volume
  - Country 는 우리 시계열 기준이라 항상 Europe.
  - 대응 안 된 문자열도 규모가 크면(출처별 대수 500 이상) Brand/Model 빈칸 + note=unmapped:<사유> 로 남긴다
    (대부분 우리 목록에 없는 차종: Ford Focus, Audi A1/Q2, MG HS, VW Multivan, Seat Ateca 등).
"""
from __future__ import annotations

import pandas as pd

from _common import HERE

OUT = HERE / "out"
UNMAPPED_MIN = 500


def main():
    parts = []
    k = pd.read_csv(OUT / "map_keys_eea_uk_ch_2025.csv")
    k = k.groupby(["source_id", "source_key", "Brand", "Model", "rule", "confidence"], dropna=False)["n"].sum()
    k = k.reset_index().rename(columns={"n": "volume"})
    k["note"] = k["source_id"].map({"EEA": "EEA co2cars 2025P Mk|Cn (모든 나라 합, 연간 대수)",
                                    "UK_DFT": "DfT VEH0160 Make|Model 2025 Q1-Q4",
                                    "CH_AUTOSCHWEIZ": "auto-schweiz Marke|Modell 2025"})
    parts.append(k)
    p = OUT / "fr_pfa_ytd.csv"
    if p.exists():
        f = pd.read_csv(p)
        f = f.sort_values("Month").groupby(["segment", "name"]).agg(
            Brand=("Brand", "last"), Model=("Model", "last"), rule=("rule", "last"),
            confidence=("confidence", "last"), volume=("ytd", "max")).reset_index()
        f["source_id"] = "FR_PFA_TOP100_" + f["segment"]
        f["source_key"] = f["name"]
        f["note"] = "PFA 상위 100 이름(누계 최대값을 volume 으로)"
        parts.append(f.drop(columns="segment"))
    p = OUT / "it_unrae_top50.csv"
    if p.exists():
        i = pd.read_csv(p)
        i = i[i["kind"] == "month"].groupby("name").agg(Brand=("Brand", "last"), Model=("Model", "last"),
                                                         rule=("rule", "last"), confidence=("confidence", "last"),
                                                         volume=("n", "sum")).reset_index()
        i["source_id"] = "IT_UNRAE_TOP50"
        i["source_key"] = i["name"]
        i["note"] = "UNRAE 상위 50 이름(받은 달 합계를 volume 으로)"
        parts.append(i.drop(columns="name"))
    for p in sorted(OUT.glob("pl_*_by_model.csv")):
        if "_d" in p.stem:  # 며칠짜리 시험 파일 제외
            continue
        g = pd.read_csv(p, dtype={"marka": str, "model": str})
        g["source_key"] = g["marka"].fillna("") + " | " + g["model"].fillna("") + " | " + g["rodzaj-paliwa"].fillna("")
        g = g.groupby(["source_key", "Brand", "Model", "rule", "confidence"], dropna=False)["n"].sum().reset_index()
        g = g.rename(columns={"n": "volume"})
        g["source_id"] = "PL_CEPIK_" + p.stem.split("_")[1]
        g["note"] = "CEPiK marka|model|paliwo (신차, 한 달)"
        parts.append(g)
    m = pd.concat(parts, ignore_index=True)
    m.insert(0, "Country", "Europe")
    unm = m["Model"].isna()
    keep = ~unm | (m["volume"] >= UNMAPPED_MIN)
    m = m[keep].copy()
    m.loc[m["Model"].isna(), "note"] = "unmapped:" + m.loc[m["Model"].isna(), "rule"].astype(str)
    m = m[["Country", "Brand", "Model", "source_id", "source_key", "rule", "confidence", "note", "volume"]]
    m = m.sort_values(["source_id", "Brand", "Model", "volume"], ascending=[True, True, True, False])
    m.to_csv(HERE / "mapping_europe_b.csv", index=False, encoding="utf-8")
    s = pd.concat(parts).assign(mapped=lambda d: d["Model"].notna())
    cov = s.groupby(["source_id", "mapped"])["volume"].sum().unstack(fill_value=0)
    cov["mapped_share"] = cov.get(True, 0) / cov.sum(axis=1)
    print(cov.round(3).to_string())
    print(f"rows={len(m)}")


if __name__ == "__main__":
    main()
