"""검증: 계산기 값 vs 우리 raw/data.csv Country=Europe (2026-07 이하 달만).

사용: python validate_europe.py [--months 2026-04,2026-05,2026-06,2026-07] [--collect]
  --collect : 각 달 europe_c 수집기를 --offline 으로 먼저 실행(캐시가 있는 나라만 만들어짐).
출력: validation.csv            차종×달(+2026-Q2 합): raw, calc, ratio, value_type, estimated_share, confidence, flag
      validation_summary.csv    달별 요약: 판매 가중 비율, WAPE, 상위 100 분포(±10/±20%), 값 만든 비율 등
      validation_country.csv    나라 수집기 점검(EEA 2025 같은 달과 대조: SE SPLY 2025-07, IE CSO/SIMI 2025-07, RO 합계)
밴(LCV_LIKE)은 std 에 없으므로 비교에서 빠진다(validation 에 flag=VAN_FLAGGED 로 우리 값만 표시).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from c_common import EB, HERE, OUT, RAW_DATA, LAST_ALLOWED_TEST_MONTH
import collect_europe as ce


def summarize(v: pd.DataFrame, label: str) -> dict:
    k = v[v["calc"].notna() & (v["raw"] > 0)]
    allr = v[(v["raw"] > 0) & (v["flag"] != "VAN_FLAGGED")]
    top = allr.sort_values("raw", ascending=False).head(100)
    tk = top[top["calc"].notna()]
    r = tk["calc"] / tk["raw"]
    return dict(period=label, n_series_raw=int(len(allr)), n_with_value=int(len(k)),
                raw_volume_with_value=round(float(k["raw"].sum() / allr["raw"].sum()), 4),
                weighted_ratio=round(float(k["calc"].sum() / k["raw"].sum()), 4),
                wape=round(float((k["calc"] - k["raw"]).abs().sum() / k["raw"].sum()), 4),
                top100_with_value=int(len(tk)),
                top100_within_10pct=round(float((r.sub(1).abs() <= 0.10).mean()), 3),
                top100_within_20pct=round(float((r.sub(1).abs() <= 0.20).mean()), 3),
                top100_ratio_p10=round(float(r.quantile(0.1)), 3), top100_ratio_median=round(float(r.median()), 3),
                top100_ratio_p90=round(float(r.quantile(0.9)), 3),
                top100_wape=round(float((tk["calc"] - tk["raw"]).abs().sum() / tk["raw"].sum()), 4))


def country_checks() -> pd.DataFrame:
    """나라 수집기 단위 점검: EEA 2025 같은 달(잠정판) 우리 차종 합계와 비교."""
    rows = []
    e = pd.read_csv(EB / "cache" / "eea_monthly_2025_ALL_ft.csv", usecols=["MS", "Month", "n"], keep_default_na=False)
    tot = e.groupby(["MS", "Month"])["n"].sum()
    for cc, fn, src in [("SE", "se_model_month_2025-07_sply.csv", "Mobility Sweden SPLY"),
                        ("IE", "ie_model_month_2025-07.csv", "CSO TEM20")]:
        p = OUT / fn
        if p.exists():
            d = pd.read_csv(p)
            rows.append(dict(country=cc, month="2025-07", source=src, collector_total=float(d["units"].sum()),
                             eea_total=float(tot.get((cc, "2025-07"), np.nan))))
    rows.append(dict(country="IE", month="2025-07", source="SIMI (표본 json, 2025 열)", collector_total=26738.0,
                     eea_total=float(tot.get(("IE", "2025-07"), np.nan))))
    for cc, m in [("RO", "2026-07")]:
        mp = OUT / f"ro_meta_{m}.json"
        if mp.exists():
            j = json.loads(mp.read_text(encoding="utf-8"))
            rows.append(dict(country="RO", month="2025-07", source="DGPCI Acea 시트 전년 같은 달",
                             collector_total=j.get("acea_sheet_m1_same_month_last_year"),
                             eea_total=float(tot.get(("RO", "2025-07"), np.nan))))
    for cc, m in [("LU", "2026-07"), ("FI", "2026-07"), ("CH", "2026-07")]:
        mp = OUT / f"{cc.lower()}_meta_{m}.json"
        if mp.exists():
            j = json.loads(mp.read_text(encoding="utf-8"))
            rows.append(dict(country=cc, month=m, source=j.get("source_id"), collector_total=j.get("units_total"),
                             eea_total=float(tot.get((cc, "2025-07"), np.nan)), note="eea_total = 2025-07(전년 같은 달, 참고)"))
    x = pd.DataFrame(rows)
    x["ratio"] = (x["collector_total"] / x["eea_total"]).round(4)
    # SE 2025-07: 차종별 대조(우리 차종 대응 후)
    p = OUT / "se_model_month_2025-07_sply.csv"
    if p.exists():
        d = pd.read_csv(p)
        d[["Brand", "Model", "rule", "conf"]] = [ce.classify_c(a, b) for a, b in zip(d["make"], d["model"])]
        se = d.dropna(subset=["Model"]).groupby(["Brand", "Model"])["units"].sum()
        sh = pd.read_csv(EB / "cache" / "eea_monthly_2025_ALL_ft.csv", keep_default_na=False)
        sh = sh[(sh["MS"] == "SE") & (sh["Month"] == "2025-07")]
        sys.path.insert(0, str(EB))
        from europe_map import classify_frame
        sh = classify_frame(sh, "Mk", "Cn", "Ft").dropna(subset=["Model"]).groupby(["Brand", "Model"])["n"].sum()
        j = pd.concat([se.rename("se"), sh.rename("eea")], axis=1).fillna(0)
        j = j[j.max(axis=1) >= 200]
        x = pd.concat([x, pd.DataFrame([dict(country="SE", month="2025-07", source="차종별(200대 이상 차종)",
                                             collector_total=float(j["se"].sum()), eea_total=float(j["eea"].sum()),
                                             ratio=round(float(j["se"].sum() / j["eea"].sum()), 4),
                                             note=f"{len(j)}개 차종, 차종 비율 중앙값 {np.nanmedian(j['se'] / j['eea'].replace(0, np.nan)):.3f}")])])
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="2026-04,2026-05,2026-06,2026-07")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    months = a.months.split(",")
    assert max(months) <= LAST_ALLOWED_TEST_MONTH, "검증은 2026-07 이하 달만"
    raw = pd.read_csv(RAW_DATA)
    eu = raw[raw["Country"] == "Europe"]
    allv, summ = [], []
    for m in months:
        if a.collect:
            ce.run_collectors(m, offline=True)
        std, vans, log, unm = ce.build(m)
        std.to_csv(OUT / f"std_europe_{m}.csv", index=False, encoding="utf-8-sig")
        r = eu[eu["Month"] == m][["Brand", "Model", "Sales"]].rename(columns={"Sales": "raw"})
        v = r.merge(std.rename(columns={"Sales": "calc"}), on=["Brand", "Model"], how="outer")
        v["Month"] = m
        v["flag"] = ""
        vk = set(zip(vans["Brand"], vans["Model"])) if len(vans) else set()
        v.loc[[(b, mo) in vk for b, mo in zip(v["Brand"], v["Model"])], "flag"] = "VAN_FLAGGED"
        cov = ";".join(f"{x.key}:{x.status}" for x in log[log["check"] == "source"].itertuples())
        v["sources"] = cov
        allv.append(v)
        summ.append(dict(summarize(v, m), sources=cov))
    V = pd.concat(allv, ignore_index=True)
    V["raw"] = V["raw"].fillna(0)
    q = [m for m in months if m in ("2026-04", "2026-05", "2026-06")]
    if len(q) == 3:
        Q = V[V["Month"].isin(q)].groupby(["Brand", "Model"], as_index=False).agg(
            raw=("raw", "sum"), calc=("calc", lambda s: s.sum() if s.notna().all() else np.nan),
            flag=("flag", "max"))
        Q["Month"] = "2026-Q2"
        V = pd.concat([V, Q], ignore_index=True)
        summ.append(dict(summarize(Q, "2026-Q2"), sources="sum of 04-06 (값이 세 달 모두 있는 차종)"))
    V["ratio"] = (V["calc"] / V["raw"].replace(0, np.nan)).round(4)
    V["abs_err"] = (V["calc"] - V["raw"]).abs().round(1)

    def flag(r):
        if r["flag"]:
            return r["flag"]
        if pd.isna(r["calc"]):
            return "NO_VALUE" if r["raw"] > 0 else ""
        if r["raw"] <= 0:
            return "NO_RAW"
        x = r["ratio"]
        return "ok10" if abs(x - 1) <= 0.10 else "ok20" if abs(x - 1) <= 0.20 else "HIGH" if x > 1 else "LOW"
    V["flag"] = V.apply(flag, axis=1)
    cols = ["Month", "Brand", "Model", "raw", "calc", "ratio", "abs_err", "flag", "value_type", "estimated_share",
            "confidence", "note"]
    V = V.sort_values(["Month", "raw"], ascending=[True, False])
    V[cols].to_csv(HERE / "validation.csv", index=False, encoding="utf-8-sig")
    S = pd.DataFrame(summ)
    S.to_csv(HERE / "validation_summary.csv", index=False, encoding="utf-8-sig")
    C = country_checks()
    C.to_csv(HERE / "validation_country.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(S.drop(columns=["sources"]).to_string(index=False))
    print(C.to_string(index=False))


if __name__ == "__main__":
    main()
