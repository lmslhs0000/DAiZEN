"""EEA CO2 승용차 모니터링 자료(Discodata SQL API)에서 나라 × 제조사(Mk) × 상품명(Cn) 월별 신차 대수를 뽑는다.

자료: https://discodata.eea.europa.eu/sql  (CC BY 4.0, 차량 1대 = 1행, R=1, Dr=등록일)
표:   [CO2Emission].[latest].[co2cars_<연도><P|F>v<n>]  (2026-10-15 확인: 2025 = co2cars_2025Pv31 잠정)
범위: EU27 + NO + IS (UK·CH 없음)

제약(시험으로 확인): LEFT()/CONVERT()·ORDER BY 는 막혀 있다(errorcode 10002). MONTH(Dr) 는 허용된다(2026-10-15, FR 2025 월합 확인).
MONTH() 가 막히는 경우를 대비해 mode="by_day" 로 Dr 일자별로 묶어 받은 뒤 파이썬에서 월로 합산하는 경로도 둔다.
시스템 표(INFORMATION_SCHEMA)는 막혀 있어서 표 이름은 EEA datahub 안내를 따른다.

사용 예:
  python eea_query.py --year 2025 --countries FR IT            # 월별 FR·IT → cache/eea_monthly_2025_FR-IT.csv
  python eea_query.py --year 2025 --annual                      # 전 나라 연간 MS×Mk×Cn → cache/eea_annual_2025.csv
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from _common import CACHE, PoliteSession, cache_path

ENDPOINT = "https://discodata.eea.europa.eu/sql"
TABLES = {
    2025: "co2cars_2025Pv31",  # 잠정(provisional). 확정판이 나오면 바꾼다.
    2024: "co2cars_2024Fv30",
    2023: "co2cars_2023Fv28",
}
EEA_COUNTRIES = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "HR", "HU", "IE", "IT",
    "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK", "NO", "IS",
]

_sess: PoliteSession | None = None


def _session() -> PoliteSession:
    global _sess
    if _sess is None:
        _sess = PoliteSession()
    return _sess


def table(year: int) -> str:
    return f"[CO2Emission].[latest].[{TABLES[year]}]"


def run_sql(query: str, page_size: int = 50000, use_cache: bool = True) -> list[dict]:
    """쪽 나눔(p, nrOfHits)으로 끝까지 받는다. 결과는 질의문 해시로 캐시."""
    cp = cache_path("eea", query + f"|{page_size}", "json")
    if use_cache and cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))
    rows: list[dict] = []
    page = 1
    while True:
        r = _session().get(ENDPOINT, params={"query": query, "p": page, "nrOfHits": page_size}, timeout=600)
        r.raise_for_status()
        j = r.json()
        if "errors" in j:
            raise RuntimeError(f"EEA SQL 오류: {j['errors']}  query={query[:200]}")
        res = j.get("results", [])
        rows.extend(res)
        if len(res) < page_size:
            break
        page += 1
    cp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def _in_list(codes) -> str:
    return ",".join("'" + c.replace("'", "") + "'" for c in codes)


def monthly_counts(year: int, countries=None, mode: str = "month", with_fuel: bool = False) -> pd.DataFrame:
    """나라 × Mk × Cn (× Ft) × 월 신차 대수. 열: MS, Mk, Cn, [Ft], Month(YYYY-MM), n.

    mode="month": SQL 안에서 MONTH(Dr)로 묶음(빠름).
    mode="by_day": Dr 일자로 묶어 받고 파이썬에서 월 합산(MONTH 가 막힌 경우 대비, 행 수 많음).
    with_fuel=True: 연료(Ft)도 나눔(Volvo XC40/EX40, Audi A6/A6 e-tron 같은 연료로 갈리는 차종에 필요).
    받은 뒤 같은 조건의 SUM(R) 총계와 맞는지 확인한다(쪽 나눔 누락 점검; ORDER BY 를 못 써서).
    """
    where = f"WHERE MS IN ({_in_list(countries)})" if countries else ""
    keys = ["MS", "Mk", "Cn"] + (["Ft"] if with_fuel else [])
    kq = ", ".join(keys)
    if mode == "month":
        q = f"SELECT {kq}, MONTH(Dr) AS m, SUM(R) AS n FROM {table(year)} {where} GROUP BY {kq}, MONTH(Dr)"
        df = pd.DataFrame(run_sql(q))
    elif mode == "by_day":
        q = f"SELECT {kq}, Dr, SUM(R) AS n FROM {table(year)} {where} GROUP BY {kq}, Dr"
        df = pd.DataFrame(run_sql(q))
        df["m"] = pd.to_datetime(df["Dr"]).dt.month
        df = df.groupby(keys + ["m"], dropna=False, as_index=False)["n"].sum()
    else:
        raise ValueError(mode)
    if df.empty:
        return pd.DataFrame(columns=keys + ["Month", "n"])
    _check_total(year, where, int(df["n"].sum()))
    df["Month"] = df["m"].map(lambda m: f"{year}-{int(m):02d}" if pd.notna(m) else f"{year}-NA")
    return df[keys + ["Month", "n"]]


def annual_counts(year: int, countries=None, with_fuel: bool = False) -> pd.DataFrame:
    """나라 × Mk × Cn (× Ft) 연간 신차 대수. 열: MS, Mk, Cn, [Ft], n."""
    where = f"WHERE MS IN ({_in_list(countries)})" if countries else ""
    kq = "MS, Mk, Cn" + (", Ft" if with_fuel else "")
    df = pd.DataFrame(run_sql(f"SELECT {kq}, SUM(R) AS n FROM {table(year)} {where} GROUP BY {kq}"))
    _check_total(year, where, int(df["n"].sum()))
    return df


def monthly_counts_with_type(year: int, makes: list[str]) -> pd.DataFrame:
    """지정 제조사(Mk)만 형식 유형(T)까지 나눈 월별 대수. Panda/Grande Panda 처럼 Cn 이 같은 차종을 가르는 데 쓴다."""
    where = f"WHERE Mk IN ({_in_list(makes)})"
    q = (f"SELECT MS, Mk, Cn, Ft, T, MONTH(Dr) AS m, SUM(R) AS n FROM {table(year)} {where} "
         f"GROUP BY MS, Mk, Cn, Ft, T, MONTH(Dr)")
    df = pd.DataFrame(run_sql(q))
    _check_total(year, where, int(df["n"].sum()))
    df["Month"] = df["m"].map(lambda m: f"{year}-{int(m):02d}" if pd.notna(m) else f"{year}-NA")
    return df[["MS", "Mk", "Cn", "Ft", "T", "Month", "n"]]


def _check_total(year: int, where: str, got: int) -> None:
    tot = run_sql(f"SELECT SUM(R) AS n FROM {table(year)} {where}")[0]["n"]
    if int(tot) != got:
        raise RuntimeError(f"EEA 쪽 나눔 합계 불일치: 받은 합 {got} ≠ 총계 {tot} ({where})")


def country_month_totals(year: int) -> pd.DataFrame:
    q = (f"SELECT MS, MONTH(Dr) AS m, SUM(R) AS n FROM {table(year)} "
         f"GROUP BY MS, MONTH(Dr)")
    df = pd.DataFrame(run_sql(q))
    df["Month"] = df["m"].map(lambda m: f"{year}-{int(m):02d}")
    return df[["MS", "Month", "n"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--countries", nargs="*", default=None, help="예: FR IT (없으면 전 나라)")
    ap.add_argument("--annual", action="store_true", help="월 대신 연간 MS×Mk×Cn")
    ap.add_argument("--mode", default="month", choices=["month", "by_day"])
    ap.add_argument("--fuel", action="store_true", help="연료(Ft)별로도 나눔")
    a = ap.parse_args()
    tag = "-".join(a.countries) if a.countries else "ALL"
    if a.annual:
        df = annual_counts(a.year, a.countries, a.fuel)
        out = CACHE / f"eea_annual_{a.year}_{tag}{'_ft' if a.fuel else ''}.csv"
    else:
        df = monthly_counts(a.year, a.countries, a.mode, a.fuel)
        out = CACHE / f"eea_monthly_{a.year}_{tag}{'_ft' if a.fuel else ''}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"{out}  rows={len(df)}  total={df['n'].sum():,}")


if __name__ == "__main__":
    main()
