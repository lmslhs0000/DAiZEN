"""브랜드 국적 무리 특징 2개(2026-10-03 사전 고정 (bw)).

국적 = brand_origin.csv(브랜드를 가진 회사의 본거지·전통 국적, 2026년 기준; 판단이 갈린 것은 note). 정적 표라 기준월 누설 없음.
기준 시점 a(학습 행 = 대상월 − 1, 재귀 예측 = 기준월 고정), 시장 c, 국적 무리 g:
- origin_share_12m: a−11…a 시장 c 판매 중 g 무리 브랜드 판매의 비중
- origin_share_change_12m: log(origin_share_12m(a) ÷ origin_share_12m(a−12))
필요한 달에 시장 판매 관측이 없거나 분모·비중이 0이면 NaN. 표에 없는 브랜드는 오류.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TABLE = Path(__file__).resolve().parent / "brand_origin.csv"
COLUMNS = ["origin_share_12m", "origin_share_change_12m"]
_ORIGINS = None


def load_origins():
    global _ORIGINS
    if _ORIGINS is None:
        table = pd.read_csv(TABLE, encoding="utf-8")
        if table["Brand"].duplicated().any() or table["origin"].isna().any():
            raise ValueError("국적 표에 중복 브랜드나 빈 국적이 있습니다.")
        _ORIGINS = dict(zip(table["Brand"], table["origin"]))
    return _ORIGINS


def origin_stats(frame):
    """(시장, 국적, 달) 판매 합과 (시장, 달) 시장 합. 공란 판매량은 더하지 않는다."""
    origins = load_origins()
    present = frame.loc[frame["Sales"].notna(), ["Country", "Brand", "Month", "Sales"]].copy()
    unknown = sorted(set(present["Brand"]) - set(origins))
    if unknown:
        raise ValueError(f"국적 표에 없는 브랜드: {unknown}")
    present["Month"] = pd.PeriodIndex(present["Month"], freq="M")
    present["origin"] = present["Brand"].map(origins)
    groups = present.groupby(["Country", "origin", "Month"])["Sales"].sum().to_dict()
    markets = present.groupby(["Country", "Month"])["Sales"].sum().to_dict()
    return {"groups": groups, "markets": markets}


def _share(stats, country, origin, end):
    months = [end - k for k in range(12)]
    totals = [stats["markets"].get((country, month)) for month in months]
    if any(total is None for total in totals) or sum(totals) <= 0:
        return np.nan
    group = sum(stats["groups"].get((country, origin, month), 0.0) for month in months)
    return group / sum(totals)


def origin_features(countries, brands, as_of_months, stats):
    origins = load_origins()
    cache, rows = {}, []
    for country, brand, as_of in zip(countries, brands, pd.PeriodIndex(list(as_of_months), freq="M")):
        if brand not in origins:
            raise ValueError(f"국적 표에 없는 브랜드: {brand}")
        key = (country, origins[brand], as_of)
        if key not in cache:
            now, before = _share(stats, *key), _share(stats, country, origins[brand], as_of - 12)
            change = np.log(now / before) if now > 0 and before > 0 else np.nan
            cache[key] = (now, change)
        rows.append(cache[key])
    return pd.DataFrame(rows, columns=COLUMNS)
