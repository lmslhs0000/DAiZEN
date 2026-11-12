"""브랜드 흐름 특징 2개(2026-10-03 사전 고정 (bc)). market_momentum_v1/market_features.py의 정의를 시장 대신
(Country, Brand) 합계에 적용했다. 기준 시점 a(학습 행 = 대상월 − 1, 재귀 예측 = 기준월 고정)까지의 합계만 쓴다.

- brand_change_3m: log1p(a−2~a 합) − log1p(a−14~a−12 합)
- brand_change_12m: log1p(a−11~a 합) − log1p(a−23~a−12 합)

필요한 달 중 하나라도 그 브랜드 판매 관측이 없으면 NaN.
"""

import numpy as np
import pandas as pd

COLUMNS = ["brand_change_3m", "brand_change_12m"]


def brand_totals(frame):
    """(Country, Brand, Month) → 판매 합계. 공란 판매량은 더하지 않는다."""
    present = frame.loc[frame["Sales"].notna()]
    months = pd.PeriodIndex(present["Month"], freq="M")
    totals = present.assign(Month=months).groupby(["Country", "Brand", "Month"])["Sales"].sum()
    return {key: float(value) for key, value in totals.items()}


def _window_sum(totals, country, brand, end, length):
    values = [totals.get((country, brand, end - k)) for k in range(length)]
    return np.nan if any(v is None for v in values) else float(sum(values))


def brand_features(countries, brands, as_of_months, totals):
    rows, cache = [], {}
    for country, brand, as_of in zip(countries, brands, pd.PeriodIndex(list(as_of_months), freq="M")):
        key = (country, brand, as_of)
        if key not in cache:
            recent3, prior3 = (_window_sum(totals, country, brand, as_of, 3),
                               _window_sum(totals, country, brand, as_of - 12, 3))
            recent12, prior12 = (_window_sum(totals, country, brand, as_of, 12),
                                 _window_sum(totals, country, brand, as_of - 12, 12))
            cache[key] = (np.log1p(recent3) - np.log1p(prior3), np.log1p(recent12) - np.log1p(prior12))
        rows.append(cache[key])
    return pd.DataFrame(rows, columns=COLUMNS)
