"""시장 신차 비중 특징 2개(2026-10-03 사전 고정 (ay)).

차종 첫 달 = 자료상 첫 관측 달. 그 시장 자료의 첫 달에 이미 있던 차종은 출시 시점을 모르므로 기존 차종으로 본다.
기준 시점 a(학습 행 = 대상월 − 1, 재귀 예측 = 기준월 고정)까지의 자료만 쓴다.

- entrant_share_24m: a−11…a 시장 판매 중 첫 달이 a−23…a인 차종의 비중
- entrant_share_12m: a−2…a 시장 판매 중 첫 달이 a−11…a인 차종의 비중

a−23이 그 시장 자료 첫 달보다 앞이거나, 창의 어느 달에 시장 판매 관측이 없으면 NaN.
"""

import numpy as np
import pandas as pd

KEY = ["Country", "Brand", "Model"]
COLUMNS = ["entrant_share_24m", "entrant_share_12m"]
WINDOWS = {"entrant_share_24m": (12, 24), "entrant_share_12m": (3, 12)}  # (판매 창 길이, 출시 창 길이)


def entrant_stats(frame):
    """(시장, 달, 출시 달) 판매 합 표와 시장별 자료 첫 달. 출시 달은 기존 차종이면 NaT."""
    present = frame.loc[frame["Sales"].notna(), KEY + ["Month", "Sales"]].copy()
    present["Month"] = pd.PeriodIndex(present["Month"], freq="M")
    first = present.groupby(KEY)["Month"].transform("min")
    start = present.groupby("Country")["Month"].transform("min")
    present["launch"] = first.where(first.ne(start))
    starts = present.groupby("Country")["Month"].min().to_dict()
    totals = present.groupby(["Country", "Month"])["Sales"].sum()
    by_launch = present.dropna(subset=["launch"]).groupby(["Country", "Month", "launch"])["Sales"].sum()
    return {"starts": starts, "totals": totals.to_dict(), "by_launch": by_launch}


def _share(stats, country, as_of, sales_length, launch_length):
    start = stats["starts"].get(country)
    if start is None or as_of - 23 < start:
        return np.nan
    months = [as_of - k for k in range(sales_length)]
    totals = [stats["totals"].get((country, month)) for month in months]
    if any(total is None for total in totals) or sum(totals) <= 0:
        return np.nan
    entrants = 0.0
    table = stats["by_launch"]
    if country in table.index.get_level_values(0):
        part = table.loc[country]
        for month in months:
            if month in part.index.get_level_values(0):
                launches = part.loc[month]
                entrants += float(launches[(launches.index >= as_of - (launch_length - 1)) & (launches.index <= as_of)].sum())
    return entrants / sum(totals)


def entrant_features(countries, as_of_months, stats):
    cache, rows = {}, []
    for country, as_of in zip(countries, pd.PeriodIndex(list(as_of_months), freq="M")):
        if (country, as_of) not in cache:
            cache[(country, as_of)] = tuple(_share(stats, country, as_of, *WINDOWS[name]) for name in COLUMNS)
        rows.append(cache[(country, as_of)])
    return pd.DataFrame(rows, columns=COLUMNS)
