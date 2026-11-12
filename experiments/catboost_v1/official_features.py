"""공식 시장 판매 특징 3개(2026-10-03 사전 고정 (by)).

공식 표 = research/*.csv(시장별 공식 월간 신차 판매, 출처·정의는 각 행). 한국은 note의 'double-counted)=N'(GM 한국 수입차가
국내 5사 내수와 KAIDA 양쪽에 들어간 대수)를 빼서 쓴다. 빈 값은 빈 채로 둔다(채우지 않음).
기준 시점 a(학습 = 대상월 − 1, 재귀 예측 = 기준월 고정), 시장 c:
- official_change_3m = log1p(공식 a−2…a 합) − log1p(공식 a−14…a−12 합)
- official_change_12m = log1p(공식 a−11…a 합) − log1p(공식 a−23…a−12 합)
- data_coverage_12m = 우리 자료 a−11…a 합 ÷ 공식 a−11…a 합
필요한 달의 공식 값(또는 우리 자료 합)이 하나라도 없으면 NaN. 공식 값은 2026-10-03에 받은 수정 반영본이다(한계).
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[0] / "official_totals_v1" / "research"  # 공식 표 원본 폴더를 그대로 읽음
COLUMNS = ["official_change_3m", "official_change_12m", "data_coverage_12m"]
_DOUBLE = re.compile(r"double-counted\)=(\d+)")
_OFFICIAL = None


def parse_table(table):
    """(Country, Month) → 공식 판매(한국 중복 차감). 빈 값은 넣지 않는다."""
    values = {}
    for country, month, sales, note in zip(table["Country"], table["Month"], table["official_sales"], table["note"]):
        text = str(sales).strip()
        if text in ("", "nan"):
            continue
        value = float(text)
        if country == "South Korea":
            found = _DOUBLE.search(str(note))
            if found:
                value -= float(found.group(1))
        values[(country, pd.Period(str(month), freq="M"))] = value
    return values


def load_official():
    global _OFFICIAL
    if _OFFICIAL is None:
        values = {}
        for path in sorted(RESEARCH.glob("*.csv")):
            table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            values.update(parse_table(table))
        _OFFICIAL = values
    return _OFFICIAL


def official_stats(frame, official=None):
    """우리 자료 (시장, 달) 합과 공식 표."""
    present = frame.loc[frame["Sales"].notna(), ["Country", "Month", "Sales"]].copy()
    present["Month"] = pd.PeriodIndex(present["Month"], freq="M")
    ours = present.groupby(["Country", "Month"])["Sales"].sum().to_dict()
    return {"ours": ours, "official": load_official() if official is None else official}


def _sum(table, country, end, length):
    values = [table.get((country, end - k)) for k in range(length)]
    return np.nan if any(v is None for v in values) else float(sum(values))


def official_features(countries, as_of_months, stats):
    cache, rows = {}, []
    for country, as_of in zip(countries, pd.PeriodIndex(list(as_of_months), freq="M")):
        if (country, as_of) not in cache:
            official = stats["official"]
            r3, p3 = _sum(official, country, as_of, 3), _sum(official, country, as_of - 12, 3)
            r12, p12 = _sum(official, country, as_of, 12), _sum(official, country, as_of - 12, 12)
            ours12 = _sum(stats["ours"], country, as_of, 12)
            coverage = ours12 / r12 if np.isfinite(r12) and r12 > 0 and np.isfinite(ours12) else np.nan
            cache[(country, as_of)] = (np.log1p(r3) - np.log1p(p3), np.log1p(r12) - np.log1p(p12), coverage)
        rows.append(cache[(country, as_of)])
    return pd.DataFrame(rows, columns=COLUMNS)
