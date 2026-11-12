"""(fp) 중국 신차 출시 압력 특징(2026-10-06 사전 고정).

launch_pressure(브랜드 b, 대상 달 m, 기준 시점 a) = 브랜드 b에서 첫 판매월이 (m−12, m]인 신차 수.
단 첫 판매월이 a보다 뒤인 신차는 공개일 ≤ a일 때만 센다(기준 시점에 알 수 있는 것만). 공개일이 없으면 첫 판매월에 알게 된 것으로 본다.
표 = china_lifecycle_v2/research/launch_dates_merged.csv(조사 에이전트 결과를 합친 것; Brand, Model, first_sale_month, announced_date ...).
"""

from pathlib import Path

import numpy as np
import pandas as pd

TABLE = Path(__file__).resolve().parents[1] / "china_lifecycle_v2" / "research" / "launch_dates_merged.csv"
COLUMNS = ["launch_pressure"]
WINDOW = 12
_TABLE = None


def _month(value):
    text = str(value).strip()
    return pd.Period(text[:7], freq="M") if text else pd.NaT


def prepare(table):
    launch = [_month(v) for v in table["first_sale_month"]]
    announced = [_month(v) for v in table["announced_date"]]
    known = [a if (a is not pd.NaT and not pd.isna(a)) else l for a, l in zip(announced, launch)]
    return pd.DataFrame({"Brand": table["Brand"].astype(str).to_numpy(), "launch": [p.ordinal for p in launch],
                         "known": [min(k.ordinal, l.ordinal) for k, l in zip(known, launch)]})


def load_table(path=None):
    return prepare(pd.read_csv(path or TABLE, dtype=str, keep_default_na=False, encoding="utf-8-sig"))


def launch_pressure(brands, target_months, as_of_months, table=None):
    global _TABLE
    if table is None:
        if _TABLE is None:
            _TABLE = load_table()
        table = _TABLE
    by_brand = {b: (g["launch"].to_numpy(), g["known"].to_numpy()) for b, g in table.groupby("Brand")}
    out = np.zeros(len(brands))
    for i, (brand, m, a) in enumerate(zip(brands, target_months, as_of_months)):
        entry = by_brand.get(str(brand))
        if entry is None:
            continue
        launch, known = entry
        mo, ao = m.ordinal, a.ordinal
        window = (launch > mo - WINDOW) & (launch <= mo)
        visible = (launch <= ao) | (known <= ao)
        out[i] = float((window & visible).sum())
    return pd.Series(out, name="launch_pressure")
