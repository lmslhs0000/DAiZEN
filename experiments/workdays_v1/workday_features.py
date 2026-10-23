"""시장별 대상월 근무일 수 조회(workdays_v1.csv, make_workdays.py가 holidays 0.105로 만든 정적 표).

표에 없는 시장·월은 오류로 멈춘다(빈칸을 0이나 평균으로 채우지 않는다).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TABLE = HERE / "workdays_v1.csv"
COLUMNS = ["workdays"]
DEFINITION = {"unit": "working_days_per_month", "table": TABLE.name,
              "table_sha256": hashlib.sha256(TABLE.read_bytes()).hexdigest(),
              "generator": json.loads(TABLE.with_suffix(".json").read_text(encoding="utf-8"))["rule"],
              "holidays_version": json.loads(TABLE.with_suffix(".json").read_text(encoding="utf-8"))["holidays_version"],
              "value_type": "KNOWN_STATUTORY_CALENDAR", "target_month": "forecast_target_month",
              "missing_policy": "unknown_market_or_month_is_error", "sales_unit_changed": False}
_VALUES = None


def _values():
    global _VALUES
    if _VALUES is None:
        frame = pd.read_csv(TABLE, dtype={"Country": str, "Month": str})
        _VALUES = {(c, m): float(v) for c, m, v in zip(frame["Country"], frame["Month"], frame["workdays"])}
    return _VALUES


def lookup(countries, months):
    """(시장, 대상월) 쌍마다 근무일 수를 돌려준다."""
    values = _values()
    out = []
    for country, month in zip(countries, months):
        key = (str(country), str(month))
        if key not in values:
            raise ValueError(f"No workdays for {key}")
        out.append(values[key])
    return np.asarray(out, dtype=float)
