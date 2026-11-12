"""작년 그대로(Seasonal Naive) 24개월 예측. AGENTS.md 4절 정의를 24개월로 잇는다."""

import pandas as pd

from common import KEY

PRED_COLUMNS = KEY + ["Month", "horizon", "prediction"]
STATE_COLUMNS = KEY + ["prediction_status", "reason"]


def seasonal_naive(data, origin, horizon):
    """h<=12는 12개월 전 실제값, h>12는 앞서 만든 예측값을 반복한다. 기준월 이후 실제값은 읽지 않는다."""
    origin = pd.Period(origin, freq="M")
    observed = data.loc[data["Month"].le(origin)]
    required = pd.period_range(origin - 11, origin, freq="M")
    predictions, states = [], []
    for key, vehicle in observed.groupby(KEY, sort=True):
        sales = vehicle.set_index("Month")["Sales"]
        if sales.index.min() > required[0]:
            status = "INSUFFICIENT_HISTORY"
        elif sales.reindex(required).isna().any():
            status = "INCOMPLETE_HISTORY"
        else:
            status = "OK"
        states.append({**dict(zip(KEY, key)), "prediction_status": status, "reason": "" if status == "OK" else status})
        if status != "OK":
            continue
        values = {month: float(sales[month]) for month in required}
        for step in range(1, horizon + 1):
            month = origin + step
            values[month] = values[month - 12]
            predictions.append({**dict(zip(KEY, key)), "Month": month, "horizon": step, "prediction": values[month]})
    return pd.DataFrame(predictions, columns=PRED_COLUMNS), pd.DataFrame(states, columns=STATE_COLUMNS)


def seasonal_mean_profile(data, origin, horizon, years=3):
    """(eo) 월 배분 기준 모양: 대상월 t(거리 h)마다 k0 = ceil(h/12)부터 years개 해 t−12k의 실제값 평균.
    모두 기준월 이하 실제값만 쓰고, 하나라도 없으면 NaN(그 묶음은 reshape에서 모델 모양을 그대로 둔다)."""
    origin = pd.Period(origin, freq="M")
    observed = data.loc[data["Month"].le(origin)]
    predictions, states = [], []
    for key, vehicle in observed.groupby(KEY, sort=True):
        sales = vehicle.set_index("Month")["Sales"]
        states.append({**dict(zip(KEY, key)), "prediction_status": "OK", "reason": ""})
        for step in range(1, horizon + 1):
            month = origin + step
            first = -(-step // 12)
            past = sales.reindex([month - 12 * k for k in range(first, first + years)])
            value = float(past.mean()) if past.notna().all() else float("nan")
            predictions.append({**dict(zip(KEY, key)), "Month": month, "horizon": step, "prediction": value})
    return pd.DataFrame(predictions, columns=PRED_COLUMNS), pd.DataFrame(states, columns=STATE_COLUMNS)
