"""(ew) 부분 변경(major_facelift) 특징 2개(2026-10-05 사전 고정). 사건표 = catboost_v1/lifecycle_features_g3.load_events()
(중국 l2 ∪ 조사 v2 + 전 시장 조사 표). 기준 시점 as_of까지 공개(announced ≤ as_of)된 사건만 쓴다.

- lc_months_since_facelift: 대상월 − 가장 최근에 출시된(사건월 ≤ as_of) 부분 변경월, 0~60으로 자름. 없으면 NaN.
- lc_months_to_facelift: 아직 출시 전(사건월 > as_of)인 가장 이른 부분 변경월 − 대상월, −24~24로 자름. 없으면 NaN.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["lc_months_since_facelift", "lc_months_to_facelift"]
G3_FILE = Path(__file__).resolve().parents[1] / "catboost_v1" / "lifecycle_features_g3.py"
_EVENTS = None


def load_events():
    spec = importlib.util.spec_from_file_location("lifecycle_features_g3", G3_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_events()


def facelift_features(countries, brands, models, target_months, as_of_months, events=None):
    global _EVENTS
    if events is None:
        if _EVENTS is None:
            _EVENTS = load_events()
        events = _EVENTS
    kept = events.loc[events["event_type"].eq("major_facelift") & events["announced"].notna()]
    by_series = {key: group for key, group in kept.groupby(["Country", "Brand", "Model"], sort=False)}
    rows = []
    for country, brand, model, month, known in zip(countries, brands, models, target_months, as_of_months):
        since = upcoming = np.nan
        group = by_series.get((country, brand, model))
        if group is not None:
            seen = group.loc[[a <= known for a in group["announced"]]]
            done = [e for e in seen["event"] if e <= known]
            if done:
                since = float(np.clip(month.ordinal - max(done).ordinal, 0, 60))
            coming = [e for e in seen["event"] if e > known]
            if coming:
                upcoming = float(np.clip(min(coming).ordinal - month.ordinal, -24, 24))
        rows.append((since, upcoming))
    return pd.DataFrame(rows, columns=COLUMNS)
