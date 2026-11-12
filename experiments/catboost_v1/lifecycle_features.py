"""중국 차종 수명 특징 3개(2026-10-02 사전 고정 (l2)).

기준 시점 as_of(학습 행 = 대상월 − 1, 재귀 예측 = 기준월 고정)까지 **처음 공개된 날짜가 지난 사건만** 쓴다.
공개 날짜가 없는 사건과 부분 변경(major_facelift)은 쓰지 않는다. 중국 외 시장은 모두 NaN이다.

- lc_months_to_replacement: 대상월부터 알려진 가장 이른 단종·후속 차종 사건월까지 개월(사건월이 대상월 24개월 전보다
  이르면 무시, −24~24로 자름). 없으면 NaN.
- lc_months_since_newgen: 대상월 − 알려진 가장 최근 같은 이름 신세대 출시월(출시월 ≤ as_of인 것만, 0~120으로 자름).
  없으면 NaN.
- lc_researched: 중국 차종이 조사 대상(targets_v2, 2022-06 이전 판매량 상위 150)이면 1, 아니면 0.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TABLE = HERE / "lifecycle_events_l2.csv"  # 다른 세션의 lifecycle_events.csv와 겹치지 않게
TARGETS = HERE / "research" / "targets_v2.csv"
COLUMNS = ["lc_months_to_replacement", "lc_months_since_newgen", "lc_researched"]
EVENT_TYPES = {"discontinued", "successor_launch", "new_generation", "major_facelift"}
REPLACEMENT = {"discontinued", "successor_launch"}
MARKET = "China"
_EVENTS = None
_TARGETS = None


def _period(value, field):
    text = str(value).strip()
    match = re.fullmatch(r"(\d{4})-(\d{2})(?:-(\d{2}))?", text)
    if match is None or not 1 <= int(match.group(2)) <= 12:
        raise ValueError(f"{field} 날짜 형식이 잘못됐습니다: {value!r}")
    return pd.Period(f"{match.group(1)}-{match.group(2)}", freq="M")


def load_events(path=None):
    """사건 표를 읽어 검사한다. 공개 날짜가 없는 행은 남겨 두되 announced = NaT."""
    table = pd.read_csv(TABLE if path is None else path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = {"Brand", "Model", "event_type", "event_month", "announced_date"} - set(table.columns)
    if missing:
        raise ValueError(f"사건 표에 열이 없습니다: {sorted(missing)}")
    bad = sorted(set(table["event_type"]) - EVENT_TYPES)
    if bad:
        raise ValueError(f"알 수 없는 사건 종류: {bad}")
    table = table.copy()
    table["event"] = [_period(v, "event_month") for v in table["event_month"]]
    table["announced"] = [_period(v, "announced_date") if str(v).strip() else pd.NaT for v in table["announced_date"]]
    return table


def load_targets(path=None):
    """조사 대상 차종 (Brand, Model) 집합."""
    table = pd.read_csv(TARGETS if path is None else path, dtype=str, encoding="utf-8-sig")
    return set(zip(table["Brand"], table["Model"]))


def _usable(events):
    """공개 날짜가 있는 단종·후속·신세대 사건만, 차종별로 묶는다."""
    kept = events.loc[events["event_type"].ne("major_facelift") & events["announced"].notna()]
    return {key: group for key, group in kept.groupby(["Brand", "Model"], sort=False)}


def lifecycle_features(countries, brands, models, target_months, as_of_months, events=None, targets=None):
    global _EVENTS, _TARGETS
    if events is None:
        if _EVENTS is None:
            _EVENTS = load_events()
        events = _EVENTS
    if targets is None:
        if _TARGETS is None:
            _TARGETS = load_targets()
        targets = _TARGETS
    by_series = _usable(events)
    months = pd.PeriodIndex(list(target_months), freq="M")
    as_of = pd.PeriodIndex(list(as_of_months), freq="M")
    rows = []
    for country, brand, model, month, known in zip(countries, brands, models, months, as_of):
        if country != MARKET:
            rows.append((np.nan, np.nan, np.nan))
            continue
        group = by_series.get((brand, model))
        replacement = since = np.nan
        if group is not None:
            seen = group.loc[[a <= known for a in group["announced"]]]
            ends = [e for t, e in zip(seen["event_type"], seen["event"]) if t in REPLACEMENT and e >= month - 24]
            if ends:
                replacement = float(np.clip(min(ends).ordinal - month.ordinal, -24, 24))
            launched = [e for t, e in zip(seen["event_type"], seen["event"]) if t == "new_generation" and e <= known]
            if launched:
                since = float(np.clip(month.ordinal - max(launched).ordinal, 0, 120))
        rows.append((replacement, since, 1.0 if (brand, model) in targets else 0.0))
    return pd.DataFrame(rows, columns=COLUMNS)
