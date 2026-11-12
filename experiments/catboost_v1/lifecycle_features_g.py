"""전 시장 차종 수명 특징 4개(2026-10-04 사전 고정 (dg)).

사건 표 = 중국 표(lifecycle_events_l2.csv, market_scope 없음 → China) + 조사 표(lifecycle_events_global.csv,
market_scope = ';'로 구분한 시장들). 기준 시점 as_of(학습 행 = 대상월 − 1, 재귀 예측 = 기준월 고정)까지
**공개 날짜가 지난 사건만** 쓴다. 공개 날짜가 없는 사건과 부분 변경(major_facelift)은 쓰지 않는다.

- lc_months_to_replacement: 대상월부터 알려진 가장 이른 단종·후속 사건월까지 개월(사건월이 대상월 24개월 전보다
  이르면 무시, −24~24로 자름). 없으면 NaN. (중국판과 같은 정의)
- lc_months_since_newgen: 대상월 − 알려진 가장 최근 신세대 출시월(출시월 ≤ as_of인 것만, 0~120). 없으면 NaN.
- lc_months_to_newgen: 공개됐지만 as_of에는 아직 출시 전인 신세대 중 가장 이른 출시월 − 대상월(−24~24). 없으면 NaN.
- lc_researched: (시장, 브랜드, 차종)이 조사 대상이면 1, 아니면 0.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CHINA_TABLE = HERE / "lifecycle_events_l2.csv"
GLOBAL_TABLE = HERE / "lifecycle_events_global.csv"
CHINA_TARGETS = HERE / "research" / "targets_v2.csv"
GLOBAL_TARGETS = HERE / "lifecycle_targets_global.csv"
COLUMNS = ["lc_months_to_replacement", "lc_months_since_newgen", "lc_months_to_newgen", "lc_researched"]
EVENT_TYPES = {"discontinued", "successor_launch", "new_generation", "major_facelift"}
REPLACEMENT = {"discontinued", "successor_launch"}
_EVENTS = None
_TARGETS = None


def _period(value, field):
    text = str(value).strip()
    match = re.fullmatch(r"(\d{4})-(\d{2})(?:-(\d{2}))?", text)
    if match is None or not 1 <= int(match.group(2)) <= 12:
        raise ValueError(f"{field} 날짜 형식이 잘못됐습니다: {value!r}")
    return pd.Period(f"{match.group(1)}-{match.group(2)}", freq="M")


def prepare_events(table):
    """market_scope(';' 구분) 행을 시장마다 펼치고 날짜를 검사한다. 공개 날짜가 없으면 announced = NaT."""
    missing = {"market_scope", "Brand", "Model", "event_type", "event_month", "announced_date"} - set(table.columns)
    if missing:
        raise ValueError(f"사건 표에 열이 없습니다: {sorted(missing)}")
    bad = sorted(set(table["event_type"]) - EVENT_TYPES)
    if bad:
        raise ValueError(f"알 수 없는 사건 종류: {bad}")
    rows = []
    for scope, brand, model, kind, month, announced in zip(table["market_scope"], table["Brand"], table["Model"],
                                                           table["event_type"], table["event_month"],
                                                           table["announced_date"]):
        event = _period(month, "event_month")
        known = _period(announced, "announced_date") if str(announced).strip() else pd.NaT
        for market in [m.strip() for m in str(scope).split(";") if m.strip()]:
            rows.append({"Country": market, "Brand": brand, "Model": model, "event_type": kind, "event": event,
                         "announced": known})
    return pd.DataFrame(rows, columns=["Country", "Brand", "Model", "event_type", "event", "announced"])


def load_events():
    china = pd.read_csv(CHINA_TABLE, dtype=str, keep_default_na=False, encoding="utf-8-sig").assign(market_scope="China")
    tables = [china]
    if GLOBAL_TABLE.exists():
        tables.append(pd.read_csv(GLOBAL_TABLE, dtype=str, keep_default_na=False, encoding="utf-8-sig"))
    columns = ["market_scope", "Brand", "Model", "event_type", "event_month", "announced_date"]
    return prepare_events(pd.concat([t[columns] for t in tables], ignore_index=True))


def load_targets():
    china = pd.read_csv(CHINA_TARGETS, dtype=str, encoding="utf-8-sig")
    targets = {("China", b, m) for b, m in zip(china["Brand"], china["Model"])}
    if GLOBAL_TARGETS.exists():
        other = pd.read_csv(GLOBAL_TARGETS, dtype=str, encoding="utf-8-sig")
        targets |= set(zip(other["Country"], other["Brand"], other["Model"]))
    return targets


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
    kept = events.loc[events["event_type"].ne("major_facelift") & events["announced"].notna()]
    by_series = {key: group for key, group in kept.groupby(["Country", "Brand", "Model"], sort=False)}
    months = pd.PeriodIndex(list(target_months), freq="M")
    as_of = pd.PeriodIndex(list(as_of_months), freq="M")
    rows = []
    for country, brand, model, month, known in zip(countries, brands, models, months, as_of):
        replacement = since = upcoming = np.nan
        group = by_series.get((country, brand, model))
        if group is not None:
            seen = group.loc[[a <= known for a in group["announced"]]]
            ends = [e for t, e in zip(seen["event_type"], seen["event"]) if t in REPLACEMENT and e >= month - 24]
            if ends:
                replacement = float(np.clip(min(ends).ordinal - month.ordinal, -24, 24))
            launched = [e for t, e in zip(seen["event_type"], seen["event"]) if t == "new_generation" and e <= known]
            if launched:
                since = float(np.clip(month.ordinal - max(launched).ordinal, 0, 120))
            coming = [e for t, e in zip(seen["event_type"], seen["event"]) if t == "new_generation" and e > known]
            if coming:
                upcoming = float(np.clip(min(coming).ordinal - month.ordinal, -24, 24))
        rows.append((replacement, since, upcoming, 1.0 if (country, brand, model) in targets else 0.0))
    return pd.DataFrame(rows, columns=COLUMNS)
