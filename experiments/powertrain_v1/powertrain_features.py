"""차종(Brand, Model)별 동력원을 그 시점 기준 숫자 입력으로 바꾼다(2026-10-01 사전 고정).

표(powertrain.csv): Brand, Model, powertrain(BEV/PHEV/MIX/ICE/UNKNOWN), plugin_from(플러그인 버전이 처음 나온 해),
nev_only_from(플러그인만 팔기 시작한 해), confidence, basis. 조사 에이전트 3개가 브랜드 규칙·이름·웹 확인으로 만들었다.

그 해의 범주
- plugin_from 전(또는 없음) → ICE (UNKNOWN 표시는 연도가 없으면 UNKNOWN, 연도가 있으면 그 해부터 플러그인: NEV 전용이면 PHEV 코드, 아니면 MIX)
- nev_only_from 이후 → BEV 표시면 BEV, 아니면 PHEV
- 그 밖 → 표의 범주(BEV/PHEV/MIX)
코드: ICE 0, MIX 1, PHEV 2, BEV 3, UNKNOWN은 NaN(결측). 플러그인 여부 = 그 해 범주가 BEV·PHEV·MIX.
기준 시점: 학습 행은 대상월 직전 달의 해, 재귀 예측은 기준월의 해(미래의 신형 플러그인 출시는 모른다고 본다).
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TABLE = HERE / "powertrain.csv"
CODES = {"ICE": 0.0, "MIX": 1.0, "PHEV": 2.0, "BEV": 3.0}
_TABLE = None


def load_table(path=None):
    table = pd.read_csv(TABLE if path is None else path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return _prepare(table)


def _year(value):
    return int(value) if str(value).strip() else None


def _prepare(table):
    if not set(table["powertrain"]) <= set(CODES) | {"UNKNOWN"}:
        raise ValueError("동력원 범주가 규칙 밖입니다.")
    if table.duplicated(["Brand", "Model"]).any():
        raise ValueError("같은 차종이 두 번 있습니다.")
    return {(b, m): (p, _year(f), _year(n)) for b, m, p, f, n in
            zip(table["Brand"], table["Model"], table["powertrain"], table["plugin_from"], table["nev_only_from"])}


def category(entry, year):
    powertrain, plugin_from, nev_only_from = entry
    if powertrain == "UNKNOWN":
        # 범주를 모르더라도 플러그인 연도가 있으면(전기 전용 브랜드 등) 그 해부터 플러그인으로 본다.
        if plugin_from is None:
            return "UNKNOWN"
        if year < plugin_from:
            return "ICE"
        return "PHEV" if nev_only_from is not None and year >= nev_only_from else "MIX"
    if plugin_from is None or year < plugin_from:
        return "ICE"
    if nev_only_from is not None and year >= nev_only_from:
        return "BEV" if powertrain == "BEV" else "PHEV"
    return powertrain if powertrain in ("BEV", "PHEV", "MIX") else "ICE"


def powertrain_codes(brands, models, years, table=None):
    """(코드 배열, 플러그인 여부 배열). 표에 없는 차종은 오류."""
    global _TABLE
    if table is None:
        if _TABLE is None:
            _TABLE = load_table()
        lookup = _TABLE
    else:
        lookup = _prepare(table)
    code, plugin = [], []
    for brand, model, year in zip(brands, models, years):
        key = (str(brand), str(model))
        if key not in lookup:
            raise ValueError(f"동력원 표에 없는 차종: {key}")
        name = category(lookup[key], int(year))
        code.append(np.nan if name == "UNKNOWN" else CODES[name])
        plugin.append(1.0 if name in ("BEV", "PHEV", "MIX") else 0.0)
    return np.asarray(code, float), np.asarray(plugin, float)
