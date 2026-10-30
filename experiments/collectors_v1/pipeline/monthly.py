"""매달 새 실적 묶음 — 수집기 출력 맞추기·빈 차종 채우기·원본 반영 (순수 함수, 네트워크 없음).

collect_month.py 와 merge_month.py 가 이 함수들을 쓴다. 규칙은 AGENTS.md 2절(2026-10-15 매달 재예측 운영).
"""
import pandas as pd

KEY = ["Country", "Brand", "Model"]
RAW_COLUMNS = ["Country", "Month", "Brand", "Model", "Sales"]
STD_COLUMNS = ["Country", "Brand", "Model", "Month", "Sales", "value_type", "estimated_share",
               "source_id", "confidence", "note"]
FILL_COLUMNS = ["Country", "Brand", "Model", "Month", "Sales", "source_run", "filled_at", "fill_type"]
EST_THRESHOLD = 0.30   # 추정 몫이 이 비율 미만이면 실제 값 취급(raw.csv), 이상이면 채움 표(2026-10-15 사용자: 20~30% 미만 → 30% 미만)
FILL_TYPE = "filled_forecast"


def _std(df, month, value_type, source_col="source_id", note_col=None):
    out = pd.DataFrame({
        "Country": df["Country"], "Brand": df["Brand"], "Model": df["Model"], "Month": month,
        "Sales": df["value"].astype(float), "value_type": value_type,
        "estimated_share": 0.0,
        "source_id": df[source_col] if source_col in df else "",
        "confidence": df["confidence"] if "confidence" in df else "high",
        "note": df["status"].astype(str) if note_col is None else df[note_col]})
    return out[STD_COLUMNS].reset_index(drop=True)


def adapt_china(df, month):
    """collect_china.py 출력: status ok* 이고 값 있는 줄만, ';allocation' 이면 배분."""
    d = df[(df["Month"].astype(str) == month) & df["value"].notna()
           & df["status"].astype(str).str.startswith("ok")]
    vt = d["status"].astype(str).str.contains("allocation").map({True: "allocated", False: "reported"})
    out = _std(d, month, vt)
    out["source_id"] = "CN_GASGOO_CPCA"
    if "flags" in d:
        out["note"] = (out["note"] + " " + d["flags"].fillna("").astype(str).values).str.strip()
    return out


GENERIC_TYPES = {"ALLOCATED": "allocated", "SPLIT": "split", "ESTIMATED": "estimated",
                 "FILLED_ALLOC": "estimated"}   # 일본: 창 잔여를 작년 비중으로 나눈 가정값
PENDING_STATUSES = {"NOT_POSTED", "ZERO_PLACEHOLDER"}   # 북미: 분기 발표 회사의 달이 아직 안 올라옴


def adapt_generic(df, month):
    """독일·일본·북미 출력(value, status 열): 값 없는 줄(NO_SOURCE 등)은 넣지 않는다."""
    d = df[(df["Month"].astype(str) == month) & df["value"].notna()]
    vt = d["status"].astype(str).map(lambda s: GENERIC_TYPES.get(s, "reported"))
    out = _std(d, month, vt)
    out.loc[out["value_type"] == "estimated", "estimated_share"] = 1.0
    return out


def pending(df, month):
    """값은 없지만 출처가 나중에 채울 줄(발표 대기). 다음 달 재수집 때 채움을 실제 값으로 바꿀 대상."""
    if "status" not in df:
        return pd.DataFrame(columns=KEY + ["Month", "status"])
    d = df[(df["Month"].astype(str) == month) & df["status"].astype(str).isin(PENDING_STATUSES)]
    return d[KEY + ["Month", "status"]].reset_index(drop=True)


def adapt_std(df, month):
    """한국·유럽처럼 이미 표준 계약으로 나오는 출력."""
    missing = [c for c in STD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"표준 열 없음: {missing}")
    return df[df["Month"].astype(str) == month][STD_COLUMNS].reset_index(drop=True) if len(df) else df[STD_COLUMNS]


def combine(frames):
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=STD_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    dup = out[out.duplicated(KEY + ["Month"], keep=False)]
    if len(dup):
        raise ValueError(f"같은 차종·달이 두 번 나옴:\n{dup[KEY + ['Month', 'source_id']].to_string()}")
    return out


def universe(raw, month):
    """대상 달 직전 12개월 안에 판매가 있었던 계열 = 이번 달에 값이 있어야 하는 계열."""
    m = pd.Period(month, freq="M")
    window = {str(m - i) for i in range(1, 13)}
    r = raw[raw["Month"].astype(str).isin(window) & (raw["Sales"] > 0)]
    return r[KEY].drop_duplicates().sort_values(KEY).reset_index(drop=True)


def fill_missing(got, univ, forecast, month, run_id):
    """수집 못 한 계열을 직전 납품 예측으로 채움. 예측도 없으면 missing 으로 돌려준다."""
    have = set(map(tuple, got[KEY].values))
    need = univ[[tuple(k) not in have for k in univ[KEY].values]]
    fc = forecast[forecast["Month"].astype(str) == month][KEY + ["prediction"]]
    j = need.merge(fc, on=KEY, how="left")
    ok = j[j["prediction"].notna()]
    filled = pd.DataFrame({
        "Country": ok["Country"], "Brand": ok["Brand"], "Model": ok["Model"], "Month": month,
        "Sales": ok["prediction"].astype(float), "value_type": FILL_TYPE, "estimated_share": 1.0,
        "source_id": run_id, "confidence": "low", "note": "직전 납품 예측으로 채움"})[STD_COLUMNS]
    missing = j[j["prediction"].isna()][KEY].reset_index(drop=True)
    return filled.reset_index(drop=True), missing


def empty_fills():
    return pd.DataFrame(columns=FILL_COLUMNS)


def _key(row):
    return (row["Country"], row["Brand"], row["Model"], str(row["Month"]))


def destination(row, threshold=EST_THRESHOLD):
    """한 줄이 어디로 가는지: 'raw'(실제·분기나눔·배분·추정 몫 < threshold) / 'estimated' / 'forecast'(채움 표)."""
    vt = row.get("value_type")
    if vt == FILL_TYPE:
        return "forecast"
    if vt == "estimated":
        share = row.get("estimated_share")
        share = 1.0 if share is None or pd.isna(share) else float(share)
        return "raw" if share < threshold else "estimated"
    return "raw"


def merge(raw, incoming, fills, filled_at="", threshold=EST_THRESHOLD):
    """incoming 을 원본(raw.csv)과 채움 표(filled_values.csv)로 나눠 넣는다(2026-10-15 사용자 규칙).
    - 원본에 이미 있는 칸은 다시 조사하지 않는다(값이 달라도 그대로; keep_existing_actual).
    - 실제·분기나눔·배분·추정 몫 < threshold → 원본. 채움에 있던 칸이면 채움에서 빼고 옮긴다(replace_fill_with_actual).
    - 추정 몫 ≥ threshold → 채움(fill_type=estimated), 직전 예측 → 채움(fill_type=forecast). 다음 재수집 때 다시 본다."""
    keys = list(zip(raw["Country"], raw["Brand"], raw["Model"], raw["Month"].astype(str)))
    raw_keys = set(keys)
    fills = fills.copy()
    if "fill_type" not in fills.columns:
        fills["fill_type"] = "forecast"
    fill_keys = {_key(r): i for i, r in fills.iterrows()}
    add, new_fill, log = [], [], []
    for _, r in incoming.iterrows():
        k = _key(r)
        new = float(r["Sales"])
        dest = destination(r, threshold)
        if k in raw_keys:
            action = "keep_existing_actual"
        elif dest == "raw":
            if k in fill_keys:
                fills = fills.drop(index=fill_keys.pop(k))
                action = "replace_fill_with_actual"
            else:
                action = "add_actual"
            add.append({"Country": k[0], "Month": k[3], "Brand": k[1], "Model": k[2], "Sales": new})
        else:
            if k in fill_keys:
                fills = fills.drop(index=fill_keys.pop(k))
                action = "update_fill"
            else:
                action = "add_fill"
            new_fill.append({"Country": k[0], "Brand": k[1], "Model": k[2], "Month": k[3], "Sales": new,
                             "source_run": r.get("source_id", ""), "filled_at": filled_at, "fill_type": dest})
        log.append({"Country": k[0], "Brand": k[1], "Model": k[2], "Month": k[3], "Sales": new,
                    "value_type": r.get("value_type"), "destination": dest, "action": action})
    new_raw = pd.concat([raw, pd.DataFrame(add, columns=RAW_COLUMNS)[list(raw.columns)]], ignore_index=True) if add else raw.copy()
    fills = pd.concat([fills, pd.DataFrame(new_fill, columns=FILL_COLUMNS)], ignore_index=True) if new_fill else fills.reset_index(drop=True)
    return new_raw, fills, pd.DataFrame(log, columns=KEY + ["Month", "Sales", "value_type", "destination", "action"])


def training_frame(raw, fills):
    """학습 입력 = 원본 + 채움(채움은 채점에서 빠지지만 재귀 입력으로는 쓴다)."""
    return pd.concat([raw[RAW_COLUMNS], fills[RAW_COLUMNS]], ignore_index=True)


def fill_streaks(fills):
    """계열별로 마지막 채움 달까지 연속 채움 개월 수. 3 이상이면 경보."""
    rows = []
    if "fill_type" in fills.columns:          # 추정 채움은 그 달 실제 자료가 섞인 값 → 경보 대상 아님
        fills = fills[fills["fill_type"].fillna("forecast") != "estimated"]
    for k, g in fills.groupby(KEY):
        ps = sorted(pd.Period(m, freq="M") for m in g["Month"].astype(str))
        n = 1
        for a, b in zip(ps[::-1][1:], ps[::-1]):
            if (b - a).n != 1:
                break
            n += 1
        rows.append({"Country": k[0], "Brand": k[1], "Model": k[2], "months": n, "last_month": str(ps[-1])})
    return pd.DataFrame(rows, columns=KEY + ["months", "last_month"])


def basis_mismatch(src_prev, raw, month, ratio=1.5, floor=300):
    """출처의 직전 달 값(src_prev)이 우리 원본 직전 달 값과 ratio 배 넘게 다르면(둘 중 큰 값 floor 이상) 그 계열은
    대응·기준이 어긋난 것으로 보고 이번 달 수집 값을 넣지 않는다(예측으로 채우고 확인 목록에 남김)."""
    prev = str(pd.Period(month, freq="M") - 1)
    r = raw[raw["Month"].astype(str) == prev][KEY + ["Sales"]].rename(columns={"Sales": "raw_prev"})
    j = src_prev.merge(r, on=KEY, how="inner")
    big = j[["src_prev", "raw_prev"]].max(axis=1) >= floor
    off = (j["src_prev"] > ratio * j["raw_prev"].clip(lower=1)) | (j["raw_prev"] > ratio * j["src_prev"].clip(lower=1))
    return j[big & off].reset_index(drop=True)


def drop_keys(df, keys):
    if not len(keys):
        return df
    k = set(map(tuple, keys[KEY].values))
    return df[[tuple(x) not in k for x in df[KEY].values]].reset_index(drop=True)


def month_range(first, last):
    a, b = pd.Period(first, freq="M"), pd.Period(last, freq="M")
    if b < a:
        raise ValueError(f"끝 달 {last} 이 시작 달 {first} 보다 앞섬")
    return [str(p) for p in pd.period_range(a, b, freq="M")]


def with_prior(raw, priors):
    """원본에 아직 넣지 않은 앞 달 수집 결과(incoming, 실제·채움 모두)를 붙여 다음 달 점검(대상 계열·직전 달 비교)에 쓴다.
    메모리 안에서만 쓰고 원본 파일은 바꾸지 않는다."""
    add = [p[RAW_COLUMNS] for p in priors if len(p)]
    if not add:
        return raw
    return pd.concat([raw] + [a[list(raw.columns)] for a in add], ignore_index=True)


def auto_months(raw_last, today, fill_months=(), recollect_window=3):
    """자동으로 모을 달 = (원본 마지막 달 다음 달, 아직 채움이 남은 가장 이른 달) 중 이른 달 ~ 오늘 기준 지난달.
    채움이 남은 달을 다시 모아야 나중에 발표된 실제 값으로 바뀐다. 단 채움 달은 지난달부터 recollect_window 개월
    안의 것만 다시 모은다(분기 발표·한 달 늦은 발표는 3개월이면 따라잡음). 계속 못 구하는 차종 때문에 오래된 달을
    매번 다시 모으는 일을 막는다. 원본 뒤 빈 달은 기간과 상관없이 모은다."""
    nxt = pd.Period(raw_last, freq="M") + 1
    prev = pd.Period(pd.Timestamp(today), freq="M") - 1
    recent = [pd.Period(x, freq="M") for x in fill_months if pd.Period(x, freq="M") > prev - recollect_window]
    if recent:
        nxt = min(nxt, min(recent))
    return [] if prev < nxt else [str(p) for p in pd.period_range(nxt, prev, freq="M")]


def pick_forecast(cands, month):
    """채울 달(month)보다 앞선 기준월의 예측 중 기준월이 가장 늦은 것(같으면 가장 최근 파일). 없으면 None."""
    c = cands[cands["origin"].astype(str) < month]
    if not len(c):
        return None
    return c.sort_values(["origin", "mtime"]).iloc[-1]["path"]
