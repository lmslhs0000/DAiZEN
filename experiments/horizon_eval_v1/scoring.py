"""goal v2 채점: 부분별 묶음·달마다 오차, ABC 그룹, 점수표, 챔피언 규칙, 허용 폭 보정.

주 점수는 종합(combined) = (묶음 WAPE + 달마다 WAPE) ÷ 2다(2026-10-01 첫 챔피언 전 규칙 변경).
"""

import numpy as np
import pandas as pd

from common import KEY

PARTS = [("front", 1, 3, 1), ("middle", 4, 12, 3), ("back", 13, 24, 12)]
PART_NAMES = [name for name, *_ in PARTS]
GROUPS = ["A", "B", "C"]
ALARM_WAPE = 50.0
DEFAULT_TOLERANCE = {"part": 1.0, "monthly": 1.0, "group": 1.0, "alarm": 1.0, "market": 2.0}
MAIN = "combined"
METRICS = {"pooled_wape": "abs_error", "monthly_wape": "abs_error_monthly"}
MONTHLY_GUARD_PARTS = ["middle", "back"]  # 앞부분은 묶음과 달마다가 같아 한 번만 본다
SCORE_COLUMNS = ["scope", "scope_value", "part", "metric", "value", "n"]


def horizon_map(max_horizon):
    """예측 거리 → (부분, 묶음 번호). max_horizon 안에서 끝나는 부분만 포함한다."""
    rows = []
    for name, start, end, size in PARTS:
        if end > max_horizon:
            continue
        for h in range(start, end + 1):
            rows.append({"horizon": h, "part": name, "bucket": (h - start) // size, "bucket_size": size})
    return pd.DataFrame(rows)


def part_errors(predictions, data, max_horizon):
    """차종·기준월·부분별 실제 합과 묶음 절대오차 합. 부분 전체가 유효할 때만 OK."""
    mapping = horizon_map(max_horizon)
    actual = data[KEY + ["Month", "Sales"]].rename(columns={"Sales": "actual"})
    table = predictions.merge(mapping, on="horizon").merge(actual, on=KEY + ["Month"], how="left")
    table["month_error"] = (table["prediction"] - table["actual"]).abs()
    ids = ["model", "origin"] + KEY + ["part"]
    buckets = (table.groupby(ids + ["bucket", "bucket_size"], sort=True)
               .agg(actual_n=("actual", "count"), prediction_n=("prediction", "count"),
                    actual_sum=("actual", "sum"), prediction_sum=("prediction", "sum"),
                    month_error=("month_error", "sum"))
               .reset_index())
    buckets["missing_actual"] = buckets["actual_n"].lt(buckets["bucket_size"])
    buckets["missing_prediction"] = buckets["prediction_n"].lt(buckets["bucket_size"])
    buckets["abs_error"] = (buckets["prediction_sum"] - buckets["actual_sum"]).abs()
    parts = (buckets.groupby(ids, sort=True)
             .agg(buckets=("bucket", "size"), missing_actual=("missing_actual", "any"),
                  missing_prediction=("missing_prediction", "any"),
                  actual_sum=("actual_sum", "sum"), abs_error=("abs_error", "sum"),
                  abs_error_monthly=("month_error", "sum"))
             .reset_index())
    expected = mapping.groupby("part")["bucket"].nunique()
    parts["missing_prediction"] |= parts["buckets"].lt(parts["part"].map(expected))
    parts["status"] = np.select([parts["missing_prediction"], parts["missing_actual"]],
                                ["MISSING_PREDICTION", "MISSING_ACTUAL"], "OK")
    ok = parts["status"].eq("OK")
    parts.loc[~ok, ["actual_sum", "abs_error", "abs_error_monthly"]] = np.nan
    positive = ok & parts["actual_sum"].gt(0)
    for error, wape in (("abs_error", "wape"), ("abs_error_monthly", "wape_monthly")):
        parts[wape] = np.nan
        parts.loc[positive, wape] = 100 * parts.loc[positive, error] / parts.loc[positive, "actual_sum"]
    return parts[ids + ["status", "actual_sum", "abs_error", "abs_error_monthly", "wape", "wape_monthly"]]


def abc_groups(data, origin):
    """기준월 직전 12개월 판매로 시장마다 A(앞선 누적 80% 미만)/B(95% 미만)/C를 정한다."""
    origin = pd.Period(origin, freq="M")
    window = data.loc[data["Month"].between(origin - 11, origin)]
    totals = window.groupby(KEY, sort=True)["Sales"].sum(min_count=1).fillna(0.0).reset_index()
    tables = []
    for _, market in totals.groupby("Country", sort=True):
        market = market.sort_values(["Sales"] + KEY, ascending=[False, True, True, True])
        total = market["Sales"].sum()
        before = (market["Sales"].cumsum() - market["Sales"]) / total if total > 0 else pd.Series(1.0, index=market.index)
        tables.append(market.assign(abc=np.select([before.lt(0.80), before.lt(0.95)], ["A", "B"], "C")))
    result = pd.concat(tables, ignore_index=True) if tables else totals.assign(abc=pd.Series(dtype=str))
    result.loc[result["Sales"].le(0), "abc"] = "C"
    result.insert(0, "origin", str(origin))
    return result[["origin"] + KEY + ["abc"]].sort_values(KEY).reset_index(drop=True)


def _pooled(frame, error="abs_error"):
    actual = frame["actual_sum"].sum()
    return 100 * frame[error].sum() / actual if actual > 0 else np.nan


def score_model(parts, groups):
    """한 모델의 OK 부분 행과 ABC 그룹으로 전체·그룹·시장·기준월 점수표(긴 형식)를 만든다."""
    table = parts.merge(groups, on=["origin"] + KEY, how="left")
    if table["abc"].isna().any():
        raise ValueError("ABC 그룹이 없는 차종·기준월이 있습니다.")
    rows = []

    def add(scope, value, part, metric, number, count):
        rows.append({"scope": scope, "scope_value": str(value), "part": part, "metric": metric,
                     "value": float(number), "n": int(count)})

    def add_pooled(scope, value, subset):
        """부분마다 묶음·달마다·종합 점수, 세 부분이 다 있으면 각각의 main(세 부분 평균)."""
        found = {metric: [] for metric in (*METRICS, MAIN)}
        for part in PART_NAMES:
            chunk = subset.loc[subset["part"].eq(part)]
            if not len(chunk):
                continue
            values = {metric: _pooled(chunk, error) for metric, error in METRICS.items()}
            values[MAIN] = np.mean(list(values.values()))
            for metric, number in values.items():
                found[metric].append(number)
                add(scope, value, part, metric, number, len(chunk))
        if len(found[MAIN]) == len(PART_NAMES):
            for metric, numbers in found.items():
                add(scope, value, "main", metric, np.mean(numbers), len(subset))

    add_pooled("overall", "all", table)
    for group in GROUPS:
        subset = table.loc[table["abc"].eq(group)]
        add_pooled("group", group, subset)
        for part in PART_NAMES:
            valid = subset.loc[subset["part"].eq(part), "wape"].dropna()
            add("group", group, part, "median_wape", valid.median() if len(valid) else np.nan, len(valid))
        valid = subset["wape"].dropna()
        add("group", group, "all", "alarm_share", 100 * valid.gt(ALARM_WAPE).mean() if len(valid) else np.nan, len(valid))
    for country, subset in table.groupby("Country", sort=True):
        add_pooled("market", country, subset)
    for origin, subset in table.groupby("origin", sort=True):
        add_pooled("origin", origin, subset)
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def cell(scores, scope, value, part, metric="pooled_wape"):
    hit = scores.loc[scores["scope"].eq(scope) & scores["scope_value"].eq(str(value))
                     & scores["part"].eq(part) & scores["metric"].eq(metric), "value"]
    return float(hit.iloc[0]) if len(hit) == 1 else np.nan


COVERAGE_MIN = 0.995


def learnable_keys(data, origin):
    """기준월까지 최근 13개월(학습 1행 = 입력 12 + 정답 1)이 모두 관측된 차종. 학습형 모델이 규칙상 예측할 수 있는 범위."""
    origin = pd.Period(origin, freq="M")
    window = data.loc[data["Month"].between(origin - 12, origin) & data["Sales"].notna()]
    counts = window.groupby(KEY)["Month"].nunique()
    keys = counts[counts.eq(13)].index.to_frame(index=False)
    keys.insert(0, "origin", str(origin))
    return keys


def coverage(parts, eligible=None):
    """공통 대상으로 좁히기 전, 모델별 예측에 성공한(OK) 기준월·차종·부분 수.

    eligible(origin+차종)을 주면 그 범위 안에서만 센다. 12개월 이력 신차처럼 학습형 모델이 규칙상
    예측할 수 없는 차종 때문에 기준선과의 예측 범위가 벌어지는 것을 빼기 위해서다.
    """
    ok = parts.loc[parts["status"].eq("OK")]
    if eligible is not None:
        ok = ok.merge(eligible, on=["origin"] + KEY)
    return {model: int(n) for model, n in ok.groupby("model").size().items()}


def _values(scores, scope):
    return sorted(scores.loc[scores["scope"].eq(scope), "scope_value"].unique())


def decide(champion, challenger, baseline, tolerances=None, coverage=None):
    """② 규칙: 모든 검사를 통과해야 채택한다. (채택 여부, 검사표)를 반환한다."""
    tolerances = tolerances or {}
    checks = []

    def limit(name):
        return float(tolerances.get(name, DEFAULT_TOLERANCE[name.split(":")[0]]))

    def record(name, before, after, allowed, passed):
        checks.append({"check": name, "champion": before, "challenger": after,
                       "delta": after - before, "allowed": allowed, "passed": bool(passed)})

    def guard(name, before, after):
        allowed = limit(name)
        both_missing = np.isnan(before) and np.isnan(after)
        record(name, before, after, allowed, both_missing or after - before <= allowed)

    before, after = cell(champion, "overall", "all", "main", MAIN), cell(challenger, "overall", "all", "main", MAIN)
    record("main_improved", before, after, 0.0, after < before)
    for part in PART_NAMES:
        guard(f"part:{part}", cell(champion, "overall", "all", part), cell(challenger, "overall", "all", part))
    for part in MONTHLY_GUARD_PARTS:
        guard(f"monthly:{part}", cell(champion, "overall", "all", part, "monthly_wape"),
              cell(challenger, "overall", "all", part, "monthly_wape"))
    for group in GROUPS:
        guard(f"group:{group}", cell(champion, "group", group, "main", MAIN),
              cell(challenger, "group", group, "main", MAIN))
        guard(f"alarm:{group}", cell(champion, "group", group, "all", "alarm_share"),
              cell(challenger, "group", group, "all", "alarm_share"))
    for market in _values(champion, "market"):
        guard(f"market:{market}", cell(champion, "market", market, "main", MAIN),
              cell(challenger, "market", market, "main", MAIN))
    for part in PART_NAMES:
        before, after = cell(baseline, "overall", "all", part, MAIN), cell(challenger, "overall", "all", part, MAIN)
        record(f"beats_baseline:{part}", before, after, 0.0, after < before)
    origins = _values(champion, "origin")
    wins = sum(cell(challenger, "origin", o, "main", MAIN) < cell(champion, "origin", o, "main", MAIN)
               for o in origins)
    record("origin_majority", len(origins) / 2, float(wins), 0.0, wins > len(origins) / 2)
    if coverage is not None:
        before, after = (float(value) for value in coverage)
        record("coverage", before, after, COVERAGE_MIN, after >= COVERAGE_MIN * before)
    table = pd.DataFrame(checks)
    return bool(table["passed"].all()), table


def choose_initial_champion(scores_by_model, baseline_name, coverage=None):
    """모든 부분의 종합 점수가 기준선보다 낮고 예측 범위가 기준선의 99.5% 이상인 후보 중 종합 주 점수 최저. 없으면 기준선."""
    base = scores_by_model[baseline_name]
    eligible = []
    for name, scores in sorted(scores_by_model.items()):
        if name == baseline_name:
            continue
        if coverage is not None and coverage.get(name, 0) < COVERAGE_MIN * coverage[baseline_name]:
            continue
        if all(cell(scores, "overall", "all", p, MAIN) < cell(base, "overall", "all", p, MAIN) for p in PART_NAMES):
            eligible.append((cell(scores, "overall", "all", "main", MAIN), name))
    return min(eligible)[1] if eligible else baseline_name


def bootstrap_tolerances(parts, groups, n=1000, seed=42):
    """차종 단위 복원 추출로 칸별 표준편차를 재고, 허용 폭 = max(기본값, 2×표준편차)."""
    table = parts.merge(groups, on=["origin"] + KEY, how="left").reset_index(drop=True)
    vehicles = table[KEY].drop_duplicates().reset_index(drop=True)
    owner = table.merge(vehicles.reset_index().rename(columns={"index": "vehicle"}), on=KEY, how="left")["vehicle"].to_numpy()
    actual = table["actual_sum"].to_numpy(float)
    error = table["abs_error"].to_numpy(float)
    error_monthly = table["abs_error_monthly"].to_numpy(float)
    wape = table["wape"].to_numpy(float)
    valid = np.isfinite(wape)
    alarm = valid & (np.nan_to_num(wape) > ALARM_WAPE)
    part, abc, market = (table[c].to_numpy() for c in ("part", "abc", "Country"))
    part_masks = {p: part == p for p in PART_NAMES}

    def pooled(w, mask, errors=error):
        denominator = (w * actual)[mask].sum()
        return 100 * (w * errors)[mask].sum() / denominator if denominator > 0 else np.nan

    def main(w, mask):
        """종합 주 점수: 부분마다 (묶음 + 달마다) ÷ 2를 세 부분 평균."""
        return float(np.mean([(pooled(w, mask & part_masks[p]) + pooled(w, mask & part_masks[p], error_monthly)) / 2
                              for p in PART_NAMES]))

    def alarm_share(w, mask):
        denominator = (w * valid)[mask].sum()
        return 100 * (w * alarm)[mask].sum() / denominator if denominator > 0 else np.nan

    cells = {f"part:{p}": (lambda w, p=p: pooled(w, part_masks[p])) for p in PART_NAMES}
    cells.update({f"monthly:{p}": (lambda w, p=p: pooled(w, part_masks[p], error_monthly))
                  for p in MONTHLY_GUARD_PARTS})
    cells.update({f"group:{g}": (lambda w, g=g: main(w, abc == g)) for g in GROUPS})
    cells.update({f"alarm:{g}": (lambda w, g=g: alarm_share(w, abc == g)) for g in GROUPS})
    cells.update({f"market:{m}": (lambda w, m=m: main(w, market == m)) for m in sorted(set(market))})
    rng = np.random.default_rng(seed)
    uniform = np.full(len(vehicles), 1 / len(vehicles))
    draws = {name: [] for name in cells}
    for _ in range(n):
        weights = rng.multinomial(len(vehicles), uniform)[owner].astype(float)
        for name, function in cells.items():
            draws[name].append(function(weights))
    result = {}
    for name, values in draws.items():
        values = np.asarray(values, float)
        finite = values[np.isfinite(values)]
        std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        default = DEFAULT_TOLERANCE[name.split(":")[0]]
        result[name] = {"default": default, "bootstrap_std": std, "tolerance": max(default, 2 * std)}
    return result
