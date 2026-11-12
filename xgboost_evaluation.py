"""완성된 재귀 예측에 정답을 연결하고 차종별 WAPE와 공통 비교를 만든다."""

import math

import pandas as pd

from baseline import calculate_wape


SERIES_KEY = ["Country", "Brand", "Model"]
METADATA_COLUMNS = ["algorithm", "method", "split", "train_end", "origin"]
MONTHLY_COLUMNS = METADATA_COLUMNS + SERIES_KEY + [
    "Month", "horizon", "actual", "prediction", "absolute_error",
    "prediction_status", "status", "reason", "supplementary_reasons",
]
METRIC_COLUMNS = METADATA_COLUMNS + SERIES_KEY + [
    "expected_months", "observed_months", "actual_sum", "absolute_error_sum", "WAPE",
    "status", "reason", "prediction_status", "supplementary_reasons",
]


def checked_copy(table, columns, unique_key, name):
    """필요한 열·중복을 검사한 복사본을 만들어 원본 표를 보존한다."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{name} 필수 열이 없습니다: {missing}")
    result = table[columns].copy()
    if result.duplicated(unique_key).any():
        raise ValueError(f"{name}에 중복 차종·월 또는 차종 키가 있습니다.")
    return result


def verify_members(table, universe, name):
    """대상 목록 밖의 예측을 조용히 버리지 않고 입력 오류로 알린다."""
    membership = table[SERIES_KEY].merge(universe, on=SERIES_KEY, how="left", indicator=True)
    if membership["_merge"].eq("left_only").any():
        raise ValueError(f"{name}에 평가 대상 목록에 없는 차종이 있습니다.")


def evaluate_forecast(predictions, statuses, actuals, universe, *, method, split,
                      train_end, origin, horizon):
    """예측 완료 표·상태·정답·전체 차종을 받아 월별 결과와 차종별 성적을 반환한다."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 12:
        raise ValueError("평가 horizon은 1~12개월 정수여야 합니다.")
    origin_month = pd.Period(origin, freq="M")
    train_end_month = pd.Period(train_end, freq="M")
    if train_end_month > origin_month:
        raise ValueError("학습 종료월은 관측 기준월보다 뒤일 수 없습니다.")
    universe = checked_copy(universe, SERIES_KEY, SERIES_KEY, "평가 대상")
    predictions = checked_copy(
        predictions, SERIES_KEY + ["Month", "horizon", "prediction"], SERIES_KEY + ["Month"], "예측표"
    )
    statuses = checked_copy(
        statuses, SERIES_KEY + ["prediction_status", "reason"], SERIES_KEY, "예측 상태표"
    )
    actuals = checked_copy(actuals, SERIES_KEY + ["Month", "Sales"], SERIES_KEY + ["Month"], "실제값 표")
    verify_members(predictions, universe, "예측표")
    verify_members(statuses, universe, "예측 상태표")

    target_months = pd.period_range(origin_month + 1, periods=horizon, freq="M")
    predictions["Month"] = pd.PeriodIndex(predictions["Month"], freq="M")
    actuals["Month"] = pd.PeriodIndex(actuals["Month"], freq="M")
    # 문자열과 Period가 섞여 있어도 같은 달로 정규화한 뒤 중복을 다시 검사한다.
    for table, name in ((predictions, "예측표"), (actuals, "실제값 표")):
        if table.duplicated(SERIES_KEY + ["Month"]).any():
            raise ValueError(f"{name}에 같은 차종·월 중복이 있습니다.")
    expected_steps = {month: step for step, month in enumerate(target_months, start=1)}
    correct_steps = predictions["Month"].map(expected_steps)
    if not predictions["horizon"].eq(correct_steps).all():
        raise ValueError("예측 대상 월과 horizon의 대응이 다르거나 평가 기간 밖의 예측입니다.")

    actuals["Sales"] = pd.to_numeric(actuals["Sales"], errors="raise").astype(float)
    valid_actuals = actuals["Sales"].dropna()
    if not valid_actuals.map(math.isfinite).all() or valid_actuals.lt(0).any():
        raise ValueError("실제 판매량은 유한한 0 이상 수치여야 합니다.")
    predictions["prediction"] = pd.to_numeric(predictions["prediction"], errors="coerce").astype(float)

    # 예측이 없는 차종도 요청한 모든 목표월을 남긴다. 실제값은 이 시점에 처음 연결한다.
    calendar = pd.DataFrame({"Month": target_months, "horizon": range(1, horizon + 1)})
    grid = universe.merge(calendar, how="cross")
    grid = grid.merge(predictions.drop(columns="horizon"), on=SERIES_KEY + ["Month"], how="left", validate="one_to_one")
    grid = grid.merge(statuses, on=SERIES_KEY, how="left", validate="many_to_one")
    target_actuals = actuals.loc[actuals["Month"].isin(target_months)].rename(columns={"Sales": "actual"})
    grid = grid.merge(target_actuals, on=SERIES_KEY + ["Month"], how="left", validate="one_to_one")
    metadata = {"algorithm": "xgboost", "method": method, "split": split,
                "train_end": train_end_month, "origin": origin_month}
    monthly_tables = []
    metric_records = []
    for identifiers, group in grid.groupby(SERIES_KEY, sort=True):
        group = group.sort_values("Month").copy()
        first_row = group.iloc[0]
        prediction_status = first_row["prediction_status"]
        if pd.isna(prediction_status) or not str(prediction_status).strip():
            prediction_status = "INSUFFICIENT_HISTORY"
        reasons = []
        if pd.notna(first_row["reason"]) and str(first_row["reason"]).strip():
            reasons.append(str(first_row["reason"]))
        reason_codes = []
        if prediction_status != "OK":
            reason_codes.append(prediction_status)
        valid_predictions = group["prediction"].map(lambda value: pd.notna(value) and math.isfinite(value) and value >= 0)
        if prediction_status == "OK" and not valid_predictions.all():
            prediction_status = "PREDICTION_ERROR"
            reason_codes.append("PREDICTION_ERROR")
            reasons.append("예측 대상 월 누락 또는 비유한·음수 예측값이 있습니다.")
        # 실패 이전에 만들어진 유효한 값은 진단용으로 남겨도 부분 WAPE는 계산하지 않는다.
        group.loc[~valid_predictions, "prediction"] = float("nan")
        if prediction_status not in ("OK", "PREDICTION_ERROR"):
            group["prediction"] = float("nan")
        group["absolute_error"] = (group["actual"] - group["prediction"]).abs()
        observed_months = int(group["actual"].notna().sum())
        if observed_months != horizon:
            missing_months = group.loc[group["actual"].isna(), "Month"].astype(str).tolist()
            reason_codes.append("MISSING_ACTUAL")
            reasons.append(f"평가 실제값 누락·공란: {missing_months}")
        elif group["actual"].eq(0).all():
            reason_codes.append("ZERO_ACTUAL_SUM")
            reasons.append("평가 기간 실제 판매량 합계가 0입니다.")

        metric = {"expected_months": horizon, "observed_months": observed_months,
                  "actual_sum": None, "absolute_error_sum": None, "WAPE": None,
                  "status": prediction_status}
        if prediction_status == "OK":
            vehicle_actuals = group[SERIES_KEY + ["Month", "actual"]].rename(columns={"actual": "Sales"})
            try:
                _, metric = calculate_wape(vehicle_actuals, group[["Month", "prediction"]])
            except ValueError as error:
                # 합계·나눗셈의 수치 범위 초과를 정상 점수나 무한대 WAPE로 저장하지 않는다.
                metric["status"] = "PREDICTION_ERROR"
                reason_codes.append("NUMERIC_ERROR")
                reasons.append(f"평가 수치 계산 오류: {error}")
            else:
                if metric["status"] == "TARGET_MISSING":
                    metric["status"] = "MISSING_ACTUAL"
        if not reasons and prediction_status != "OK":
            reasons.append(f"예측 불가 상태: {prediction_status}")
        supplementary = ";".join(dict.fromkeys(code for code in reason_codes if code != metric["status"]))
        details = {"prediction_status": prediction_status, "status": metric["status"],
                   "reason": "; ".join(reasons), "supplementary_reasons": supplementary}
        for column, value in {**metadata, **details}.items():
            group[column] = value
        monthly_tables.append(group[MONTHLY_COLUMNS])
        metric_records.append({**metadata, **dict(zip(SERIES_KEY, identifiers)), **metric, **details})

    monthly_results = pd.concat(monthly_tables, ignore_index=True) if monthly_tables else pd.DataFrame(columns=MONTHLY_COLUMNS)
    metrics = pd.DataFrame(metric_records, columns=METRIC_COLUMNS)
    return monthly_results, metrics


def finite_nonnegative_mean(values):
    """큰 유한 값의 중간 합계가 넘치지 않도록 최대값으로 나눠 평균을 구한다."""
    maximum = max(values)
    if maximum == 0:
        return 0.0
    normalized_mean = math.fsum(value / maximum for value in values) / len(values)
    return maximum * normalized_mean


def score_validation_candidate(metrics, method, cohort):
    """독립적으로 정한 동일 차종 집합으로 후보의 검증 WAPE를 동일 가중 평균한다."""
    if method not in ("A", "B"):
        raise ValueError("검증 방식은 A 또는 B여야 합니다.")
    cohort = checked_copy(cohort, SERIES_KEY, SERIES_KEY, "검증 코호트")
    result = {"status": "NO_VALIDATION_SCORE", "validation_score": None,
              "evaluable_series": 0, "reason": "고정 검증 대상 차종이 없습니다."}
    if cohort.empty:
        return result
    required_splits = ["A_validation_1", "A_validation_2", "A_validation_3"] if method == "A" else ["B_validation"]
    validation = metrics.loc[metrics["method"].eq(method) & metrics["split"].isin(required_splits)].copy()
    result.update(status="CANDIDATE_ERROR", reason="후보 검증 오류")
    if validation.duplicated(SERIES_KEY + ["split"]).any():
        result["reason"] = "후보 검증 차종·구간 중복이 있습니다."
        return result
    prediction_errors = validation["status"].eq("PREDICTION_ERROR")
    if "prediction_status" in validation:
        prediction_errors |= validation["prediction_status"].eq("PREDICTION_ERROR")
    if prediction_errors.any():
        result["reason"] = "예측 실패가 있는 후보는 차종을 제외해 점수를 계산하지 않습니다."
        return result

    expected = cohort.merge(pd.DataFrame({"split": required_splits}), how="cross")
    scored = expected.merge(validation[SERIES_KEY + ["split", "status", "WAPE"]],
                            on=SERIES_KEY + ["split"], how="left", validate="one_to_one", indicator=True)
    if scored["_merge"].eq("left_only").any():
        result["reason"] = "고정 검증 코호트의 차종·구간 결과가 누락되었습니다."
        return result
    scores = pd.to_numeric(scored["WAPE"], errors="coerce")
    finite_scores = scores.map(lambda value: pd.notna(value) and math.isfinite(value) and value >= 0)
    if not (scored["status"].eq("OK") & finite_scores).all():
        result["reason"] = "고정 검증 코호트의 모든 차종·구간이 유효해야 합니다."
        return result
    scored["WAPE"] = scores
    # 먼저 차종 안에서 fold 평균을 내고, 다음에 차종마다 같은 가중치를 준다.
    series_scores = []
    for _, group in scored.groupby(SERIES_KEY, sort=True):
        series_scores.append(finite_nonnegative_mean(group["WAPE"].tolist()))
    validation_score = finite_nonnegative_mean(series_scores)
    if not math.isfinite(validation_score):
        result["reason"] = "후보 검증 평균이 유한하지 않습니다."
        return result
    return {"status": "OK", "validation_score": validation_score,
            "evaluable_series": len(series_scores), "reason": ""}


def compare_common_test(baseline_metrics, xgb_metrics, universe):
    """동일 6개월에 세 결과가 모두 유효한 차종을 표시하고 WAPE 차이를 계산한다."""
    comparison = checked_copy(universe, SERIES_KEY, SERIES_KEY, "공통 비교 대상")
    for label, table, method, split in (
        ("baseline", baseline_metrics, "A", "A_test"),
        ("A", xgb_metrics, "A", "A_test"),
        ("B", xgb_metrics, "B", "B_common_test_6m"),
    ):
        selected = table.loc[table["split"].eq(split)].copy()
        if selected.duplicated(SERIES_KEY).any():
            raise ValueError(f"{label} 공통 비교에 차종 중복이 있습니다.")
        correct_period = (selected["method"].eq(method) & selected["origin"].astype(str).eq("2026-01")
                          & selected["expected_months"].eq(6))
        if not correct_period.all():
            raise ValueError(f"{label} 공통 비교는 2026-01 기준 6개월 결과만 허용합니다.")
        scores = pd.to_numeric(selected["WAPE"], errors="coerce")
        valid_scores = scores.map(lambda value: pd.notna(value) and math.isfinite(value) and value >= 0)
        invalid_ok = selected["status"].eq("OK") & ~valid_scores
        selected.loc[invalid_ok, "status"] = "INVALID_WAPE"
        selected["WAPE"] = scores.where(selected["status"].eq("OK") & valid_scores)
        selected = selected[SERIES_KEY + ["WAPE", "status"]].rename(
            columns={"WAPE": f"{label}_WAPE", "status": f"{label}_status"}
        )
        comparison = comparison.merge(selected, on=SERIES_KEY, how="left", validate="one_to_one")
        comparison[f"{label}_status"] = comparison[f"{label}_status"].fillna("NOT_EVALUATED")
    comparison["is_common"] = (
        comparison["baseline_status"].eq("OK")
        & comparison["A_status"].eq("OK")
        & comparison["B_status"].eq("OK")
    )
    comparison["exclusion_reason"] = comparison.apply(
        lambda row: ";".join(f"{label}:{row[f'{label}_status']}" for label in ("baseline", "A", "B")
                             if row[f"{label}_status"] != "OK"), axis=1
    )
    for method in ("A", "B"):
        # 차이는 공통 집합에서만 해석한다. 음수는 XGBoost WAPE가 더 작다는 뜻이다.
        comparison[f"{method}_delta_WAPE_pp"] = (
            comparison[f"{method}_WAPE"] - comparison["baseline_WAPE"]
        ).where(comparison["is_common"])
    return comparison
