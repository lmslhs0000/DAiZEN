"""원래 11입력을 유지하고 log1p 정답을 학습하는 시장별 XGBoost 실험이다."""

import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from baseline import SERIES_KEY


FEATURE_COLUMNS = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "month", "Brand", "Model",
]
NUMERIC_FEATURES = FEATURE_COLUMNS[:9]
PREDICTION_COLUMNS = SERIES_KEY + ["Month", "horizon", "prediction"]
STATUS_COLUMNS = SERIES_KEY + ["prediction_status", "reason"]
AUDIT_COLUMNS = SERIES_KEY + ["Month", "training_status", "reason"]


def _month(value):
    """문자열 또는 월 Period를 월 단위 값으로 검사한다."""
    if isinstance(value, pd.Period) and value.freqstr == "M":
        return value
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{1,2}", value) is None:
        raise ValueError("월은 YYYY-MM 형식이어야 합니다.")
    return pd.to_datetime(value, format="%Y-%m").to_period("M")


def build_training_frame(data, train_end):
    """검증된 월별 판매표에서 정답 이전 12개월 입력과 월별 제외 사유를 만든다."""
    train_end = _month(train_end)
    if data.duplicated(SERIES_KEY + ["Month"]).any():
        raise ValueError("학습 입력에 차종·월 중복이 있습니다.")
    # 범주 목록도 이후 fit의 유효 학습 행에서만 정한다.
    observed = data.loc[data["Month"].between(pd.Period("2016-01", "M"), train_end)]
    training_tables = []
    audit_tables = []
    for key, vehicle in observed.groupby(SERIES_KEY, sort=True):
        monthly = vehicle.set_index("Month")["Sales"].astype(float)
        calendar = pd.period_range(monthly.index.min(), train_end, freq="M")
        sales = monthly.reindex(calendar)
        # 달력에 맞춘 다음 shift해야 누락 월을 건너뛴 이전 행을 전월로 쓰지 않는다.
        table = pd.DataFrame(index=calendar)
        for lag in (1, 2, 3, 6, 12):
            table[f"lag_{lag}"] = sales.shift(lag)
        previous_sales = sales.shift(1)
        for window in (3, 6, 12):
            table[f"rolling_mean_{window}"] = previous_sales.rolling(window, min_periods=window).mean()
        table["month"] = calendar.month.astype("int32")
        table["Sales"] = sales
        table["Month"] = calendar
        for column, value in zip(SERIES_KEY, key):
            table[column] = value

        enough_calendar = np.arange(len(calendar)) >= 12
        complete_history = previous_sales.rolling(12, min_periods=1).count().eq(12).to_numpy()
        target_valid = sales.notna().to_numpy()
        valid = enough_calendar & complete_history & target_valid
        training_tables.append(table.loc[valid].reset_index(drop=True))

        audit = table[SERIES_KEY + ["Month"]].copy()
        audit["training_status"] = "OK"
        audit["reason"] = ""
        invalid_positions = np.flatnonzero(~valid)
        for position in invalid_positions:
            target_month = calendar[position]
            reasons = []
            if not target_valid[position]:
                label = "정답 누락" if target_month not in monthly.index else "정답 공란"
                reasons.append(f"{label}: {target_month}")
            if not enough_calendar[position]:
                reasons.append("정답월 이전 연속 12개월의 관측 기간이 부족합니다.")
                status = "INSUFFICIENT_HISTORY"
            elif not complete_history[position]:
                required = pd.period_range(target_month - 12, target_month - 1, freq="M")
                missing = required.difference(monthly.index)
                blank = required.intersection(monthly.index)
                blank = blank[monthly.reindex(blank).isna()]
                if len(missing):
                    reasons.append(f"이력 누락: {missing.astype(str).tolist()}")
                if len(blank):
                    reasons.append(f"이력 공란: {blank.astype(str).tolist()}")
                status = "INCOMPLETE_HISTORY"
            else:
                status = "MISSING_TARGET"
            if not target_valid[position]:
                status = "MISSING_TARGET"
            audit.loc[target_month, ["training_status", "reason"]] = [status, "; ".join(reasons)]
        audit_tables.append(audit.reset_index(drop=True))

    columns = SERIES_KEY + ["Month", "Sales"] + NUMERIC_FEATURES
    training_frame = pd.concat(training_tables, ignore_index=True)[columns] if training_tables else pd.DataFrame(columns=columns)
    row_audit = pd.concat(audit_tables, ignore_index=True)[AUDIT_COLUMNS] if audit_tables else pd.DataFrame(columns=AUDIT_COLUMNS)
    if not training_frame.empty and not np.isfinite(training_frame[NUMERIC_FEATURES + ["Sales"]].to_numpy(dtype=float)).all():
        raise ValueError("학습 입력 또는 정답에 유한하지 않은 값이 있습니다.")
    return training_frame, row_audit


def _validate_schema(schema):
    """열 순서와 범주 목록·시계열 키의 일관성을 확인한다."""
    if schema.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("저장된 입력 스키마의 열 순서가 다릅니다.")
    for name, expected in (("target_transform", "log1p"), ("prediction_inverse", "expm1"),
                           ("prediction_unit", "sales")):
        if schema.get(name) != expected:
            raise ValueError(f"로그 정답 모델의 변환·출력 단위 스키마가 다릅니다: {name}")
    levels = schema.get("categorical_levels", {})
    for column in ("Brand", "Model"):
        values = levels.get(column)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values) or len(values) != len(set(values)):
            raise ValueError(f"범주 스키마가 잘못되었습니다: {column}")
    _month(schema["train_end"])
    for key in schema["seen_series"]:
        if (len(key) != 3 or key[0] != schema["country"]
                or key[1] not in levels["Brand"] or key[2] not in levels["Model"]):
            raise ValueError("학습 시계열과 범주 스키마가 일치하지 않습니다.")


def _feature_frame(frame, schema):
    """학습 때의 열 순서·자료형·범주를 예측에도 동일하게 적용한다."""
    features = frame[FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        features[column] = features[column].astype("int32" if column == "month" else "float64")
    for column in ("Brand", "Model"):
        features[column] = pd.Categorical(features[column], categories=schema["categorical_levels"][column])
        if features[column].isna().any():
            raise ValueError(f"학습에 없던 범주입니다: {column}")
    return features


def _schema_digest(schema):
    text = json.dumps(schema, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_series_weights(rows):
    """유효 학습 행의 입력 순서를 유지하며 차종별 총 가중치를 같게 만든다.

    시장·train_end 필터와 유효 학습 창 판정은 호출 전에 완료해야 한다.
    판매량 크기와 무관하게 N / (K * N_i)를 적용하고 검산 표를 반환한다.
    """
    if rows.empty:
        raise ValueError("NO_TRAINING_ROWS")
    if rows[SERIES_KEY].isna().any().any():
        raise ValueError("학습 가중치 계산에 차종 식별키 누락이 있습니다.")
    groups = rows.groupby(SERIES_KEY, sort=True, observed=True)
    audit = groups.size().reset_index(name="training_rows")
    total_rows = len(rows)
    series_count = len(audit)
    row_counts = groups["Model"].transform("size").to_numpy(dtype=float)
    weights = total_rows / (series_count * row_counts)
    audit["row_weight"] = total_rows / (series_count * audit["training_rows"])
    audit["total_weight"] = audit["training_rows"] * audit["row_weight"]
    return weights, audit


def fit_market(training_frame, country, parameters, train_end, n_jobs=4, *, weighting="row"):
    """종료월 이하 시장 모델을 학습한다. 기본 행 균등 또는 승인된 차종 균등 방식이다."""
    if weighting not in ("row", "series"):
        raise ValueError("학습 가중치 방식은 row 또는 series여야 합니다.")
    train_end = _month(train_end)
    rows = training_frame.loc[training_frame["Country"].eq(country) & training_frame["Month"].le(train_end)].copy()
    if rows.empty:
        raise ValueError("NO_TRAINING_ROWS")
    if rows.duplicated(SERIES_KEY + ["Month"]).any():
        raise ValueError("학습 표에 차종·월 중복이 있습니다.")
    rows = rows.sort_values(SERIES_KEY + ["Month"]).reset_index(drop=True)
    fixed_parameters = {
        "objective": "reg:squarederror", "tree_method": "hist", "enable_categorical": True,
        "device": "cpu", "random_state": 42, "subsample": 1.0, "colsample_bytree": 1.0,
    }
    for name, value in fixed_parameters.items():
        if name in parameters and parameters[name] != value:
            raise ValueError(f"실험의 고정 학습 설정을 바꿀 수 없습니다: {name}")
    if parameters.get("early_stopping_rounds") is not None:
        raise ValueError("초기 탐색에서는 early stopping을 사용하지 않습니다.")
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1:
        raise ValueError("CPU 작업 수는 양의 정수여야 합니다.")
    settings = {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 200, **parameters, **fixed_parameters, "n_jobs": n_jobs}
    schema = {
        "country": country, "feature_columns": FEATURE_COLUMNS.copy(),
        "target_transform": "log1p", "prediction_inverse": "expm1", "prediction_unit": "sales",
        "model_output_unit": "log1p_sales", "input_sales_unit": "sales",
        "categorical_levels": {column: sorted(rows[column].unique().tolist()) for column in ("Brand", "Model")},
        "seen_series": rows[SERIES_KEY].drop_duplicates().values.tolist(),
        "train_end": str(train_end),
        "feature_dtypes": {column: ("category" if column in ("Brand", "Model") else "int32" if column == "month" else "float64") for column in FEATURE_COLUMNS},
        "training_rows": len(rows), "training_target_start": str(rows["Month"].min()),
        "training_target_end": str(rows["Month"].max()),
    }
    _validate_schema(schema)
    features = _feature_frame(rows, schema)
    target = rows["Sales"].to_numpy(dtype=float)
    if not np.isfinite(features[NUMERIC_FEATURES].to_numpy(dtype=float)).all() or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("유효하지 않은 학습 입력 또는 정답입니다.")
    model = XGBRegressor(**settings)
    # 입력 lag/평균과 원본 정답 표는 판매 대수를 유지하고 fit에 전달하는 정답만 변환한다.
    if weighting == "series":
        weights, weight_audit = build_series_weights(rows)
        schema["weighting"] = {
            "method": "series", "formula": "N / (K * N_i)",
            "training_rows": len(rows), "training_series": len(weight_audit),
            "mean_row_weight": float(weights.mean()),
            "min_row_weight": float(weights.min()), "max_row_weight": float(weights.max()),
            "per_series_total_weight": len(rows) / len(weight_audit),
        }
        model.fit(features, np.log1p(target), sample_weight=weights)
    else:
        # 기존 모델 재현을 위해 기본 방식은 sample_weight 인자 자체를 전달하지 않는다.
        model.fit(features, np.log1p(target))
    # XGBoost의 missing 기본값 NaN은 유효한 JSON 숫자가 아니므로 의미를 문자열로 기록한다.
    schema["parameters"] = {
        name: ("NaN" if isinstance(value, float) and np.isnan(value) else value)
        for name, value in model.get_params().items()
    }
    schema["booster_config"] = json.loads(model.get_booster().save_config())
    # 스키마를 모델 내부 해시에 연결해 다른 범주 목록과 잘못 조합하지 않게 한다.
    model.get_booster().set_attr(daizen_feature_schema_sha256=_schema_digest(schema))
    return model, schema


def forecast_recursive(model, history, origin, horizon, schema):
    """고정된 관측 기준월에서 이전 예측을 이어 붙이며 시장 차종들을 배치 예측한다."""
    origin = _month(origin)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("예측 개월 수는 양의 정수여야 합니다.")
    _validate_schema(schema)
    if _month(schema["train_end"]) > origin:
        raise ValueError("모델의 학습 종료월이 관측 기준월보다 뒤입니다.")
    if history["Month"].gt(origin).any():
        raise ValueError("기준월 이후 미래 실제값을 예측 이력에 넣을 수 없습니다.")
    if history.duplicated(SERIES_KEY + ["Month"]).any():
        raise ValueError("예측 이력에 차종·월 중복이 있습니다.")
    required = pd.period_range(origin - 11, origin, freq="M")
    seen = {tuple(key) for key in schema["seen_series"]}
    status_rows = []
    eligible_keys = []
    initial_values = []
    status_positions = []
    for key, vehicle in history.groupby(SERIES_KEY, sort=True):
        reasons = []
        status = "OK"
        monthly = vehicle.set_index("Month")["Sales"]
        if monthly.index.min() > required[0]:
            status = "INSUFFICIENT_HISTORY"
            reasons.append("기준월까지 최소 12개월의 관측 기간이 필요합니다.")
        else:
            missing = required.difference(monthly.index)
            present = required.intersection(monthly.index)
            blank = present[monthly.reindex(present).isna()]
            if len(missing) or len(blank):
                status = "INCOMPLETE_HISTORY"
                if len(missing):
                    reasons.append(f"이력 누락: {missing.astype(str).tolist()}")
                if len(blank):
                    reasons.append(f"이력 공란: {blank.astype(str).tolist()}")
        if key not in seen:
            if status == "OK":
                status = "UNSEEN_SERIES"
            reasons.append("해당 시장 모델의 학습 행에 없던 정확한 시계열 키입니다.")
        record = dict(zip(SERIES_KEY, key))
        record.update(prediction_status=status, reason="; ".join(reasons))
        status_rows.append(record)
        if status == "OK":
            values = monthly.loc[required].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"예측 이력에 유효하지 않은 판매량이 있습니다: {key}")
            eligible_keys.append(key)
            initial_values.append(values)
            status_positions.append(len(status_rows) - 1)

    if not eligible_keys:
        return pd.DataFrame(columns=PREDICTION_COLUMNS), pd.DataFrame(status_rows, columns=STATUS_COLUMNS)
    buffers = np.asarray(initial_values, dtype=float)
    forecasts = np.full((len(eligible_keys), horizon), np.nan)
    active = np.arange(len(eligible_keys))
    for step in range(horizon):
        if not len(active):
            break
        target_month = origin + step + 1
        current = buffers[active]
        batch = pd.DataFrame({f"lag_{lag}": current[:, -lag] for lag in (1, 2, 3, 6, 12)})
        for window in (3, 6, 12):
            batch[f"rolling_mean_{window}"] = np.sum(current[:, -window:] / window, axis=1)
        batch["month"] = target_month.month
        batch["Brand"] = [eligible_keys[index][1] for index in active]
        batch["Model"] = [eligible_keys[index][2] for index in active]
        try:
            inputs = _feature_frame(batch, schema)
            if not np.isfinite(inputs[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
                raise ValueError("재귀 입력값이 유한하지 않습니다.")
            raw_predictions = np.asarray(model.predict(inputs), dtype=float)
            if raw_predictions.shape != (len(active),):
                raise ValueError("모델의 예측 행 수가 요청 차종 수와 다릅니다.")
        except Exception as error:
            for index in active:
                status_rows[status_positions[index]].update(prediction_status="PREDICTION_ERROR", reason=f"{target_month} 예측 실패: {type(error).__name__}: {error}")
            break

        # raw 출력은 로그 단위다. 복원한 대수만 다음 재귀 입력과 평가에 사용한다.
        # 오버플로를 임의 상한으로 잘라 숨기지 않고 해당 차종의 예측 실패로 기록한다.
        with np.errstate(over="ignore", invalid="ignore"):
            sales_predictions = np.expm1(raw_predictions)
        finite = np.isfinite(raw_predictions) & np.isfinite(sales_predictions)
        for index in active[~finite]:
            status_rows[status_positions[index]].update(prediction_status="PREDICTION_ERROR", reason=f"{target_month} 원시 로그 예측 또는 판매 대수 복원값이 유한하지 않습니다.")
        good = active[finite]
        clipped = np.maximum(0.0, sales_predictions[finite])
        forecasts[good, step] = clipped
        # 실패한 차종은 제외하고 각 차종의 최근 12개월만 독립적으로 갱신한다.
        buffers[good, :-1] = buffers[good, 1:]
        buffers[good, -1] = clipped
        active = good

    prediction_rows = []
    for index, key in enumerate(eligible_keys):
        if status_rows[status_positions[index]]["prediction_status"] != "OK":
            continue
        for step, prediction in enumerate(forecasts[index], start=1):
            row = dict(zip(SERIES_KEY, key))
            row.update(Month=origin + step, horizon=step, prediction=float(prediction))
            prediction_rows.append(row)
    return pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS), pd.DataFrame(status_rows, columns=STATUS_COLUMNS)


def save_market_model(model, schema, model_path):
    """새 UBJSON 모델과 같은 폴더의 feature_schema.json을 기존 파일 덮어쓰기 없이 저장한다."""
    _validate_schema(schema)
    model_path = Path(model_path)
    if model_path.suffix.lower() != ".ubj":
        raise ValueError("범주형 모델은 .ubj 파일로 저장해야 합니다.")
    schema_path = model_path.with_name("feature_schema.json")
    if model_path.exists() or schema_path.exists():
        raise FileExistsError("기존 모델이나 입력 스키마는 덮어쓰지 않습니다.")
    if model.get_booster().attr("daizen_feature_schema_sha256") != _schema_digest(schema):
        raise ValueError("모델과 입력 스키마가 일치하지 않습니다.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".model_staging_", dir=model_path.parent) as temporary:
        staged_model = Path(temporary) / "model.ubj"
        model.save_model(staged_model)
        schema_text = json.dumps(schema, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        created = []
        try:
            with model_path.open("xb") as destination:
                created.append(model_path)
                destination.write(staged_model.read_bytes())
            with schema_path.open("x", encoding="utf-8") as destination:
                created.append(schema_path)
                destination.write(schema_text)
        except BaseException:
            # 이 호출이 새로 만든 파일만 지운다. 기존 다른 실행 파일에는 손대지 않는다.
            for path in created:
                path.unlink(missing_ok=True)
            raise


def load_market_model(model_path):
    """UBJSON 모델과 범주 스키마를 읽고 열 순서·스키마 결합을 확인한다."""
    model_path = Path(model_path)
    schema = json.loads(model_path.with_name("feature_schema.json").read_text(encoding="utf-8"))
    _validate_schema(schema)
    model = XGBRegressor()
    model.load_model(model_path)
    booster = model.get_booster()
    if booster.feature_names != FEATURE_COLUMNS or booster.attr("daizen_feature_schema_sha256") != _schema_digest(schema):
        raise ValueError("저장 모델과 입력 스키마·범주가 일치하지 않습니다.")
    if booster.feature_types[-2:] != ["c", "c"]:
        raise ValueError("저장 모델의 범주형 입력 정보가 다릅니다.")
    # 모델 파일에는 작업 스레드 수 같은 실행 설정이 유지되지 않는다.
    # 가중치나 디스크 파일을 바꾸지 않고 검증된 스키마의 CPU 실행 설정만 복구한다.
    runtime_parameters = schema.get("parameters", {})
    n_jobs = runtime_parameters.get("n_jobs")
    device = runtime_parameters.get("device")
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1 or device != "cpu":
        raise ValueError("저장된 CPU 실행 설정의 n_jobs 또는 device가 잘못되었습니다.")
    model.set_params(n_jobs=n_jobs, device=device)
    return model, schema
