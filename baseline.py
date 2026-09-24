import argparse
import hashlib
import io
import math
from pathlib import Path
import sys

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "raw" / "data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "reports" / "baseline"
REQUIRED_COLUMNS = ["Country", "Brand", "Model", "Month", "Sales"]
SERIES_KEY = ["Country", "Brand", "Model"]


DEFAULT_ORIGIN = "2026-01"
DEFAULT_HORIZON = 6


EVALUATION_PERIODS = [
    ("A", "A_validation_1", "2024-07", 6),
    ("A", "A_validation_2", "2025-01", 6),
    ("A", "A_validation_3", "2025-07", 6),
    ("A", "A_test", DEFAULT_ORIGIN, DEFAULT_HORIZON),
    ("B", "B_validation", "2024-12", 12),
    ("B", "B_test", "2025-12", 7),
    ("B", "B_common_test_6m", DEFAULT_ORIGIN, DEFAULT_HORIZON),
]


class HistoryUnavailableError(ValueError):


    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status


def read_sales_csv(csv_path):


    data = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)

    missing_columns = []
    for column in REQUIRED_COLUMNS:
        if column not in data.columns:
            missing_columns.append(column)
    if missing_columns:
        raise ValueError(f"필수 열이 없습니다: {missing_columns}")
    data = data[REQUIRED_COLUMNS].copy()
    if data.empty:
        raise ValueError("CSV에 판매 데이터 행이 없습니다.")

    for column in SERIES_KEY:
        blank_key = data[column].str.strip().eq("")
        if blank_key.any():
            raise ValueError(f"{column}이 비어 있는 행:\n{data.loc[blank_key].to_string(index=False)}")


    month_text = data["Month"].str.strip()
    month_format = month_text.str.fullmatch(r"\d{4}-\d{1,2}")
    parsed_month = pd.to_datetime(month_text, format="%Y-%m", errors="coerce")
    invalid_month = ~month_format | parsed_month.isna()
    if invalid_month.any():
        raise ValueError(f"Month 형식이 잘못된 행:\n{data.loc[invalid_month].to_string(index=False)}")
    data["Month"] = parsed_month.dt.to_period("M")


    sales_text = data["Sales"].str.strip()
    sales_number = pd.to_numeric(sales_text, errors="coerce")
    not_numeric = sales_text.ne("") & sales_number.isna()
    infinite = sales_number.isin([float("inf"), float("-inf")])
    invalid_sales = not_numeric | infinite | sales_number.lt(0)
    if invalid_sales.any():
        raise ValueError(f"Sales가 잘못된 행:\n{data.loc[invalid_sales].to_string(index=False)}")
    data["Sales"] = sales_number

    row_key = SERIES_KEY + ["Month"]
    duplicates = data.duplicated(row_key, keep=False)
    if duplicates.any():
        raise ValueError(f"차종·월 중복 행:\n{data.loc[duplicates].to_string(index=False)}")

    data = data.sort_values(row_key).reset_index(drop=True)
    return data


def select_vehicle(data, country, brand, model):

    same_country = data["Country"] == country
    same_brand = data["Brand"] == brand
    same_model = data["Model"] == model


    selected_rows = same_country & same_brand & same_model
    vehicle_data = data.loc[selected_rows].copy()
    if vehicle_data.empty:
        raise ValueError(f"차종을 찾을 수 없습니다: {country} / {brand} / {model}")

    vehicle_data = vehicle_data.sort_values("Month").reset_index(drop=True)
    return vehicle_data


def predict_seasonal_naive(vehicle_data, origin, horizon):


    if not isinstance(horizon, int) or not 1 <= horizon <= 12:
        raise ValueError("예측 개월 수는 1~12 사이의 정수여야 합니다.")
    if len(vehicle_data[SERIES_KEY].drop_duplicates()) > 1:
        raise ValueError("예측 함수에는 한 차종의 이력만 넣어야 합니다.")

    origin_month = pd.Period(origin, freq="M")

    history = vehicle_data.loc[vehicle_data["Month"] <= origin_month].copy()
    reference_start = origin_month - 11
    if history.empty or history["Month"].min() > reference_start:
        raise HistoryUnavailableError(
            "INSUFFICIENT_HISTORY", "이력 부족: 기준월까지 최소 12개월의 관측 기간이 필요합니다."
        )
    if history["Month"].duplicated().any():
        raise ValueError("이력에 중복 월이 있습니다.")


    history_by_month = history.set_index("Month")["Sales"]
    required_months = pd.period_range(reference_start, origin_month, freq="M")
    missing_months = required_months.difference(history_by_month.index)
    if len(missing_months) > 0:
        raise HistoryUnavailableError("HISTORY_MISSING", f"이력 누락: {missing_months.astype(str).tolist()}")
    recent_sales = history_by_month.loc[required_months]
    blank_months = recent_sales.index[recent_sales.isna()]
    if len(blank_months) > 0:
        raise HistoryUnavailableError("HISTORY_BLANK", f"이력 공란: {blank_months.astype(str).tolist()}")

    predictions = []
    for step in range(1, horizon + 1):
        target_month = origin_month + step
        reference_month = target_month - 12
        prediction = history_by_month.loc[reference_month]
        predictions.append({
            "origin": origin_month,
            "Month": target_month,
            "horizon": step,
            "reference_month": reference_month,
            "prediction": prediction,
        })

    return pd.DataFrame(predictions)


def calculate_wape(vehicle_data, predictions):

    if len(vehicle_data[SERIES_KEY].drop_duplicates()) > 1:
        raise ValueError("WAPE 계산에는 한 차종의 이력만 넣어야 합니다.")
    if predictions.empty:
        raise ValueError("예측표가 비어 있습니다.")

    actuals = vehicle_data[["Month", "Sales"]].rename(columns={"Sales": "actual"})


    comparison = predictions.merge(actuals, on="Month", how="left", validate="one_to_one")

    comparison["absolute_error"] = (comparison["actual"].astype(float) - comparison["prediction"].astype(float)).abs()

    expected_months = len(comparison)

    observed_months = int(comparison["actual"].notna().sum())
    metrics = {
        "expected_months": expected_months,
        "observed_months": observed_months,
        "actual_sum": None,
        "absolute_error_sum": None,
        "WAPE": None,
        "status": "TARGET_MISSING",
    }

    if observed_months != expected_months:
        return comparison, metrics

    try:

        actual_sum = math.fsum(comparison["actual"])
        absolute_error_sum = math.fsum(comparison["absolute_error"])
    except OverflowError as error:
        raise ValueError("판매량·오차 합계가 계산 가능한 수치 범위를 넘었습니다.") from error
    if not math.isfinite(actual_sum) or not math.isfinite(absolute_error_sum):
        raise ValueError("판매량·오차 합계가 유한하지 않습니다.")
    metrics["actual_sum"] = actual_sum
    metrics["absolute_error_sum"] = absolute_error_sum
    if actual_sum == 0:
        metrics["status"] = "ZERO_ACTUAL_SUM"
    else:
        wape = absolute_error_sum / actual_sum * 100
        if not math.isfinite(wape):
            raise ValueError("WAPE가 계산 가능한 수치 범위를 넘었습니다.")
        metrics["WAPE"] = wape
        metrics["status"] = "OK"

    return comparison, metrics


def evaluate_all_vehicles(data, origin, horizon):

    if data.empty:
        raise ValueError("평가할 데이터가 없습니다.")
    if not isinstance(horizon, int) or not 1 <= horizon <= 12:
        raise ValueError("예측 개월 수는 1~12 사이의 정수여야 합니다.")
    if data.duplicated(SERIES_KEY + ["Month"]).any():
        raise ValueError("전체 평가 입력에 차종·월 중복 행이 있습니다.")

    origin_month = pd.Period(origin, freq="M")
    target_months = pd.period_range(origin_month + 1, periods=horizon, freq="M")
    comparison_tables = []
    metric_records = []


    for (country, brand, model), vehicle_data in data.groupby(SERIES_KEY, sort=True):
        vehicle_data = vehicle_data.sort_values("Month")
        identifiers = {
            "Country": country,
            "Brand": brand,
            "Model": model,
            "origin": origin_month,
        }
        reason = ""
        try:
            predictions = predict_seasonal_naive(vehicle_data, origin_month, horizon)
        except HistoryUnavailableError as error:

            target_actuals = vehicle_data.loc[vehicle_data["Month"].isin(target_months), "Sales"]
            metrics = {
                "expected_months": horizon,
                "observed_months": int(target_actuals.notna().sum()),
                "actual_sum": None,
                "absolute_error_sum": None,
                "WAPE": None,
                "status": error.status,
            }
            reason = str(error)
        else:
            comparison, metrics = calculate_wape(vehicle_data, predictions)
            if metrics["status"] == "TARGET_MISSING":
                missing_actuals = comparison.loc[comparison["actual"].isna(), "Month"]
                reason = f"평가 실제값 누락·공란: {missing_actuals.astype(str).tolist()}"
            elif metrics["status"] == "ZERO_ACTUAL_SUM":
                reason = "평가 기간 실제 판매량 합계가 0입니다."

            for column, value in identifiers.items():
                comparison[column] = value
            comparison_tables.append(comparison)

        record = identifiers.copy()
        record.update(metrics)
        record["reason"] = reason
        metric_records.append(record)

    comparison_columns = SERIES_KEY + [
        "origin", "Month", "horizon", "reference_month", "prediction", "actual", "absolute_error",
    ]
    if comparison_tables:
        all_comparisons = pd.concat(comparison_tables, ignore_index=True)[comparison_columns]
    else:

        all_comparisons = pd.DataFrame(columns=comparison_columns)
    all_metrics = pd.DataFrame(metric_records)
    return all_comparisons, all_metrics


def build_validation_summary(metrics):

    summary = metrics[SERIES_KEY].drop_duplicates().set_index(SERIES_KEY).sort_index()
    a_score_columns = []
    a_valid_columns = []
    for split in ["A_validation_1", "A_validation_2", "A_validation_3", "B_validation"]:
        scores = metrics.loc[metrics["split"].eq(split)].set_index(SERIES_KEY)
        if scores.index.has_duplicates:
            raise ValueError(f"검증 요약에 중복 차종이 있습니다: {split}")

        summary[f"{split}_WAPE"] = scores["WAPE"].reindex(summary.index).astype(float)
        summary[f"{split}_status"] = scores["status"].reindex(summary.index).fillna("NOT_EVALUATED")
        if split.startswith("A_"):
            a_score_columns.append(f"{split}_WAPE")
            a_valid_columns.append(summary[f"{split}_status"].eq("OK") & summary[f"{split}_WAPE"].notna())

    summary["A_valid_folds"] = pd.concat(a_valid_columns, axis=1).sum(axis=1)
    all_folds_valid = summary["A_valid_folds"].eq(3)


    summary["A_validation_mean_WAPE"] = (summary[a_score_columns] / 3).sum(axis=1, min_count=3).where(all_folds_valid)
    summary["A_validation_status"] = "INCOMPLETE_VALIDATION"
    summary.loc[all_folds_valid, "A_validation_status"] = "OK"
    columns = SERIES_KEY.copy()
    for split in ["A_validation_1", "A_validation_2", "A_validation_3"]:
        columns.extend([f"{split}_WAPE", f"{split}_status"])
    columns.extend(["A_valid_folds", "A_validation_mean_WAPE", "A_validation_status", "B_validation_WAPE", "B_validation_status"])
    return summary.reset_index()[columns]


def evaluate_baseline(data):

    cached_evaluations = {}
    prediction_tables = []
    metric_tables = []
    for method, split, origin, horizon in EVALUATION_PERIODS:

        cache_key = (origin, horizon)
        if cache_key not in cached_evaluations:
            cached_evaluations[cache_key] = evaluate_all_vehicles(data, origin, horizon)
        comparisons, metrics = cached_evaluations[cache_key]
        comparisons = comparisons.copy()
        metrics = metrics.copy()
        for table in (comparisons, metrics):
            table.insert(0, "method", method)
            table.insert(1, "split", split)
        prediction_tables.append(comparisons)
        metric_tables.append(metrics)

    prediction_columns = ["method", "split", "origin"] + SERIES_KEY + [
        "Month", "horizon", "reference_month", "actual", "prediction", "absolute_error",
    ]
    metric_columns = ["method", "split", "origin"] + SERIES_KEY + [
        "expected_months", "observed_months", "actual_sum", "absolute_error_sum", "WAPE", "status", "reason",
    ]
    nonempty_predictions = [table for table in prediction_tables if not table.empty]
    if nonempty_predictions:
        predictions = pd.concat(nonempty_predictions, ignore_index=True)[prediction_columns]
    else:
        predictions = pd.DataFrame(columns=prediction_columns)
    metrics = pd.concat(metric_tables, ignore_index=True)[metric_columns]
    validation_summary = build_validation_summary(metrics)
    return predictions, metrics, validation_summary


def validate_evaluation_results(predictions, metrics, validation_summary=None):

    metric_key = ["method", "split"] + SERIES_KEY
    if metrics.duplicated(metric_key).any():
        raise ValueError("성적표에 차종·평가 구간 중복이 있습니다.")
    if predictions.duplicated(metric_key + ["Month"]).any():
        raise ValueError("예측표에 차종·평가 구간·월 중복이 있습니다.")
    for table, columns in [
        (predictions, ["actual", "prediction", "absolute_error"]),
        (metrics, ["actual_sum", "absolute_error_sum", "WAPE"]),
    ]:
        for column in columns:
            if not table[column].dropna().map(math.isfinite).all():
                raise ValueError(f"계산 결과에 유한하지 않은 값이 있습니다: {column}")
            if table[column].dropna().lt(0).any():
                raise ValueError(f"계산 결과에 음수가 있습니다: {column}")
    if not metrics["status"].eq("OK").eq(metrics["WAPE"].notna()).all():
        raise ValueError("평가 상태와 WAPE의 계산 가능 여부가 다릅니다.")


    for table, sort_columns in [(predictions, SERIES_KEY + ["Month"]), (metrics, SERIES_KEY)]:
        a_common = table.loc[table["split"].eq("A_test")].drop(columns=["method", "split"])
        b_common = table.loc[table["split"].eq("B_common_test_6m")].drop(columns=["method", "split"])
        a_common = a_common.sort_values(sort_columns).reset_index(drop=True)
        b_common = b_common.sort_values(sort_columns).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(a_common, b_common)
        except AssertionError as error:
            raise ValueError("A/B 공통 6개월 테스트 결과가 다릅니다.") from error

    for row in predictions.itertuples(index=False):
        if row.Month != row.origin + row.horizon or row.reference_month != row.Month - 12:
            raise ValueError("예측 대상 월과 전년 동월의 대응이 잘못되었습니다.")
        if row.reference_month > row.origin:
            raise ValueError("예측에 기준월 이후 판매량이 사용되었습니다.")
    checks = {
        "unique_metric_keys": True,
        "unique_prediction_keys": True,
        "finite_results": True,
        "status_matches_wape": True,
        "common_test_predictions_and_metrics_equal": True,
        "reference_months_before_or_at_origin": True,
    }
    if validation_summary is not None:
        for column in [name for name in validation_summary if name.endswith("WAPE")]:
            values = validation_summary[column].dropna()
            if not values.map(math.isfinite).all() or values.lt(0).any():
                raise ValueError(f"검증 요약 점수가 유한한 0 이상 수치가 아닙니다: {column}")
        complete = validation_summary["A_valid_folds"].eq(3)
        if not complete.eq(validation_summary["A_validation_mean_WAPE"].notna()).all():
            raise ValueError("A 검증 유효 구간 수와 평균 계산 여부가 다릅니다.")
        checks["finite_validation_summary"] = True
        checks["a_mean_requires_three_valid_folds"] = True
    return checks


def print_evaluation(all_comparisons, all_metrics, split_name, origin, horizon, summary_only=False):

    origin_month = pd.Period(origin, freq="M")
    print(f"\n[{split_name}]")
    print(f"기준월: {origin_month} / 평가 기간: {origin_month + 1}~{origin_month + horizon} ({horizon}개월)")
    status_labels = {
        "OK": "WAPE 계산 가능",
        "INSUFFICIENT_HISTORY": "최근 12개월 이력 부족",
        "HISTORY_MISSING": "최근 12개월 중 누락 월 존재",
        "HISTORY_BLANK": "최근 12개월 중 판매량 공란 존재",
        "TARGET_MISSING": "평가 실제값 누락·공란",
        "ZERO_ACTUAL_SUM": "평가 실제값 합계 0",
    }
    print(f"전체 차종 시계열 수: {len(all_metrics):,}")
    print(f"생성된 월별 예측 행 수: {len(all_comparisons):,}")
    for status, count in all_metrics["status"].value_counts().items():
        print(f"{status_labels[status]}: {count:,}개 차종 시계열")

    if summary_only:
        return
    print("\n전체 차종별 WAPE (%, 국가·브랜드·차종 순):")
    display_metrics = all_metrics[SERIES_KEY + ["WAPE", "status", "reason"]].copy()
    display_metrics["WAPE"] = display_metrics["WAPE"].map(
        lambda value: f"{value:.2f}%" if pd.notna(value) else "계산 불가"
    )

    print(display_metrics.to_string(index=False))


def main(argv=None):

    parser = argparse.ArgumentParser(description="모든 시장·차종의 계절 나이브 A/B 평가와 결과 CSV 저장을 실행합니다.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="평가할 CSV 경로 (기본: raw/data.csv)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="실행별 결과 폴더를 만들 상위 경로 (기본: reports/baseline)")
    parser.add_argument("--summary-only", action="store_true", help="터미널에는 구간별 개수만 표시 (전체 차종 결과는 CSV에 저장)")
    arguments = parser.parse_args(argv)

    try:

        input_bytes = arguments.csv.read_bytes()
        input_sha256 = hashlib.sha256(input_bytes).hexdigest()
        sales_data = read_sales_csv(io.BytesIO(input_bytes))
    except (OSError, ValueError) as error:
        print(f"입력 오류: {error}")
        return 2

    print(f"입력 파일: {arguments.csv.resolve()}")
    print(f"행 수: {len(sales_data):,}")
    print(f"기간: {sales_data['Month'].min()} ~ {sales_data['Month'].max()}")
    print(f"포함된 국가·시장: {', '.join(sorted(sales_data['Country'].unique()))}")
    print(f"판매량 결측 행 수: {sales_data['Sales'].isna().sum():,}")

    print("\n전체 A/B 평가를 계산하고 결과 파일을 저장합니다...", flush=True)
    try:
        predictions, metrics, validation_summary = evaluate_baseline(sales_data)
        checks = validate_evaluation_results(predictions, metrics, validation_summary)
        from baseline_outputs import save_results

        input_info = {
            "rows": len(sales_data),
            "series_count": len(sales_data[SERIES_KEY].drop_duplicates()),
            "countries": sorted(sales_data["Country"].unique().tolist()),
            "month_start": str(sales_data["Month"].min()),
            "month_end": str(sales_data["Month"].max()),
            "missing_sales_rows": int(sales_data["Sales"].isna().sum()),
        }
        run_dir = save_results(
            predictions, metrics, validation_summary,
            input_path=arguments.csv, input_sha256=input_sha256, input_info=input_info,
            evaluation_periods=EVALUATION_PERIODS, checks=checks, output_dir=arguments.output_dir,
        )
    except (OSError, ValueError) as error:
        print(f"실행 오류: {error}")
        return 2

    for method, split, origin, horizon in EVALUATION_PERIODS:
        split_predictions = predictions.loc[predictions["split"].eq(split)]
        split_metrics = metrics.loc[metrics["split"].eq(split)]
        print_evaluation(split_predictions, split_metrics, split, origin, horizon, arguments.summary_only)
    complete_validations = int(validation_summary["A_valid_folds"].eq(3).sum())
    print(f"\nA 검증 3구간 평균 계산 가능: {complete_validations:,}개 차종 시계열")
    print("A 평균은 세 구간 모두 계산 가능한 경우에만 저장합니다.")
    print("A_test와 B_common_test_6m의 결과 일치 및 원본 CSV 보존 확인 완료")
    print(f"결과 폴더: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
