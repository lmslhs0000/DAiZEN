from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import tempfile
from uuid import uuid4

import pandas as pd


def file_sha256(path):

    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_input_unchanged(input_path, expected_sha256):

    if file_sha256(input_path) != expected_sha256:
        raise ValueError("입력 원본이 실행 중 변경되었습니다. 해시가 달라 결과를 저장하지 않습니다.")


def summarize_periods(predictions, metrics, evaluation_periods):

    period_summaries = []
    for method, split, origin, horizon in evaluation_periods:
        origin_month = pd.Period(origin, freq="M")
        metric_rows = metrics.loc[
            metrics["method"].eq(method)
            & metrics["split"].eq(split)
            & metrics["origin"].astype(str).eq(str(origin_month))
        ]
        prediction_rows = predictions.loc[
            predictions["method"].eq(method)
            & predictions["split"].eq(split)
            & predictions["origin"].astype(str).eq(str(origin_month))
        ]
        target_months = pd.period_range(origin_month + 1, periods=horizon, freq="M")
        status_counts = {
            str(status): int(count)
            for status, count in metric_rows["status"].value_counts().sort_index().items()
        }
        evaluable_series = status_counts.get("OK", 0)
        period_summaries.append({
            "method": method,
            "split": split,
            "origin": str(origin_month),
            "horizon": horizon,
            "target_start": str(target_months[0]),
            "target_end": str(target_months[-1]),
            "target_months": target_months.astype(str).tolist(),
            "series_count": len(metric_rows),
            "evaluable_series": evaluable_series,
            "unavailable_series": len(metric_rows) - evaluable_series,
            "status_counts": status_counts,
            "prediction_rows": len(prediction_rows),
        })
    return period_summaries


def save_results(
    predictions,
    metrics,
    validation_summary,
    *,
    input_path,
    input_sha256,
    input_info,
    evaluation_periods,
    checks,
    output_dir,
):

    input_path = Path(input_path).resolve(strict=True)

    verify_input_unchanged(input_path, input_sha256)
    failed_checks = [name for name, passed in checks.items() if passed is not True]
    if failed_checks:
        raise ValueError(f"실행 검산을 통과하지 못했습니다: {failed_checks}")

    timestamp = datetime.now(timezone.utc)
    summary = {
        "schema_version": 1,
        "timestamp": timestamp.isoformat(),
        "input": {"path": str(input_path), "sha256": input_sha256, "info": input_info},
        "settings": {
            "model": "seasonal_naive",
            "series_key": ["Country", "Brand", "Model"],
            "minimum_history_months": 12,
            "history_requirement": "기준월까지 연속 12개월의 유효한 실제 판매량",
            "prediction_rule": "actual_sales(target_month - 12 months)",
            "zero_imputation": False,
            "within_period_actual_updates": False,
            "cross_market_aggregation": False,
            "future_forecast_months": 0,
        },
        "metric_rules": {
            "metric": "WAPE",
            "unit": "percent",
            "formula": "100 * sum(abs(actual - prediction)) / sum(actual)",
            "prediction_rounding": False,
            "unavailable_WAPE": "blank",
            "zero_actual_sum": "WAPE는 공란, status=ZERO_ACTUAL_SUM",
            "A_required_valid_folds": 3,
            "A_validation_mean": "A 검증 3개 구간의 상태가 모두 OK일 때만 WAPE 산술평균",
            "B_test_and_common_test": "7개월 테스트와 공통 6개월 테스트를 각각 저장",
        },
        "evaluation_periods": summarize_periods(predictions, metrics, evaluation_periods),
        "checks": {**checks, "input_unchanged": True},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__},
        "outputs": {},
    }

    json.dumps(summary, ensure_ascii=False, allow_nan=False)

    output_base = Path(output_dir).resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    run_name = f"run_{timestamp.strftime('%Y%m%dT%H%M%S_%fZ')}_{uuid4().hex}"
    run_directory = output_base / run_name
    if run_directory.exists():
        raise FileExistsError(f"이미 있는 결과 폴더는 덮어쓰지 않습니다: {run_directory}")

    output_tables = {
        "baseline_predictions.csv": predictions,
        "baseline_metrics.csv": metrics,
        "baseline_validation_summary.csv": validation_summary,
    }

    with tempfile.TemporaryDirectory(prefix=".baseline_staging_", dir=output_base) as temporary:
        staged_directory = Path(temporary) / "result"
        staged_directory.mkdir()
        for filename, table in output_tables.items():
            output_path = staged_directory / filename

            table.to_csv(output_path, index=False, encoding="utf-8-sig", na_rep="")
            summary["outputs"][filename] = {
                "rows": len(table),
                "sha256": file_sha256(output_path),
            }

        verify_input_unchanged(input_path, input_sha256)
        summary_text = json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2)
        (staged_directory / "run_summary.json").write_text(summary_text + "\n", encoding="utf-8")

        verify_input_unchanged(input_path, input_sha256)
        staged_directory.rename(run_directory)

    return run_directory
