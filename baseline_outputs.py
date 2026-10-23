"""원본과 이전 결과를 보존하며 베이스라인 결과표와 실행 근거를 저장한다."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import tempfile
from uuid import uuid4

import pandas as pd


def file_sha256(path):
    """파일 경로를 받아 내용 전체의 SHA-256 해시를 반환한다."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_input_unchanged(input_path, expected_sha256):
    """현재 원본 해시가 평가에 사용한 바이트의 해시와 다르면 저장을 중단한다."""
    if file_sha256(input_path) != expected_sha256:
        raise ValueError("입력 원본이 실행 중 변경되었습니다. 해시가 달라 결과를 저장하지 않습니다.")


def summarize_periods(predictions, metrics, evaluation_periods):
    """평가 구간별 대상 월과 차종 수·평가 상태 수를 JSON용 목록으로 반환한다."""
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
    """세 결과표와 실행 정보를 받아 네 파일이 들어 있는 새 실행 폴더를 반환한다."""
    input_path = Path(input_path).resolve(strict=True)
    # 첫 검사에 실패하면 출력 폴더도 만들지 않는다.
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
    # NaN·무한대 또는 JSON으로 표현할 수 없는 실행 정보를 조용히 저장하지 않는다.
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
    # 이 함수가 만든 임시 폴더만 정리한다. 기존 결과·원본 파일은 삭제하지 않는다.
    with tempfile.TemporaryDirectory(prefix=".baseline_staging_", dir=output_base) as temporary:
        staged_directory = Path(temporary) / "result"
        staged_directory.mkdir()
        for filename, table in output_tables.items():
            output_path = staged_directory / filename
            # float_format을 지정하지 않아 정밀도를 유지하고, 결측은 공란으로 쓴다.
            table.to_csv(output_path, index=False, encoding="utf-8-sig", na_rep="")
            summary["outputs"][filename] = {
                "rows": len(table),
                "sha256": file_sha256(output_path),
            }

        verify_input_unchanged(input_path, input_sha256)
        summary_text = json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2)
        (staged_directory / "run_summary.json").write_text(summary_text + "\n", encoding="utf-8")
        # 요약 파일을 쓰는 동안 원본이 바뀐 경우에도 완성 결과로 공개하지 않는다.
        verify_input_unchanged(input_path, input_sha256)
        staged_directory.rename(run_directory)

    return run_directory
