"""학습 없는 도전자: 기본 XGBoost와 작년 그대로의 결합, 추세 보정 작년 그대로.

규칙은 점수를 보기 전에 고정했다(2026-09-30 사용자 승인).
- combo_equal: 모든 예측 거리에서 XGBoost 0.5 + 작년 그대로 0.5
- combo_horizon: 1~12개월은 XGBoost 1.0, 13~24개월은 0.5 + 0.5 (1단계 진단을 본 뒤 정한 규칙이라 확인용 기준월을 특히 주의)
- naive_trend: 작년 그대로 × 최근 1년 증감률 g(1~12개월 g, 13~24개월 g²), g는 [0.5, 1.5]로 제한

사용: PY experiments/horizon_eval_v1/challengers.py <XGBoost run> [--only combo_horizon] [--suffix _v2]
"""

import argparse

import numpy as np
import pandas as pd

from common import (CHECK_HORIZON, CHECK_ORIGIN, CSV_PATH, HERE, HORIZON, KEY, ORIGINS, PREDICTION_COLUMNS,
                    STATUS_COLUMNS, file_sha256, guarded, new_run_dir, require_unchanged, stack_outputs, write_csv,
                    write_json)
import baseline
from naive import seasonal_naive
from score import read_run

RULES = {"combo_equal": (lambda h: 0.5, "XGBoost 0.5 + 작년 그대로 0.5 (모든 예측 거리)"),
         "combo_horizon": (lambda h: 1.0 if h <= 12 else 0.5, "1~12개월 XGBoost 1.0, 13~24개월 0.5 + 0.5")}
TREND_RULE = "작년 그대로 × g (1~12개월) 또는 g² (13~24개월), g = 최근 12개월 합 / 그 전 12개월 합, [0.5, 1.5]로 제한"


def combine(first, second, model_name, weight):
    """두 예측을 예측 거리별 가중치 weight(h)로 섞는다: weight × 첫째 + (1 − weight) × 둘째. 둘 다 OK인 차종만 낸다."""
    (p1, s1), (p2, s2) = first, second
    ids = ["origin"] + KEY
    merged = p1.drop(columns="model").merge(p2.drop(columns="model"), on=ids + ["Month", "horizon"],
                                            suffixes=("_a", "_b"))
    w = merged["horizon"].map(weight).astype(float)
    merged["prediction"] = w * merged["prediction_a"] + (1 - w) * merged["prediction_b"]
    predictions = merged.assign(model=model_name)[PREDICTION_COLUMNS]
    states = s1.drop(columns="model").merge(s2.drop(columns="model"), on=ids, how="outer", suffixes=("_a", "_b"))
    status_a = states["prediction_status_a"].fillna("MISSING_SOURCE")
    status_b = states["prediction_status_b"].fillna("MISSING_SOURCE")
    states["prediction_status"] = np.where(status_a.ne("OK"), status_a, status_b)
    states["reason"] = np.where(states["prediction_status"] == "OK", "", states["prediction_status"])
    statuses = states.assign(model=model_name)[STATUS_COLUMNS]
    ok = statuses.loc[statuses["prediction_status"].eq("OK"), ids]
    return predictions.merge(ok, on=ids).reset_index(drop=True), statuses.reset_index(drop=True)


def naive_trend(data, origin, horizon, low=0.5, high=1.5):
    """작년 그대로 × 최근 1년 증감률. 그 전 1년 자료가 없거나 합이 0이면 g=1로 두고 reason='trend_fallback'."""
    origin = pd.Period(origin, freq="M")
    predictions, statuses = seasonal_naive(data, origin, horizon)
    observed = data.loc[data["Month"].le(origin)]
    recent_months = pd.period_range(origin - 11, origin, freq="M")
    previous_months = pd.period_range(origin - 23, origin - 12, freq="M")
    growth = {}
    for key, vehicle in observed.groupby(KEY, sort=True):
        sales = vehicle.set_index("Month")["Sales"]
        recent, previous = sales.reindex(recent_months), sales.reindex(previous_months)
        if recent.isna().any() or previous.isna().any() or previous.sum() <= 0:
            growth[key] = np.nan
        else:
            growth[key] = float(np.clip(recent.sum() / previous.sum(), low, high))
    keys = list(zip(*(predictions[c] for c in KEY)))
    g = pd.Series([growth.get(k, np.nan) for k in keys], index=predictions.index, dtype=float).fillna(1.0)
    power = np.where(predictions["horizon"].le(12), 1, 2)
    predictions = predictions.assign(prediction=predictions["prediction"] * g ** power)
    fallback = {k for k, v in growth.items() if np.isnan(v)}
    statuses = statuses.copy()
    in_fallback = pd.Series([k in fallback for k in zip(*(statuses[c] for c in KEY))], index=statuses.index)
    statuses.loc[in_fallback & statuses["prediction_status"].eq("OK"), "reason"] = "trend_fallback"
    return predictions, statuses


def read_statuses(run_dir):
    return pd.read_csv(guarded(run_dir) / "statuses.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)


def plan_outputs(only, suffix):
    """만들 도전자: {출력 모델 이름: 규칙 이름}. only가 없으면 세 규칙 모두."""
    rules = list(RULES) + ["naive_trend"]
    chosen = rules if not only else list(only)
    unknown = [name for name in chosen if name not in rules]
    if unknown:
        raise ValueError(f"알 수 없는 규칙: {unknown}")
    return {name + suffix: name for name in chosen}


def run(xgb_run, csv_path=CSV_PATH, output_base=HERE / "reports", only=None, suffix="", origins=None, check=True):
    plan = plan_outputs(only, suffix)
    input_hash = file_sha256(csv_path)
    data = baseline.read_sales_csv(guarded(csv_path))
    xgb = (read_run(xgb_run), read_statuses(xgb_run))
    frames = {"naive": ([], []), "naive_trend": ([], [])}
    plan_origins = list(origins) if origins else list(ORIGINS)
    for origin, horizon in [(o, HORIZON) for o in plan_origins] + ([(CHECK_ORIGIN, CHECK_HORIZON)] if check else []):
        period = pd.Period(origin, "M")
        for name, function in (("naive", seasonal_naive), ("naive_trend", naive_trend)):
            p, s = function(data, origin, horizon)
            p, s = stack_outputs([p], [s], "seasonal_naive" if name == "naive" else name, period)
            frames[name][0].append(p)
            frames[name][1].append(s)
    stacked = {name: (pd.concat(p, ignore_index=True), pd.concat(s, ignore_index=True)) for name, (p, s) in frames.items()}
    descriptions = {**{name: text for name, (_, text) in RULES.items()}, "naive_trend": TREND_RULE}
    require_unchanged(csv_path, input_hash)
    run_dirs = []
    for output, rule in plan.items():
        if rule == "naive_trend":
            predictions, statuses = (table.assign(model=output) for table in stacked["naive_trend"])
        else:
            predictions, statuses = combine(xgb, stacked["naive"], output, RULES[rule][0])
        run_dir = new_run_dir(output_base, output)
        write_csv(run_dir / "predictions.csv", predictions)
        write_csv(run_dir / "statuses.csv", statuses)
        write_json(run_dir / "run_summary.json", {
            "model": output, "rule_name": rule, "rule": descriptions[rule], "source_run": str(xgb_run),
            "input_sha256": input_hash, "origins": plan_origins, "check_origin": CHECK_ORIGIN if check else None, "trained_models": 0,
            "status_counts": statuses.groupby(["origin", "prediction_status"]).size().rename("n").reset_index().to_dict("records")})
        run_dirs.append(run_dir)
        print(run_dir)
    return run_dirs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xgb_run")
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--only", nargs="*", help="만들 규칙 이름(combo_equal, combo_horizon, naive_trend)")
    parser.add_argument("--suffix", default="", help="출력 모델 이름 뒤에 붙일 말(예: _v2)")
    parser.add_argument("--origins", nargs="*", help="(hz) 이 기준월들만(기본 ORIGINS + 확인용)")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args(argv)
    run(args.xgb_run, args.csv, only=args.only, suffix=args.suffix, origins=args.origins, check=not args.no_check)


if __name__ == "__main__":
    main()
