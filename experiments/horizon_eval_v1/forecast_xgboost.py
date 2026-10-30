"""기본 XGBoost(AGENTS.md 5절 11입력, A 방식 시장별 선택 설정)의 기준월별 24개월 백테스트 예측."""

import argparse

import numpy as np
import pandas as pd

from common import (CHECK_HORIZON, CHECK_ORIGIN, CSV_PATH, HERE, HORIZON, KEY, ORIGINS, file_sha256, guarded,
                    new_run_dir, read_json, require_unchanged, stack_outputs, write_csv, write_json)
import baseline
import xgboost_core as core

MODEL_NAME = "basic_xgboost"
SELECTED = "reports/xgboost/run_20260921T175602_851343Z_28c46a24e7d2/selected_parameters.json"


def load_parameters(path=SELECTED):
    chosen = {r["Country"]: r["parameters"] for r in read_json(path) if r["method"] == "A" and r["status"] == "OK"}
    if len(chosen) != 7:
        raise ValueError(f"A 방식 선택 설정이 7개 시장이 아닙니다: {sorted(chosen)}")
    return chosen


def forecast_origin(data, origin, parameters, horizon, n_jobs=4, model_dir=None, model_name=MODEL_NAME):
    """기준월까지 자료로만 시장별 모델을 새로 학습하고 horizon개월을 재귀 예측한다."""
    origin = pd.Period(origin, freq="M")
    history = data.loc[data["Month"].le(origin)].copy()
    training, _ = core.build_training_frame(history, origin)
    predictions, statuses = [], []
    for country in sorted(history["Country"].unique()):
        market = history.loc[history["Country"].eq(country)]
        try:
            model, schema = core.fit_market(training, country, parameters[country], origin, n_jobs=n_jobs)
        except ValueError as exc:
            if str(exc) != "NO_TRAINING_ROWS":
                raise
            statuses.append(market[KEY].drop_duplicates().assign(prediction_status="NO_TRAINING_ROWS",
                                                                 reason="NO_TRAINING_ROWS"))
            continue
        if model_dir is not None:
            core.save_market_model(model, schema, guarded(model_dir) / str(origin) / country / "model.ubj")
        p, s = core.forecast_recursive(model, market, origin, horizon, schema)
        predictions.append(p)
        statuses.append(s)
    return stack_outputs(predictions, statuses, model_name, origin)


def check_reload(data, model_dir, origin, country, horizon, expected):
    """저장 모델을 다시 불러 같은 예측이 나오는지 확인한다."""
    origin = pd.Period(origin, freq="M")
    model, schema = core.load_market_model(guarded(model_dir) / str(origin) / country / "model.ubj")
    history = data.loc[data["Month"].le(origin) & data["Country"].eq(country)]
    again, _ = core.forecast_recursive(model, history, origin, horizon, schema)
    saved = expected.loc[expected["origin"].eq(str(origin)) & expected["Country"].eq(country)]
    order = KEY + ["Month"]
    np.testing.assert_allclose(again.sort_values(order)["prediction"].to_numpy(float),
                               saved.sort_values(order)["prediction"].to_numpy(float), rtol=1e-12, atol=1e-9)


def run(csv_path=CSV_PATH, output_base=HERE / "reports", n_jobs=4, origins=ORIGINS, check=True,
        parameters=None, model_name=MODEL_NAME):
    input_hash = file_sha256(csv_path)
    data = baseline.read_sales_csv(guarded(csv_path))
    parameters = parameters or load_parameters()
    run_dir = new_run_dir(output_base, model_name)
    jobs = [(o, HORIZON) for o in origins] + ([(CHECK_ORIGIN, CHECK_HORIZON)] if check else [])
    frames_p, frames_s = [], []
    for origin, horizon in jobs:
        p, s = forecast_origin(data, origin, parameters, horizon, n_jobs, run_dir / "models", model_name)
        frames_p.append(p)
        frames_s.append(s)
        print(f"{model_name} {origin}: OK {int(s['prediction_status'].eq('OK').sum())} / {len(s)}", flush=True)
    predictions = pd.concat(frames_p, ignore_index=True)
    statuses = pd.concat(frames_s, ignore_index=True)
    first = predictions.iloc[0]
    check_reload(data, run_dir / "models", first["origin"], first["Country"],
                 HORIZON if first["origin"] in origins else CHECK_HORIZON, predictions)
    require_unchanged(csv_path, input_hash)
    write_csv(run_dir / "predictions.csv", predictions)
    write_csv(run_dir / "statuses.csv", statuses)
    write_json(run_dir / "run_summary.json", {
        "model": model_name, "input_sha256": input_hash, "origins": list(origins),
        "check_origin": CHECK_ORIGIN if check else None, "parameters": parameters,
        "status_counts": statuses.groupby(["origin", "prediction_status"]).size().rename("n").reset_index().to_dict("records"),
        "reload_check": "PASS"})
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--output-base", default=str(HERE / "reports"))
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--origins", nargs="*", default=ORIGINS)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.csv, args.output_base, args.n_jobs, args.origins, not args.no_check))


if __name__ == "__main__":
    main()
