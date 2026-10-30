"""ETS(차종별 통계 모델) 백테스트 예측.

experiments/ets_v1/ets_core.py를 그대로 쓴다. 후보는 ANN·AAdN·ANA·AAdA이고, 12~23개월 이력은 비계절 후보만,
24개월 이상은 4개 모두 학습해 학습 AICc로 고른다. 기준월까지의 마지막 연속 구간만 쓰며, 누락은 채우지 않는다.
상태식으로 24개월(확인용 기준월은 12개월)을 한 번에 예측하고 0 미만은 0으로 자른다.
--log(ets_log, 2026-10-01 사전 고정): log1p(판매량)에 같은 ETS를 맞추고 예측을 expm1로 되돌린다. statsmodels가 있는
그 폴더 전용 환경으로 실행한다(루트 환경은 바꾸지 않음).
--theta(theta, 2026-10-03 사전 고정 (ab)): experiments/theta_v1/theta_core(statsmodels ThetaModel θ=2, 같은 이력 규칙,
24개월 이상이면 가법 계절 조정)로 같은 방식의 예측을 낸다.
--arima(arima, 2026-10-03 사전 고정 (aq)): experiments/arima_v1/arima_core(statsforecast AutoARIMA, 같은 이력 규칙,
24개월 이상이면 season_length 12). statsforecast가 있는 experiments/arima_v1/.venv로 실행한다.

사용: experiments/ets_v1/.venv/Scripts/python.exe -X utf8 -B experiments/horizon_eval_v1/forecast_ets.py [--workers 4]
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import importlib.util

import numpy as np
import pandas as pd

from common import (CHECK_HORIZON, CHECK_ORIGIN, CSV_PATH, HERE, HORIZON, KEY, ORIGINS, ROOT, file_sha256, guarded,
                    new_run_dir, require_unchanged, stack_outputs, write_csv, write_json)
import baseline

MODEL_NAME = "ets"
_CORES = {}
CORE_FILES = {False: ("ets_core", ROOT / "experiments" / "ets_v1" / "ets_core.py"),
              True: ("theta_core", ROOT / "experiments" / "theta_v1" / "theta_core.py"),
              "arima": ("arima_core", ROOT / "experiments" / "arima_v1" / "arima_core.py")}


def core(theta=False):
    """ets_v1의 ets_core(theta=True면 theta_v1의 theta_core, "arima"면 arima_v1의 arima_core)를 파일 경로로 불러온다."""
    if theta not in _CORES:
        name, path = CORE_FILES[theta]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CORES[theta] = module
    return _CORES[theta]


def _transform(rows, log):
    return rows.assign(Sales=np.log1p(rows["Sales"])) if log else rows


def forecast_one(job):
    """한 차종·한 기준월: 학습 AICc로 고른 ETS로 horizon개월 예측. (예측 행들, 상태 행, 선택 후보)를 반환한다."""
    key, rows, origin, horizon, log, *rest = job
    ets = core(rest[0] if rest else False)
    bundle = ets.fit_series(_transform(rows, log), origin, key=list(key))
    state = {**dict(zip(KEY, key)), "prediction_status": bundle["status"], "reason": bundle["reason"]}
    if bundle["status"] != "OK":
        return [], state, None
    try:
        values = ets.forecast_bundle(bundle, horizon)
        values = np.expm1(values) if log else values
    except ValueError as error:
        state.update(prediction_status="PREDICTION_ERROR", reason=str(error))
        return [], state, None
    start = pd.Period(origin, freq="M")
    predictions = [{**dict(zip(KEY, key)), "Month": start + h, "horizon": h, "prediction": float(v)}
                   for h, v in enumerate(values, start=1)]
    return predictions, state, bundle["metadata"].get("selected_candidate")


def check_reload(job, expected, run_dir):
    """한 차종의 ETS를 저장했다가 다시 불러 같은 예측이 나오는지 확인한다."""
    key, rows, origin, horizon, log, *rest = job
    ets = core(rest[0] if rest else False)
    bundle = ets.fit_series(_transform(rows, log), origin, key=list(key))
    info = ets.save_bundle(bundle, guarded(run_dir) / "models" / "reload_check.pkl")
    loaded = ets.load_bundle(info["model_path"], info["model_sha256"])
    again = ets.forecast_bundle(loaded, horizon)
    np.testing.assert_allclose(np.expm1(again) if log else again, expected, rtol=1e-12, atol=1e-9)


def run(csv_path=CSV_PATH, output_base=HERE / "reports", workers=4, origins=ORIGINS, check=True, log=False, theta=False,
        arima=False):
    input_hash = file_sha256(csv_path)
    data = baseline.read_sales_csv(guarded(csv_path))
    if (theta or arima) and log or (theta and arima):
        raise ValueError("--theta·--arima·--log는 함께 쓰지 않습니다.")
    kind = "arima" if arima else theta
    name = "arima" if arima else "theta" if theta else "ets_log" if log else MODEL_NAME
    run_dir = new_run_dir(output_base, name)
    plan = [(o, HORIZON) for o in origins] + ([(CHECK_ORIGIN, CHECK_HORIZON)] if check else [])
    jobs = []
    for origin, horizon in plan:
        history = data.loc[data["Month"].le(pd.Period(origin, freq="M"))]
        for key, rows in history.groupby(KEY, sort=True):
            jobs.append((key, rows[KEY + ["Month", "Sales"]].reset_index(drop=True), origin, horizon, log, kind))
    if workers == 0:
        results = [forecast_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(forecast_one, jobs, chunksize=16))
    frames_p, frames_s = [], []
    for origin, _ in plan:
        picked = [result for job, result in zip(jobs, results) if job[2] == origin]
        rows = [row for predictions, _, _ in picked for row in predictions]
        columns = KEY + ["Month", "horizon", "prediction"]
        p, s = stack_outputs([pd.DataFrame(rows, columns=columns)], [pd.DataFrame([state for _, state, _ in picked])],
                             name, pd.Period(origin, freq="M"))
        frames_p.append(p)
        frames_s.append(s)
        print(f"{name} {origin}: OK {int(s['prediction_status'].eq('OK').sum())} / {len(s)}", flush=True)
    predictions, statuses = pd.concat(frames_p, ignore_index=True), pd.concat(frames_s, ignore_index=True)
    first = next(i for i, (rows, _, _) in enumerate(results) if rows)
    check_reload(jobs[first], np.array([row["prediction"] for row in results[first][0]]), run_dir)
    require_unchanged(csv_path, input_hash)
    write_csv(run_dir / "predictions.csv", predictions)
    write_csv(run_dir / "statuses.csv", statuses)
    write_json(run_dir / "run_summary.json", {
        "model": name, "log_target": log, "input_sha256": input_hash, "origins": list(origins),
        "check_origin": CHECK_ORIGIN if check else None, "ets_core": ("experiments/arima_v1/arima_core.py" if arima else "experiments/theta_v1/theta_core.py" if theta
                                 else "experiments/ets_v1/ets_core.py"),
        "selected_candidates": dict(Counter(choice for _, _, choice in results if choice)),
        "status_counts": statuses.groupby(["origin", "prediction_status"]).size().rename("n").reset_index().to_dict("records"),
        "reload_check": "PASS"})
    print(run_dir)
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--output-base", default=str(HERE / "reports"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--origins", nargs="*", default=ORIGINS)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--log", action="store_true", help="log1p(판매량)에 맞추고 expm1로 되돌린다(ets_log)")
    parser.add_argument("--theta", action="store_true", help="ETS 대신 Theta(theta_v1/theta_core)")
    parser.add_argument("--arima", action="store_true", help="ETS 대신 AutoARIMA(arima_v1/arima_core, 그 폴더 환경)")
    args = parser.parse_args(argv)
    run(args.csv, args.output_base, args.workers, args.origins, not args.no_check, args.log, args.theta, args.arima)


if __name__ == "__main__":
    main()
