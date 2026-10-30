"""basic_xgboost_cal: 기본 XGBoost(AGENTS.md 5절 11입력)에 대상월 달력 8개를 더한 시장별 모델.

달력(점수를 보기 전에 고정, 2026-10-01): rich_cal과 같은 정의인 experiments/calendar_exposure_v1/calendar_features.py의
대상월 달력 일수(days_in_month)와 월~일요일 개수(weekday_count_0~6). 공휴일 가정 없음. 미래 달력은 결정적으로 알 수 있다.
--kind wd(basic_xgboost_wd, 2026-10-01 사전 고정): 달력 8개 대신 experiments/workdays_v1 표의 시장별 대상월 근무일
(공휴일 반영) 1개만 더한다.
--kind age(basic_xgboost_age, 2026-10-02 사전 고정 (e)): 달력 대신 series_age 1개 = 대상월 − 그 차종이 기준월까지
처음 판매가 관측된 달(개월, 첫 달 0). 첫 관측월은 기준월 이하 자료로만 정하고, 재귀 예측에서도 대상월로 계산한다.
--kind tw(basic_xgboost_tw, 2026-10-02 사전 고정 (i)): 입력은 11개 그대로, objective만 reg:tweedie(power 1.5 고정),
base_score = 그 시장 학습 판매량 평균. 출력은 판매 대수(log 링크).
--kind lvw(basic_xgboost_lvw, 2026-10-03 사전 고정 (ai)): 입력 11개 그대로, 정답 log1p(판매량)(예측은 expm1 후 0 하한),
학습 가중치 = 차종 학습 행 평균 판매량 ÷ 그 시장 차종 평균.
--kind clvw(basic_xgboost_clvw, 2026-10-03 사전 고정 (aj)): lvw에 대상월 달력 8개(cal과 같은 정의)를 더함.
--kind elvw(basic_xgboost_elvw, 2026-10-03 사전 고정 (ba)): lvw에 시장 신차 비중 2개(experiments/entrant_v1/entrant_features)
를 더함. 학습 행은 대상월 직전 달 기준, 재귀 예측은 기준월 고정(그 시장의 기준월 이하 자료로 계산).
--kind blvw(basic_xgboost_blvw, 2026-10-03 사전 고정 (bp)): lvw에 브랜드 흐름 2개(experiments/brand_momentum_v1/
brand_features)를 같은 기준 시점 규칙으로 더함.
설정: basic_xgboost_v2의 시장별 선택 설정(+ AGENTS.md 5절 고정 설정). 재귀 예측, 0 미만은 0으로 자른다.

사용: PY experiments/horizon_eval_v1/xgb_cal.py --v2-run <reports/basic_xgboost_v2_...> [--kind cal|wd]
"""

import argparse
import importlib.util
import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from common import (CHECK_HORIZON, CHECK_ORIGIN, CSV_PATH, HERE, HORIZON, KEY, ORIGINS, PROJECT_ROOT, ROOT, file_sha256,
                    guarded, new_run_dir, read_json, require_unchanged, stack_outputs, write_csv, write_json)
import baseline
import xgboost_core as core

MODEL_NAMES = {"cal": "basic_xgboost_cal", "wd": "basic_xgboost_wd", "age": "basic_xgboost_age", "tw": "basic_xgboost_tw",
               "lvw": "basic_xgboost_lvw", "clvw": "basic_xgboost_clvw", "elvw": "basic_xgboost_elvw",
               "blvw": "basic_xgboost_blvw"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cf = _load("calendar_features", ROOT / "experiments" / "calendar_exposure_v1" / "calendar_features.py")
wf = _load("workday_features", ROOT / "experiments" / "workdays_v1" / "workday_features.py")
ef = _load("entrant_features", ROOT / "experiments" / "entrant_v1" / "entrant_features.py")
bfm = _load("brand_features", ROOT / "experiments" / "brand_momentum_v1" / "brand_features.py")
CALENDAR = list(cf.COLUMNS)
EXTRAS = {"cal": CALENDAR, "wd": list(wf.COLUMNS), "age": ["series_age"], "tw": [], "lvw": [], "clvw": CALENDAR,
          "elvw": list(ef.COLUMNS), "blvw": list(bfm.COLUMNS)}
LOG_KINDS = ("lvw", "clvw", "elvw", "blvw")
DEFINITIONS = {"cal": cf.DEFINITION, "wd": wf.DEFINITION,
               "age": "series_age = target month - first month with non-blank Sales up to the origin (months, first = 0)",
               "tw": "no extra inputs; objective reg:tweedie (variance power 1.5), base_score = market training mean Sales",
               "lvw": "no extra inputs; target log1p(Sales), sample weight = series mean Sales / market mean over series",
               "clvw": {"calendar": cf.DEFINITION, "target": "log1p(Sales)",
                        "sample_weight": "series mean Sales / market mean over series"},
               "elvw": {"entrant_shares": "experiments/entrant_v1/entrant_features.py (as of target month - 1 in training, "
                                          "origin during recursion)", "target": "log1p(Sales)",
                        "sample_weight": "series mean Sales / market mean over series"},
               "blvw": {"brand_changes": "experiments/brand_momentum_v1/brand_features.py (as of target month - 1 in "
                                         "training, origin during recursion)", "target": "log1p(Sales)",
                        "sample_weight": "series mean Sales / market mean over series"}}
TWEEDIE = {"objective": "reg:tweedie", "tweedie_variance_power": 1.5}


def numeric(kind):
    return core.NUMERIC_FEATURES + EXTRAS[kind]


def features(kind):
    return numeric(kind) + ["Brand", "Model"]
FIXED = {"objective": "reg:squarederror", "tree_method": "hist", "enable_categorical": True, "device": "cpu",
         "random_state": 42, "subsample": 1.0, "colsample_bytree": 1.0}
PRED_COLUMNS = KEY + ["Month", "horizon", "prediction"]
STATE_COLUMNS = KEY + ["prediction_status", "reason"]


def series_weights(rows):
    """학습 행마다 그 차종의 학습 행 평균 판매량 ÷ (같은 행들 안) 차종 평균 — 차종 단위로 평균 1."""
    means = rows.groupby(KEY, sort=False)["Sales"].transform("mean")
    return (means / float(rows.groupby(KEY, sort=False)["Sales"].mean().mean())).to_numpy(float)


def first_months(history):
    """차종별로 판매량이 비어 있지 않은 첫 달. history는 기준월 이하 자료만 담아야 한다."""
    present = history.loc[history["Sales"].notna()]
    return {key: month for key, month in present.groupby(KEY, sort=True)["Month"].min().items()}


def add_calendar(rows, kind="cal", firsts=None, stats=None, as_of=None):
    """Month(대상월) 열로 달력 8개(cal), 시장별 근무일 1개(wd), 차종 나이 1개(age, firsts 필요) 또는
    신차 비중 2개(elvw, stats 필요; as_of가 없으면 대상월 직전 달)를 붙인다."""
    rows = rows.copy()
    if kind in ("tw", "lvw"):
        return rows
    if kind == "elvw":
        months = pd.PeriodIndex(rows["Month"], freq="M")
        when = months - 1 if as_of is None else pd.PeriodIndex([as_of] * len(rows), freq="M")
        shares = ef.entrant_features(rows["Country"], when, stats)
        for name in ef.COLUMNS:
            rows[name] = shares[name].to_numpy(float)
        return rows
    if kind == "blvw":
        months = pd.PeriodIndex(rows["Month"], freq="M")
        when = months - 1 if as_of is None else pd.PeriodIndex([as_of] * len(rows), freq="M")
        changes = bfm.brand_features(rows["Country"], rows["Brand"], when, stats)
        for name in bfm.COLUMNS:
            rows[name] = changes[name].to_numpy(float)
        return rows
    if kind == "age":
        keys = list(rows[KEY].itertuples(index=False, name=None))
        months = pd.PeriodIndex(rows["Month"], freq="M")
        rows["series_age"] = [float(month.ordinal - firsts[key].ordinal) for key, month in zip(keys, months)]
        return rows
    if kind == "wd":
        rows["workdays"] = wf.lookup(rows["Country"], pd.PeriodIndex(rows["Month"], freq="M").astype(str))
        return rows
    table = {month: cf.month_features(str(month)) for month in rows["Month"].unique()}
    for name in CALENDAR:
        rows[name] = [table[month][name] for month in rows["Month"]]
    return rows


def build_training(data, origin, kind="cal"):
    """기본 11입력 학습 행(정답월 ≤ 기준월)에 대상월 달력(kind)을 붙인다."""
    origin = pd.Period(origin, freq="M")
    history = data.loc[data["Month"].le(origin)].copy()
    base, _ = core.build_training_frame(history, origin)
    firsts = first_months(history) if kind == "age" else None
    stats = ef.entrant_stats(history) if kind == "elvw" else bfm.brand_totals(history) if kind == "blvw" else None
    return add_calendar(base, kind, firsts, stats).sort_values(KEY + ["Month"]).reset_index(drop=True)


def _frame(rows, schema):
    frame = rows[schema["features"]].copy()
    for name in numeric(schema["kind"]):
        frame[name] = frame[name].astype("int32" if name == "month" else "float64")
    for name in ("Brand", "Model"):
        frame[name] = pd.Categorical(frame[name], categories=schema["categories"][name])
    return frame


def fit_cal(training, country, parameters, origin, kind="cal"):
    origin = pd.Period(origin, freq="M")
    rows = training.loc[training["Country"].eq(country) & training["Month"].le(origin)]
    if rows.empty:
        raise ValueError("NO_TRAINING_ROWS")
    settings = {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 200, **parameters, **FIXED, "n_jobs": 4}
    if kind == "tw":
        settings.update(TWEEDIE, base_score=float(rows["Sales"].mean()))
    schema = {"country": country, "kind": kind, "features": features(kind), "settings": settings,
              "train_end": str(origin), "calendar_definition": DEFINITIONS[kind],
              "categories": {name: sorted(rows[name].unique().tolist()) for name in ("Brand", "Model")},
              "seen_series": rows[KEY].drop_duplicates().values.tolist(), "training_rows": int(len(rows))}
    model = XGBRegressor(**settings)
    if kind in LOG_KINDS:
        schema["target"] = "log1p"
        model.fit(_frame(rows, schema), np.log1p(rows["Sales"].to_numpy(float)), sample_weight=series_weights(rows))
    else:
        schema["target"] = "sales"
        model.fit(_frame(rows, schema), rows["Sales"].to_numpy(float))
    return model, schema


def step_features(keys, buffers, month, kind="cal", first_months=None, stats=None, as_of=None):
    """재귀 한 단계의 입력: 버퍼(12개월) 마지막 값이 대상월 직전 달이다. 학습 행 특징과 같은 정의."""
    batch = pd.DataFrame(keys, columns=KEY)
    batch["Month"] = month
    for lag in (1, 2, 3, 6, 12):
        batch[f"lag_{lag}"] = buffers[:, -lag]
    for window in (3, 6, 12):
        batch[f"rolling_mean_{window}"] = buffers[:, -window:].mean(axis=1)
    batch["month"] = month.month
    return add_calendar(batch, kind, first_months, stats, as_of)[KEY + ["Month"] + numeric(kind)]


def forecast_cal(model, schema, history, origin, horizon):
    """기준월까지 자료로 한 시장의 차종들을 horizon개월 재귀 예측한다."""
    origin = pd.Period(origin, freq="M")
    observed = history.loc[history["Month"].le(origin) & history["Country"].eq(schema["country"])]
    seen = {tuple(key) for key in schema["seen_series"]}
    required = pd.period_range(origin - 11, origin, freq="M")
    keys, buffers, states = [], [], []
    for key, vehicle in observed.groupby(KEY, sort=True):
        monthly = vehicle.set_index("Month")["Sales"].astype(float)
        status = ("INSUFFICIENT_HISTORY" if monthly.index.min() > required[0]
                  else "INCOMPLETE_HISTORY" if monthly.reindex(required).isna().any() else "OK")
        if status == "OK" and key not in seen:
            status = "UNSEEN_SERIES"
        states.append({**dict(zip(KEY, key)), "prediction_status": status, "reason": "" if status == "OK" else status})
        if status == "OK":
            keys.append(key)
            buffers.append(monthly.reindex(required).to_numpy(float))
    rows = []
    firsts = first_months(observed) if schema["kind"] == "age" else None
    stats = (ef.entrant_stats(observed) if schema["kind"] == "elvw"  # 그 시장의 기준월 이하 자료(예측 중 고정)
             else bfm.brand_totals(observed) if schema["kind"] == "blvw" else None)
    if keys:
        buffers = np.asarray(buffers, float)
        for step in range(horizon):
            month = origin + step + 1
            batch = step_features(keys, buffers, month, schema["kind"], firsts, stats, origin)
            raw = np.asarray(model.predict(_frame(batch, schema)), float)
            prediction = np.maximum(0.0, np.expm1(raw) if schema.get("target") == "log1p" else raw)
            if not np.isfinite(prediction).all():
                raise ValueError(f"PREDICTION_ERROR at {month}")
            rows += [{**dict(zip(KEY, key)), "Month": month, "horizon": step + 1, "prediction": value}
                     for key, value in zip(keys, prediction)]
            buffers[:, :-1] = buffers[:, 1:]
            buffers[:, -1] = prediction
    return pd.DataFrame(rows, columns=PRED_COLUMNS), pd.DataFrame(states, columns=STATE_COLUMNS)


def run(v2_run, csv_path=CSV_PATH, output_base=HERE / "reports", origins=ORIGINS, check=True, kind="cal"):
    input_hash = file_sha256(csv_path)
    model_name = MODEL_NAMES[kind]
    data = baseline.read_sales_csv(guarded(csv_path))
    parameters = {c: item["parameters"] for c, item in read_json(guarded(v2_run) / "run_summary.json")["choice"].items()}
    run_dir = new_run_dir(output_base, model_name)
    plan = [(o, HORIZON) for o in origins] + ([(CHECK_ORIGIN, CHECK_HORIZON)] if check else [])
    frames_p, frames_s, first = [], [], None
    for origin, horizon in plan:
        training = build_training(data, origin, kind)
        predictions, statuses = [], []
        for country in sorted(parameters):
            model, schema = fit_cal(training, country, parameters[country], origin, kind)
            path = run_dir / "models" / origin / country / "model.ubj"
            path.parent.mkdir(parents=True)
            model.save_model(path)
            path.with_name("schema.json").write_text(json.dumps(schema, ensure_ascii=False, default=str), encoding="utf-8")
            p, s = forecast_cal(model, schema, data, origin, horizon)
            if first is None:
                first = (path, origin, horizon, p)
            predictions.append(p)
            statuses.append(s)
        p, s = stack_outputs(predictions, statuses, model_name, pd.Period(origin, freq="M"))
        frames_p.append(p)
        frames_s.append(s)
        print(f"{model_name} {origin}: OK {int(s['prediction_status'].eq('OK').sum())} / {len(s)}", flush=True)
    path, origin, horizon, expected = first
    reloaded = XGBRegressor()
    reloaded.load_model(path)
    again, _ = forecast_cal(reloaded, json.loads(path.with_name("schema.json").read_text(encoding="utf-8")), data,
                            origin, horizon)
    np.testing.assert_allclose(again["prediction"].to_numpy(float), expected["prediction"].to_numpy(float),
                               rtol=1e-12, atol=1e-9)
    predictions, statuses = pd.concat(frames_p, ignore_index=True), pd.concat(frames_s, ignore_index=True)
    require_unchanged(csv_path, input_hash)
    write_csv(run_dir / "predictions.csv", predictions)
    write_csv(run_dir / "statuses.csv", statuses)
    write_json(run_dir / "run_summary.json", {
        "model": model_name, "kind": kind, "input_sha256": input_hash, "origins": list(origins),
        "check_origin": CHECK_ORIGIN if check else None, "features": features(kind), "parameters": parameters,
        "calendar_definition": DEFINITIONS[kind], "v2_run": str(guarded(v2_run).relative_to(PROJECT_ROOT)),
        "status_counts": statuses.groupby(["origin", "prediction_status"]).size().rename("n").reset_index().to_dict("records"),
        "reload_check": "PASS"})
    print(run_dir)
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-run", required=True)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--origins", nargs="*", default=ORIGINS)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--kind", default="cal", choices=sorted(MODEL_NAMES))
    parser.add_argument("--output-base", default=str(HERE / "reports"), help="결과 폴더(납품 재학습은 production)")
    args = parser.parse_args(argv)
    run(args.v2_run, args.csv, output_base=args.output_base, origins=args.origins, check=not args.no_check, kind=args.kind)


if __name__ == "__main__":
    main()
