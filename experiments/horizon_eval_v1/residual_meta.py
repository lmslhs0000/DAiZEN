"""(gf) 잔차 보정 학습(2026-10-06 사전 고정): (기준월, 차종, 묶음)의 log((실제 합+1)/(예측 합+1))을 기준월 시점 특징으로
LightGBM(L1, 예측 합 가중)에 학습해, 그 묶음 달들에 exp(0.5 × ŷ)를 곱한다. 묶음 = 중간 분기 3개(4~6·7~9·10~12)와
뒤 1년(13~24). 앞 1~3개월은 그대로. 백테스트 기준월은 그 기준월을 뺀 나머지로 학습, 확인용은 백테스트 모두로.

사용: experiments/lightgbm_v1/.venv/Scripts/python.exe -X utf8 -B experiments/horizon_eval_v1/residual_meta.py
      --first <run> --name <이름>
"""

import argparse

import numpy as np
import pandas as pd

from common import (CSV_PATH, HERE, KEY, ORIGINS, PREDICTION_COLUMNS, STATUS_COLUMNS, file_sha256, guarded, new_run_dir,
                    write_csv, write_json)

GROUPS = [("middle", 1, 4, 6), ("middle", 2, 7, 9), ("middle", 3, 10, 12), ("back", 1, 13, 24)]
FEATURES = ["Country", "part", "bucket", "log_pred", "log_last12", "log_growth12", "log_dip", "hist_len", "brand_share",
            "log_brand_growth12"]
SHRINK = 0.5
LGB_PARAMS = {"objective": "l1", "num_leaves": 15, "n_estimators": 200, "learning_rate": 0.05, "min_data_in_leaf": 50,
              "random_state": 42, "verbose": -1}


def _sum(frame, low, high):
    return frame.loc[frame["Month"].between(low, high)].groupby(KEY)["Sales"].sum()


def series_features(data, origin):
    origin = pd.Period(origin, freq="M")
    rows = data.loc[data["Month"].le(origin)].dropna(subset=["Sales"])
    last12, prev12 = _sum(rows, origin - 11, origin), _sum(rows, origin - 23, origin - 12)
    last3 = rows.loc[rows["Month"].gt(origin - 3)].groupby(KEY)["Sales"].mean()
    before = rows.loc[rows["Month"].between(origin - 14, origin - 3)].groupby(KEY)["Sales"].mean()
    keys = rows[KEY].drop_duplicates().set_index(KEY).index
    table = pd.DataFrame(index=keys)
    table["last12"] = last12.reindex(keys).fillna(0.0)
    table["prev12"] = prev12.reindex(keys).fillna(0.0)
    table["log_last12"] = np.log(table["last12"] + 1)
    table["log_growth12"] = np.log((table["last12"] + 1) / (table["prev12"] + 1))
    table["log_dip"] = np.log((last3.reindex(keys).fillna(0.0) + 1) / (before.reindex(keys).fillna(0.0) + 1))
    lengths = {}
    for key, g in rows.groupby(KEY, sort=False):
        months = np.array(sorted(m.ordinal for m in g["Month"]))
        if months[-1] != origin.ordinal:
            lengths[key] = 0
            continue
        breaks = np.nonzero(np.diff(months) != 1)[0]
        lengths[key] = min(120, len(months) - (breaks[-1] + 1 if len(breaks) else 0))
    table["hist_len"] = [lengths[k] for k in table.index]
    brand = table.groupby(level=["Country", "Brand"])[["last12", "prev12"]].transform("sum")
    table["brand_share"] = np.where(brand["last12"] > 0, table["last12"] / brand["last12"].replace(0, np.nan), 0.0)
    table["log_brand_growth12"] = np.log((brand["last12"] + 1) / (brand["prev12"] + 1))
    return table.drop(columns=["last12", "prev12"]).reset_index()


def bucket_table(predictions, data):
    """예측 표 → (기준월, 차종, 묶음) 표: pred_sum, actual_sum, y(실제가 다 있을 때), 특징."""
    actual = data[KEY + ["Month", "Sales"]].dropna(subset=["Sales"]).copy()
    actual["Month"] = actual["Month"].astype(str)
    frames = []
    for origin, rows in predictions.groupby(predictions["origin"].astype(str), sort=False):
        rows = rows.assign(Month=rows["Month"].astype(str)).merge(actual, on=KEY + ["Month"], how="left")
        for part, bucket, low, high in GROUPS:
            chunk = rows.loc[rows["horizon"].between(low, high)]
            g = chunk.groupby(KEY).agg(n=("prediction", "count"), pred_sum=("prediction", "sum"),
                                        n_actual=("Sales", "count"), actual_sum=("Sales", "sum"))
            g = g.loc[g["n"].eq(high - low + 1)].reset_index()
            g["y"] = np.where(g["n_actual"].eq(high - low + 1),
                              np.log((g["actual_sum"] + 1) / (g["pred_sum"] + 1)), np.nan)
            frames.append(g.assign(origin=origin, part=part, bucket=bucket))
        frames[-4:] = [f.merge(series_features(data, origin), on=KEY, how="left") for f in frames[-4:]]
    table = pd.concat(frames, ignore_index=True)
    table["log_pred"] = np.log(table["pred_sum"] + 1)
    return table


def loo_predict(table, fitter, backtest_origins=ORIGINS):
    backtest = [str(o) for o in backtest_origins]
    out = []
    for origin in dict.fromkeys(table["origin"].astype(str)):
        test = table.loc[table["origin"].astype(str).eq(origin)]
        train = table.loc[table["origin"].astype(str).isin([o for o in backtest if o != origin]) & table["y"].notna()]
        out.append(test.assign(yhat=fitter(train, test)))
    return pd.concat(out, ignore_index=True)


def time_respecting_predict(table, fitter, min_rows=50):
    """(gf2) 시험 기준월 O의 학습 행 = 정답이 있고 묶음 끝 달(기준월' + 끝 거리) ≤ O인 행. min_rows 미만이면 보정 0."""
    ends = {(part, bucket): high for part, bucket, _, high in GROUPS}
    end = [pd.Period(str(o), freq="M") + ends[(p, b)] for o, p, b in zip(table["origin"], table["part"], table["bucket"])]
    end_ordinal = np.array([e.ordinal for e in end])
    out = []
    for origin in dict.fromkeys(table["origin"].astype(str)):
        test = table.loc[table["origin"].astype(str).eq(origin)]
        keep = table["y"].notna().to_numpy() & (end_ordinal <= pd.Period(origin, freq="M").ordinal)
        train = table.loc[keep]
        yhat = fitter(train, test) if len(train) >= min_rows else np.zeros(len(test))
        out.append(test.assign(yhat=yhat))
    return pd.concat(out, ignore_index=True)


def lgb_fitter(train, test):
    import lightgbm as lgb
    def frame(t):
        x = t[FEATURES].copy()
        for c in ("Country", "part"):
            x[c] = pd.Categorical(x[c], categories=sorted(set(train[c]) | set(test[c])))
        return x
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(frame(train), train["y"], sample_weight=train["pred_sum"].clip(lower=1.0))
    return model.predict(frame(test))


def correct(predictions, yhat, shrink=SHRINK):
    out = predictions.copy()
    out["_o"] = out["origin"].astype(str)
    factor = np.ones(len(out))
    look = {(str(o), c, b, m, p, k): v for o, c, b, m, p, k, v in
            zip(yhat["origin"], yhat["Country"], yhat["Brand"], yhat["Model"], yhat["part"], yhat["bucket"], yhat["yhat"])}
    h = out["horizon"].to_numpy()
    for part, bucket, low, high in GROUPS:
        mask = (h >= low) & (h <= high)
        keys = zip(out["_o"][mask], out["Country"][mask], out["Brand"][mask], out["Model"][mask])
        values = np.array([look.get((o, c, b, m, part, bucket), 0.0) for o, c, b, m in keys])
        factor[mask] = np.exp(shrink * values)
    out["prediction"] = out["prediction"].to_numpy(float) * factor
    return out.drop(columns="_o")


def main(argv=None):
    import baseline
    from challengers import read_statuses
    from score import read_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--output-base", default=str(HERE / "reports"))
    parser.add_argument("--backtest-origins", nargs="*", default=ORIGINS)
    parser.add_argument("--time-respecting", action="store_true", help="(gf2) 묶음 끝 달 ≤ 시험 기준월인 행만으로 학습")
    args = parser.parse_args(argv)
    data = baseline.read_sales_csv(guarded(args.csv))
    predictions = read_run(args.first)
    table = bucket_table(predictions, data)
    yhat = (time_respecting_predict(table, lgb_fitter) if args.time_respecting
            else loo_predict(table, lgb_fitter, args.backtest_origins))
    out = correct(predictions, yhat).assign(model=args.name)[PREDICTION_COLUMNS]
    run_dir = new_run_dir(args.output_base, args.name)
    write_csv(run_dir / "predictions.csv", out)
    write_csv(run_dir / "statuses.csv", read_statuses(args.first).assign(model=args.name)[STATUS_COLUMNS])
    write_csv(run_dir / "yhat.csv", yhat[["origin"] + KEY + ["part", "bucket", "pred_sum", "y", "yhat"]])
    write_json(run_dir / "run_summary.json", {
        "model": args.name, "rule": "(gf) residual meta-correction, LightGBM l1 weighted by pred sum, "
        + ("(gf2) time-respecting: train rows whose bucket ended by the test origin" if args.time_respecting
           else "leave-one-origin-out (overlapping target windows — leaky)"),
        "features": FEATURES, "params": LGB_PARAMS, "shrink": SHRINK, "source": str(args.first),
        "input_sha256": file_sha256(args.csv), "backtest_origins": list(args.backtest_origins), "trained_models": "meta"})
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
