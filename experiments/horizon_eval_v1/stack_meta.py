"""(ic) 차종×부분별 조립 가중 메타모델(2026-10-13 사전 고정, progress.md).

세 조립(재료) 중 어느 것이 (기준월, 차종, 부분)에서 절대오차 합이 가장 작은지를 LightGBM 다중분류로 배우고,
예측 확률을 가중치로 재료를 섞는다. 특징 = residual_meta.series_features(기준월 시점) + 시장 + 부분.
학습 행은 시간 준수: 시험 기준월 O의 학습 행 = 부분 창 끝(앞 O'+3·중간 O'+12·뒤 O'+24) ≤ O인 행.
학습 행이 min_rows 미만이면 동일 가중 평균.

사용: experiments/lightgbm_v1/.venv/Scripts/python.exe -X utf8 -B stack_meta.py --runs <A> <B> <F> [--train-runs <A_old> <B_old> <F_old>]
      --origins 2022-07 ... 2025-07 --name ic_meta
"""
import argparse

import numpy as np
import pandas as pd

from challengers import read_statuses
from common import CSV_PATH, HERE, KEY, PREDICTION_COLUMNS, STATUS_COLUMNS, file_sha256, guarded, new_run_dir, write_csv, write_json
from residual_meta import series_features
from score import read_run
from stacking import _statuses

PART_END = {"front": 3, "middle": 12, "back": 24}
QUARTER_END = {"front": 3, "q1": 6, "q2": 9, "q3": 12, "back": 24}


def part_of(h, buckets="parts"):
    """예측 거리 → 부분 이름. buckets='quarters'((if))면 중간을 분기 셋으로 나눈다."""
    h = np.asarray(h)
    if buckets == "quarters":
        return np.where(h <= 3, "front", np.where(h <= 6, "q1", np.where(h <= 9, "q2", np.where(h <= 12, "q3", "back"))))
    return np.where(h <= 3, "front", np.where(h <= 12, "middle", "back"))


def part_end(part, buckets="parts"):
    return (QUARTER_END if buckets == "quarters" else PART_END)[part]
FEATURES = ["Country", "part", "log_last12", "log_growth12", "log_dip", "hist_len", "brand_share", "log_brand_growth12"]
LGB_PARAMS = {"num_leaves": 15, "n_estimators": 200, "learning_rate": 0.05, "min_data_in_leaf": 50,
              "random_state": 42, "verbose": -1}


def _part(h):
    return part_of(h, "parts")


SEG_COLS = ["segment", "powertrain"]


def train_run_members(train_runs, members):
    """(ig) 옛 기준월 run 목록을 같은 자리 재료 이름과 짝짓는다. 수는 재료 수의 배수여야 한다."""
    if len(train_runs) % len(members):
        raise ValueError("--train-runs 수는 --runs 수의 배수(같은 순서로 반복)")
    return [(r, members[i % len(members)]) for i, r in enumerate(train_runs)]


def load_segments(path):
    """(ie) 차급 표 → KEY + segment + powertrain (문자열)."""
    t = pd.read_csv(guarded(path), encoding="utf-8-sig", dtype=str)
    return t[KEY + SEG_COLS].drop_duplicates(KEY)


def feature_columns(segments):
    return FEATURES + (SEG_COLS if segments is not None else [])


def categorical_columns(segments):
    return ["Country", "part"] + (SEG_COLS if segments is not None else [])


def bucket_table(predictions, data, members, segments=None, buckets="parts"):
    """(기준월, 차종, 부분) 행: 재료별 절대오차 합(err_<m>), best(가장 작은 재료 번호; 정답 없으면 NaN), 특징."""
    p = predictions.loc[predictions["model"].isin(members)].copy()
    p["origin"] = p["origin"].astype(str)
    p["Month"] = pd.PeriodIndex(p["Month"], freq="M")
    actual = data[KEY + ["Month", "Sales"]].copy()
    actual["Month"] = pd.PeriodIndex(actual["Month"], freq="M")
    p = p.merge(actual, on=KEY + ["Month"], how="left")
    p["part"] = part_of(p["horizon"].astype(int).to_numpy(), buckets)
    p["err"] = (p["prediction"] - p["Sales"]).abs()
    ids = ["origin"] + KEY + ["part"]
    wide = p.pivot_table(index=ids, columns="model", values="err", aggfunc="sum")      # dropna=False는 전체 조합을 만들어 폭발
    counts = p.pivot_table(index=ids, columns="model", values="Sales", aggfunc="count")
    expected = p.groupby(ids)["horizon"].nunique()
    full = (counts.min(axis=1) == expected.reindex(counts.index)).reindex(wide.index).fillna(False)   # 모든 달의 정답이 있을 때만 best
    wide = wide.reindex(columns=members)
    table = wide.rename(columns={m: f"err_{m}" for m in members}).reset_index()
    best = wide.to_numpy(float).argmin(axis=1).astype(float)
    best[~full.to_numpy()] = np.nan
    table["best"] = best
    feats = []
    for origin in table["origin"].unique():
        f = series_features(data, origin)
        f["origin"] = origin
        feats.append(f)
    table = table.merge(pd.concat(feats, ignore_index=True), on=["origin"] + KEY, how="left")
    if segments is not None:
        table = table.merge(segments, on=KEY, how="left")
        table[SEG_COLS] = table[SEG_COLS].fillna("unknown")
    return table


def training_rows(table, origin, buckets="parts"):
    o = pd.Period(origin, "M")
    end = [pd.Period(r.origin, "M") + part_end(r.part, buckets) for r in table.itertuples()]
    keep = table["best"].notna().to_numpy() & np.array([e <= o for e in end], dtype=bool)
    return table.loc[keep]


def _frame(t, cats, feats):
    x = t[feats].copy()
    for c, values in cats.items():
        x[c] = pd.Categorical(x[c], categories=values)
    return x


def fit_weights(train, test, n_members, segments=None):
    import lightgbm as lgb
    feats = feature_columns(segments)
    cats = {c: sorted(set(train[c]) | set(test[c])) for c in categorical_columns(segments)}   # 학습·적용에 같은 범주 목록
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(_frame(train, cats, feats), train["best"].astype(int))
    proba = model.predict_proba(_frame(test, cats, feats))
    out = np.zeros((len(test), n_members))
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = proba[:, j]
    return out


def stack_meta(predictions, data, members, origins, name, min_rows=200, segments=None, buckets="parts"):
    table = bucket_table(predictions, data, members, segments=segments, buckets=buckets)
    p = predictions.loc[predictions["model"].isin(members)].copy()
    p["origin"] = p["origin"].astype(str)
    p["part"] = part_of(p["horizon"].astype(int).to_numpy(), buckets)
    frames, info = [], []
    for origin in origins:
        test = table.loc[table["origin"].eq(origin), list(dict.fromkeys(["origin"] + KEY + ["part"] + feature_columns(segments)))]
        train = training_rows(table, origin, buckets)
        train = train.loc[train["origin"] != origin]
        if len(train) >= min_rows:
            w = fit_weights(train, test, len(members), segments)
        else:
            w = np.full((len(test), len(members)), 1.0 / len(members))
        weights = test[["origin"] + KEY + ["part"]].assign(**{f"w_{m}": w[:, i] for i, m in enumerate(members)})
        rows = p.loc[p["origin"].eq(origin)]
        wide = rows.pivot_table(index=["origin"] + KEY + ["Month", "horizon", "part"], columns="model", values="prediction")
        wide = wide.reindex(columns=members).dropna().reset_index().merge(weights, on=["origin"] + KEY + ["part"], how="left")
        for m in members:
            wide[f"w_{m}"] = wide[f"w_{m}"].fillna(1.0 / len(members))
        wide["prediction"] = sum(wide[m] * wide[f"w_{m}"] for m in members)
        frames.append(wide[["origin"] + KEY + ["Month", "horizon", "prediction"]].assign(model=name))
        info.append({"origin": origin, "train_rows": int(len(train)), "mean_weights": {m: float(w[:, i].mean()) for i, m in enumerate(members)}})
    out = pd.concat(frames, ignore_index=True)
    return out[PREDICTION_COLUMNS], None, info


def delivery_weights(table, data, origin, members, keys, min_rows, segments=None, buckets="parts"):
    """납품용: 기준월 origin까지 끝난 학습 행으로 메타모델을 맞추고 keys(차종)×부분 가중치 표를 돌려준다."""
    train = training_rows(table, origin, buckets)
    feats = series_features(data, origin)
    test = keys.merge(feats, on=KEY, how="left")
    parts = QUARTER_END if buckets == "quarters" else PART_END
    test = pd.concat([test.assign(part=part) for part in parts], ignore_index=True)
    test["Country"] = test["Country"].astype(str)
    if segments is not None:
        test = test.merge(segments, on=KEY, how="left")
        test[SEG_COLS] = test[SEG_COLS].fillna("unknown")
    cols = [c for c in FEATURES if c not in ("Country", "part")]
    test[cols] = test[cols].fillna(0.0)
    if len(train) >= min_rows:
        w = fit_weights(train, test, len(members), segments)
    else:
        w = np.full((len(test), len(members)), 1.0 / len(members))
    return test[KEY + ["part"]].assign(**{f"w_{m}": w[:, i] for i, m in enumerate(members)}), int(len(train))


def apply_delivery(deliveries, weights, members, buckets="parts"):
    """세 납품표(같은 행 구성)를 가중 평균. 셋 다 유한한 예측이 있는 행만 섞고, 아니면 첫 납품표 행 그대로."""
    from pathlib import Path
    tables = [pd.read_csv(next(Path(guarded(d)).glob("*_future_24m.csv")), encoding="utf-8-sig", dtype={"origin": str, "Month": str}) for d in deliveries]
    base = tables[0].copy()
    ids = KEY + ["Month"]
    preds = base[ids].copy()
    for m, t in zip(members, tables):
        preds = preds.merge(t[ids + ["prediction"]].rename(columns={"prediction": m}), on=ids, how="left")
    preds["part"] = part_of(base["horizon"].astype(int).to_numpy(), buckets)
    preds = preds.merge(weights, on=KEY + ["part"], how="left")
    for m in members:
        preds[f"w_{m}"] = preds[f"w_{m}"].fillna(1.0 / len(members))
    allok = np.all(np.isfinite(preds[members].to_numpy(float)), axis=1)
    mixed = sum(preds[m] * preds[f"w_{m}"] for m in members)
    base["prediction"] = np.where(allok, mixed, base["prediction"])
    return base, int(allok.sum())


def main(argv=None):
    import baseline
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", help="적용할 조립 셋(정규 기준월) — 납품 모드에서는 학습 행 출처")
    parser.add_argument("--train-runs", nargs="+", help="같은 순서의 옛 기준월 조립 셋(학습 행 전용)")
    parser.add_argument("--deliveries", nargs="+", help="납품 모드: 같은 순서의 납품 조립 폴더 셋(*_future_24m.csv)")
    parser.add_argument("--origins", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--output-base", default=None)
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--buckets", default="parts", choices=["parts", "quarters"], help="(if) 메타 단위: 부분 3개 또는 묶음 5개")
    parser.add_argument("--segments", help="(ie) 차급 표 csv(segment·powertrain 범주 특징 추가)")
    parser.add_argument("--weights", help="납품 모드: 미리 계산한 weights.csv(학습 없이 적용; --runs는 이름 순서 확인용)")
    args = parser.parse_args(argv)
    input_hash = file_sha256(args.csv)
    if args.weights:
        if not args.deliveries or len(args.origins) != 1:
            raise ValueError("--weights는 --deliveries 셋과 납품 기준월 하나와 함께 씁니다.")
        weights = pd.read_csv(guarded(args.weights), encoding="utf-8-sig")
        members = [c[2:] for c in weights.columns if c.startswith("w_")]
        buckets = "quarters" if "q1" in set(weights["part"]) else "parts"
        out, n_mixed = apply_delivery(args.deliveries, weights, members, buckets)
        run_dir = new_run_dir(args.output_base or HERE / "production", f"delivery_{args.name}")
        out = out.assign(run_id=run_dir.name, algorithm=args.name)
        write_csv(run_dir / f"{args.name}_future_24m.csv", out)
        write_json(run_dir / "run_summary.json", {"model": args.name, "rule": "(ic) precomputed per-series x part meta weights applied to three delivery assemblies",
                                                  "members": members, "weights": str(args.weights), "weights_sha256": file_sha256(args.weights),
                                                  "deliveries": list(args.deliveries), "origin": args.origins[0], "mixed_rows": n_mixed,
                                                  "input_sha256": input_hash, "trained_models": 0})
        print(args.origins[0], "mixed_rows", n_mixed, flush=True)
        print(run_dir)
        return run_dir
    data = baseline.read_sales_csv(guarded(args.csv))
    data["Month"] = pd.PeriodIndex(data["Month"], freq="M")
    segments = load_segments(args.segments) if args.segments else None
    members = [read_run(r)["model"].iloc[0] for r in args.runs]
    frames = [read_run(r) for r in args.runs]
    if args.train_runs:
        for r, m in train_run_members(args.train_runs, members):      # 옛 기준월 run은 같은 자리의 이름으로 바꿔 합침
            f = read_run(r)
            f["model"] = m
            frames.append(f)
    predictions = pd.concat(frames, ignore_index=True)
    if args.deliveries:
        if len(args.origins) != 1:
            raise ValueError("납품 모드의 --origins는 납품 기준월 하나")
        from pathlib import Path
        origin = args.origins[0]
        table = bucket_table(predictions, data, members, segments=segments, buckets=args.buckets)
        first = pd.read_csv(next(Path(guarded(args.deliveries[0])).glob("*_future_24m.csv")), encoding="utf-8-sig", dtype=str)
        keys = first[KEY].drop_duplicates().reset_index(drop=True)
        weights, n_train = delivery_weights(table, data, origin, members, keys, args.min_rows, segments=segments, buckets=args.buckets)
        out, n_mixed = apply_delivery(args.deliveries, weights, members, args.buckets)
        run_dir = new_run_dir(args.output_base or HERE / "production", f"delivery_{args.name}")
        out = out.assign(run_id=run_dir.name, algorithm=args.name)
        write_csv(run_dir / f"{args.name}_future_24m.csv", out)
        write_csv(run_dir / "weights.csv", weights)
        write_json(run_dir / "run_summary.json", {"model": args.name, "rule": "(ic) per-series x part meta weights applied to three delivery assemblies",
                                                  "members": members, "sources": list(args.runs), "train_runs": args.train_runs, "deliveries": list(args.deliveries),
                                                  "origin": origin, "train_rows": n_train, "mixed_rows": n_mixed, "features": feature_columns(segments), "segments": args.segments, "params": LGB_PARAMS,
                                                  "min_rows": args.min_rows, "input_sha256": input_hash, "trained_models": "meta"})
        print(origin, "train_rows", n_train, "mixed_rows", n_mixed, flush=True)
        print(run_dir)
        return run_dir
    out, _, info = stack_meta(predictions, data, members, list(args.origins), args.name, min_rows=args.min_rows, segments=segments, buckets=args.buckets)
    statuses = pd.concat([read_statuses(r) for r in args.runs], ignore_index=True)
    states = _statuses(statuses, members, args.name)
    ok = states.loc[states["prediction_status"].eq("OK"), ["origin"] + KEY]
    out = out.merge(ok, on=["origin"] + KEY)
    out["Month"] = out["Month"].astype(str)
    run_dir = new_run_dir(args.output_base or HERE / "reports", args.name)
    write_csv(run_dir / "predictions.csv", out.sort_values(["origin"] + KEY + ["horizon"])[PREDICTION_COLUMNS])
    write_csv(run_dir / "statuses.csv", states[STATUS_COLUMNS])
    write_json(run_dir / "run_summary.json", {"model": args.name, "rule": "(ic) per-series x part member weighting by LightGBM multiclass meta-model (time-respecting)",
                                              "members": members, "sources": list(args.runs), "train_runs": args.train_runs, "features": feature_columns(segments), "segments": args.segments, "buckets": args.buckets,
                                              "params": LGB_PARAMS, "min_rows": args.min_rows, "info": info, "input_sha256": input_hash, "trained_models": "meta"})
    for i in info:
        print(i["origin"], "train_rows", i["train_rows"], {k: round(v, 3) for k, v in i["mean_weights"].items()}, flush=True)
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
