"""(fu) 짧은 이력 신차 보조 예측(2026-10-06 사전 고정). 주 점수·챔피언과 별개.

대상 = 기준월 O에 마지막 연속 구간(빈 달 없음, O에서 끝남) 길이 L이 3~11개월인 차종.
- flat3: 마지막 3개월 평균을 그대로.
- launch_curve: O 이전에 완전히 관측된 과거 신차(첫 판매월 F > 2016-01, F..F+k−1+24 빈 달 없음, F+k−1+24 ≤ O)로
  R_k(h) = Σ판매[F+k−1+h] ÷ Σ평균(판매[F+k−3..F+k−1]) (중국 / 중국 밖 따로), 예측 = 마지막 3개월 평균 × R_k(h),
  k = min(L, 11). 그 묶음·k에 과거 신차가 없거나 분모가 0이면 비율 1(flat3과 같음).
- chronos_short는 short_chronos.py(chronos_ft_v1 환경)가 만든다.

사용: short_history.py --origins ... --name short_flat3/short_curve (백테스트 run 형식으로 저장)
"""

import argparse

import numpy as np
import pandas as pd

from common import (CHECK_ORIGIN, CSV_PATH, HERE, KEY, ORIGINS, PREDICTION_COLUMNS, STATUS_COLUMNS, file_sha256, guarded,
                    new_run_dir, require_unchanged, write_csv, write_json)
import baseline

MIN_LEN, MAX_LEN, MAX_K = 3, 11, 11
DATA_START = pd.Period("2016-01", "M")


def _runs(data, origin):
    """기준월까지 값이 있는 행을 차종별 (달 ordinal 배열, 값 배열)로."""
    rows = data.loc[data["Month"].le(origin)].dropna(subset=["Sales"]).sort_values(KEY + ["Month"])
    for key, g in rows.groupby(KEY, sort=True):
        yield key, np.array([m.ordinal for m in g["Month"]]), g["Sales"].to_numpy(float)


def short_histories(data, origin, min_len=MIN_LEN, max_len=MAX_LEN):
    """{(Country, Brand, Model): 마지막 연속 구간 값} — 구간이 O에서 끝나고 길이가 min_len~max_len인 차종만."""
    origin = pd.Period(origin, freq="M")
    found = {}
    for key, months, values in _runs(data, origin):
        if months[-1] != origin.ordinal:
            continue
        breaks = np.nonzero(np.diff(months) != 1)[0]
        start = breaks[-1] + 1 if len(breaks) else 0
        run = values[start:]
        if min_len <= len(run) <= max_len:
            found[tuple(key)] = run
    return found


def flat3(values, horizon):
    return np.full(horizon, float(np.mean(values[-3:])))


def launch_curve_table(data, origin, horizon=24):
    """(china, k, horizon) → ratio, launches. O 이전에 완전히 관측된 과거 신차로만."""
    origin = pd.Period(origin, freq="M")
    sums = {}
    for key, months, values in _runs(data, origin):
        positive = np.nonzero(values > 0)[0]
        if not len(positive):
            continue
        first = positive[0]
        if months[first] <= DATA_START.ordinal:
            continue
        months, values = months[first:], values[first:]
        breaks = np.nonzero(np.diff(months) != 1)[0]
        run = len(values) if not len(breaks) else breaks[0] + 1  # 첫 달부터 빈 달 없이 이어진 길이
        china = key[0] == "China"
        for k in range(MIN_LEN, MAX_K + 1):
            if run < k + horizon:
                break
            anchor = values[k - 3:k].mean()
            target = values[k:k + horizon]
            total = sums.setdefault((china, k), [0.0, np.zeros(horizon), 0])
            total[0] += anchor
            total[1] += target
            total[2] += 1
    rows = []
    for (china, k), (anchor, target, n) in sorted(sums.items()):
        for h in range(1, horizon + 1):
            rows.append({"china": china, "k": k, "horizon": h, "ratio": target[h - 1] / anchor if anchor > 0 else np.nan,
                         "launches": n})
    return pd.DataFrame(rows, columns=["china", "k", "horizon", "ratio", "launches"])


def forecast_short(data, origin, horizon):
    """대상 차종마다 flat3·launch_curve 예측(KEY, Month, horizon, L, k, flat3, launch_curve)."""
    origin = pd.Period(origin, freq="M")
    table = launch_curve_table(data, origin)
    ratio = {(c, k, h): r for c, k, h, r in zip(table["china"], table["k"], table["horizon"], table["ratio"])}
    rows = []
    for key, values in short_histories(data, origin).items():
        k = min(len(values), MAX_K)
        base = flat3(values, horizon)
        for h in range(1, horizon + 1):
            r = ratio.get((key[0] == "China", k, h), np.nan)
            r = 1.0 if not np.isfinite(r) else r
            rows.append({**dict(zip(KEY, key)), "Month": origin + h, "horizon": h, "L": len(values), "k": k,
                         "flat3": base[h - 1], "launch_curve": max(0.0, base[h - 1] * r)})
    return pd.DataFrame(rows, columns=KEY + ["Month", "horizon", "L", "k", "flat3", "launch_curve"])


def build_supplement(delivery, short, name, status="OK_SHORT_HISTORY"):
    """(fu) 보조 납품 표: 본 납품에서 OK가 아닌 차종 중 짧은 이력 예측이 있는 것만, 본 납품과 같은 열·행 형식."""
    keys = KEY + ["Month", "horizon"]
    missing = delivery.loc[delivery["status"].ne("OK")].copy()
    missing["Month"] = missing["Month"].astype(str)
    pred = short[keys + ["prediction"]].copy()
    pred["Month"] = pred["Month"].astype(str)
    out = missing.drop(columns="prediction").merge(pred, on=keys, how="inner")
    out = out.assign(algorithm=name, status=status)
    return out[delivery.columns].sort_values(KEY + ["horizon"]).reset_index(drop=True)


def fill_from_supplement(delivery, supplement, name):
    """(fu) 본 납품 행 순서 그대로, 예측이 빈(OK 아님) 행만 보조 예측 값·상태로 채운 합친 판."""
    keys = KEY + ["Month", "horizon"]
    sup = supplement[keys + ["prediction", "status"]].rename(columns={"prediction": "_p", "status": "_s"})
    sup = sup.assign(Month=sup["Month"].astype(str))
    out = delivery.assign(Month=delivery["Month"].astype(str)).merge(sup, on=keys, how="left")
    fill = out["status"].ne("OK") & out["_p"].notna()
    out.loc[fill, "prediction"] = out.loc[fill, "_p"]
    out.loc[fill, "status"] = out.loc[fill, "_s"]
    return out.drop(columns=["_p", "_s"]).assign(algorithm=name)[delivery.columns]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--output-base", default=str(HERE / "reports"))
    parser.add_argument("--origins", nargs="*", default=ORIGINS + [CHECK_ORIGIN])
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args(argv)
    input_hash = file_sha256(args.csv)
    data = baseline.read_sales_csv(guarded(args.csv))
    frames = [forecast_short(data, o, args.horizon).assign(origin=str(o)) for o in args.origins]
    table = pd.concat(frames, ignore_index=True)
    require_unchanged(args.csv, input_hash)
    out = []
    for column, name in (("flat3", "short_flat3"), ("launch_curve", "short_curve")):
        run_dir = new_run_dir(args.output_base, name)
        predictions = table.assign(model=name, prediction=table[column])[PREDICTION_COLUMNS]
        statuses = table.drop_duplicates(["origin"] + KEY).assign(model=name, prediction_status="OK", reason="")
        write_csv(run_dir / "predictions.csv", predictions)
        write_csv(run_dir / "statuses.csv", statuses.reindex(columns=STATUS_COLUMNS))
        write_json(run_dir / "run_summary.json", {
            "model": name, "rule": "(fu) short-history (3-11 consecutive months ending at origin) " + column,
            "input_sha256": input_hash, "origins": [str(o) for o in args.origins], "horizon": args.horizon,
            "series_per_origin": table.drop_duplicates(["origin"] + KEY).groupby("origin").size().to_dict(),
            "trained_models": 0})
        out.append(run_dir)
        print(run_dir)
    return out


if __name__ == "__main__":
    main()
