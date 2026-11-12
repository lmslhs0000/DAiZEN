"""예측 범위(2026-10-06): 백테스트에서 실제로 틀린 정도로 80% 범위를 붙인다. 정확도를 바꾸지 않는 보조 정보.

- 잔차 r = log((실제 + 1) / (예측 + 1)), 묶음 = (시장, 부분 앞·중간·뒤, 규모 그룹 A·B·C). 묶음 행이 min_rows보다 적으면
  (시장, 부분), 그것도 적으면 (부분) 전체로 대체.
- 범위: 하한 = max(0, (예측 + 1)·e^{q10} − 1), 상한 = (예측 + 1)·e^{q90} − 1.
- 검증(시간 순서): 백테스트 잔차 중 달 ≤ 2025-07인 행만으로 분위수를 정하고, 확인용 기준월 2025-07(1~12개월)에서 실제가
  범위 안에 든 비율을 잰다.
- 보조 예측 차종(OK_SHORT_HISTORY·OK_CHRONOS_ONLY)은 짧은 이력 Chronos 백테스트 잔차로 (중국 여부, 부분) 묶음.

사용: prediction_intervals.py --run <챔피언 백테스트 run> --short-run <short_chronos 백테스트 run> --delivery <납품 폴더>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import CHECK_ORIGIN, CSV_PATH, HERE, KEY, ORIGINS, file_sha256, guarded, new_run_dir, write_csv, write_json
import baseline
from scoring import abc_groups

LEVELS = (0.1, 0.9)
MIN_ROWS = 200


def _part(h):
    return np.select([h <= 3, h <= 12], ["front", "middle"], "back")


def residual_rows(predictions, data, max_month=None):
    actual = data[KEY + ["Month", "Sales"]].dropna(subset=["Sales"]).copy()
    actual["Month"] = actual["Month"].astype(str)
    p = predictions.assign(Month=predictions["Month"].astype(str)).merge(actual, on=KEY + ["Month"], how="inner")
    p = p.dropna(subset=["prediction"])
    p["Month"] = p["Month"].map(lambda m: pd.Period(m, freq="M"))
    if max_month is not None:
        p = p.loc[p["Month"].le(pd.Period(str(max_month), freq="M"))]
    p["part"] = _part(p["horizon"].to_numpy())
    p["r"] = np.log((p["Sales"] + 1.0) / (p["prediction"] + 1.0))
    return p.reset_index(drop=True)


def fit_quantiles(res, levels=LEVELS, min_rows=MIN_ROWS):
    out = {}
    for country in sorted(res["Country"].unique()):
        for part in sorted(res["part"].unique()):
            for abc in ("A", "B", "C"):
                cell = res.loc[res["Country"].eq(country) & res["part"].eq(part) & res["abc"].eq(abc), "r"]
                if len(cell) < min_rows:
                    cell = res.loc[res["Country"].eq(country) & res["part"].eq(part), "r"]
                if len(cell) < min_rows:
                    cell = res.loc[res["part"].eq(part), "r"]
                out[(country, part, abc)] = tuple(float(np.quantile(cell, lv)) for lv in levels)
    return out


def apply_intervals(table, quantiles, key_columns=("Country", "part", "abc")):
    out = table.copy()
    out["part"] = _part(out["horizon"].to_numpy())
    lo, hi = [], []
    for row in zip(*(out[c] for c in key_columns)):
        q = quantiles.get(tuple(row))
        lo.append(np.nan if q is None else q[0])
        hi.append(np.nan if q is None else q[1])
    base = out["prediction"].to_numpy(float) + 1.0
    out["lower"] = np.maximum(0.0, base * np.exp(np.array(lo)) - 1.0)
    out["upper"] = base * np.exp(np.array(hi)) - 1.0
    return out


def coverage(frame):
    inside = (frame["actual"] >= frame["lower"]) & (frame["actual"] <= frame["upper"])
    return float(inside.mean()), float((frame["actual"] * inside).sum() / frame["actual"].sum())


def _with_abc(table, data, origin_column="origin"):
    frames = []
    for origin, rows in table.groupby(table[origin_column].astype(str), sort=False):
        frames.append(rows.merge(abc_groups(data, origin)[KEY + ["abc"]], on=KEY, how="left"))
    out = pd.concat(frames, ignore_index=True)
    out["abc"] = out["abc"].fillna("C")
    return out


def main(argv=None):
    from score import read_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--short-run", required=True)
    parser.add_argument("--delivery", required=True)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--output-base", default=str(HERE / "production"))
    args = parser.parse_args(argv)
    data = baseline.read_sales_csv(guarded(args.csv))
    champ = read_run(args.run)
    champ["origin"] = champ["origin"].astype(str)
    back = _with_abc(residual_rows(champ.loc[champ["origin"].isin(ORIGINS)], data), data)
    # 시간 순서 검증: 달 ≤ 확인용 기준월인 잔차만으로 정하고 2025-07 기준 1~12개월에서 적중률
    early = back.loc[back["Month"].le(pd.Period(CHECK_ORIGIN, freq="M"))]
    q_check = fit_quantiles(early)
    check = _with_abc(residual_rows(champ.loc[champ["origin"].eq(CHECK_ORIGIN)], data), data)
    check = apply_intervals(check, q_check).rename(columns={"Sales": "actual"})
    report = []
    for (part,), g in check.groupby(["part"]):
        report.append({"scope": "part", "value": part, **dict(zip(("coverage", "coverage_volume"), coverage(g)))})
    for (country,), g in check.groupby(["Country"]):
        report.append({"scope": "market", "value": country, **dict(zip(("coverage", "coverage_volume"), coverage(g)))})
    report.append({"scope": "all", "value": "all", **dict(zip(("coverage", "coverage_volume"), coverage(check)))})
    # 납품 범위: 백테스트 잔차 전부
    q_all = fit_quantiles(back)
    short = read_run(args.short_run)
    short["origin"] = short["origin"].astype(str)
    sres = residual_rows(short.loc[short["origin"].isin(ORIGINS)], data)
    sres["china"] = np.where(sres["Country"].eq("China"), "China", "rest")
    q_short = {(c, p): tuple(float(np.quantile(g["r"], lv)) for lv in LEVELS)
               for (c, p), g in sres.groupby(["china", "part"])}
    path = next(Path(guarded(args.delivery)).glob("*_future_24m.csv"))
    delivery = pd.read_csv(path, encoding="utf-8-sig", dtype={"origin": str, "train_end": str, "Month": str})
    main_rows = _with_abc(delivery, data)
    with_main = apply_intervals(main_rows, q_all)
    sup = with_main["status"].isin(["OK_SHORT_HISTORY", "OK_CHRONOS_ONLY", "OK_SHORT_PUBLIC"])  # (gv) 공개 실적 덧붙인 신차 보조 예측
    tmp = with_main.loc[sup].assign(china=np.where(with_main.loc[sup, "Country"].eq("China"), "China", "rest"))
    tmp = apply_intervals(tmp.drop(columns=["lower", "upper"]), q_short, key_columns=("china", "part"))
    with_main.loc[sup, ["lower", "upper"]] = tmp[["lower", "upper"]].to_numpy()
    with_main["interval_basis"] = np.where(sup, "short_history_backtest", np.where(with_main["status"].eq("OK"),
                                                                                    "champion_backtest", ""))
    with_main.loc[with_main["prediction"].isna(), ["lower", "upper"]] = np.nan
    out = with_main[list(delivery.columns) + ["lower", "upper", "interval_basis"]].rename(
        columns={"lower": "lower_80", "upper": "upper_80"})
    run_dir = new_run_dir(args.output_base, "intervals")
    write_csv(run_dir / "future_24m_with_intervals.csv", out)
    write_csv(run_dir / "check_coverage.csv", pd.DataFrame(report))
    write_json(run_dir / "run_summary.json", {
        "rule": "80% interval from empirical quantiles (0.1, 0.9) of log((actual+1)/(pred+1)) by (Country, part, abc), "
                "fallback (Country, part) then (part) below 200 rows; supplement rows use short_chronos backtest by (China?, part)",
        "champion_run": str(args.run), "short_run": str(args.short_run), "delivery": str(args.delivery),
        "delivery_sha256": file_sha256(path), "check": "quantiles from residual months <= 2025-07, coverage on origin 2025-07 h1-12"})
    print(pd.DataFrame(report).round(3).to_string(index=False))
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
