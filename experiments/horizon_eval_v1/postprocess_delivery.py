"""납품 파일 후처리(2026-10-03 (ca)·(cb)): 한 시장의 13~24개월(1년 묶음) 예측 합계를 기준 run(production cohort_sn_a12)
쪽으로 비례 조정한 새 납품 폴더를 만든다. 원래 납품 파일은 바꾸지 않는다.

목표 합 = weight × 기준 합 + (1 − weight) × 원래 합(두 쪽 모두 OK인 차종 기준). 그 시장 OK 차종의 13~24개월 예측에
같은 비율을 곱한다. 1~12개월·다른 시장·상태는 그대로.

사용: PY experiments/horizon_eval_v1/postprocess_delivery.py --delivery <production/delivery_...> --cohort-run
      <production/cohort_sn_a12_...> --name <새 이름> [--market China] [--weight 1.0]

(cf) 여러 시장 × 부분: --reference naive(입력 CSV로 만든 작년 그대로) 또는 run 폴더, --market <시장들>
     --part middle back --weight 0.5. 4~12개월은 분기 묶음, 13~24개월은 1년 묶음마다 같은 식으로 조정한다.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import CSV_PATH, HERE, KEY, PROJECT_ROOT, file_sha256, guarded, new_run_dir, stack_outputs, write_csv, write_json
import baseline  # common이 프로젝트 루트를 경로에 넣은 뒤에 불러온다
from naive import seasonal_naive
from scoring import horizon_map

BACK = (13, 24)


def scale_future_table(table, reference, market, weight):
    """table: 납품 미래 표, reference: 기준 run 예측(KEY, horizon, prediction)."""
    out = table.copy()
    chosen = out["Country"].eq(market) & out["horizon"].between(*BACK) & out["status"].eq("OK")
    ref = reference.loc[reference["Country"].eq(market) & reference["horizon"].between(*BACK),
                        KEY + ["horizon", "prediction"]].rename(columns={"prediction": "ref"})
    both = out.loc[chosen, KEY + ["horizon", "prediction"]].merge(ref, on=KEY + ["horizon"], how="inner")
    base, target = float(both["prediction"].sum()), float(both["ref"].sum())
    if base <= 0:
        return out
    factor = (weight * target + (1 - weight) * base) / base
    out.loc[chosen, "prediction"] = out.loc[chosen, "prediction"].to_numpy(float) * factor
    return out


def scale_future_buckets(table, reference, markets, parts, weight):
    """(cf) markets × parts의 묶음마다 OK 예측 합을 weight × 기준 합 + (1 − weight) × 원래 합으로 비례 조정한다
    (합은 두 쪽 모두 예측한 차종 기준, 배율은 그 시장·묶음의 OK 차종 전체에 적용)."""
    out = table.copy()
    mapping = horizon_map(24)[["horizon", "part", "bucket"]]
    where = out[["horizon"]].merge(mapping, on="horizon", how="left").set_index(out.index)
    ref = reference.merge(mapping, on="horizon")
    for market in markets:
        for part in parts:
            for bucket in sorted(mapping.loc[mapping["part"].eq(part), "bucket"].unique()):
                chosen = (out["Country"].eq(market) & where["part"].eq(part) & where["bucket"].eq(bucket)
                          & out["status"].eq("OK"))
                r = ref.loc[ref["Country"].eq(market) & ref["part"].eq(part) & ref["bucket"].eq(bucket),
                            KEY + ["horizon", "prediction"]].rename(columns={"prediction": "ref"})
                both = out.loc[chosen, KEY + ["horizon", "prediction"]].merge(r, on=KEY + ["horizon"], how="inner")
                base, target = float(both["prediction"].sum()), float(both["ref"].sum())
                if base <= 0:
                    continue
                factor = (weight * target + (1 - weight) * base) / base
                out.loc[chosen, "prediction"] = out.loc[chosen, "prediction"].to_numpy(float) * factor
    return out


def naive_reference(data, origin, horizon=24):
    """기준월까지의 실제값으로 만든 작년 그대로 예측표(KEY, horizon, prediction)."""
    p, s = seasonal_naive(data, origin, horizon)
    return stack_outputs([p], [s], "seasonal_naive", pd.Period(origin, "M"))[0]


def read_delivery(delivery):
    files = sorted(Path(delivery).glob("*_future_24m.csv"))
    if len(files) != 1:
        raise ValueError(f"납품 파일이 하나가 아닙니다: {delivery}")
    return files[0], pd.read_csv(files[0], encoding="utf-8-sig", dtype={"origin": str, "train_end": str, "Month": str})


def run_buckets(delivery, reference, label, name, output_base, markets, parts, weight):
    path, table = read_delivery(delivery)
    origins = set(table["origin"]) | set(reference["origin"].astype(str))
    if len(origins) != 1:
        raise ValueError(f"기준월이 서로 다릅니다: {origins}")
    out = scale_future_buckets(table, reference, markets, parts, weight)
    out_dir = new_run_dir(output_base, f"delivery_{name}")
    out = out.assign(run_id=out_dir.name, algorithm=name)
    write_csv(out_dir / f"{name}_future_24m.csv", out)
    write_json(out_dir / "run_summary.json", {
        "model": name, "base_delivery": str(delivery), "base_sha256": file_sha256(path), "reference": label,
        "markets": list(markets), "parts": list(parts), "weight": weight,
        "rule": f"{list(markets)} {list(parts)} buckets: scaled to {weight} x {label} + {1 - weight} x base on common OK series"})
    print(out_dir)
    return out_dir


def route_parts(delivery, other, parts, name, output_base, markets=None):
    """(dz) 고른 부분(앞 1~3·중간 4~12·뒤 13~24개월)만 other 납품 파일의 예측으로 바꾼다. 행·상태는 delivery 그대로,
    other에 같은 (차종, 월) 예측이 없으면 바꾸지 않는다."""
    path, table = read_delivery(delivery)
    other_path, alt = read_delivery(other)
    if set(table["origin"]) != set(alt["origin"]):
        raise ValueError("기준월이 서로 다릅니다.")
    mapping = horizon_map(24)[["horizon", "part"]]
    keys = KEY + ["Month", "horizon"]
    merged = table.merge(mapping, on="horizon", how="left").merge(
        alt[keys + ["prediction"]].rename(columns={"prediction": "alt"}), on=keys, how="left")
    chosen = merged["part"].isin(parts) & merged["alt"].notna() & merged["status"].eq("OK")
    if markets:
        chosen &= merged["Country"].isin(markets)  # (es) 목록 시장만
    out = table.copy()
    out.loc[chosen.to_numpy(), "prediction"] = merged.loc[chosen, "alt"].to_numpy()
    out_dir = new_run_dir(output_base, f"delivery_{name}")
    out = out.assign(run_id=out_dir.name, algorithm=name)
    write_csv(out_dir / f"{name}_future_24m.csv", out)
    write_json(out_dir / "run_summary.json", {
        "model": name, "base_delivery": str(delivery), "base_sha256": file_sha256(path), "route_from": str(other),
        "route_sha256": file_sha256(other_path), "route_parts": list(parts),
        "route_markets": list(markets) if markets else None,
        "rule": f"{list(parts)} predictions from route_from delivery; other parts and statuses from base"})
    print(out_dir)
    return out_dir


def run(delivery, cohort_run, name, output_base=HERE / "production", market="China", weight=1.0):
    delivery, cohort_run = Path(delivery), Path(cohort_run)
    files = sorted(delivery.glob("*_future_24m.csv"))
    if len(files) != 1:
        raise ValueError(f"납품 파일이 하나가 아닙니다: {delivery}")
    table = pd.read_csv(files[0], encoding="utf-8-sig", dtype={"origin": str, "train_end": str, "Month": str})
    reference = pd.read_csv(cohort_run / "predictions.csv", encoding="utf-8-sig", dtype={"origin": str, "Month": str})
    origins = set(table["origin"]) | set(reference["origin"])
    if len(origins) != 1:
        raise ValueError(f"기준월이 서로 다릅니다: {origins}")
    out = scale_future_table(table, reference, market, weight)
    out_dir = new_run_dir(output_base, f"delivery_{name}")
    out = out.assign(run_id=out_dir.name, algorithm=name)
    write_csv(out_dir / f"{name}_future_24m.csv", out)
    write_json(out_dir / "run_summary.json", {
        "model": name, "base_delivery": str(delivery), "base_sha256": file_sha256(files[0]),
        "cohort_run": str(cohort_run), "market": market, "weight": weight,
        "rule": f"{market} horizons 13-24: scaled to {weight} x cohort + {1 - weight} x base on common OK series"})
    print(out_dir)
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", required=True)
    parser.add_argument("--cohort-run", help="(ca) 한 시장 13~24개월을 이 run 쪽으로")
    parser.add_argument("--route-from", help="(dz) --part 부분만 이 납품 폴더의 예측으로 바꿈")
    parser.add_argument("--reference", help="(cf) naive 또는 기준 run 폴더")
    parser.add_argument("--csv", default=str(CSV_PATH), help="--reference naive일 때 작년 그대로를 만들 입력 CSV")
    parser.add_argument("--name", required=True)
    parser.add_argument("--market", nargs="+", default=["China"])
    parser.add_argument("--part", nargs="+", choices=["front", "middle", "back"])
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--output-base", default=str(HERE / "production"))
    parser.add_argument("--route-markets", nargs="+", help="(es) --route-from을 이 시장들에만(없으면 모든 시장)")
    args = parser.parse_args(argv)
    absolute = lambda p: guarded(PROJECT_ROOT / p) if not Path(p).is_absolute() else guarded(p)  # noqa: E731
    if args.route_from:
        if not args.part:
            raise ValueError("--route-from에는 --part가 필요합니다.")
        route_parts(absolute(args.delivery), absolute(args.route_from), args.part, args.name, args.output_base,
                    args.route_markets)
        return
    if args.cohort_run:
        if len(args.market) != 1:
            raise ValueError("--cohort-run에는 시장 하나만 줍니다.")
        run(absolute(args.delivery), absolute(args.cohort_run), args.name, args.output_base, args.market[0], args.weight)
        return
    if not (args.reference and args.part):
        raise ValueError("--cohort-run 또는 --reference·--part가 필요합니다.")
    delivery = absolute(args.delivery)
    _, table = read_delivery(delivery)
    if args.reference == "naive":
        data = baseline.read_sales_csv(guarded(args.csv))
        reference, label = naive_reference(data, table["origin"].iloc[0]), "naive"
    else:
        reference = pd.read_csv(absolute(args.reference) / "predictions.csv", encoding="utf-8-sig",
                                dtype={"origin": str, "Month": str})
        label = str(args.reference)
    run_buckets(delivery, reference, label, args.name, args.output_base, args.market, args.part, args.weight)


if __name__ == "__main__":
    main()
