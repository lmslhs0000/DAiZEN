"""두 예측 run을 조합하는 학습 없는 도전자.

규칙(점수를 보기 전에 고정, 2026-10-01 사용자 승인):
- route: 기준월 시점 규모 그룹(abc_groups, 직전 12개월 판매)이 first_groups면 첫째 run, 아니면 둘째 run의 예측·상태를 쓴다.
  abc_route = A는 combo_horizon_v2, B·C는 tft_a.
- average: 모든 차종·예측 거리에서 두 run을 0.5 + 0.5로 평균한다. ens_tft_combo = tft_a와 combo_horizon_v2.
- average_many(2026-10-01 사전 고정): 여러 run을 같은 비중으로 평균한다. ens3 = tft_a, combo_horizon_v2, rich49를 1/3씩.
- reshape(2026-10-01 사전 고정): 분기(4~12)·1년(13~24) 묶음 합은 모델 그대로 두고 묶음 안 월 배분을 작년 그대로 예측의
  월 비율로 바꾼다. 작년 그대로 묶음 합이 0이거나 없으면 원래 배분을 유지하고, 1~3개월은 그대로 둔다. shape_naive = 챔피언에 적용.

- fill(2026-10-03 사전 고정 (af)): 첫째 run이 OK인 (기준월, 차종)은 첫째, 아니면 둘째 run의 예측·상태를 쓴다.
  nhits_e24f = nhits_e24(24개월 미만 차종 예측 없음)를 nhits_global로 채움.

사용:
  PY experiments/horizon_eval_v1/ensembles.py fill --first <run> --second <run> --name nhits_e24f
  PY experiments/horizon_eval_v1/ensembles.py route --first <run> --second <run> --first-groups A --name abc_route
  PY experiments/horizon_eval_v1/ensembles.py average --first <run> --second <run> --name ens_tft_combo
  PY experiments/horizon_eval_v1/ensembles.py average_many --runs <run> <run> <run> --name ens3
  PY experiments/horizon_eval_v1/ensembles.py reshape --first <모델 run> --name shape_naive
"""

import argparse

import numpy as np
import pandas as pd

from common import (CHECK_HORIZON, CHECK_ORIGIN, CSV_PATH, HERE, HORIZON, KEY, ORIGINS, PREDICTION_COLUMNS,
                    STATUS_COLUMNS, file_sha256, guarded, new_run_dir, require_unchanged, stack_outputs, write_csv,
                    write_json)
import baseline
from challengers import combine, read_statuses
from naive import seasonal_mean_profile, seasonal_naive
from score import read_run
from scoring import abc_groups, horizon_map


def route_by_group(first, second, groups, first_groups, model_name):
    """규모 그룹이 first_groups인 (기준월, 차종)은 첫째, 나머지는 둘째 run의 예측과 상태를 쓴다."""
    ids = ["origin"] + KEY
    chosen = groups.loc[groups["abc"].isin(first_groups), ids].assign(_first=True)

    def pick(table, from_first):
        marked = table.merge(chosen, on=ids, how="left")
        mask = marked["_first"].fillna(False).astype(bool)
        return marked.loc[mask if from_first else ~mask].drop(columns="_first")

    (p1, s1), (p2, s2) = first, second
    predictions = pd.concat([pick(p1, True), pick(p2, False)], ignore_index=True).assign(model=model_name)
    statuses = pd.concat([pick(s1, True), pick(s2, False)], ignore_index=True).assign(model=model_name)
    predictions = predictions[PREDICTION_COLUMNS].sort_values(ids + ["Month"]).reset_index(drop=True)
    statuses = statuses[STATUS_COLUMNS].sort_values(ids).reset_index(drop=True)
    return predictions, statuses


def fill_from(first, second, model_name):
    """첫째 run이 OK인 (기준월, 차종)은 첫째 run, 그 밖은 둘째 run의 예측과 상태를 쓴다."""
    ids = ["origin"] + KEY
    (p1, s1), (p2, s2) = first, second
    ok_first = s1.loc[s1["prediction_status"].eq("OK"), ids].assign(_first=True)

    def pick(table, from_first):
        marked = table.merge(ok_first, on=ids, how="left")
        mask = marked["_first"].fillna(False).astype(bool)
        return marked.loc[mask if from_first else ~mask].drop(columns="_first")

    predictions = pd.concat([pick(p1, True), pick(p2, False)], ignore_index=True).assign(model=model_name)
    statuses = pd.concat([pick(s1, True), pick(s2, False)], ignore_index=True).assign(model=model_name)
    predictions = predictions[PREDICTION_COLUMNS].sort_values(ids + ["Month"]).reset_index(drop=True)
    statuses = statuses[STATUS_COLUMNS].sort_values(ids).reset_index(drop=True)
    return predictions, statuses


def route_by_market(first, second, first_markets, model_name):
    """first_markets 시장의 차종은 first run, 나머지 시장은 second run의 예측과 상태를 쓴다((ax), 2026-10-03)."""
    ids = ["origin"] + KEY
    (p1, s1), (p2, s2) = first, second
    chosen = set(first_markets)
    predictions = pd.concat([p1.loc[p1["Country"].isin(chosen)], p2.loc[~p2["Country"].isin(chosen)]],
                            ignore_index=True).assign(model=model_name)
    statuses = pd.concat([s1.loc[s1["Country"].isin(chosen)], s2.loc[~s2["Country"].isin(chosen)]],
                         ignore_index=True).assign(model=model_name)
    predictions = predictions[PREDICTION_COLUMNS].sort_values(ids + ["Month"]).reset_index(drop=True)
    statuses = statuses[STATUS_COLUMNS].sort_values(ids).reset_index(drop=True)
    return predictions, statuses


def scale_part_to_reference(model, reference, market, part, weight, model_name):
    """(ca) market 시장의 part 부분 묶음(기준월·묶음)마다 모델 예측을 비례 조정한다. 목표 = weight × 기준 합 +
    (1 − weight) × 모델 합(두 run이 모두 예측한 차종 기준). 묶음 안 차종 비율과 달 모양, 그 밖 부분·시장은 그대로."""
    (p, s), (r, _) = model, reference
    ids = ["origin"] + KEY
    mapping = horizon_map(HORIZON)[["horizon", "part", "bucket"]]
    table = p.merge(mapping, on="horizon", how="left")
    chosen = table["Country"].eq(market) & table["part"].eq(part)
    ref = r.loc[r["Country"].eq(market)].merge(mapping, on="horizon")
    ref = ref.loc[ref["part"].eq(part), ids + ["horizon", "bucket", "prediction"]].rename(columns={"prediction": "ref"})
    both = table.loc[chosen].merge(ref, on=ids + ["horizon", "bucket"], how="inner")
    sums = both.groupby(["origin", "bucket"])[["prediction", "ref"]].sum()
    factor = ((weight * sums["ref"] + (1 - weight) * sums["prediction"]) / sums["prediction"]).where(sums["prediction"] > 0)
    keys = pd.MultiIndex.from_frame(table.loc[chosen, ["origin", "bucket"]])
    scale = factor.reindex(keys).fillna(1.0).to_numpy()
    table.loc[chosen, "prediction"] = table.loc[chosen, "prediction"].to_numpy() * scale
    predictions = table.assign(model=model_name)[PREDICTION_COLUMNS].reset_index(drop=True)
    return predictions, s.assign(model=model_name)[STATUS_COLUMNS].reset_index(drop=True)


def route_parts(first, second, parts, model_name):
    """(eb) 고른 부분(앞·중간·뒤)만 second 예측으로 바꾼다(같은 기준월·차종·달이 있을 때). 상태는 first 그대로."""
    (p, s), (q, _) = first, second
    mapping = horizon_map(HORIZON)[["horizon", "part"]]
    ids = ["origin"] + KEY + ["Month", "horizon"]
    table = p.merge(mapping, on="horizon", how="left").merge(
        q[ids + ["prediction"]].rename(columns={"prediction": "alt"}), on=ids, how="left")
    chosen = table["part"].isin(parts) & table["alt"].notna()
    table.loc[chosen, "prediction"] = table.loc[chosen, "alt"]
    predictions = table.assign(model=model_name)[PREDICTION_COLUMNS].reset_index(drop=True)
    return predictions, s.assign(model=model_name)[STATUS_COLUMNS].reset_index(drop=True)


def top_down_child(model, data, parent, child, weight, model_name):
    """(dv) child 시장 예측 = weight × parent 예측 × child 몫 + (1 − weight) × child 예측. 몫 = 기준월 o−11~o child 판매 합 ÷
    parent 합(두 계열 모두 12개월 관측·parent 합 > 0일 때만, 기준월 이하 자료만). parent 짝이 없거나 몫이 없으면 그대로."""
    p, s = model
    p = p.copy()
    shares = {}
    for origin in p["origin"].unique():
        o = pd.Period(origin, freq="M")
        window = data.loc[data["Month"].between(o - 11, o) & data["Country"].isin([parent, child])]
        counts = window.groupby(["Country", "Brand", "Model"])["Sales"].count()
        sums = window.groupby(["Country", "Brand", "Model"])["Sales"].sum()
        for (country, brand, name), total in sums.items():
            if country != child or counts[(country, brand, name)] != 12:
                continue
            key = (parent, brand, name)
            if key in sums.index and counts[key] == 12 and sums[key] > 0:
                shares[(origin, brand, name)] = total / sums[key]
    ids = ["origin", "Brand", "Model", "Month"]
    parents = p.loc[p["Country"].eq(parent), ids + ["prediction"]].rename(columns={"prediction": "parent"})
    rows = p.loc[p["Country"].eq(child)].reset_index().merge(parents, on=ids, how="left")
    share = np.asarray([shares.get((o, b, m), np.nan) for o, b, m in zip(rows["origin"], rows["Brand"], rows["Model"])])
    usable = rows["parent"].notna().to_numpy() & np.isfinite(share)
    blended = weight * rows["parent"].to_numpy() * share + (1 - weight) * rows["prediction"].to_numpy()
    p.loc[rows.loc[usable, "index"].to_numpy(), "prediction"] = blended[usable]
    predictions = p.assign(model=model_name)[PREDICTION_COLUMNS].reset_index(drop=True)
    return predictions, s.assign(model=model_name)[STATUS_COLUMNS].reset_index(drop=True)


def brand_feedback(model, data, model_name, alpha=0.5, min_months=6, clip=(0.5, 2.0), parts=("middle", "back")):
    """(ch) 브랜드 오차 되먹임. 기준월 O마다 O보다 앞선 기준월들의 예측 중 대상월이 O−11…O인 행을 실제값과 맞춰
    (시장, 브랜드)별 r = Σ실제 ÷ Σ예측을 구하고, 서로 다른 대상월이 min_months개 이상이면 f = clip(r^alpha), 아니면 1.
    O의 parts(기본 4~24개월) 예측에 f를 곱한다. 대상월 ≤ O인 실제값만 쓰고, 입력은 조정 전 예측이다."""
    p, s = model
    out = p.copy()
    mapping = horizon_map(HORIZON)[["horizon", "part"]]
    part = out[["horizon"]].merge(mapping, on="horizon", how="left").set_index(out.index)["part"]
    month = pd.PeriodIndex(out["Month"], freq="M")
    origin = pd.PeriodIndex(out["origin"].astype(str), freq="M")
    table = out.assign(Month=month, origin_p=origin)
    actual = data[KEY + ["Month", "Sales"]].assign(Month=lambda d: pd.PeriodIndex(d["Month"], freq="M"))
    joined = table.merge(actual, on=KEY + ["Month"], how="inner").dropna(subset=["prediction", "Sales"])
    groups = ["Country", "Brand"]
    for o in sorted(origin.unique()):
        prior = joined.loc[joined["origin_p"].lt(o) & joined["Month"].ge(o - 11) & joined["Month"].le(o)]
        if prior.empty:
            continue
        stats = prior.groupby(groups).agg(act=("Sales", "sum"), pred=("prediction", "sum"), months=("Month", "nunique"))
        usable = stats["months"].ge(min_months) & stats["pred"].gt(0)
        ratio = (stats["act"] / stats["pred"].where(stats["pred"] > 0)).clip(lower=0)
        factor = (ratio ** alpha).clip(*clip).where(usable, 1.0).rename("f").reset_index()
        rows = (origin == o) & part.isin(parts).to_numpy()
        f = out.loc[rows, groups].merge(factor, on=groups, how="left")["f"].fillna(1.0).to_numpy()
        out.loc[rows, "prediction"] = out.loc[rows, "prediction"].to_numpy(float) * f
    predictions = out.assign(model=model_name)[PREDICTION_COLUMNS].reset_index(drop=True)
    return predictions, s.assign(model=model_name)[STATUS_COLUMNS].reset_index(drop=True)


def scale_parts_to_reference(model, reference, markets, parts, weight, model_name):
    """(cf) 여러 시장 × 부분에 scale_part_to_reference를 차례로 적용한다. 각 조정은 서로 다른 행만 바꾼다."""
    for market in markets:
        for part in parts:
            model = scale_part_to_reference(model, reference, market, part, weight, model_name)
    return model


def average_runs(first, second, model_name):
    """두 run을 모든 예측 거리에서 반반 평균한다. 둘 다 OK인 차종만 예측을 낸다."""
    return combine(first, second, model_name, lambda h: 0.5)


def average_many(sources, model_name):
    """여러 run을 같은 비중으로 평균한다. 모든 run이 OK인 차종만 예측을 내고, 상태는 앞선 run의 실패를 우선한다."""
    return _combine_many(sources, model_name, "mean")


def iqm_many(sources, model_name):
    """(co) 사분위 절사평균: 점마다 아래·위에서 floor(n/4)개씩 빼고 남은 값의 평균. 상태 규칙은 median_many와 같다."""
    return _combine_many(sources, model_name, "iqm")


def median_many(sources, model_name):
    """여러 run의 점별 중앙값(짝수 개면 가운데 두 값의 평균). 나머지 규칙은 average_many와 같다."""
    return _combine_many(sources, model_name, "median")


def _combine_many(sources, model_name, how):
    ids = ["origin"] + KEY
    keys = ids + ["Month", "horizon"]
    merged = None
    states = None
    for i, (p, s) in enumerate(sources):
        part = p[keys + ["prediction"]].rename(columns={"prediction": f"p{i}"})
        merged = part if merged is None else merged.merge(part, on=keys)
        state = s[ids + ["prediction_status"]].rename(columns={"prediction_status": f"s{i}"})
        states = state if states is None else states.merge(state, on=ids, how="outer")
    columns = [f"p{i}" for i in range(len(sources))]
    if how == "iqm":
        ordered = np.sort(merged[columns].to_numpy(float), axis=1)
        cut = len(sources) // 4
        merged["prediction"] = ordered[:, cut:len(sources) - cut].mean(axis=1)
    else:
        merged["prediction"] = merged[columns].agg(how, axis=1)
    status = pd.Series("OK", index=states.index)
    for i in reversed(range(len(sources))):  # 뒤에서부터 덮어써서 앞선 run의 실패가 남게 한다
        column = states[f"s{i}"].fillna("MISSING_SOURCE")
        status = column.where(column.ne("OK"), status)
    statuses = states[ids].assign(model=model_name, prediction_status=status,
                                  reason=status.where(status.ne("OK"), ""))[STATUS_COLUMNS]
    ok = statuses.loc[statuses["prediction_status"].eq("OK"), ids]
    predictions = merged.assign(model=model_name)[PREDICTION_COLUMNS].merge(ok, on=ids)
    return (predictions.sort_values(keys).reset_index(drop=True),
            statuses.sort_values(ids).reset_index(drop=True))


def reshape_by_reference(model, reference, model_name):
    """분기(4~12)·1년(13~24) 묶음 합은 model 그대로 두고, 묶음 안 월 배분을 reference 비율로 바꾼다. 1~3개월은 그대로."""
    (p, s), (r, _) = model, reference
    keys = ["origin"] + KEY + ["Month", "horizon"]
    table = p.merge(horizon_map(HORIZON), on="horizon", how="left").merge(
        r[keys + ["prediction"]].rename(columns={"prediction": "reference"}), on=keys, how="left")
    grouped = table.groupby(["origin"] + KEY + ["part", "bucket"])
    total = grouped["prediction"].transform("sum")
    reference_total = grouped["reference"].transform("sum")
    complete = grouped["reference"].transform("count").eq(grouped["reference"].transform("size"))
    reshape = table["part"].ne("front") & complete & reference_total.gt(0)
    table.loc[reshape, "prediction"] = total[reshape] * table.loc[reshape, "reference"] / reference_total[reshape]
    predictions = table.assign(model=model_name)[PREDICTION_COLUMNS]
    return predictions.reset_index(drop=True), s.assign(model=model_name)[STATUS_COLUMNS].reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["route", "average", "average_many", "median_many", "iqm_many", "reshape", "fill",
                                         "route_market", "scale_part", "brand_feedback", "top_down", "route_parts"])
    parser.add_argument("--first")
    parser.add_argument("--parent", help="(dv) top_down: 위 시장(예: Europe)")
    parser.add_argument("--child", help="(dv) top_down: 아래 시장(예: Germany)")
    parser.add_argument("--second")
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--name", required=True)
    parser.add_argument("--first-groups", nargs="*", default=["A"])
    parser.add_argument("--first-markets", nargs="*", default=[], help="route_market: first run을 쓸 시장")
    parser.add_argument("--market", nargs="+", help="scale_part: 조정할 시장(여러 개 가능)")
    parser.add_argument("--part", nargs="+", choices=["front", "middle", "back"], help="scale_part: 조정할 부분(여러 개 가능)")
    parser.add_argument("--weight", type=float, help="scale_part: 기준 run(second) 쪽 비중")
    parser.add_argument("--profile-years", type=int, help="(eo) reshape: 최근 N년 같은 달 평균 모양(없으면 작년 그대로)")
    parser.add_argument("--reference", help="reshape의 월 배분 기준 run(없으면 작년 그대로)")
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--output-base", default=str(HERE / "reports"))
    args = parser.parse_args(argv)
    input_hash = file_sha256(args.csv)
    runs = args.runs if args.mode in ("average_many", "median_many", "iqm_many") else [r for r in (args.first, args.second) if r]
    sources = [(read_run(run), read_statuses(run)) for run in runs]
    if args.mode == "average_many":
        predictions, statuses = average_many(sources, args.name)
        rule = f"{len(sources)}개 run 같은 비중 평균"
    elif args.mode == "iqm_many":
        predictions, statuses = iqm_many(sources, args.name)
        rule = f"{len(sources)}개 run 점별 사분위 절사평균(아래·위 {len(sources) // 4}개씩 제외)"
    elif args.mode == "median_many":
        predictions, statuses = median_many(sources, args.name)
        rule = f"{len(sources)}개 run 점별 중앙값"
    elif args.mode == "reshape" and args.reference:
        runs.append(args.reference)
        reference = (read_run(args.reference), read_statuses(args.reference))
        predictions, statuses = reshape_by_reference(sources[0], reference, args.name)
        rule = f"분기·1년 묶음 합은 모델, 묶음 안 월 배분은 기준 run({args.reference}) 비율(1~3개월 그대로)"
    elif args.mode == "reshape":
        data = baseline.read_sales_csv(guarded(args.csv))
        years = args.profile_years
        shape = (lambda o, h: seasonal_mean_profile(data, o, h, years=years)) if years else (
            lambda o, h: seasonal_naive(data, o, h))
        origins = sorted(sources[0][0]["origin"].astype(str).unique())
        plan = [(o, CHECK_HORIZON if o == CHECK_ORIGIN else HORIZON) for o in origins] if years else (
            [(o, HORIZON) for o in ORIGINS] + [(CHECK_ORIGIN, CHECK_HORIZON)])
        frames = [stack_outputs(*map(lambda t: [t], shape(o, h)), "seasonal_naive", pd.Period(o, "M")) for o, h in plan]
        reference = (pd.concat([f[0] for f in frames], ignore_index=True),
                     pd.concat([f[1] for f in frames], ignore_index=True))
        predictions, statuses = reshape_by_reference(sources[0], reference, args.name)
        rule = (f"분기·1년 묶음 합은 모델, 묶음 안 월 배분은 최근 {years}년 같은 달 평균 비율(1~3개월 그대로, (eo))" if years
                else "분기·1년 묶음 합은 모델, 묶음 안 월 배분은 작년 그대로 비율(1~3개월 그대로)")
    elif args.mode == "scale_part":
        if not (args.market and args.part and args.weight is not None):
            raise ValueError("scale_part에는 --market, --part, --weight가 필요합니다.")
        predictions, statuses = scale_parts_to_reference(*sources, args.market, args.part, args.weight, args.name)
        rule = (f"{args.market}의 {args.part} 묶음 합 = {args.weight} × second + {1 - args.weight} × first "
                "(공통 차종 기준, 묶음 안 비율 유지)")
    elif args.mode == "route_parts":
        if not args.part:
            raise ValueError("route_parts에는 --part가 필요합니다.")
        predictions, statuses = route_parts(*sources, args.part, args.name)
        rule = f"(eb) {args.part} 부분은 second 예측, 나머지·상태는 first"
    elif args.mode == "top_down":
        if not (args.parent and args.child and args.weight is not None):
            raise ValueError("top_down에는 --parent, --child, --weight가 필요합니다.")
        data = baseline.read_sales_csv(guarded(args.csv))
        predictions, statuses = top_down_child(sources[0], data, args.parent, args.child, args.weight, args.name)
        rule = (f"(dv) {args.child} = {args.weight} × {args.parent} 예측 × {args.child} 몫(기준월 직전 12개월 합 비) + "
                f"{1 - args.weight} × {args.child} 예측")
    elif args.mode == "brand_feedback":
        if args.weight is None:
            raise ValueError("brand_feedback에는 --weight(보정 지수)가 필요합니다.")
        data = baseline.read_sales_csv(guarded(args.csv))
        predictions, statuses = brand_feedback(sources[0], data, args.name, alpha=args.weight)
        rule = (f"(ch) 앞선 기준월 예측의 최근 12개월 (시장, 브랜드) r = 실제/예측, 6개월 이상이면 "
                f"f = clip(r^{args.weight}, 0.5, 2.0)을 4~24개월에 곱함")
    elif args.mode == "route_market":
        if not args.first_markets:
            raise ValueError("route_market에는 --first-markets가 필요합니다.")
        predictions, statuses = route_by_market(*sources, args.first_markets, args.name)
        rule = f"시장 {args.first_markets}는 first, 나머지 시장은 second"
    elif args.mode == "fill":
        predictions, statuses = fill_from(*sources, args.name)
        rule = "first가 OK인 차종은 first, 아니면 second"
    elif args.mode == "route":
        data = baseline.read_sales_csv(guarded(args.csv))
        groups = pd.concat([abc_groups(data, o) for o in ORIGINS + [CHECK_ORIGIN]], ignore_index=True)
        predictions, statuses = route_by_group(*sources, groups, args.first_groups, args.name)
        rule = f"규모 그룹 {args.first_groups}는 first, 나머지는 second"
    else:
        predictions, statuses = average_runs(*sources, args.name)
        rule = "first 0.5 + second 0.5 (모든 예측 거리)"
    require_unchanged(args.csv, input_hash)
    run_dir = new_run_dir(args.output_base, args.name)
    write_csv(run_dir / "predictions.csv", predictions)
    write_csv(run_dir / "statuses.csv", statuses)
    write_json(run_dir / "run_summary.json", {
        "model": args.name, "mode": args.mode, "rule": rule, "sources": [str(r) for r in runs],
        "input_sha256": input_hash, "origins": ORIGINS, "check_origin": CHECK_ORIGIN, "trained_models": 0,
        "status_counts": statuses.groupby(["origin", "prediction_status"]).size().rename("n").reset_index().to_dict("records")})
    print(run_dir)


if __name__ == "__main__":
    main()
