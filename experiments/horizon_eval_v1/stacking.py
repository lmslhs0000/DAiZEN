"""부분별 결합(기준월 하나 빼기) — ens4_stack_loo (2026-10-01 사전 고정).

재료 여러 개의 예측을 부분(앞 1~3·중간 4~12·뒤 13~24)마다 다르게 결합한다. 후보는 점별 중앙값과 0.1 간격의 비음수·합 1
가중 평균(재료 4개면 286개)이고, 중앙값·격자 순서로 동률을 깬다. 기준월 o에 쓰는 결합은 o를 뺀 나머지 기준월들에서
그 부분의 (묶음 WAPE + 달마다 WAPE) ÷ 2가 가장 낮은 후보다. 확인용 기준월과 납품은 백테스트 기준월 전체로 고른다.
네 재료가 모두 예측했고 실제값이 있는 차종·월만으로 고른다.

사용: PY experiments/horizon_eval_v1/stacking.py --runs <재료 run ...> --name ens4_stack_loo
"""

import argparse
import itertools

import numpy as np
import pandas as pd

from common import (CHECK_ORIGIN, CSV_PATH, HERE, HORIZON, KEY, ORIGINS, file_sha256, guarded, new_run_dir,
                    require_unchanged, write_csv, write_json)
import baseline
from challengers import read_statuses
from scoring import horizon_map
from score import read_run

CANDIDATES_MEDIAN = "median"
PARTS = ["front", "middle", "back"]


def weight_grid(members, step=0.1):
    """비음수·합 1 가중치 격자(itertools.product 순서)."""
    n = int(round(1 / step))
    return np.asarray([c for c in itertools.product(range(n + 1), repeat=members) if sum(c) == n], float) / n


def combine_part(values, choice):
    """values: (행, 재료) 예측. choice: 'median' 또는 가중치 벡터."""
    if isinstance(choice, str):
        return np.median(values, axis=1)
    return values @ np.asarray(choice, float)


def _wide(predictions, actual, members, origins):
    """재료별 예측을 한 행(기준월·차종·월)에 모으고 실제값과 부분·묶음을 붙인다. 모든 재료와 실제값이 있는 행만."""
    p = predictions.loc[predictions["model"].isin(members) & predictions["origin"].isin(origins)]
    wide = p.pivot_table(index=["origin"] + KEY + ["Month", "horizon"], columns="model", values="prediction")
    wide = wide.reindex(columns=members).dropna().reset_index()
    wide = wide.merge(actual[KEY + ["Month", "Sales"]], on=KEY + ["Month"], how="inner")
    return wide.merge(horizon_map(HORIZON), on="horizon")


def _scores(part_rows, members, candidates):
    """후보마다 그 부분의 (묶음 + 달마다) WAPE ÷ 2."""
    values = part_rows[members].to_numpy(float)
    actual = part_rows["Sales"].to_numpy(float)
    total = actual.sum()
    groups = part_rows.groupby(["origin"] + KEY + ["bucket"], sort=False).ngroup().to_numpy()
    bucket_actual = np.bincount(groups, weights=actual)
    out = []
    for choice in candidates:
        combined = combine_part(values, choice)
        monthly = np.abs(combined - actual).sum()
        bucket = np.abs(np.bincount(groups, weights=combined) - bucket_actual).sum()
        out.append(100 * (bucket + monthly) / (2 * total) if total > 0 else np.inf)
    return np.asarray(out)


def choose(predictions, actual, members, origins, pure=False):
    """origins의 자료로 부분마다 가장 좋은 결합을 고른다. 반환: {부분: 'median' 또는 가중치 벡터}.
    (dz) pure=True면 후보는 재료 하나씩(섞지 않음), 동률이면 앞 재료."""
    candidates = list(np.eye(len(members))) if pure else [CANDIDATES_MEDIAN] + list(weight_grid(len(members)))
    wide = _wide(predictions, actual, members, origins)
    chosen = {}
    for part in PARTS:
        rows = wide.loc[wide["part"].eq(part)]
        if rows.empty:
            continue
        chosen[part] = candidates[int(np.argmin(_scores(rows, members, candidates)))]
    return chosen


def _apply(predictions, members, origin, choice, name):
    wide = predictions.loc[predictions["model"].isin(members) & predictions["origin"].eq(origin)]
    wide = wide.pivot_table(index=["origin"] + KEY + ["Month", "horizon"], columns="model", values="prediction")
    wide = wide.reindex(columns=members).dropna().reset_index().merge(horizon_map(HORIZON), on="horizon")
    parts = []
    for part, rows in wide.groupby("part", sort=False):
        parts.append(rows.assign(prediction=combine_part(rows[members].to_numpy(float), choice[part])))
    out = pd.concat(parts, ignore_index=True).assign(model=name)
    return out[["model", "origin"] + KEY + ["Month", "horizon", "prediction"]]


def _statuses(statuses, members, name):
    """모든 재료가 OK일 때만 OK. 아니면 앞선 재료의 실패 상태를 남긴다."""
    ids = ["origin"] + KEY
    table = None
    for i, member in enumerate(members):
        part = statuses.loc[statuses["model"].eq(member), ids + ["prediction_status"]].rename(
            columns={"prediction_status": f"s{i}"})
        table = part if table is None else table.merge(part, on=ids, how="outer")
    status = pd.Series("OK", index=table.index)
    for i in reversed(range(len(members))):
        column = table[f"s{i}"].fillna("MISSING_SOURCE")
        status = column.where(column.ne("OK"), status)
    return table[ids].assign(model=name, prediction_status=status, reason=status.where(status.ne("OK"), ""))[
        ["model"] + ids + ["prediction_status", "reason"]]


def _describe(choice):
    return {part: c if isinstance(c, str) else ",".join(f"{w:.1f}" for w in c) for part, c in choice.items()}


def stack_loo(predictions, statuses, actual, members, origins, name, check_origin=None, pure=False):
    """기준월마다 그 기준월을 뺀 나머지로 고른 결합을 적용한다. check_origin은 전체 기준월로 고른다."""
    frames, records = [], []
    plan = [(o, [x for x in origins if x != o]) for o in origins]
    if check_origin is not None and predictions["origin"].eq(check_origin).any():  # --no-check 백테스트면 건너뜀
        plan.append((check_origin, list(origins)))
    for origin, used in plan:
        choice = choose(predictions, actual, members, used, pure=pure)
        frames.append(_apply(predictions, members, origin, choice, name))
        records += [{"origin": origin, "part": part, "choice": text, "used_origins": ",".join(used)}
                    for part, text in _describe(choice).items()]
    out = pd.concat(frames, ignore_index=True).sort_values(["origin"] + KEY + ["horizon"]).reset_index(drop=True)
    states = _statuses(statuses.loc[statuses["origin"].isin([o for o, _ in plan])], members, name)
    ok = states.loc[states["prediction_status"].eq("OK"), ["origin"] + KEY]
    return out.merge(ok, on=["origin"] + KEY), states, pd.DataFrame(records)


def stack_loo_split(predictions, statuses, actual, members, origins, name, first_markets, check_origin=None,
                    pure=False):
    """(es) 시장 묶음(first_markets / 나머지)마다 따로 stack_loo. choices에 group 열(first_markets 이름 또는 rest)."""
    first = set(first_markets)
    outs, states, choices = [], [], []
    for group, pick in (("+".join(first_markets), lambda c: c.isin(first)), ("rest", lambda c: ~c.isin(first))):
        out, state, choice = stack_loo(predictions.loc[pick(predictions["Country"])],
                                       statuses.loc[pick(statuses["Country"])], actual.loc[pick(actual["Country"])],
                                       members, origins, name, check_origin, pure=pure)
        outs.append(out)
        states.append(state)
        choices.append(choice.assign(group=group))
    out = pd.concat(outs, ignore_index=True).sort_values(["origin"] + KEY + ["horizon"]).reset_index(drop=True)
    return out, pd.concat(states, ignore_index=True), pd.concat(choices, ignore_index=True)


def stack_loo_per_market(predictions, statuses, actual, members, origins, name, check_origin=None, pure=False):
    """(hx) 시장 7개 각각을 묶음으로 stack_loo. choices에 group = 시장 이름."""
    outs, states, choices = [], [], []
    for market in sorted(predictions["Country"].unique()):
        pick = lambda c, m=market: c.eq(m)
        out, state, choice = stack_loo(predictions.loc[pick(predictions["Country"])],
                                       statuses.loc[pick(statuses["Country"])], actual.loc[pick(actual["Country"])],
                                       members, origins, name, check_origin, pure=pure)
        outs.append(out)
        states.append(state)
        choices.append(choice.assign(group=market))
    out = pd.concat(outs, ignore_index=True).sort_values(["origin"] + KEY + ["horizon"]).reset_index(drop=True)
    return out, pd.concat(states, ignore_index=True), pd.concat(choices, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--pure", action="store_true", help="(dz) 부분마다 재료 하나만 고름(섞지 않음, 동률이면 앞 run)")
    parser.add_argument("--split-markets", nargs="*", default=None,
                        help="(es) 이 시장 묶음과 나머지 시장 묶음을 따로 고름(예: China)")
    parser.add_argument("--per-market", action="store_true", help="(hx) 시장마다 따로 고름(--split-markets와 함께 쓰지 않음)")
    args = parser.parse_args(argv)
    if args.per_market and args.split_markets:
        raise ValueError("--per-market와 --split-markets는 함께 쓰지 않습니다.")
    input_hash = file_sha256(args.csv)
    data = baseline.read_sales_csv(guarded(args.csv))
    predictions = pd.concat([read_run(r) for r in args.runs], ignore_index=True)
    predictions["origin"] = predictions["origin"].astype(str)
    statuses = pd.concat([read_statuses(r) for r in args.runs], ignore_index=True)
    members = [read_run(r)["model"].iloc[0] for r in args.runs]
    if args.per_market:
        out, states, choices = stack_loo_per_market(predictions, statuses, data, members, ORIGINS, args.name, CHECK_ORIGIN,
                                                    pure=args.pure)
    elif args.split_markets:
        out, states, choices = stack_loo_split(predictions, statuses, data, members, ORIGINS, args.name,
                                               args.split_markets, CHECK_ORIGIN, pure=args.pure)
    else:
        out, states, choices = stack_loo(predictions, statuses, data, members, ORIGINS, args.name, CHECK_ORIGIN,
                                         pure=args.pure)
    require_unchanged(args.csv, input_hash)
    run_dir = new_run_dir(HERE / "reports", args.name)
    write_csv(run_dir / "predictions.csv", out)
    write_csv(run_dir / "statuses.csv", states)
    write_csv(run_dir / "choices.csv", choices)
    write_json(run_dir / "run_summary.json", {
        "model": args.name, "rule": "part-wise median or 0.1-grid weighted mean, chosen leave-one-origin-out",
        "split_markets": args.split_markets, "per_market": args.per_market,
        "members": members, "sources": args.runs, "input_sha256": input_hash, "origins": ORIGINS,
        "check_origin": CHECK_ORIGIN, "trained_models": 0})
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
