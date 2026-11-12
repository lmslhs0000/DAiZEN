"""(et) 단종 발표 뒤 감소 규칙(2026-10-05 사전 고정): 중국 밖 시장에서, 기준월 O까지 공개된(공개월 ≤ O) 이름판 종료
사건(discontinued·successor_launch, 같은 이름이 이어지지 않음)의 종료월 E가 O < E ≤ O+24이면 대상월 t > E의 예측에
max(0, 1 − (t−E)/12)를 곱한다(12개월에 걸쳐 0으로). 차종마다 가장 이른 사건 하나. 상태 표는 그대로.

사용: discontinuation_rule.py --first <백테스트 run> --events <조사 CSV ...> --name <이름>
      discontinuation_rule.py --delivery <납품 폴더> --events <조사 CSV ...> --name <이름>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (HERE, KEY, PREDICTION_COLUMNS, STATUS_COLUMNS, file_sha256, guarded, new_run_dir, write_csv,
                    write_json)
from challengers import read_statuses
from score import read_run

END_TYPES = {"discontinued", "successor_launch"}
RUNOFF_MONTHS = 12
HORIZON_LIMIT = 24
EXCLUDED_MARKETS = {"China"}


def _month(value):
    text = str(value).strip()
    return pd.Period(text[:7], freq="M") if text else pd.NaT


def prepare_events(table):
    """market_scope(';' 구분)를 시장마다 펼친다. 종류·날짜를 검사한다."""
    bad = sorted(set(table["event_type"]) - END_TYPES)
    if bad:
        raise ValueError(f"종료 사건 종류가 아닙니다: {bad}")
    rows = []
    for scope, brand, model, kind, month, announced, same in zip(
            table["market_scope"], table["Brand"], table["Model"], table["event_type"], table["event_month"],
            table["announced_date"], table["continues_same_name"]):
        for market in [m.strip() for m in str(scope).split(";") if m.strip()]:
            rows.append({"Country": market, "Brand": str(brand), "Model": str(model), "event_type": kind,
                         "event": _month(month), "announced": _month(announced),
                         "same_name": str(same).strip().lower() == "yes"})
    return pd.DataFrame(rows, columns=KEY + ["event_type", "event", "announced", "same_name"])


def load_events(paths):
    tables = [pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig") for p in paths]
    return prepare_events(pd.concat(tables, ignore_index=True))


def _end_months(events, origin, past_months=0, include_china=False):
    """기준월 origin에 쓸 수 있는 차종별 가장 이른 종료월. (fq) past_months > 0이면 이미 끝난 사건(origin − past_months < E ≤ origin)도.
    (ft) include_china면 중국 사건도 쓴다."""
    excluded = set() if include_china else EXCLUDED_MARKETS
    usable = events.loc[events["announced"].notna() & ~events["same_name"]
                        & ~events["Country"].isin(excluded)].copy()
    usable = usable.loc[usable["announced"].le(origin) & usable["event"].gt(origin - past_months)
                        & usable["event"].le(origin + HORIZON_LIMIT)]
    return usable.groupby(KEY, as_index=False)["event"].min()


def apply_runoff(predictions, events, model_name=None, past_months=0, include_china=False):
    """예측 표(origin·Country·Brand·Model·Month·prediction)에 감소 계수를 곱한다. 다른 열은 그대로."""
    frames = []
    for origin, rows in predictions.groupby("origin", sort=False):
        o = pd.Period(str(origin), freq="M")
        ends = _end_months(events, o, past_months, include_china)
        table = rows.merge(ends, on=KEY, how="left")
        months = table["Month"].map(lambda m: pd.Period(str(m), freq="M"))
        gap = np.array([(m - e).n if pd.notna(e) else 0 for m, e in zip(months, table["event"])], dtype=float)
        factor = np.where(gap > 0, np.clip(1.0 - gap / RUNOFF_MONTHS, 0.0, 1.0), 1.0)
        table["prediction"] = table["prediction"] * factor
        frames.append(table.drop(columns="event"))
    out = pd.concat(frames, ignore_index=True)
    if model_name is not None and "model" in out.columns:
        out["model"] = model_name
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", help="백테스트 run")
    parser.add_argument("--delivery", help="납품 폴더(*_future_24m.csv)")
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-base")
    parser.add_argument("--past-months", type=int, default=0, help="(fq) 기준월 전 이 개월 안에 이미 끝난 사건도 적용")
    parser.add_argument("--include-china", action="store_true", help="(ft) 중국 사건에도 적용")
    args = parser.parse_args(argv)
    if bool(args.first) == bool(args.delivery):
        raise ValueError("--first와 --delivery 중 하나만 줍니다.")
    events = load_events([guarded(p) for p in args.events])
    summary = {"model": args.name, "rule": f"(et) linear run-off to 0 over {RUNOFF_MONTHS} months after announced "
               f"nameplate end (announced <= origin < end <= origin+{HORIZON_LIMIT}), "
               f"{'all markets' if args.include_china else 'markets except China'}",
               "events": {str(p): file_sha256(guarded(p)) for p in args.events}, "past_months": args.past_months,
               "include_china": args.include_china,
               "trained_models": 0}
    if args.first:
        predictions = read_run(args.first)
        out = apply_runoff(predictions, events, args.name, args.past_months, args.include_china)[PREDICTION_COLUMNS]
        run_dir = new_run_dir(args.output_base or HERE / "reports", args.name)
        write_csv(run_dir / "predictions.csv", out)
        write_csv(run_dir / "statuses.csv", read_statuses(args.first).assign(model=args.name)[STATUS_COLUMNS])
        summary.update(source=str(args.first), origins=sorted(out["origin"].astype(str).unique()))
    else:
        path = next(Path(guarded(args.delivery)).glob("*_future_24m.csv"))
        table = pd.read_csv(path, encoding="utf-8-sig", dtype={"origin": str, "Month": str})
        out = apply_runoff(table, events, past_months=args.past_months, include_china=args.include_china)[table.columns]
        run_dir = new_run_dir(args.output_base or HERE / "production", f"delivery_{args.name}")
        out = out.assign(run_id=run_dir.name, algorithm=args.name)
        write_csv(run_dir / f"{args.name}_future_24m.csv", out)
        summary.update(base_delivery=str(args.delivery), base_sha256=file_sha256(path))
    write_json(run_dir / "run_summary.json", summary)
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
