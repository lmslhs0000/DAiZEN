"""(hq) 사건 규칙(2026-10-10 사전 고정, progress.md): 챔피언 예측 위에 얹는 학습 없는 규칙.

기준월 O마다 발표일 ≤ O이고 사건 달 L이 O 뒤인 사건만 적용한다.
  sibling_launch(같은 브랜드 새 차종이 수요를 가져감): L 이후 달 × r_sib
  new_generation / powertrain_switch: L 이후 달 × r_up, L 직전 3달(O 뒤인 달만) × r_pre
계수는 사건 종류별로, 기준월 O까지 결과가 끝난 과거 사건만으로 구한다(뒤 6달 L..L+5 ≤ O, 직전 창 L−1 ≤ O):
  비율 = 사건 차종의 (뒤 창 월평균 ÷ 앞 창 월평균) ÷ 같은 시장 전체의 같은 비율, 사건 종류별 중앙값,
  3건 미만이면 1, LOW~HIGH로 자른다. 앞·뒤 창: after = L−6..L−1 대 L..L+5, pre = L−9..L−4 대 L−3..L−1.

사용: event_rules.py --first <규칙 전 run> --events <csv ...> --name hq_events [--parts back]
      event_rules.py --delivery <납품 폴더> --events <csv ...> --name <이름> [--parts back]   (납품 파일에 같은 규칙)
"""

import argparse

import numpy as np
import pandas as pd

from common import CSV_PATH, HERE, PREDICTION_COLUMNS, STATUS_COLUMNS, file_sha256, guarded, new_run_dir, require_unchanged, write_csv, write_json

LOW, HIGH, MIN_EVENTS = 0.3, 3.0, 3
TYPES = ["sibling_launch", "new_generation", "powertrain_switch"]
WITH_PRE = ["new_generation", "powertrain_switch"]


def _expand(events):
    ev = events.copy()
    ev["market"] = ev["market_scope"].astype(str).str.split(";")
    ev = ev.explode("market")
    ev["market"] = ev["market"].str.strip()
    ev["L"] = [pd.Period(str(v)[:7], "M") for v in ev["event_month"]]
    ev["pub"] = [pd.Period(pd.Timestamp(v), "M") if pd.notna(v) and str(v).strip() else None for v in ev["announced_date"]]
    return ev[ev["event_type"].isin(TYPES)]


def _window_mean(s, start, end):
    w = s.reindex(pd.period_range(start, end, freq="M"))
    return float(w.mean()) if w.notna().all() and len(w) else np.nan


def event_ratio(data, market, brand, model, L, kind="after"):
    mk = data[data["Country"] == market]
    series = mk[(mk["Brand"] == brand) & (mk["Model"] == model)].set_index("Month")["Sales"]
    total = mk.groupby("Month")["Sales"].sum()
    if kind == "after":
        a, b = (L - 6, L - 1), (L, L + 5)
    else:
        a, b = (L - 9, L - 4), (L - 3, L - 1)
    s0, s1 = _window_mean(series, *a), _window_mean(series, *b)
    t0, t1 = _window_mean(total, *a), _window_mean(total, *b)
    if not (s0 > 0 and t0 > 0 and t1 > 0) or np.isnan(s1):
        return np.nan
    return (s1 / s0) / (t1 / t0)


def learn_factors(data, events, origin, per_market=False):
    """per_market=True((hq4))면 키가 (시장, 종류): 그 시장 사건이 MIN_EVENTS 이상이면 시장 중앙값, 아니면 전 시장 중앙값."""
    o = pd.Period(origin, "M")
    d = data[pd.PeriodIndex(data["Month"], freq="M") <= o].copy()
    d["Month"] = pd.PeriodIndex(d["Month"], freq="M")
    ev = _expand(events).drop_duplicates(["market", "Brand", "Model", "event_type", "L"])
    markets = sorted(set(d["Country"]))
    out = {}
    for t in TYPES:
        kinds = [("after", t)] + ([("pre", t + "_pre")] if t in WITH_PRE else [])
        for kind, key in kinds:
            done = ev[(ev["event_type"] == t).to_numpy() & np.array([(L + 5 <= o) if kind == "after" else (L - 1 <= o) for L in ev["L"]], dtype=bool)]
            ratios = {r.market: [] for r in done.itertuples()}
            for r in done.itertuples():
                v = event_ratio(d, r.market, r.Brand, r.Model, r.L, kind)
                if np.isfinite(v):
                    ratios[r.market].append(v)
            all_ratios = [x for v in ratios.values() for x in v]
            glob = float(np.clip(np.median(all_ratios), LOW, HIGH)) if len(all_ratios) >= MIN_EVENTS else 1.0
            if not per_market:
                out[key] = glob
                continue
            for m in markets:
                own = ratios.get(m, [])
                out[(m, key)] = float(np.clip(np.median(own), LOW, HIGH)) if len(own) >= MIN_EVENTS else glob
    return out


def _factor(factors, market, key):
    return factors.get((market, key), factors.get(key, 1.0))


PART_HORIZONS = {"front": range(1, 4), "middle": range(4, 13), "back": range(13, 25)}


def apply(preds, events, origin, factors, parts=None):
    """parts(예: ['back'])를 주면 그 부분의 예측 거리 달에만 적용한다((hq3))."""
    o = pd.Period(origin, "M")
    out = preds.copy()
    months = pd.PeriodIndex(out["Month"], freq="M")
    allowed = np.ones(len(out), dtype=bool) if not parts else         out["horizon"].astype(int).isin([h for part in parts for h in PART_HORIZONS[part]]).to_numpy()
    ev = _expand(events)
    ev = ev[np.array([p is not None and p <= o for p in ev["pub"]], dtype=bool) & np.array([L > o for L in ev["L"]], dtype=bool)]
    ev = ev.drop_duplicates(["market", "Brand", "Model", "event_type", "L"])
    for r in ev.itertuples():
        key = (out["Country"] == r.market) & (out["Brand"] == r.Brand) & (out["Model"] == r.Model) & allowed
        f_after = _factor(factors, r.market, r.event_type)
        out.loc[key & (months >= r.L), "prediction"] *= f_after
        if r.event_type in WITH_PRE:
            f_pre = _factor(factors, r.market, r.event_type + "_pre")
            out.loc[key & (months >= r.L - 3) & (months <= r.L - 1), "prediction"] *= f_pre
    return out


def main(argv=None):
    from challengers import read_statuses
    from score import read_run
    import baseline
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", help="백테스트 run")
    parser.add_argument("--delivery", help="납품 폴더(*_future_24m.csv) — 기준월은 파일의 origin, 행·상태 그대로")
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--parts", nargs="+", choices=sorted(PART_HORIZONS), help="(hq3) 이 부분에만 적용")
    parser.add_argument("--per-market", action="store_true", help="(hq4) 계수를 시장별로 학습(3건 미만이면 전 시장 값)")
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--output-base", default=str(HERE / "reports"))
    args = parser.parse_args(argv)
    if bool(args.first) == bool(args.delivery):
        raise ValueError("--first와 --delivery 중 하나만 줍니다.")
    input_hash = file_sha256(args.csv)
    data = baseline.read_sales_csv(guarded(args.csv))
    cols = ["market_scope", "Brand", "Model", "event_type", "event_month", "announced_date"]
    events = pd.concat([pd.read_csv(guarded(p), encoding="utf-8-sig")[cols] for p in args.events], ignore_index=True)
    summary = {"model": args.name, "rule": "(hq) event rules: sibling_launch / new_generation / powertrain_switch factors learned from past events finished by origin",
               "parts": args.parts, "per_market": args.per_market, "events": {p: file_sha256(guarded(p)) for p in args.events}, "input_sha256": input_hash, "trained_models": 0}
    if args.first:
        pred = read_run(args.first)
        pred["origin"] = pred["origin"].astype(str)
        frames, info = [], []
        for origin, rows in pred.groupby("origin", sort=True):
            f = learn_factors(data, events, origin, per_market=args.per_market)
            frames.append(apply(rows, events, origin, f, parts=args.parts))
            info.append({"origin": origin, **{str(k): v for k, v in f.items()}})
            print(origin, {str(k): round(v, 3) for k, v in f.items()}, flush=True)
        out = pd.concat(frames, ignore_index=True).assign(model=args.name)
        out["Month"] = out["Month"].astype(str)
        require_unchanged(args.csv, input_hash)
        run_dir = new_run_dir(args.output_base, args.name)
        write_csv(run_dir / "predictions.csv", out[PREDICTION_COLUMNS])
        write_csv(run_dir / "statuses.csv", read_statuses(args.first).assign(model=args.name)[STATUS_COLUMNS])
        summary.update(source=str(args.first))
    else:
        from pathlib import Path
        path = next(Path(guarded(args.delivery)).glob("*_future_24m.csv"))
        table = pd.read_csv(path, encoding="utf-8-sig", dtype={"origin": str, "Month": str})
        origins = sorted(table["origin"].unique())
        if len(origins) != 1:
            raise ValueError(f"납품 파일의 기준월이 하나여야 합니다: {origins}")
        f = learn_factors(data, events, origins[0], per_market=args.per_market)
        print(origins[0], {str(k): round(v, 3) for k, v in f.items()}, flush=True)
        out = apply(table, events, origins[0], f, parts=args.parts)[table.columns]
        info = [{"origin": origins[0], **{str(k): v for k, v in f.items()}}]
        require_unchanged(args.csv, input_hash)
        run_dir = new_run_dir(args.output_base if args.output_base != str(HERE / "reports") else HERE / "production", f"delivery_{args.name}")
        out = out.assign(run_id=run_dir.name, algorithm=args.name)
        write_csv(run_dir / f"{args.name}_future_24m.csv", out)
        summary.update(base_delivery=str(args.delivery), base_sha256=file_sha256(path))
    write_csv(run_dir / "factors.csv", pd.DataFrame(info))
    write_json(run_dir / "run_summary.json", summary)
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
