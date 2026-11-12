"""Quarter / 2-month-block -> month split rules for quarterly-only NA OEM series, with backtests.

Rule priority (applied per series and period):
 (a) quarter_split:gcbc_shape - GCBC model page posts monthly values for the same model and country in that
     period: month_i = official_total * gcbc_i / sum(gcbc) (if sum(gcbc) == 0 -> fall through to (b)).
 (b) quarter_split:trailing - shares from the series own history in our raw data, using only months
     before the period: the variant chosen below on 2022Q1..2024Q4 (selection), reported on 2025Q1..2026Q2.
     Fallback to equal split when no complete earlier same-period history with total > 0 exists.
Period types: 'quarter' (3 months) for all quarterly OEMs; '2month' for US_TMNA (Toyota/Lexus print the
quarter-end month, the two inner months come as one block: Jan+Feb, Apr+May, Jul+Aug, Oct+Nov).
Outputs: split_rules.csv, split_backtest_detail.csv, split_variant_selection.csv"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RAW = os.path.join(ROOT, "raw", "data.csv")
COV = os.path.join(ROOT, "experiments", "data_sources_v1", "group_1", "coverage.csv")
GCBC_LONG = os.path.join(HERE, "gcbc_monthly_long.csv")  # written by collect_na.py (all parsed GCBC months)

VARIANTS = ["equal", "caldays", "weekdays", "mon_sat", "prev1", "avg2", "avg3", "pooled2", "pooled3"]
SEL_PERIODS = [f"{y}Q{q}" for y in (2022, 2023, 2024) for q in (1, 2, 3, 4)]
TEST_PERIODS = [f"{y}Q{q}" for y, q in [(2025, 1), (2025, 2), (2025, 3), (2025, 4), (2026, 1), (2026, 2)]]


def months_of(period, ptype):
    y, q = int(period[:4]), int(period[-1])
    m0 = 3 * (q - 1) + 1
    ms = [m0, m0 + 1] if ptype == "2month" else [m0, m0 + 1, m0 + 2]
    return [f"{y}-{m:02d}" for m in ms]


def shares(hist, period, ptype, variant):
    """hist: dict month->sales (raw). Only same-period months of earlier years are used."""
    n = 2 if ptype == "2month" else 3
    if variant == "equal":
        return np.ones(n) / n, "equal"
    if variant in ("caldays", "weekdays", "mon_sat"):
        w = []
        for mo in months_of(period, ptype):
            days = pd.date_range(mo + "-01", periods=pd.Period(mo).days_in_month, freq="D")
            if variant == "caldays":
                w.append(len(days))
            elif variant == "weekdays":
                w.append(int((days.dayofweek < 5).sum()))
            else:
                w.append(int((days.dayofweek < 6).sum()))
        w = np.array(w, float)
        return w / w.sum(), variant
    y = int(period[:4])
    k = {"prev1": 1, "avg2": 2, "avg3": 3, "pooled2": 2, "pooled3": 3}[variant]
    vecs = []
    for back in range(1, k + 1):
        ms = months_of(f"{y-back}Q{period[-1]}", ptype)
        v = [hist.get(m) for m in ms]
        if any(x is None or pd.isna(x) for x in v):
            continue
        v = np.array(v, float)
        if v.sum() > 0:
            vecs.append(v)
    if not vecs:
        return np.ones(n) / n, "equal_fallback"
    if variant.startswith("pooled"):
        s = np.sum(vecs, axis=0)
        return s / s.sum(), f"{variant}({len(vecs)}y)"
    return np.mean([v / v.sum() for v in vecs], axis=0), f"{variant}({len(vecs)}y)"


COLS = ["Country", "Brand", "Model", "source_id", "period_type", "variant", "period",
        "Month", "actual", "pred", "abs_err", "shares_used"]


def backtest(series, raw, periods, variant):
    out = []
    for (c, b, m, sid, ptype) in series:
        h = raw.get((c, b, m), {})
        for p in periods:
            ms = months_of(p, ptype)
            act = [h.get(x) for x in ms]
            if any(a is None or pd.isna(a) for a in act):
                continue
            act = np.array(act, float)
            sh, used = shares(h, p, ptype, variant)
            pred = act.sum() * sh
            for x, a, pr in zip(ms, act, pred):
                out.append([c, b, m, sid, ptype, variant, p, x, a, pr, abs(a - pr), used])
    return pd.DataFrame(out, columns=COLS)


def wape(d):
    s = d.actual.sum()
    return 100 * d.abs_err.sum() / s if s > 0 else np.nan


def load_gcbc():
    if not os.path.exists(GCBC_LONG):
        return {}
    g = pd.read_csv(GCBC_LONG, encoding="utf-8-sig")
    g = g[(g.kind == "M") & g.value.notna()]
    return {(r.Country, r.Brand, r.Model, r.period): r.value for r in g.itertuples()}


def gcbc_shape_backtest(series, raw, periods, gd):
    out = []
    for (c, b, m, sid, ptype) in series:
        h = raw.get((c, b, m), {})
        for p in periods:
            ms = months_of(p, ptype)
            act = [h.get(x) for x in ms]
            gv = [gd.get((c, b, m, x)) for x in ms]
            if any(a is None or pd.isna(a) for a in act) or any(v is None for v in gv):
                continue
            act, gv = np.array(act, float), np.array(gv, float)
            if gv.sum() <= 0:
                continue
            pred = act.sum() * gv / gv.sum()
            for x, a, pr in zip(ms, act, pred):
                out.append([c, b, m, sid, ptype, "gcbc_shape", p, x, a, pr, abs(a - pr), "gcbc"])
    return pd.DataFrame(out, columns=COLS)


def main():
    cov = pd.read_csv(COV, encoding="utf-8-sig")
    q = cov[cov.frequency == "quarterly"].copy()
    q["period_type"] = np.where(q.source_id == "US_TMNA", "2month", "quarter")
    series = list(q[["Country", "Brand", "Model", "source_id", "period_type"]].itertuples(index=False, name=None))
    r = pd.read_csv(RAW, encoding="utf-8-sig")
    r = r[r.Country.isin(["United States", "Canada"]) & (r.Month <= "2026-07")]
    raw = {k: dict(zip(d.Month, d.Sales)) for k, d in r.groupby(["Country", "Brand", "Model"])}

    # 1) choose the trailing variant on the selection quarters (per period type)
    sel = []
    for v in VARIANTS:
        for per_name, pers in [("2022Q1..2024Q4 (selection)", SEL_PERIODS), ("2025Q1..2026Q2 (report only)", TEST_PERIODS)]:
            d = backtest(series, raw, pers, v)
            for pt, dd in d.groupby("period_type"):
                sel.append([pt, v, per_name, len(dd), round(wape(dd), 3)])
    sel = pd.DataFrame(sel, columns=["period_type", "variant", "periods", "n_months", "wape_pct"])
    sel.to_csv(os.path.join(HERE, "split_variant_selection.csv"), index=False, encoding="utf-8-sig")
    print(sel.to_string())
    s0 = sel[sel.periods.str.contains("selection")]
    best = {pt: d.sort_values(["wape_pct", "variant"]).variant.iloc[0] for pt, d in s0.groupby("period_type")}
    print("chosen:", best)

    # 2) test-period backtest with chosen variant (+ equal baseline, + gcbc shape where available)
    gd = load_gcbc()
    det = []
    for pt, v in best.items():
        ss = [x for x in series if x[4] == pt]
        det.append(backtest(ss, raw, TEST_PERIODS, v))
        if v != "equal":
            det.append(backtest(ss, raw, TEST_PERIODS, "equal"))
    det.append(gcbc_shape_backtest(series, raw, TEST_PERIODS, gd))
    det = pd.concat([d for d in det if len(d)])
    det.to_csv(os.path.join(HERE, "split_backtest_detail.csv"), index=False, encoding="utf-8-sig")
    win = det[det.Month.isin(["2026-05", "2026-06"])]
    gl = {(k[0], k[1], k[2]) for k in gd if "2025-01" <= k[3] <= "2026-07"}

    rows = []
    for (c, b, m, sid, pt) in series:
        v = best[pt]
        dd = det[(det.Country == c) & (det.Brand == b) & (det.Model == m)]
        tr, eq, gs = dd[dd.variant == v], dd[dd.variant == "equal"], dd[dd.variant == "gcbc_shape"]
        w = win[(win.Country == c) & (win.Brand == b) & (win.Model == m) & (win.variant == v)]
        has_g = (c, b, m) in gl
        base = "block_split" if pt == "2month" else "quarter_split"
        fb = {"equal": "equal", "caldays": "calendar_days", "weekdays": "weekdays", "mon_sat": "mon_sat_days"}.get(v, "trailing")
        rule = f"{base}:gcbc_shape" if has_g else f"{base}:{fb}"
        if fb == "trailing":
            tparam = (f"trailing {v}: month shares of the same quarter/block in up to {v[-1] if v[-1].isdigit() else 1} "
                      f"previous year(s) of our raw series, info before the period only; equal split if none")
        elif fb == "equal":
            tparam = "equal split (beat own-history trailing shares in the 2022Q1..2024Q4 selection backtest)"
        else:
            tparam = (f"{fb}: month share = number of {fb.replace('_', ' ')} in the month / in the quarter "
                      f"(beat own-history trailing shares in the 2022Q1..2024Q4 selection backtest)")
        params = f"primary=GCBC monthly scaled to official total; fallback={tparam}" if has_g else tparam
        note = f"backtest 2025Q1..2026Q2 months={len(tr)}; equal-split WAPE={wape(eq):.2f}%"
        note += f"; gcbc_shape WAPE={wape(gs):.2f}% (n={len(gs)})" if len(gs) else "; no GCBC monthly for shape test"
        if len(w):
            note += (f"; window 2026-05..06 WAPE={wape(w):.2f}% (2026-07 not testable: its quarter/block needs "
                     f"Aug/Sep > 2026-07)")
        if pt == "2month":
            note += "; TMNA prints the quarter-end month, only the 2 inner months are split"
        sl = cov[(cov.Country == c) & (cov.Brand == b) & (cov.Model == m)].sales_last12.iloc[0]
        auto = cov[(cov.Country == c) & (cov.Brand == b) & (cov.Model == m)].automatable.iloc[0]
        rows.append([c, b, m, sid, pt, rule, params, round(wape(tr), 3) if len(tr) else np.nan,
                     round(wape(gs), 3) if len(gs) else np.nan, sl, auto, note])
    out = pd.DataFrame(rows, columns=["Country", "Brand", "Model", "source_id", "period_type", "rule", "params",
                                      "backtest_wape", "backtest_wape_gcbc_shape", "sales_last12",
                                      "coverage_automatable", "note"])
    out = out.sort_values(["Country", "sales_last12"], ascending=[False, False])
    out.to_csv(os.path.join(HERE, "split_rules.csv"), index=False, encoding="utf-8-sig")

    for pt, v in best.items():
        for c in ["United States", "Canada"]:
            for auto in ["partial", "all"]:
                keys = set(zip(out.Country, out.Brand, out.Model)) if auto == "all" else \
                    set(zip(*out[out.coverage_automatable == "partial"][["Country", "Brand", "Model"]].values.T)) \
                    if (out.coverage_automatable == "partial").any() else set()
                for name, vv in [("trailing " + v, v), ("equal", "equal"), ("gcbc_shape", "gcbc_shape")]:
                    dd = det[(det.Country == c) & (det.period_type == pt) & (det.variant == vv)]
                    dd = dd[[k in keys for k in zip(dd.Country, dd.Brand, dd.Model)]]
                    if len(dd):
                        ww = dd[dd.Month.isin(["2026-05", "2026-06"])]
                        print(f"{c:14s} {pt:8s} {auto:8s} {name:16s} months={len(dd):5d} WAPE={wape(dd):6.2f}%  "
                              f"window 2026-05..06 WAPE={wape(ww):6.2f}%")


if __name__ == "__main__":
    main()
