"""매달 성적 확인 — 지난번에 낸 예측 vs 그 뒤 들어온 실제 값(raw/data.csv, 채움은 실제가 아니라 제외).

사용: python monitor.py [--raw raw/data.csv] [--out raw/monitoring.csv] [--report raw/monitoring.md]
  merge_month.py --apply 가 끝에 자동으로 부른다.

- 비교 대상: 납품 목록(raw/deliveries.csv)의 예측 — 7월 납품 + 이후 step2 가 만든 납품.
- 시장·달마다: 모델 WAPE, '작년 같은 달 그대로'(기준월에 알던 값) WAPE, 쏠림(예측 합 − 실제 합)/실제 합.
- 경보(그 달 직전 기준월의 가장 최근 예측 기준):
  - worse_than_naive_3m: 한 시장에서 3개월 연속 모델이 '작년 그대로'보다 못함
  - bias_over_15pct_2m: 2개월 연속 같은 방향으로 15% 넘게 쏠림
- 참고값: 백테스트(챔피언)의 1~3개월 시장별 WAPE. 매달 값은 차종 수·계절에 따라 출렁이므로 한 달 값으로 판단하지 않는다.
- 차종별 실시간 성적(2026-10-22): 차종마다 최근 12개 채점 달의 WAPE·작년 그대로·쏠림과 등급(납품과 같은 구간, 3개월 이상일 때만)
  → raw/monitoring_series.csv + monitoring.md '차종별 실시간 성적' 절. 백테스트 평가가 없던 신차도 몇 달 지나면 자기 등급이 생긴다.
  매달 재학습하면 대부분 1~3개월 앞 예측의 성적이라 백테스트 24개월 점수(25.86%)와 직접 비교하지 않는다.
"""
import argparse
from pathlib import Path
import sys

import pandas as pd

KEY = ["Country", "Brand", "Model"]
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GRADES = [(20, "A(<20%)"), (35, "B(20-35%)"), (50, "C(35-50%)")]   # 납품 grade 와 같은 구간, 그 위는 D(50%+)
LETTER = [g for _, g in GRADES] + ["D(50%+)"]
MIN_MONTHS = 3      # 등급을 매기는 최소 채점 달 수
WINDOW = 12         # 차종마다 최근 몇 개 채점 달로 계산하나
SERIES_COLUMNS = KEY + ["status", "backtest_wape", "no_backtest", "last_origin", "months_scored", "first_month", "last_month",
                        "actual_sum", "abs_error_sum", "live_wape", "naive_wape", "bias_pct", "live_grade"]
EXPECTED_1_3 = {"United States": 13, "Canada": 14, "Europe": 18, "Japan": 19, "South Korea": 24, "Germany": 25, "China": 28}


def seasonal_naive(raw, origin, months):
    """작년 같은 달 그대로: 대상 달 M 의 12개월 전 값. 그 달이 기준월 뒤면 1년 더 앞(기준월에 알던 값만)."""
    o = pd.Period(origin, freq="M")
    base = {}
    for m in months:
        b = pd.Period(m, freq="M") - 12
        while b > o:
            b -= 12
        base[m] = str(b)
    r = raw[raw["Month"].astype(str).isin(set(base.values()))][KEY + ["Month", "Sales"]]
    out = []
    for m, b in base.items():
        x = r[r["Month"].astype(str) == b].rename(columns={"Sales": "naive"}).drop(columns="Month")
        out.append(x.assign(Month=m))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=KEY + ["naive", "Month"])


def score(fc, raw, months=None):
    """예측 표(fc: 한 기준월 또는 여러 기준월) → 기준월·시장·달마다 점수. 실제 값(raw)이 있는 차종·달만."""
    rows = []
    act = raw[KEY + ["Month", "Sales"]].rename(columns={"Sales": "actual"}).assign(Month=lambda d: d["Month"].astype(str))
    for origin, f in fc.groupby(fc["origin"].astype(str)):
        f = f.assign(Month=f["Month"].astype(str))
        ms = sorted(set(f["Month"]) & set(act["Month"]) & (set(months) if months else set(f["Month"])))
        ms = [m for m in ms if m > origin]
        if not ms:
            continue
        j = f[f["Month"].isin(ms)][KEY + ["Month", "horizon", "prediction"]].merge(act, on=KEY + ["Month"])
        j = j.merge(seasonal_naive(raw, origin, ms), on=KEY + ["Month"], how="left")
        j = j[j["prediction"].notna() & j["naive"].notna()]          # 두 방법을 같은 차종으로 비교
        for (c, m), g in j.groupby(["Country", "Month"]):
            a = g["actual"].sum()
            if a <= 0:
                continue
            rows.append({"origin": origin, "Country": c, "Month": m, "horizon": int(g["horizon"].iloc[0]), "series": len(g),
                         "actual_sum": a, "model_wape": 100 * (g["prediction"] - g["actual"]).abs().sum() / a,
                         "naive_wape": 100 * (g["naive"] - g["actual"]).abs().sum() / a,
                         "bias_pct": 100 * (g["prediction"].sum() - a) / a})
    return pd.DataFrame(rows, columns=["origin", "Country", "Month", "horizon", "series", "actual_sum", "model_wape", "naive_wape", "bias_pct"])


def latest(hist):
    """달마다 그 달 직전 기준월의 가장 최근 예측(=실제로 쓰였을 예측)만."""
    h = hist[hist["origin"].astype(str) < hist["Month"].astype(str)]
    return h.sort_values("origin").groupby(["Country", "Month"], as_index=False).tail(1).sort_values(["Country", "Month"])


def alarms(hist):
    out = []
    for c, g in latest(hist).groupby("Country"):
        g = g.sort_values("Month")
        t3 = g.tail(3)
        if len(t3) == 3 and (t3["model_wape"] > t3["naive_wape"]).all():
            out.append({"Country": c, "alarm": "worse_than_naive_3m",
                        "detail": f"{t3['Month'].iloc[0]}~{t3['Month'].iloc[-1]} 모델 {t3['model_wape'].round(1).tolist()}% vs 작년 그대로 {t3['naive_wape'].round(1).tolist()}%"})
        t2 = g.tail(2)
        if len(t2) == 2 and ((t2["bias_pct"] > 15).all() or (t2["bias_pct"] < -15).all()):
            out.append({"Country": c, "alarm": "bias_over_15pct_2m",
                        "detail": f"{t2['Month'].iloc[0]}~{t2['Month'].iloc[-1]} 쏠림 {t2['bias_pct'].round(1).tolist()}% (+ 는 높게 예측)"})
    return pd.DataFrame(out, columns=["Country", "alarm", "detail"])


def grade(wape, months):
    """차종 실시간 등급: 채점 달이 MIN_MONTHS 미만이면 '평가 중', 실제 합이 0 이면 '평가 불가'."""
    if months < MIN_MONTHS:
        return f"평가 중({months}개월)"
    if wape is None or pd.isna(wape):
        return "평가 불가(실제 0)"
    for limit, label in GRADES:
        if wape < limit:
            return label
    return "D(50%+)"


def series_score(fc, raw, months=None):
    """차종별 실시간 성적. 달마다 그 달 직전 기준월의 가장 최근 예측(시장 표와 같은 규칙), 차종마다 최근 WINDOW 개 채점 달."""
    act = raw[KEY + ["Month", "Sales"]].rename(columns={"Sales": "actual"}).assign(Month=lambda d: d["Month"].astype(str))
    f = fc.assign(Month=fc["Month"].astype(str), origin=fc["origin"].astype(str))
    f = f[(f["Month"] > f["origin"]) & f["prediction"].notna()]
    if months:
        f = f[f["Month"].isin(set(months))]
    j = f.merge(act, on=KEY + ["Month"])
    if not len(j):
        return pd.DataFrame(columns=SERIES_COLUMNS)
    j = j.sort_values("origin").groupby(KEY + ["Month"], as_index=False).tail(1)
    parts = [g.merge(seasonal_naive(raw, o, sorted(g["Month"].unique())), on=KEY + ["Month"], how="left") for o, g in j.groupby("origin")]
    j = pd.concat(parts, ignore_index=True).sort_values("Month").groupby(KEY).tail(WINDOW)
    rows = []
    for key, g in j.groupby(KEY):
        g = g.sort_values("Month")
        a = g["actual"].sum()
        err = (g["prediction"] - g["actual"]).abs().sum()
        wape = 100 * err / a if a > 0 else float("nan")
        gn = g[g["naive"].notna()]
        an = gn["actual"].sum()
        last = g.iloc[-1]
        bw = last["backtest_wape"] if "backtest_wape" in g else float("nan")
        rows.append({**dict(zip(KEY, key)), "status": last["status"] if "status" in g else "",
                     "backtest_wape": bw, "no_backtest": bool(pd.isna(bw)), "last_origin": last["origin"],
                     "months_scored": len(g), "first_month": g["Month"].iloc[0], "last_month": g["Month"].iloc[-1],
                     "actual_sum": a, "abs_error_sum": err, "live_wape": wape,
                     "naive_wape": 100 * (gn["naive"] - gn["actual"]).abs().sum() / an if an > 0 else float("nan"),
                     "bias_pct": 100 * (g["prediction"].sum() - a) / a if a > 0 else float("nan"),
                     "live_grade": grade(wape, len(g))})
    return pd.DataFrame(rows, columns=SERIES_COLUMNS)


def series_report(ss):
    lines = ["## 차종별 실시간 성적", "",
             f"지난 납품 예측 vs 실제, 차종마다 최근 {WINDOW}개 채점 달. 등급은 채점 달 {MIN_MONTHS}개 이상일 때만(구간은 납품 grade 와 같음). "
             "매달 재학습하면 대부분 1~3개월 앞 예측의 성적이라 백테스트 24개월 점수와 직접 비교하지 않는다. 전체 표: raw/monitoring_series.csv", ""]
    if not len(ss):
        return "\n".join(lines + ["아직 채점할 실제 값이 없음.", ""]) + "\n"
    dist = ss["live_grade"].value_counts()
    order = LETTER + sorted(k for k in dist.index if k not in LETTER)
    lines += ["| 등급 | 차종 수 |", "|---|---:|"] + [f"| {k} | {int(dist[k])} |" for k in order if k in dist.index] + [""]
    new = ss[ss["no_backtest"] & ss["live_grade"].isin(LETTER)].sort_values("actual_sum", ascending=False)
    lines += [f"### 백테스트 평가가 없던 차종 중 등급이 생긴 차종 ({len(new)}개, 판매 많은 순 30개)", ""]
    lines += [f"- {r.Country} {r.Brand} {r.Model}: {r.live_grade} {r.live_wape:.1f}% ({r.months_scored}개월, 상태 {r.status})"
              for r in new.head(30).itertuples()] or ["- 없음"]
    bad = ss[ss["live_grade"] == "D(50%+)"].sort_values("actual_sum", ascending=False)
    lines += ["", f"### 판매 큰 차종 중 D 등급 ({len(bad)}개 중 15개)", ""]
    lines += [f"- {r.Country} {r.Brand} {r.Model}: {r.live_wape:.1f}% (작년 그대로 {r.naive_wape:.1f}%, 쏠림 {r.bias_pct:+.1f}%, {r.months_scored}개월)"
              for r in bad.head(15).itertuples()] or ["- 없음"]
    return "\n".join(lines) + "\n"


def forecast_files():
    """납품 목록(raw/deliveries.csv)의 예측만 — 시험·실험 묶음은 제외. 기준월마다 하나."""
    import deliveries
    return [Path(p) for p in deliveries.load()["path"] if Path(p).exists()]


def report(hist, al):
    lines = ["# 매달 성적 확인 (지난 예측 vs 실제)", "",
             "실제 값은 raw/data.csv 만(채움 제외). '작년 그대로' = 기준월에 알던 작년 같은 달 값. 낮을수록 좋음.", ""]
    if len(al):
        lines += ["## ⚠ 경보", ""] + [f"- **{r.Country}** {r.alarm}: {r.detail}" for r in al.itertuples()] + [
            "", "경보 뜻·할 일은 README_운영 '경보가 뜨면' 참고.", ""]
    else:
        lines += ["경보 없음.", ""]
    lt = latest(hist)
    if len(lt):
        lines += ["## 시장별 최근 달", "", "| 시장 | 달 | 예측 기준월 | 차종 | 모델 WAPE | 작년 그대로 | 쏠림 | 백테스트 1~3개월 참고 |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for r in lt.sort_values(["Country", "Month"]).groupby("Country").tail(3).itertuples():
            lines.append(f"| {r.Country} | {r.Month} | {r.origin} | {r.series} | {r.model_wape:.1f}% | {r.naive_wape:.1f}% | "
                         f"{r.bias_pct:+.1f}% | {EXPECTED_1_3.get(r.Country, '')}% |")
    return "\n".join(lines) + "\n"


def run(raw_path, out_path, report_path, series_path=None, forecast_paths=None):
    raw = pd.read_csv(raw_path, encoding="utf-8-sig", dtype={"Month": str})
    paths = forecast_files() if forecast_paths is None else [Path(p) for p in forecast_paths]
    fcs = [pd.read_csv(p, encoding="utf-8-sig", dtype={"Month": str, "origin": str}) for p in paths]
    empty = pd.DataFrame(columns=KEY + ["Month", "horizon", "prediction", "origin"])
    fc = pd.concat(fcs, ignore_index=True) if fcs else empty
    hist = score(fc, raw)
    hist.to_csv(out_path, index=False, encoding="utf-8-sig")
    al = alarms(hist) if len(hist) else pd.DataFrame(columns=["Country", "alarm", "detail"])
    ss = series_score(fc, raw)
    ss.to_csv(series_path or Path(out_path).with_name("monitoring_series.csv"), index=False, encoding="utf-8-sig")
    Path(report_path).write_text(report(hist, al) + "\n" + series_report(ss), encoding="utf-8")
    return hist, al


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "raw" / "data.csv"))
    ap.add_argument("--out", default=str(ROOT / "raw" / "monitoring.csv"))
    ap.add_argument("--report", default=str(ROOT / "raw" / "monitoring.md"))
    ap.add_argument("--series-out", default=str(ROOT / "raw" / "monitoring_series.csv"))
    a = ap.parse_args()
    hist, al = run(a.raw, a.out, a.report, a.series_out)
    print(f"성적 {len(hist)}줄, 경보 {len(al)}개 → {a.report}")


if __name__ == "__main__":
    main()
