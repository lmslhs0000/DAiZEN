"""고객용 설명서를 새 납품 묶음에 맞춰 쓴다. 사용: delivery_note.py <납품 묶음 폴더> --template <7월 판 설명서> --rawdir raw
2026-10-29: run_pipeline 이 7월 판 설명서를 그대로 복사해 9월 기준 판에도 '2026년 8월~2028년 7월'이 적혀 나가던 것을 고침.
기준월 2026-07 이면 원문 그대로. 그 밖에는 기간 줄·기준 문장을 새 기준월로 바꾸고, 제목 아래에 '이번 판' 블록(기준월·예측 차종 수·
8월부터 기준월까지 달마다 예측으로 채운 칸 수와 비율)을 넣는다. 정확도·세대 교체·단종 목록은 7월 판 그대로라고 밝힌다."""
import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

FIRST = "2026-07"     # 첫 납품판 기준월(설명서 원문이 이 판 기준)
OLD_BASIS = "2026년 8월 이후에 나온 실적·발표는 쓰지 않았습니다(7월 시점 예측)."
PERIOD_LINE = re.compile(r"^- \*\*기간\*\*:.*$", re.M)
OUT_NAME = "README_고객용_설명서.md"


def add_months(month, k):
    p = pd.Period(month, freq="M") + k
    return f"{p.year}-{p.month:02d}"


def period(origin, horizon=24):
    return add_months(origin, 1), add_months(origin, horizon)


def ko_month(month):
    y, m = month.split("-")
    return f"{y}년 {int(m)}월"


def fill_shares(raw, fills, origin):
    """첫 판 다음 달부터 기준월까지, 달마다 원본 실제 칸 수, 채움 표의 직전 예측 칸(filled)·수집기 추정 칸(estimated) 수와
    그 달 전체 칸(실제 + 채움 표) 중 비율(share·est_share, %). fill_type 열이 없으면 모두 직전 예측으로 본다."""
    rows = []
    months = pd.period_range(add_months(FIRST, 1), origin, freq="M").strftime("%Y-%m")
    has = fills is not None and len(fills)
    kind = fills["fill_type"].fillna("forecast") if has and "fill_type" in fills else None
    for m in months:
        actual = int((raw["Month"].astype(str) == m).sum())
        in_m = (fills["Month"].astype(str) == m) if has else None
        est = int((in_m & (kind == "estimated")).sum()) if kind is not None else 0
        filled = int(in_m.sum()) - est if has else 0
        tot = actual + filled + est
        rows.append({"Month": m, "actual": actual, "filled": filled, "estimated": est,
                     "share": 100.0 * filled / tot if tot else 0.0, "est_share": 100.0 * est / tot if tot else 0.0})
    return pd.DataFrame(rows, columns=["Month", "actual", "filled", "estimated", "share", "est_share"])


def build(template, origin, shares, n_series, n_pred, made):
    if origin == FIRST:
        return template
    start, end = period(origin)
    body = PERIOD_LINE.sub(f"- **기간**: {ko_month(start)} ~ {ko_month(end)}(24개월). {ko_month(origin)}까지의 실제 판매로 만들었습니다.",
                           template, count=1)
    body = body.replace(OLD_BASIS, f"{ko_month(start)} 이후 달의 실적은 쓰지 않았습니다({ko_month(origin)} 시점 예측. "
                                   f"{ko_month(origin)}까지의 실적은 그 뒤에 발표된 값을 모아 넣었습니다).")
    body = body.replace("몇 달 뒤인지(1 = 2026년 8월)", f"몇 달 뒤인지(1 = {ko_month(start)})")
    body = body.replace("  - 2026년 8월 이후에 나온 발표·실적.", f"  - {ko_month(start)} 이후 달의 실적, {ko_month(FIRST)}보다 뒤에 나온 "
                                                              f"신차·단종 발표(세대 교체·단종 목록은 {ko_month(FIRST)} 판 그대로).")
    filled = ", ".join(f"{ko_month(r.Month)} 직전 예측 {r.filled:,}칸({r.share:.1f}%)·수집기 추정 {r.estimated:,}칸({r.est_share:.1f}%)"
                       for r in shares.itertuples())
    block = [f"## 이번 판: {ko_month(origin)} 기준 (만든 날 {made})",
             f"- 예측 기간: {ko_month(start)} ~ {ko_month(end)}. {n_series:,}개 차종 중 {n_pred:,}개에 예측이 있습니다(나머지는 status 열에 이유).",
             f"- {ko_month(add_months(FIRST, 1))}부터는 매달 공개 출처에서 실적을 모아 넣었습니다. 공식 실적을 구하지 못한 칸은 직전 예측값으로, "
             f"수집기가 계산한 값 중 추정 몫이 30% 이상인 칸(예: 유럽 합계의 미공개 나라 몫)은 그 추정치로 원본과 따로 두고 "
             f"학습에만 씁니다. 정확도 채점에는 쓰지 않습니다. 그 달 전체 칸 중: {filled or '없음'}.",
             f"- 정확도 숫자(backtest_wape·grade 열 포함)와 세대 교체·단종 목록은 {ko_month(FIRST)} 판에서 만든 것을 그대로 씁니다. 예측 방법은 같습니다.",
             f"- 본문에 적힌 날짜는 {ko_month(FIRST)} 판 기준입니다(위 기간·기준 줄은 이번 판에 맞춰 고쳤습니다)."]
    title, _, rest = body.partition("\n")
    return title + "\n\n" + "\n".join(block) + "\n" + rest


def main(kp, template, rawdir, made=None):
    kp, rawdir = Path(kp), Path(rawdir)
    fc = pd.read_csv(kp / "client_forecast_24m.csv", encoding="utf-8-sig", dtype={"Month": str, "origin": str})
    origins = fc["origin"].dropna().unique()
    assert len(origins) == 1, f"기준월이 하나가 아님: {origins}"
    raw = pd.read_csv(rawdir / "data.csv", encoding="utf-8-sig", dtype={"Month": str})
    fp = rawdir / "filled_values.csv"
    fills = pd.read_csv(fp, encoding="utf-8-sig", dtype={"Month": str}) if fp.exists() else None
    key = ["Country", "Brand", "Model"]
    n_series = len(fc[key].drop_duplicates())
    n_pred = len(fc[fc["prediction"].notna()][key].drop_duplicates())
    text = build(Path(template).read_text(encoding="utf-8"), origins[0], fill_shares(raw, fills, origins[0]),
                 n_series, n_pred, made or f"{datetime.now():%Y-%m-%d}")
    (kp / OUT_NAME).write_text(text, encoding="utf-8")
    print(kp / OUT_NAME)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("--template", required=True)
    ap.add_argument("--rawdir", default="raw")
    a = ap.parse_args()
    main(a.package, a.template, a.rawdir)
