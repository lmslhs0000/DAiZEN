"""여러 달을 차례로 수집 — 시작 달·끝 달을 받아 한 달씩 collect_month.py 를 돌린다.

사용:
  python run_months.py                                  # 자동: 빠진 달·채움 남은 달 ~ 오늘 기준 지난달을 모아 **원본에 반영**
                                                        #   (예: 원본 2026-07까지, 오늘 10월 → 8·9월)
  python run_months.py --no-merge                       # 모으기만(raw/incoming/<달>/ 에 보고서), 원본은 안 바꿈
  python run_months.py --ask                            # 시작 달·끝 달을 물어봄
  python run_months.py --from 2026-08 --to 2026-10      # 8·9·10월 수집 + 반영
  --offline (캐시만) --outdir <시험 폴더> --forecast <직전 납품 예측>

- 원본에 넣지 않고 여러 달을 모으면, 다음 달 점검(대상 계열·직전 달 비교·중국 기준 검사)에 앞 달 수집 결과를
  메모리에서 이어 붙인다(--prior). 그래서 8월을 원본에 안 넣어도 9월 중국 보류 검사가 제대로 돈다.
- 반영(기본, 2026-10-15 사용자 지시)은 기본 폴더(raw/incoming)일 때만 된다. --outdir 를 다른 곳으로 주면 모으기만 한다.
  반영 뒤 재학습: bash experiments/horizon_eval_v1/run_pipeline.sh production (원본+채움 파일과 마지막 달 기준월을 자동으로 씀)
- 끝에 달별 요약(수집·채움·보류·발표 대기)을 표로 출력하고 <outdir>/run_months_<시작>_<끝>.csv 로 남긴다.
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd

import monthly as m
from collect_month import DEFAULT_FORECAST, INCOMING, RAW

HERE = Path(__file__).resolve().parent


def ask(prompt):
    while True:
        v = input(prompt).strip()
        if re.fullmatch(r"\d{4}-\d{2}", v):
            return v
        print("YYYY-MM 형식으로 입력 (예: 2026-08)")


def summary(outdir, month):
    d = Path(outdir) / month
    if not (d / "incoming.csv").exists():
        return {"month": month, "status": "실패(incoming 없음)"}
    inc = pd.read_csv(d / "incoming.csv", encoding="utf-8-sig")
    rep = (d / "report.md").read_text(encoding="utf-8")
    pct = re.search(r"수집 값: \d+행 \(대상 계열 판매량 기준 ([\d.]+)%\)", rep)
    n = lambda f: len(pd.read_csv(d / f, encoding="utf-8-sig")) if (d / f).exists() else 0
    return {"month": month, "status": "ok", "collected": int((inc["value_type"] != m.FILL_TYPE).sum()),
            "filled": int((inc["value_type"] == m.FILL_TYPE).sum()), "collected_volume_pct": float(pct.group(1)) if pct else None,
            "held": n("held_basis_mismatch.csv"), "pending": n("pending.csv"), "missing": n("missing.csv")}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="first")
    ap.add_argument("--to", dest="last")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-merge", dest="merge", action="store_false", help="모으기만 하고 원본은 안 바꿈")
    ap.add_argument("--outdir", default=str(INCOMING))
    ap.add_argument("--forecast", default=str(DEFAULT_FORECAST))
    ap.add_argument("--ask", action="store_true", help="시작 달·끝 달을 직접 입력")
    a = ap.parse_args()
    if a.ask:
        months = m.month_range(ask("시작 달 (YYYY-MM): "), ask("끝 달 (YYYY-MM): "))
    elif a.first or a.last:
        months = m.month_range(a.first or a.last, a.last or a.first)
    else:
        raw_last = str(pd.read_csv(RAW, encoding="utf-8-sig", usecols=["Month"])["Month"].max())
        fp = RAW.parent / "filled_values.csv"
        fill_months = sorted(pd.read_csv(fp, encoding="utf-8-sig", dtype={"Month": str})["Month"].unique()) if fp.exists() else []
        months = m.auto_months(raw_last, pd.Timestamp.today(), fill_months)
        print(f"자동: 원본 마지막 달 {raw_last}, 채움이 남은 달 {fill_months or '없음'}, 오늘 {pd.Timestamp.today():%Y-%m-%d}")
        if not months:
            print("새로 모을 달 없음(지난달까지 원본에 있음)")
            return
    outdir = Path(a.outdir).resolve()
    if a.merge and outdir != INCOMING.resolve():
        print("시험 폴더라 원본 반영은 하지 않음(모으기만)")
        a.merge = False
    print(f"수집할 달: {', '.join(months)} → {outdir}" + (" (달마다 원본 반영)" if a.merge else " (원본은 안 바꿈)"))
    rows = []
    for i, mo in enumerate(months):
        prior = [] if a.merge else [str(outdir / p / "incoming.csv") for p in months[:i]]
        cmd = [sys.executable, str(HERE / "collect_month.py"), "--month", mo, "--outdir", str(outdir),
               "--forecast", a.forecast] + (["--offline"] if a.offline else []) + (["--prior"] + prior if prior else [])
        print(f"\n=== {mo} 수집 ===", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"{mo}: 수집 실패(코드 {r.returncode}) → 여기서 멈춤")
            rows.append({"month": mo, "status": f"실패(코드 {r.returncode})"})
            break
        if a.merge:
            r = subprocess.run([sys.executable, str(HERE / "merge_month.py"), "--month", mo, "--apply"])
            if r.returncode != 0:
                print(f"{mo}: 반영 실패 → 여기서 멈춤")
                rows.append({"month": mo, "status": "반영 실패"})
                break
        rows.append(summary(outdir, mo))
    t = pd.DataFrame(rows)
    t.to_csv(outdir / f"run_months_{months[0]}_{months[-1]}.csv", index=False, encoding="utf-8-sig")
    print("\n" + t.to_string(index=False))


if __name__ == "__main__":
    main()
