"""raw/incoming/<M>/incoming.csv 를 원본에 반영 — 사용자가 report.md 를 보고 확인한 뒤에만 --apply.

규칙(2026-10-15 사용자):
- 실제 값·분기/반기/연 나눔·배분·추정 몫 30% 미만 → raw/data.csv.
- 추정 몫 30% 이상(fill_type=estimated)·직전 예측(fill_type=forecast) → raw/filled_values.csv (원본과 섞지 않음, 채점 제외).
- 원본에 이미 있는 칸은 다시 조사하지 않는다(그대로 둠). 채움에 있던 칸만 다시 보고, 실제 값이 나오면 원본으로 옮긴다.
- 반영 전 raw/backups/ 에 백업, raw/monthly_updates.csv 에 해시·건수 기록.
- 학습 파일 raw/data_with_fills.csv(원본+채움)를 새로 만든다 → 재학습 때 DAIZEN_CSV 로 지정.

사용: python merge_month.py --month 2026-08            # 미리보기만
      python merge_month.py --month 2026-08 --apply    # 반영
"""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import shutil
import sys

import pandas as pd

import monthly as m

ROOT = Path(__file__).resolve().parents[3]
RAWDIR = ROOT / "raw"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--rawdir", default=str(RAWDIR))
    a = ap.parse_args()
    rawdir = Path(a.rawdir)
    data, fills_path = rawdir / "data.csv", rawdir / "filled_values.csv"
    inc = pd.read_csv(rawdir / "incoming" / a.month / "incoming.csv", encoding="utf-8-sig")
    raw = pd.read_csv(data, encoding="utf-8-sig")
    fills = pd.read_csv(fills_path, encoding="utf-8-sig") if fills_path.exists() else m.empty_fills()
    stamp = datetime.now().strftime("%Y-%m-%d")
    new_raw, new_fills, log = m.merge(raw, inc, fills, filled_at=stamp)
    streak = m.fill_streaks(new_fills)
    alarm = streak[streak["months"] >= 3]
    print(log.groupby(["destination", "action"]).size().to_string())
    print(f"원본 {len(raw)} → {len(new_raw)}행, 채움 표 {len(fills)} → {len(new_fills)}행, 3개월 넘게 연속 채움 {len(alarm)}계열")
    if not a.apply:
        print("미리보기만 했다. 반영하려면 --apply")
        return
    before = sha(data)
    backup = rawdir / "backups" / f"data_before_{a.month}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    backup.parent.mkdir(exist_ok=True)
    shutil.copy2(data, backup)
    new_raw.to_csv(data, index=False, encoding="utf-8-sig")
    new_fills.to_csv(fills_path, index=False, encoding="utf-8-sig")
    log.to_csv(rawdir / "incoming" / a.month / "merge_log.csv", index=False, encoding="utf-8-sig")
    alarm.to_csv(rawdir / "incoming" / a.month / "fill_alarm.csv", index=False, encoding="utf-8-sig")
    m.training_frame(new_raw, new_fills).to_csv(rawdir / "data_with_fills.csv", index=False, encoding="utf-8-sig")
    upd = rawdir / "monthly_updates.csv"
    row = pd.DataFrame([{"month": a.month, "applied_at": datetime.now().isoformat(timespec="seconds"),
                         "sha_before": before, "sha_after": sha(data), "backup": backup.name,
                         **log["action"].value_counts().to_dict(), "fill_alarm": len(alarm)}])
    (pd.concat([pd.read_csv(upd, encoding="utf-8-sig"), row]) if upd.exists() else row).to_csv(upd, index=False, encoding="utf-8-sig")
    import monitor   # 매달 성적 확인(지난 예측 vs 이번에 들어온 실제)
    hist, al = monitor.run(data, rawdir / "monitoring.csv", rawdir / "monitoring.md")
    shutil.copy2(rawdir / "monitoring.md", rawdir / "incoming" / a.month / "monitoring.md")
    print(f"성적 확인: {len(hist)}줄, 경보 {len(al)}개 → raw/monitoring.md")
    for r in al.itertuples():
        print(f"  ⚠ {r.Country} {r.alarm}: {r.detail}")
    print(f"반영 완료. 백업 {backup}. 재학습: DAIZEN_CSV=raw/data_with_fills.csv ORIGIN={new_raw["Month"].astype(str).max()} bash experiments/horizon_eval_v1/run_pipeline.sh production")


if __name__ == "__main__":
    main()
