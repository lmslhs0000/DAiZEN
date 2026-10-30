"""1단계: 자료 모으기 + 원본(raw/data.csv) 반영. VS Code 에서 이 파일을 열고 ▶ 로 실행(약 1시간, 인터넷 필요).

오늘 날짜를 보고 원본 마지막 달 다음 달 ~ 지난달(그리고 예측으로 채워 둔 달)을 모은다.
실제 숫자 → raw/data.csv, 못 구해 예측으로 채운 값 → raw/filled_values.csv. 반영 전 raw/backups/ 에 백업.
모으기만(원본 안 바꿈): 터미널에서 .venv/Scripts/python.exe step1_collect.py --no-merge
보고서: raw/incoming/<달>/report.md, 로그: handover_logs/monthly/collect_<시각>.log
"""
import sys
from monthly_common import ROOT, PY, run, new_log

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    log = new_log("collect")
    code = run([PY, "experiments/collectors_v1/pipeline/run_months.py"] + sys.argv[1:], log)
    print(f"\n끝(코드 {code}). 다음: step2_train.py 로 재학습")
    sys.exit(code)
