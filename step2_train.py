"""2단계: 재학습 + 고객 예측 파일. VS Code 에서 이 파일을 열고 ▶ 로 실행(약 5시간, 그동안 VS Code 를 켜 둘 것).

원본 + 예측 채움을 합친 학습 파일로 모델 34개를 다시 학습하고, 기준월(원본 마지막 달)부터 24개월 예측을 만든다.
안에서는 기존 run_pipeline.sh production 을 그대로 부른다(파이썬 환경 5개·동시 6개 작업; Git Bash 필요).
결과: experiments/horizon_eval_v1/production/client_package_*pipeline*/client_forecast_24m.csv
로그: handover_logs/monthly/train_<시각>.log, 작업별 성공·실패: handover_logs/production/q.log
"""
import sys
from monthly_common import ROOT, find_bash, run, new_log

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    log = new_log("train")
    code = run([find_bash(), "experiments/horizon_eval_v1/run_pipeline.sh", "production"], log)
    pk = sorted((ROOT / "experiments/horizon_eval_v1/production").glob("client_package*pipeline*"), key=lambda p: p.stat().st_mtime)
    print(f"\n끝(코드 {code}). 결과: {pk[-1] / 'client_forecast_24m.csv' if pk else '없음'}")
    print("작업별 성공·실패: handover_logs/production/q.log")
    sys.exit(code)
