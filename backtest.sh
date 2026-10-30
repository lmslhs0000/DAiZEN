#!/bin/bash
# 기준월 5개(+확인용)에서 재료를 학습하고 챔피언 레시피로 조립해 채점한다(약 18시간·GPU 실측, Git Bash; 끊기면 RESUME=1 로 이어서). 점수를 다시 매길 때만.
cd "$(dirname "$0")"
bash experiments/horizon_eval_v1/run_pipeline.sh backtest
