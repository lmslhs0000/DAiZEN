#!/bin/bash
# 기준월 5개(+확인용)에서 재료를 학습하고 챔피언 레시피로 조립해 채점한다(15시간 이상, Git Bash). 점수를 다시 매길 때만.
cd "$(dirname "$0")"
bash experiments/horizon_eval_v1/run_pipeline.sh backtest
