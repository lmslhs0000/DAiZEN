#!/bin/bash
# = step2_train.py: 원본(+채움)으로 재료 34개를 처음부터 학습하고 조립해 고객 CSV를 만든다(약 4.5시간, Git Bash).
cd "$(dirname "$0")"
bash experiments/horizon_eval_v1/run_pipeline.sh production
