#!/bin/bash
# 인계용 전체 파이프라인(2026-10-12). D:\daizen에서 Git Bash로 실행.
#   bash experiments/horizon_eval_v1/run_pipeline.sh production   # 2026-07 기준 재료 34개 재학습 → 조립 → 고객 CSV
#   bash experiments/horizon_eval_v1/run_pipeline.sh backtest     # 기준월 5개(+확인용) 재료 재학습 → 챔피언 조립 → 채점 보고
#   bash experiments/horizon_eval_v1/run_pipeline.sh assemble     # 학습 없이, 이미 있는 production run으로 조립 → 고객 CSV (수 분)
#   bash experiments/horizon_eval_v1/run_pipeline.sh all          # production → backtest
# 동시 작업 K개, 시작 전 가상 메모리 여유 6GB 대기(메모리 부족으로 페이징 파일이 C 드라이브를 채우는 일 방지).
# 로그: handover_logs/<모드>/q.log(작업별 START/OK/FAIL), q.out(표준 출력), driver.log.
set -u
MODE=${1:-assemble}
cd "$(dirname "$0")/../.." || exit 1          # 프로젝트 루트(D:\daizen)
H=experiments/horizon_eval_v1; P=$H/production; R=$H/reports
PY=.venv/Scripts/python.exe; CB=experiments/catboost_v1/.venv/Scripts/python.exe; ETSPY=experiments/ets_v1/.venv/Scripts/python.exe
TFM=experiments/timesfm_v1/.venv/Scripts/python.exe; FT=experiments/chronos_ft_v1/.venv/Scripts/python.exe
# Chronos-2 미세조정 장치: GPU(CUDA)가 있으면 cuda(이 PC·지금까지와 같음), 없으면 cpu(다른 PC; 훨씬 느림)
FTDEV=$($FT -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null | tr -d '\r'); FTDEV=${FTDEV:-cuda}
# 매달 재예측(2026-10-15): production 은 원본 + 예측 채움(raw/filled_values.csv)을 합친 학습 파일을 쓰고,
# 기준월은 그 파일의 마지막 달(예: 9월까지 반영했으면 2026-09). 채움 표가 없으면 예전처럼 raw/data.csv·2026-07.
if [ "$MODE" != backtest ] && [ -z "${DAIZEN_CSV:-}" ] && [ -f raw/filled_values.csv ]; then   # production·assemble·all
  $PY -c "import sys;sys.path.insert(0,'experiments/collectors_v1/pipeline');import pandas as pd,monthly as m;m.training_frame(pd.read_csv('raw/data.csv',encoding='utf-8-sig'),pd.read_csv('raw/filled_values.csv',encoding='utf-8-sig')).to_csv('raw/data_with_fills.csv',index=False,encoding='utf-8-sig')" || exit 1
  export DAIZEN_CSV=raw/data_with_fills.csv
fi
LASTM=$($PY -c "import pandas as pd;print(pd.read_csv('${DAIZEN_CSV:-raw/data.csv}',encoding='utf-8-sig',usecols=['Month']).Month.max())" | tr -d '\r')
O=${ORIGIN:-$LASTM}; K=${K:-6}
L=handover_logs/$MODE; mkdir -p "$L"
CP="--origins $O --no-check --output-base $P"; CB_="--no-check"; W2="--workers 2"
V2=$R/basic_xgboost_v2_20260930T151336_725180Z_357a96          # XGBoost 설정 선택 run(고정)
EVENTS="$(ls experiments/global_lifecycle_v1/research/events_*.csv) experiments/china_lifecycle_v2/research/events_china_v2_merged.csv experiments/data_audit_v1/events_v3/nonchina/events_v3_nonchina.csv experiments/data_audit_v1/events_v3/china/events_v3_china.csv"
DISC_PROD=experiments/global_lifecycle_v1/research/discontinued_reviewed_v2_1.csv
DISC_CN=experiments/china_lifecycle_v2/research/discontinued_china_reviewed_production.csv
DISC_BT=experiments/global_lifecycle_v1/research/discontinued_reviewed.csv
newestp() { for d in $(ls -td $P/$1_2* 2>/dev/null); do [ -f $d/run_summary.json ] && { echo $d; return; }; done; }
newest()  { for d in $(ls -td $R/$1_2* 2>/dev/null); do [ -f $d/run_summary.json ] && { echo $d; return; }; done; }
memwait() { while true; do f=$(powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" 2>/dev/null | tr -d '\r'); [ -n "$f" ] && [ "$f" -ge 6 ] && break; sleep 60; done; }
. "$H/pipeline_lib.sh"   # run·rotate_log: 실행마다 q.log 새로 시작(지난 것은 q_<시각>.log), RESUME=1 이면 끝난 작업 건너뜀(2026-10-24)
slot() { while [ $(jobs -rp | wc -l) -ge $K ]; do sleep 30; done; memwait; sleep 30; }
log()  { echo "$(date '+%F %T') $*" | tee -a "$L/driver.log"; }
sha8() { $PY -c "import hashlib;print(hashlib.sha256(open('${DAIZEN_CSV:-raw/data.csv}','rb').read()).hexdigest()[:8])"; }
rotate_log "$(sha8 | tr -d '\r')"

NN_ARGS=("--method A" "--method A --variant seed43" "--method A --variant seed44" "--method A --variant seed45" "--method A --variant seed46"
         "--global --powertrain" "--global --powertrain --seed 43" "--global --powertrain --seed 44" "--global --deepar"
         "--global --nhits" "--global --nhits --seed 43" "--global --nhits --seed 44" "--global --nhits-vw"
         "--family nhits_pm" "--family nhits_pm --seed 43" "--family nhits_pm --seed 44")
RICH=(calendar_l2 calendar_vw calendar_vw_all tweedie_all); CBV=(catboost_all catboost_rel_all catboost_yoy_lcg catboost_yoy_lcg_s43 catboost_yoy_lcg_s44)

train_members() {   # $1 = 공통 인자(CP 또는 CB_), $2 = production|backtest
  local C="$1" kind="$2"
  if [ "$kind" = production ]; then
    ( run $FT $H/forecast_chronos.py --finetune --device $FTDEV $C
      run $FT $H/forecast_chronos.py --from-run $(newestp chronos2_ft) --lifecycle --device cpu $C
      run $PY $H/ensembles.py route_market --first $(newestp chronos2_ft_lc) --second $(newestp chronos2_ft) --first-markets China --name chronos2_ft_cnlc --output-base $P
      run $FT $H/short_chronos.py --from-run $(newestp chronos2_ft) --device $FTDEV --origins $O --output-base $P ) &
  else
    ( run $FT $H/forecast_chronos.py --finetune --device $FTDEV $C
      run $FT $H/forecast_chronos.py --from-run $(newest chronos2_ft) --lifecycle --device cpu $C
      run $PY $H/ensembles.py route_market --first $(newest chronos2_ft_lc) --second $(newest chronos2_ft) --first-markets China --name chronos2_ft_cnlc ) &
  fi
  for a in "${NN_ARGS[@]}"; do slot; run $PY $H/forecast_tft.py $a $W2 $C & done
  slot; run $ETSPY $H/forecast_ets.py --workers 2 $C &
  slot; run $TFM $H/forecast_chronos.py --timesfm $C &
  if [ "$kind" = production ]; then slot; run $PY $H/xgb_cal.py --kind blvw --v2-run $V2 $C &
  else slot; ( run $PY $H/xgb_cal.py --kind blvw --v2-run $V2 $C; run $PY $H/challengers.py $(newest basic_xgboost_blvw) --only combo_horizon --suffix _blvw ) & fi
  for v in "${RICH[@]}"; do slot; run $PY $H/forecast_rich49.py --variant $v $C & done
  slot; run $PY $H/forecast_rich49.py --variant tweedie_all --per-market $C &
  for v in "${CBV[@]}"; do slot; run $CB $H/forecast_rich49.py --variant $v $C & done
  slot; run $CB $H/forecast_rich49.py --variant catboost_all --per-market $C &
  slot; run $CB $H/forecast_rich49.py --variant catboost_yoy_all --per-market $C &
  wait
}

assemble_production() {   # 이미 있는 production run(가장 최신)으로 조립 → 고객 CSV
  local MK="Canada Europe Germany Japan"
  local BASE="--tft-run $(newestp tft_a) --v2-run $V2 --xgb-run $(newestp basic_xgboost_blvw) --rich-run $(newestp rich_cal) --ets-run $(newestp ets) --extra-run $(newestp tft_global_pt) --seed-runs $(newestp tft_a_s43) $(newestp tft_a_s44) --extra2-run $(newestp rich_tw_all) --origin $O"
  local MORE="$(newestp nhits_global) $(newestp deepar_global) $(newestp nhits_vw) $(newestp rich_cal_vw) $(newestp nhits_pm) $(newestp chronos2_ft_cnlc) $(newestp rich_cal_vw_all) $(newestp timesfm) $(newestp rich_cb_all) $(newestp rich_cb_pm_all) $(newestp rich_tw_pm_all) $(newestp rich_cb_yoy_lcg) $(newestp rich_cb_yoy_pm) $(newestp rich_cb_rel)"
  local SEEDB="$(newestp rich_cb_yoy_lcg_s43) $(newestp rich_cb_yoy_lcg_s44)"
  local SEEDF="$(newestp tft_a_s45) $(newestp tft_a_s46) $(newestp nhits_global_s43) $(newestp nhits_global_s44) $(newestp nhits_pm_s43) $(newestp nhits_pm_s44) $(newestp tft_global_pt_s43) $(newestp tft_global_pt_s44)"
  log "assemble: BASE=$BASE"; log "MORE=$MORE"; log "SEEDB=$SEEDB SEEDF=$SEEDF"
  local A B F ANV BNV FNV LOO ES FQ FR CN HQ KP
  need() { [ -d "$2" ] || { log "조립 실패: $1 결과 폴더 없음 — 로그의 Traceback 확인"; exit 1; }; }   # 2026-10-16: 조용히 넘어가지 않게
  A=$($PY $H/deliver.py --recipe ens20_lrel_cn $BASE --more-runs $MORE | tail -1 | tr -d '\r')
  B=$($PY $H/deliver.py --recipe ens22_seed_cn $BASE --more-runs $MORE $SEEDB | tail -1 | tr -d '\r')
  F=$($PY $H/deliver.py --recipe ens28_front_cn $BASE --more-runs $MORE $SEEDF | tail -1 | tr -d '\r')
  ANV=$($PY $H/postprocess_delivery.py --delivery "$A" --reference naive --name ens20_lrel_cn_nv --market $MK "South Korea" "United States" --part middle back --weight 0.5 | tail -1 | tr -d '\r')
  BNV=$($PY $H/postprocess_delivery.py --delivery "$B" --reference naive --name ens22_seed_cn_nv --market $MK "South Korea" "United States" --part middle back --weight 0.5 | tail -1 | tr -d '\r')
  FNV=$($PY $H/postprocess_delivery.py --delivery "$F" --reference naive --name ens28_front_cn_nv --market $MK "South Korea" "United States" --part middle back --weight 0.5 | tail -1 | tr -d '\r')
  # (ic) 차종×부분별 메타 가중: 백테스트 조립 run(ws_*_nv + 옛 기준월 hz_*_nv)이 있으면 새로 학습, 없으면 저장된 가중치 표(meta_weights_2026-07.csv) 사용
  local LG=experiments/lightgbm_v1/.venv/Scripts/python.exe WA WB WF OA OB OF ES
  WA=$(newest ws_a_nv); WB=$(newest ws_b_nv); WF=$(newest ws_f_nv); OA=$(newest hz_a_nv); OB=$(newest hz_b_nv); OF=$(newest hz_f_nv)
  if [ -x $LG ] && [ -n "$WA" ] && [ -n "$OA" ]; then
    ES=$($LG -X utf8 -B $H/stack_meta.py --runs $WA $WB $WF --train-runs $OA $OB $OF --deliveries "$ANV" "$BNV" "$FNV" --origins $O --name final_pipeline_meta | tail -1 | tr -d '\r')
  else
    ES=$($PY -X utf8 -B $H/stack_meta.py --deliveries "$ANV" "$BNV" "$FNV" --weights $H/meta_weights_2026-07.csv --origins $O --name final_pipeline_meta | tail -1 | tr -d '\r')
  fi
  FQ=$($PY $H/discontinuation_rule.py --delivery "$ES" --events $DISC_PROD --name fqfc_v21 --past-months 12 | tail -1 | tr -d '\r')
  $PY $H/final_delivery.py --delivery "$FQ" --short-run $(newestp short_chronos) --chronos-run $(newestp chronos2_ft) --name final_pipeline > /dev/null
  FR=$(ls -td $P/delivery_final_pipeline_2* | head -1)
  CN=$($PY -X utf8 $H/discontinuation_rule.py --delivery "$FR" --events $DISC_CN --include-china --past-months 12 --name final_pipeline_cn --output-base $P | tail -1 | tr -d '\r')
  HQ=$($PY -X utf8 $H/event_rules.py --delivery "$CN" --parts back --events $EVENTS --name final_pipeline_hq3 | tail -1 | tr -d '\r')
  KP=$($PY -X utf8 $H/package_july.py "$HQ" client_package_pipeline | tail -1 | tr -d '\r')
  [ -f "$KP/client_forecast_24m.csv" ] || { log "고객 파일 묶기 실패(package_july) — 로그의 Traceback 확인. 조립 중간 결과: $HQ"; exit 1; }
  # 고객용 설명서: 7월 판 원문을 이번 기준월·채운 칸에 맞춰 고쳐 넣음(실패하면 원문 그대로 복사)
  local TPL="$P/client_package_july_ic_20261013T035849_323513Z_9f3a75/README_고객용_설명서.md"
  $PY -X utf8 $H/delivery_note.py "$KP" --template "$TPL" --rawdir raw >> "$L/driver.log" 2>&1 \
    || { log "설명서 만들기 실패(delivery_note.py) — 7월 판 원문 복사, driver.log 확인"; cp "$TPL" "$KP/" 2>/dev/null; }
  log "고객 CSV: $KP/client_forecast_24m.csv"
  # 납품 목록(raw/deliveries.csv)에 등록 — 다음 달 빈칸 채우기와 매달 성적 확인이 이 예측을 쓴다(production 만)
  $PY -X utf8 experiments/collectors_v1/pipeline/deliveries.py add "$KP" >> "$L/driver.log" 2>&1 || log "납품 목록 등록 실패(deliveries.py) — driver.log 확인"
}

case "$MODE" in
  production|all)
    log "begin production (sha $(sha8))"
    run $PY -X utf8 $H/make_naive_member.py
    train_members "$CP" production
    grep -q FAIL "$L/q.log" && { log "재료 실패 있음 — q.log 확인 뒤 다시"; exit 1; }
    assemble_production
    [ "$MODE" = all ] && exec bash "$0" backtest
    ;;
  assemble)
    log "assemble only"
    [ -n "$(newest naive_member)" ] || run $PY -X utf8 $H/make_naive_member.py
    assemble_production
    ;;
  backtest)
    log "begin backtest (sha $(sha8))"
    [ -n "$(newest naive_member)" ] || run $PY -X utf8 $H/make_naive_member.py
    train_members "$CB_" backtest                                   # 기준월 5개
    train_members "--origins" backtest                              # 확인용 기준월만(--origins 빈 목록 = 2025-07만)
    grep -q FAIL "$L/q.log" && { log "재료 실패 있음 — q.log 확인 뒤 다시"; exit 1; }
    SHA=$(sha8)
    $PY -X utf8 $H/join_all.py $SHA $(cat $H/handover_members_join.txt) >> "$L/q.log" 2>&1 || { log "join 실패"; exit 1; }
    # 중국만 수명 공변량판을 쓰는 재료(chronos2_ft_cnlc)는 합친 lc·ft run 으로 다시 만든다 — 안 하면 확인용 기준월만 든 cnlc 가 조립에 들어가
    # stack_meta 가 빈 표로 멈춘다(2026-10-25 줄인 묶음 백테스트 시험에서 발견)
    $PY $H/ensembles.py route_market --first $(newest chronos2_ft_lc) --second $(newest chronos2_ft) --first-markets China --name chronos2_ft_cnlc >> "$L/q.out" 2>&1 || { log "cnlc 다시 만들기 실패"; exit 1; }
    # 같은 이유로 combo_horizon_blvw(합친 basic_xgboost_blvw 에서 만드는 파생 재료)도 다시 만든다
    $PY $H/challengers.py $(newest basic_xgboost_blvw) --only combo_horizon --suffix _blvw >> "$L/q.out" 2>&1 || { log "combo_horizon_blvw 다시 만들기 실패"; exit 1; }
    run $PY $H/ensembles.py average_many --runs $(newest tft_a) $(newest tft_a_s43) $(newest tft_a_s44) --name tft_a3
    bash $H/assemble_champion.sh "$SHA" > "$L/champion_score.txt" 2>&1
    log "채점 결과: $L/champion_score.txt"
    ;;
  *) echo "모드: production | backtest | assemble | all"; exit 2;;
esac
log "END"
