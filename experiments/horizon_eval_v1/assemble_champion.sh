#!/bin/bash
# 인계용: 지금 자료(SHA 앞 8자리 = $1)로 학습된 최신 백테스트 run들로 챔피언 레시피를 조립해 보고용 채점.
#   bash experiments/horizon_eval_v1/assemble_champion.sh 99341ce7
# 레시피(ic_champ, 2026-10-13): 재료 목록 handover_members_{a,b,f}.txt → iqm_many → 중국 밖 nv(작년 그대로 0.5)
#   → 차종×부분별 메타 가중(stack_meta, 시간 준수 학습) → 단종 규칙(과거 12) → 사건 규칙 뒤 부분만 → score.py --report-only(현재 챔피언과)
set -e
SHA=$1; [ -n "$SHA" ] || { echo "사용: assemble_champion.sh <자료 SHA 앞 8자리>"; exit 2; }
cd "$(dirname "$0")/../.."
H=experiments/horizon_eval_v1; R=$H/reports; PY=.venv/Scripts/python.exe
NAIVE=$(for d in $(ls -td $R/naive_member_2*); do [ -f $d/run_summary.json ] && { echo $d; break; }; done)
grep -q "$SHA" $NAIVE/run_summary.json || { echo "naive_member가 지금 자료로 만든 것이 아님 — make_naive_member.py 먼저"; exit 1; }
pick() {  # 접두사 목록 파일 → 지금 자료로 학습된 최신 run 경로들
  for prefix in $(cat "$1"); do
    found=""
    for d in $(ls -td $R/${prefix}_2026* 2>/dev/null); do
      [ -f $d/run_summary.json ] || continue
      if [ "$prefix" = tft_a3 ] || grep -q "$SHA" $d/run_summary.json; then found=$d; break; fi
    done
    [ -n "$found" ] || { echo "없음: $prefix (자료 $SHA)" >&2; exit 1; }
    echo $found
  done
}
mk() {
  local f=$1 name=$2
  $PY -X utf8 -B $H/ensembles.py iqm_many --runs $(pick $f) --name "$name" > /dev/null
  local ens=$(ls -td $R/${name}_2* | head -1)
  $PY -X utf8 -B $H/ensembles.py scale_part --first "$ens" --second $NAIVE \
     --market Canada Europe Germany Japan "South Korea" "United States" --part middle back --weight 0.5 --name "${name}_nv" > /dev/null
  ls -td $R/${name}_nv_2* | head -1
}
A=$(mk $H/handover_members_a.txt pipe_a); B=$(mk $H/handover_members_b.txt pipe_b); F=$(mk $H/handover_members_f.txt pipe_f)
# (ic) 차종×부분별 메타 가중(옛 기준월 조립 hz_*_nv가 있으면 학습 행에 추가)
LG=experiments/lightgbm_v1/.venv/Scripts/python.exe
OLD=""; for t in a b f; do d=$(for x in $(ls -td $R/hz_${t}_nv_2* 2>/dev/null); do [ -f $x/run_summary.json ] && { echo $x; break; }; done); [ -n "$d" ] && OLD="$OLD $d"; done
if [ $(echo $OLD | wc -w) -eq 3 ]; then TR="--train-runs $OLD"; else TR=""; fi
SP=$($LG -X utf8 -B $H/stack_meta.py --runs $A $B $F $TR --origins 2022-07 2023-01 2023-07 2024-01 2024-07 2025-07 --name pipe_meta | tail -1 | tr -d '\r')
D=$($PY -X utf8 -B $H/discontinuation_rule.py --first "$SP" --events experiments/global_lifecycle_v1/research/discontinued_reviewed.csv --name pipe_disc --past-months 12 | tail -1 | tr -d '\r')
EVENTS="$(ls experiments/global_lifecycle_v1/research/events_*.csv) experiments/china_lifecycle_v2/research/events_china_v2_merged.csv experiments/data_audit_v1/events_v3/nonchina/events_v3_nonchina.csv experiments/data_audit_v1/events_v3/china/events_v3_china.csv"
C=$($PY -X utf8 $H/event_rules.py --first "$D" --parts back --events $EVENTS --name pipe_champ | tail -1 | tr -d '\r')
CUR=$($PY -c "import json;print(json.load(open('$H/champion.json',encoding='utf-8'))['current']['run_dir'])" | tr -d '\r' | tr '\\' '/')
O=$($PY -X utf8 -B $H/score.py --runs "$C" "$CUR" --report-only 2>&1 | tail -1 | tr -d '\r')
echo "CHAMP $C"; echo "SCORE $O"
$PY -X utf8 - "$O" <<'PYEOF'
import sys, pandas as pd
s = pd.read_csv(sys.argv[1] + "/scorecard.csv", encoding="utf-8-sig")
c = s[(s.metric == "combined") & (s.part == "main")].pivot_table(index=["scope", "scope_value"], columns="model", values="value")
print(c.round(3).to_string())
PYEOF
echo "정식 판정(챔피언 교체 여부)은: $PY -X utf8 -B $H/score.py --runs $C  → 채택되면 $PY -X utf8 -B $H/score.py --check-origin"
