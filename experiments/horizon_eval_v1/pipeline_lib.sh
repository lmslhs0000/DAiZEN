#!/bin/bash
# run_pipeline.sh 가 source 하는 작업 로그 도구(2026-10-24). 쓰기 전에 L(로그 폴더)이 정해져 있어야 한다.
#
# rotate_log <입력 sha8>   실행을 시작할 때 한 번. 지난 q.log 를 q_<시각>.log 로 보관하고 새 q.log 로 시작한다
#                          (예전 FAIL 이 새 실행의 '재료 실패 있음' 검사에 걸리지 않게).
#                          RESUME=1 이고 지난 실행(driver.log 마지막 begin)의 입력 sha 가 같으면, 보관한 로그를 PREV 로 기억해
#                          그때 OK(또는 SKIP)로 끝난 똑같은 명령은 run 이 다시 돌리지 않는다. sha 가 다르면 이어하기 거부(처음부터).
# run <명령...>            작업 하나: START → 실행 → OK/FAIL 기록. 이어하기면 끝났던 명령은 SKIP 만 기록.

rotate_log() {
  PREV=""
  local cur="${1:-}" ts prev_sha
  ts=$(date '+%Y%m%d_%H%M%S')
  if [ "${RESUME:-0}" = 1 ]; then
    prev_sha=$(grep -a -o "begin [a-z]* (sha [0-9a-f]*)" "$L/driver.log" 2>/dev/null | tail -1 | grep -o "sha [0-9a-f]*" | cut -c5-)
    if [ -n "$cur" ] && [ "$prev_sha" != "$cur" ]; then
      echo "$(date '+%F %T') 이어하기 거부: 지난 실행 입력 sha=${prev_sha:-없음}, 지금 sha=$cur → 처음부터" >> "$L/driver.log"
      RESUME=0
    fi
  fi
  if [ -f "$L/q.log" ]; then
    mv "$L/q.log" "$L/q_$ts.log"
    [ "${RESUME:-0}" = 1 ] && PREV="$L/q_$ts.log"
  fi
  [ "${RESUME:-0}" = 1 ] && echo "$(date '+%F %T') 이어하기: 지난 로그 ${PREV:-없음} 에서 끝난 작업은 건너뜀" >> "$L/driver.log"
  return 0
}

done_before() {
  [ -n "${PREV:-}" ] || return 1
  cut -c21- "$PREV" | grep -qxF -e "OK $*" -e "SKIP $*"
}

run() {
  if done_before "$@"; then
    echo "$(date '+%F %T') SKIP $*" >> "$L/q.log"
    return 0
  fi
  echo "$(date '+%F %T') START $*" >> "$L/q.log"
  "$@" >> "$L/q.out" 2>&1 && echo "$(date '+%F %T') OK $*" >> "$L/q.log" || echo "$(date '+%F %T') FAIL $*" >> "$L/q.log"
}
