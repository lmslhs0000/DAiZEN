#!/bin/bash
cd "$(dirname "$0")/dn"
for nat in domestic export; do
for y in $(seq 2026 -1 2016); do for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
  p="$y-$m"; [[ "$p" > "2026-07" ]] && continue
  f="${nat}_$p.html"; [ -s "$f" ] && continue
  code=$(curl -s -A "Mozilla/5.0" -o "$f" -w "%{http_code}" "https://auto.danawa.com/auto/?Work=record&Tab=Model&Nation=$nat&Month=$p-00")
  echo "$nat $p $code $(wc -c < $f)" >> ../dn_log.txt
  if [ "$code" = "429" ]; then echo STOP429 >> ../dn_log.txt; rm -f "$f"; exit 1; fi
  sleep 10
done; done; done
echo DONE >> ../dn_log.txt
