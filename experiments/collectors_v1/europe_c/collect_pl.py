"""폴란드 CEPiK 신차 승용 → 제조사×모델 월 대수. **네트워크 요청을 하지 않는다.**

사용: python collect_pl.py --month 2026-07

왜 네트워크를 안 쓰나:
  - api.cepik.gov.pl 은 2026-10-15 HTTP 429 로 막혔다(europe_b/cache/pl/BLOCKED_429.txt). 이 폴더는 그 호스트를 FORBIDDEN 으로 둔다.
  - SPEC2 가 권한 dane.gov.pl 데이터셋 1558 은 확인해 보니 **CSV 묶음이 아니라 CEPiK API 링크 모음**이다
    (자료 20개 = API 문서·사전·도별 /pojazdy 질의 링크, 그리고 '파일 정보' https://api.cepik.gov.pl/pliki).
    대량 파일도 api.cepik.gov.pl(/pliki) 에 있어 같은 막힌 호스트 → 받지 않았다(2026-10-15 api.dane.gov.pl 메타데이터 2회만 조회).
입력(있는 것부터):
  1) cache/pl/bulk/ 아래 사람이 받아 둔 CEPiK 대량 CSV(들). 열 이름은 API 속성과 같다고 가정
     (marka, model, rodzaj-pojazdu, pochodzenie-pojazdu, data-pierwszej-rejestracji-w-kraju; '-' 대신 '_' 도 허용).
     필터: rodzaj = 'SAMOCHÓD OSOBOWY', pochodzenie ∈ {'NOWY ZAKUPIONY W KRAJU', 'NOWY IMPORT INDYW'},
     최초 국내 등록일이 그 달. → 파일 형식은 받아 보지 못해 **검증 안 됨(설계)**.
  2) europe_b 가 API 로 받아 둔 쪽 캐시의 집계(europe_b/out/pl_<YYYY-MM>_by_model.csv + _meta.json).
     2026-07 은 16개 도 중 12개(국내 신차) + 14개(개인 수입 신차)만 있음 → meta.complete=false.
출력 meta.complete=false 이면 collect_europe.py 는 폴란드 값을 쓰지 않고 폴란드 몫을 추정한다(부분 합을 전국으로 쓰지 않음).
"""
from __future__ import annotations

import argparse
import glob
import json

import pandas as pd

from c_common import CACHE, EB, PL_BLOCK_FILE, check_month, write_model_month

D = CACHE / "pl"
(D / "bulk").mkdir(parents=True, exist_ok=True)
NEW = {"NOWY ZAKUPIONY W KRAJU", "NOWY IMPORT INDYW"}
WOJ = ["02", "04", "06", "08", "10", "12", "14", "16", "18", "20", "22", "24", "26", "28", "30", "32"]


def from_bulk(month: str):
    files = sorted(glob.glob(str(D / "bulk" / "*.csv")))
    if not files:
        return None
    parts = []
    for f in files:
        x = pd.read_csv(f, dtype=str, sep=None, engine="python")
        x.columns = [c.strip().lower().replace("_", "-") for c in x.columns]
        need = {"marka", "model", "rodzaj-pojazdu", "pochodzenie-pojazdu", "data-pierwszej-rejestracji-w-kraju"}
        if not need <= set(x.columns):
            raise SystemExit(f"{f}: 열 없음 {sorted(need - set(x.columns))} (형식 확인 필요)")
        x = x[(x["rodzaj-pojazdu"] == "SAMOCHÓD OSOBOWY") & x["pochodzenie-pojazdu"].isin(NEW)
              & x["data-pierwszej-rejestracji-w-kraju"].astype(str).str.startswith(month)]
        if "id" in x.columns:
            x = x.drop_duplicates("id")
        parts.append(x)
    x = pd.concat(parts)
    d = x.groupby(["marka", "model"], dropna=False).size().reset_index(name="units")
    meta = dict(source_id="PL_CEPIK_BULK", files=files, complete=True,
                note="사람이 받아 둔 대량 파일; 전국 파일인지(도 16개)는 파일 목록으로 확인할 것")
    return d.rename(columns={"marka": "make"}), meta


def from_europe_b(month: str):
    p = EB / "out" / f"pl_{month}_by_model.csv"
    mp = EB / "out" / f"pl_{month}_meta.json"
    if not p.exists():
        return None
    x = pd.read_csv(p, keep_default_na=False)
    meta_b = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    d = x.groupby(["marka", "model"], as_index=False)["n"].sum().rename(columns={"marka": "make", "n": "units"})
    failed = meta_b.get("failed", [])
    complete = (not failed) and sorted(meta_b.get("woj_done", [])) == WOJ and meta_b.get("days") == [1, 31]
    meta = dict(source_id="PL_CEPIK_API_CACHE(europe_b)", file=str(p), complete=bool(complete),
                missing_parts=failed, records=meta_b.get("records"),
                note="europe_b 의 API 쪽 캐시 집계(이 스크립트는 요청하지 않음)")
    return d, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    a = ap.parse_args()
    month = check_month(a.month)
    r = from_bulk(month) or from_europe_b(month)
    if r is None:
        raise SystemExit(f"{month}: 폴란드 입력 없음(cache/pl/bulk/*.csv 또는 europe_b 캐시). API 는 요청하지 않음"
                         + (" — CEPiK 429 차단 파일 있음" if PL_BLOCK_FILE.exists() else ""))
    d, meta = r
    meta["cepik_block_file"] = PL_BLOCK_FILE.exists()
    out = d.assign(country="Poland", month=month)[["country", "month", "make", "model", "units"]]
    print(write_model_month("pl", month, out[out["units"] > 0], meta))


if __name__ == "__main__":
    main()
