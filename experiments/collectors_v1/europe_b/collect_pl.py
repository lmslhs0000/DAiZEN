"""폴란드 CEPiK API → 한 달 신규 승용차 등록 대수(제조사 × 모델 × 연료) → 우리 Europe 시계열 대응.

API: https://api.cepik.gov.pl/pojazdy  (문서 https://api.cepik.gov.pl/doc, robots.txt 없음)
 - 필수: wojewodztwo(도 코드 16개), data-od/data-do(YYYYMMDD), typ-daty=1(국내 최초 등록일)
 - filter[rodzaj-pojazdu]=SAMOCHÓD OSOBOWY (승용차), filter[pochodzenie-pojazdu]=신차 값만
   · 'NOWY ZAKUPIONY W KRAJU' (국내 구입 신차) · 'NOWY IMPORT INDYW' (개인 수입 신차)
 - tylko-zarejestrowane=false: 그 뒤 말소된 차도 그 달 등록으로 센다(true 면 재수출·폐차분이 빠짐)
 - limit=500, page=1.. (meta.count 로 쪽 수 계산)
 - 서버가 DH 키가 작아 TLS SECLEVEL=1 필요(_common._LegacyTLS)

요청 수(2026-10-15 측정): 한 도·한 달·신차 필터 1쪽 응답 13~18초. 2026-07 Wielkopolskie 국내 신차 4,874대 = 10쪽.
전국 한 달 ≈ 신차 5만~6만 대 → 약 110~130쪽 + 수입 신차 16회 ≈ 150 요청, 요청 간 2초 + 응답 15초 → 약 40분.
그래서 하루씩이 아니라 '도 × 한 달' 범위로 받는다(날짜별로 쪼개면 16×31×쪽 → 500회 이상).

사용:
  python collect_pl.py --month 2026-07                 # 전국 16개 도
  python collect_pl.py --month 2026-07 --woj 30 14     # 일부 도만(시험)
  python collect_pl.py --month 2026-07 --days 1 3      # 1~3일만(시험)
결과: cache/pl/<월>/<도>_<출처>.csv (원 레코드 요약, 다시 받지 않음), out/pl_<월>_by_model.csv, out/pl_<월>_ours.csv
"""
from __future__ import annotations

import argparse
import calendar
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

from _common import CACHE, HERE, HostBlocked, PoliteSession
from europe_map import classify_frame

API = "https://api.cepik.gov.pl/pojazdy"
WOJ = {"02": "DOLNOŚLĄSKIE", "04": "KUJAWSKO-POMORSKIE", "06": "LUBELSKIE", "08": "LUBUSKIE", "10": "ŁÓDZKIE",
       "12": "MAŁOPOLSKIE", "14": "MAZOWIECKIE", "16": "OPOLSKIE", "18": "PODKARPACKIE", "20": "PODLASKIE",
       "22": "POMORSKIE", "24": "ŚLĄSKIE", "26": "ŚWIĘTOKRZYSKIE", "28": "WARMIŃSKO-MAZURSKIE",
       "30": "WIELKOPOLSKIE", "32": "ZACHODNIOPOMORSKIE"}
ORIGINS = {"NKR": "NOWY ZAKUPIONY W KRAJU", "NIM": "NOWY IMPORT INDYW"}
KEEP = ["marka", "model", "rodzaj-paliwa", "pochodzenie-pojazdu", "data-pierwszej-rejestracji-w-kraju", "wojewodztwo-kod"]
OUT = HERE / "out"
BLOCK = CACHE / "pl" / "BLOCKED_429.txt"


class PagingError(RuntimeError):
    pass


def fetch_woj(sess: PoliteSession, woj: str, origin_key: str, d_from: str, d_to: str, log) -> pd.DataFrame:
    """한 도·한 출처·기간. 서버 쪽 나눔이 불안정하면(중간 쪽 404, 큰 도의 0건) 기간을 반으로 나눠 다시 받는다.

    2026-10-15 관측: Mazowieckie·Śląskie 한 달 범위에서 5·3쪽째에 404, Łódzkie 한 달 0건(같은 날 다른 도는 정상).
    """
    d0, d1 = pd.Timestamp(d_from), pd.Timestamp(d_to)
    try:
        df = _fetch_range(sess, woj, origin_key, d_from, d_to, log)
        suspicious = origin_key == "NKR" and len(df) == 0 and (d1 - d0).days >= 6
        if not suspicious:
            return df
        log(f"  {woj} {origin_key} {d_from}-{d_to}: 0건 의심 → 기간 나눠 다시 확인")
    except PagingError as e:
        log(f"  {woj} {origin_key} {d_from}-{d_to}: {e} → 기간 나눔")
    if d0 == d1:
        raise PagingError(f"{woj} {origin_key} {d_from}: 하루 범위에서도 실패")
    mid = d0 + (d1 - d0) / 2
    mid = mid.normalize()
    a = fetch_woj(sess, woj, origin_key, d0.strftime("%Y%m%d"), mid.strftime("%Y%m%d"), log)
    b = fetch_woj(sess, woj, origin_key, (mid + pd.Timedelta(days=1)).strftime("%Y%m%d"), d1.strftime("%Y%m%d"), log)
    return pd.concat([a, b], ignore_index=True)


def _fetch_range(sess: PoliteSession, woj: str, origin_key: str, d_from: str, d_to: str, log) -> pd.DataFrame:
    """한 도·한 출처·기간의 모든 쪽을 받아 요약 레코드로 돌려준다(캐시)."""
    tag = f"{d_from}_{d_to}"
    cdir = CACHE / "pl" / tag
    cdir.mkdir(parents=True, exist_ok=True)
    cfile = cdir / f"{woj}_{origin_key}.csv"
    if cfile.exists():
        return pd.read_csv(cfile, dtype=str, keep_default_na=False)
    params = {"wojewodztwo": woj, "data-od": d_from, "data-do": d_to, "typ-daty": "1",
              "tylko-zarejestrowane": "false", "limit": "500", "page": "1",
              "filter[rodzaj-pojazdu]": "SAMOCHÓD OSOBOWY", "filter[pochodzenie-pojazdu]": ORIGINS[origin_key]}
    # 주의(2026-10-15 관측): 첫 요청의 meta.count 가 501 처럼 임시값으로 나올 때가 있어 count 만 믿지 않는다.
    # 쪽이 꽉 차(500행) 있으면 다음 쪽을 계속 받고, 500행 미만이 나오면 끝낸다. 응답이 중간에 잘리는(JSON 오류) 경우 재시도.
    # 쪽 단위 캐시: 429·끊김 뒤 다시 실행하면 받은 쪽은 건너뛴다(쪽 순서가 세션마다 같다는 보장은 없어 끝에 id 중복 제거·count 대조)
    rows, page, count = [], 1, 0
    while True:
        params["page"] = str(page)
        pfile = cdir / f"{woj}_{origin_key}_p{page:03d}.json"
        if pfile.exists():
            pj = json.loads(pfile.read_text(encoding="utf-8"))
            rows.extend(pj["rows"])
            count = max(count, pj["count"])
            if pj["n_data"] < 500:
                break
            page += 1
            continue
        j = None
        for attempt in range(4):
            r = sess.get(API, params=params, timeout=300)
            if r.status_code == 404 and page == 1:
                break
            if r.status_code == 404:  # 중간 쪽 404: 잠시 뒤 한 번 더, 그래도면 기간 나눔
                if attempt == 0:
                    log(f"  {woj} {origin_key} p{page} HTTP 404 → 20초 뒤 재시도")
                    time.sleep(20)
                    continue
                raise PagingError(f"p{page} HTTP 404")
            if r.status_code in (500, 502, 503, 504):
                log(f"  {woj} {origin_key} p{page} HTTP {r.status_code} → 재시도 {attempt + 1}")
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            try:
                j = r.json()
                break
            except ValueError:
                log(f"  {woj} {origin_key} p{page} JSON 잘림({len(r.content)}B) → 재시도 {attempt + 1}")
                time.sleep(10 * (attempt + 1))
        if r.status_code == 404 and page == 1:  # 해당 조건 자료 없음
            log(f"  {woj} {origin_key}: 404(자료 없음) {r.text[:120]}")
            break
        if j is None:
            raise RuntimeError(f"{woj} {origin_key} p{page}: 재시도 후에도 응답 불완전")
        count = max(count, int(j.get("meta", {}).get("count", 0) or 0))
        data = j.get("data", [])
        prow = [{"id": x.get("id"), **{k: (x.get("attributes") or {}).get(k) for k in KEEP}} for x in data]
        rows.extend(prow)
        pfile.write_text(json.dumps({"count": int(j.get("meta", {}).get("count", 0) or 0), "n_data": len(data),
                                     "rows": prow}, ensure_ascii=False), encoding="utf-8")
        if page == 1 or page % 5 == 0:
            log(f"  {woj} {WOJ[woj]} {origin_key}: p{page} rows={len(rows)} count={count}")
        if len(data) < 500:
            break
        page += 1
    df = pd.DataFrame(rows, columns=["id"] + KEEP).drop_duplicates("id")
    log(f"  {woj} {origin_key}: 끝 pages={page} 고유 id={df['id'].nunique()} count={count}")
    if count and df["id"].nunique() != count:
        log(f"  경고 {woj} {origin_key}: 받은 고유 id {df['id'].nunique()} ≠ count {count}")
    df.to_csv(cfile, index=False, encoding="utf-8")
    return df.astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="YYYY-MM")
    ap.add_argument("--woj", nargs="*", default=list(WOJ), help="도 코드(기본 16개 전부)")
    ap.add_argument("--days", nargs=2, type=int, default=None, help="시험용: 시작일 끝일")
    ap.add_argument("--origins", nargs="*", default=list(ORIGINS))
    ap.add_argument("--offline", action="store_true", help="요청 없이 캐시에 있는 도만 집계")
    ap.add_argument("--after-block", action="store_true", help="429 표시(cache/pl/BLOCKED_429.txt)를 무시하고 실행")
    a = ap.parse_args()
    y, m = map(int, a.month.split("-"))
    last = calendar.monthrange(y, m)[1]
    d1, d2 = (a.days if a.days else (1, last))
    d_from, d_to = f"{y}{m:02d}{d1:02d}", f"{y}{m:02d}{d2:02d}"
    OUT.mkdir(exist_ok=True)
    logf = open(OUT / f"pl_{a.month}_log.txt", "a", encoding="utf-8")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"시작 month={a.month} days={d1}-{d2} woj={a.woj} origins={a.origins}")
    if BLOCK.exists() and not a.after_block and not a.offline:
        log(f"중단: 이전 실행에서 HTTP 429 ({BLOCK.read_text(encoding='utf-8').strip()}). "
            f"충분히 쉰 뒤 --after-block 으로 다시 실행")
        sys.exit(2)
    sess = PoliteSession()
    parts, failed = [], []
    for woj in a.woj:
        for ok in a.origins:
            cf = CACHE / "pl" / f"{d_from}_{d_to}" / f"{woj}_{ok}.csv"
            if a.offline and not cf.exists():
                failed.append((woj, ok, "not_cached"))
                continue
            try:
                parts.append(fetch_woj(sess, woj, ok, d_from, d_to, log))
            except HostBlocked as e:
                log(f"중단: {e}")
                BLOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f" {a.month} woj={woj} {ok}", encoding="utf-8")
                failed.append((woj, ok, "429"))
                break
            except Exception as e:  # 한 도 실패해도 나머지는 계속, 기록
                log(f"실패 {woj} {ok}: {e!r}")
                failed.append((woj, ok, repr(e)[:200]))
        if sess.blocked:
            break
    log(f"요청 수 {sess.n_requests}  실패 {failed}")
    if not parts:
        sys.exit(1)
    df = pd.concat(parts, ignore_index=True)
    dup = df["id"].duplicated().sum()
    df = df.drop_duplicates("id")
    tag = a.month if not a.days else f"{a.month}_d{d1:02d}-{d2:02d}"
    g = (df.groupby(["marka", "model", "rodzaj-paliwa"], dropna=False).size().rename("n").reset_index()
         .sort_values("n", ascending=False))
    g.insert(0, "Month", a.month)
    g = classify_frame(g, "marka", "model", "rodzaj-paliwa")
    g.to_csv(OUT / f"pl_{tag}_by_model.csv", index=False, encoding="utf-8")
    ours = (g[g["Model"].notna()].groupby(["Brand", "Model"], as_index=False)["n"].sum()
            .sort_values("n", ascending=False))
    ours.insert(0, "Month", a.month)
    ours.insert(0, "Country_part", "PL")
    ours.to_csv(OUT / f"pl_{tag}_ours.csv", index=False, encoding="utf-8")
    meta = {"month": a.month, "days": [d1, d2], "woj_done": sorted(set(df["wojewodztwo-kod"])),
            "records": int(len(df)), "duplicate_ids_dropped": int(dup), "mapped_share": float(
                g.loc[g["Model"].notna(), "n"].sum() / max(1, g["n"].sum())), "failed": failed,
            "requests": sess.n_requests}
    (OUT / f"pl_{tag}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"완료 records={len(df)} mapped={meta['mapped_share']:.3f}")


if __name__ == "__main__":
    main()
