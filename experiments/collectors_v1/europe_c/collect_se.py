"""스웨덴 Mobility Sweden 월간 보고서 xlsx → 신차 승용(Personbilar) 제조사×모델 월 대수.

사용: python collect_se.py --month 2026-07 [--offline] [--sply]
  --sply : 같은 파일의 'SPLY'(전년 같은 달) 열로 (month-12) 값도 따로 출력(검증용, *_sply 접미사)

출처: https://mobilitysweden.se/statistik/Nyregistreringar_per_manad_1 (robots.txt 비어 있음 = 허용)
  목록 쪽에서 '<1~5> <다음 달 이름> <연도>'에 게시된 기사 → 기사 안 'Månadsrapport Nyregistreringar <달> <연도>.xlsx'.
  게시: 다음 달 첫 영업일(예: 2026-07 → 2026-08-03).
시트 'PB - Fabrikat och modell': 열 D=Fabrikat(브랜드 합계 줄에만, 아래로 채움), F=Modell, J=그 달 Antal, K=SPLY.
  'Total' 줄(브랜드 합계)은 빼고 모델 줄만. 마지막 'Totalt' = 전체 합계 → 모델 줄 합과 대조.
  모델 이름이 두 차종을 묶는 경우가 있다('VOLVO EX/XC40', 'VOLVO S/V60', 'VW ID.7/ID.7 TOURER') → 대응·배분은 collect_europe.py.
  'ÖVRIGA'(기타) 줄은 차종을 알 수 없어 대응되지 않는다.
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

from c_common import CACHE, SAMPLES, Session, check_month, find_cached, sha256, write_model_month, prev_month

BASE = "https://mobilitysweden.se"
LISTING = BASE + "/statistik/Nyregistreringar_per_manad_1"
SV_MONTH = ["januari", "februari", "mars", "april", "maj", "juni", "juli", "augusti", "september", "oktober",
            "november", "december"]
SHEET = "PB - Fabrikat och modell"
D = CACHE / "se"
D.mkdir(parents=True, exist_ok=True)


def cached_file(month: str):
    return find_cached([D / f"se_mobilitysweden_{month}.xlsx", SAMPLES / f"se_mobilitysweden_{month}.xlsx"])


def download(month: str) -> "Path":
    y, m = int(month[:4]), int(month[5:7])
    nxt = prev_month(month, -1)
    pub_label = f"{SV_MONTH[int(nxt[5:7]) - 1]} {nxt[:4]}"
    s = Session()
    html = s.get(LISTING).text
    (D / "listing.html").write_text(html, encoding="utf-8")
    # 기사 링크와 그 뒤의 게시일
    cand, named = [], []
    for mt in re.finditer(r'href="(https://mobilitysweden\.se/statistik/Nyregistreringar_per_manad_1/'
                          r'nyregistreringar-\d{4}/[^"]+)"', html):
        if SV_MONTH[m - 1] in mt.group(1).lower():
            named.append(mt.group(1))   # 기사 제목(주소)에 대상 달 이름이 있으면 그것이 우선
        tail = re.sub(r"<[^>]+>", " ", html[mt.end():mt.end() + 800])
        if re.search(rf"\b\d{{1,2}} {pub_label}\b", tail):
            cand.append(mt.group(1))
    cand = list(dict.fromkeys(named + cand))
    if not cand:
        raise SystemExit(f"{month}: 목록에서 '{pub_label}' 게시 기사를 찾지 못함(아직 미발표?)")
    art = s.get(cand[0]).text
    (D / f"article_{month}.html").write_text(art, encoding="utf-8")
    xl = re.findall(r'href="([^"]+\.xlsx)"', art)
    from urllib.parse import unquote
    xl = [u for u in xl if f"{SV_MONTH[m - 1][:3]} {y}" in unquote(u).lower()] or          [u for u in xl if SV_MONTH[m - 1] in u.lower() or "nyregistreringar" in u.lower()] or xl
    if not xl:
        raise SystemExit(f"{month}: 기사에 xlsx 링크 없음: {cand[0]}")
    url = xl[0] if xl[0].startswith("http") else BASE + xl[0]
    r = s.get(url)
    r.raise_for_status()
    p = D / f"se_mobilitysweden_{month}.xlsx"
    p.write_bytes(r.content)
    (D / f"se_mobilitysweden_{month}.url.txt").write_text(url + "\n" + cand[0] + "\n", encoding="utf-8")
    return p


def parse(path, month: str):
    x = pd.read_excel(path, sheet_name=SHEET, header=None)
    title = " ".join(str(v) for v in x.iloc[:3].values.ravel() if isinstance(v, str))
    want = f"{SV_MONTH[int(month[5:7]) - 1]} {month[:4]}"
    if want not in title.lower():
        raise SystemExit(f"파일 제목이 {want} 가 아님: {title[:80]}")
    # 머리 줄: 'Antal' 'SPLY' 가 있는 줄
    hdr = next(i for i in range(len(x)) if "Antal" in [str(v).strip() for v in x.iloc[i].values])
    row = [str(v).strip() for v in x.iloc[hdr].values]
    c_cur, c_ly = row.index("Antal"), row.index("SPLY")
    c_brand, c_model = 3, 5
    rows, brand, total = [], None, None
    for i in range(hdr + 1, len(x)):
        b, mdl = x.iat[i, c_brand], x.iat[i, c_model]
        if isinstance(b, str) and b.strip():
            brand = b.strip()
        if not isinstance(mdl, str) or not mdl.strip():
            continue
        mdl = mdl.strip()
        cur = pd.to_numeric(x.iat[i, c_cur], errors="coerce")
        ly = pd.to_numeric(x.iat[i, c_ly], errors="coerce")
        if mdl == "Totalt":
            total = (cur, ly)
            continue
        if mdl == "Total":
            continue
        rows.append(dict(make=brand, model=mdl, units=0.0 if pd.isna(cur) else float(cur),
                         units_sply=0.0 if pd.isna(ly) else float(ly)))
    d = pd.DataFrame(rows)
    return d, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--sply", action="store_true")
    a = ap.parse_args()
    month = check_month(a.month)
    p = cached_file(month)
    if p is None:
        if a.offline:
            raise SystemExit(f"{month}: 캐시 없음(--offline)")
        p = download(month)
    d, total = parse(p, month)
    s_cur, s_ly = d["units"].sum(), d["units_sply"].sum()
    meta = dict(source_id="SE_MOBILITYSWEDEN", file=str(p), sha256=sha256(p), sheet=SHEET,
                file_total=float(total[0]) if total else None, sum_models=float(s_cur),
                total_check="ok" if total and abs(total[0] - s_cur) < 0.5 else "MISMATCH",
                definition="nyregistrerade personbilar (nya), Sverige; källa Transportstyrelsen via Mobility Sweden")
    out = d.assign(country="Sweden", month=month)[["country", "month", "make", "model", "units"]]
    out = out[out["units"] > 0]
    path = write_model_month("se", month, out, meta)
    if a.sply:
        ly = prev_month(month, 12)
        o2 = d.assign(country="Sweden", month=ly, units=d["units_sply"])[["country", "month", "make", "model", "units"]]
        o2 = o2[o2["units"] > 0]
        write_model_month("se", ly, o2, dict(meta, file_total=float(total[1]) if total else None,
                                              sum_models=float(s_ly), note="SPLY 열(전년 같은 달)"), suffix="_sply")
    print(path)


if __name__ == "__main__":
    main()
