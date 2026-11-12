"""룩셈부르크 SNCA '차량 등록부 월간 변경분(Opérations delta)' xlsx → 신차 승용 제조사×상품명 대수.

사용: python collect_lu.py --month 2026-07 [--offline]

출처: https://data.public.lu/fr/datasets/operations-delta-des-vehicules-au-luxembourg/ (CC0)
  쪽 HTML 안의 download.data.public.lu/.../operations-delta-<YYYYMM>.xlsx 링크(매달 초 게시, 약 7~8MB).
  data.public.lu robots.txt 없음. 이 쪽 HTML 과 파일만 받는다(API 미사용).
행 = 그 달 등록부 처리 1건. CODEOP: N = 신규 등록(신차), I = 수입 중고, T = 명의 이전, H/E/E1 = 말소·수출 등.
필터: CODEOP == 'N' 그리고 CATEU ∈ {M1, M1G}, DATCIRPRM(최초 운행일)이 그 달인지 확인(2026-07: 전부 그 달).
  make = LIBMRQ, model = TYPCOM(상품명·트림: 'GLC 220D 4MATIC', 'SCENIC E-TECH ELECTRIC').
설명서: 같은 쪽의 description-des-donnees-exportees-vers-opendata-v.1.9.1.pdf.
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

from c_common import CACHE, SAMPLES, Session, check_month, sha256, write_model_month

D = CACHE / "lu"
D.mkdir(parents=True, exist_ok=True)
PAGE = "https://data.public.lu/fr/datasets/operations-delta-des-vehicules-au-luxembourg/"


def get_file(month: str, offline: bool):
    ym = month.replace("-", "")
    for p in [D / f"operations-delta-{ym}.xlsx", SAMPLES / f"lu_snca_operationsdelta_{month}.xlsx"]:
        if p.exists():
            return p
    if offline:
        raise SystemExit(f"{month}: 캐시 없음(--offline)")
    s = Session()
    html = s.get(PAGE).text
    (D / "dataset_page.html").write_text(html, encoding="utf-8")
    links = sorted(set(re.findall(r'https://download\.data\.public\.lu/resources/operations-delta-des-vehicules-au-'
                                  rf'luxembourg/[0-9\-]+/operations[-_]delta[-_]{ym}[^"<\s]*\.xlsx', html, flags=re.I)))
    if not links:
        raise SystemExit(f"{month}: 쪽에 operations-delta-{ym}.xlsx 없음(아직 미게시?)")
    r = s.get(links[-1], timeout=300)
    r.raise_for_status()
    p = D / f"operations-delta-{ym}.xlsx"
    p.write_bytes(r.content)
    (D / f"operations-delta-{ym}.url.txt").write_text(links[-1], encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    month = check_month(a.month)
    p = get_file(month, a.offline)
    import warnings
    warnings.filterwarnings("ignore")
    x = pd.read_excel(p, usecols=["CODEOP", "CATEU", "LIBMRQ", "TYPCOM", "DATCIRPRM", "PAYPVN"])
    ym = int(month.replace("-", ""))
    n = x[(x["CODEOP"] == "N") & x["CATEU"].isin(["M1", "M1G"])].copy()
    in_month = (pd.to_numeric(n["DATCIRPRM"], errors="coerce") // 100 == ym)
    d = n[in_month].groupby(["LIBMRQ", "TYPCOM"], dropna=False).size().reset_index(name="units")
    d = d.rename(columns={"LIBMRQ": "make", "TYPCOM": "model"})
    d["make"] = d["make"].fillna("").astype(str)
    d["model"] = d["model"].fillna("").astype(str)
    meta = dict(source_id="LU_SNCA_DELTA", file=str(p), sha256=sha256(p), rows_file=int(len(x)),
                new_m1=int(len(n)), new_m1_first_circ_in_month=int(in_month.sum()),
                new_m1_foreign_origin=int((n["PAYPVN"] != "LU").sum()),
                definition="SNCA opérations delta: CODEOP N (nouvelle immatriculation), CATEU M1/M1G")
    out = d.assign(country="Luxembourg", month=month)[["country", "month", "make", "model", "units"]]
    print(write_model_month("lu", month, out, meta))


if __name__ == "__main__":
    main()
