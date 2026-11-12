"""핀란드 Traficom PxWeb 표 105(henkilöautojen mallisarjojen ensirekisteröinnit alueittain kuukausittain)
→ 그 달 신차 승용 제조사×모델 대수.

사용: python collect_fi.py --month 2026-07 [--offline]

출처: https://trafi2.stat.fi/PXWeb/api/v1/fi/TraFi/Ensirekisteroinnit/050_ensirek_tau_105.px
  (robots.txt 없음(404) → 제한 없음. POST 질의 1번, 응답 60~70초로 느림.)
질의: Alue ∈ {MA1 Manner-Suomi, 200 Ulkomaat, X Tuntematon}(표에 있는 전국 구분 셋; 올란드는 이 표에 없음),
     Mallisarja = 전부, Käyttövoima = YH(합계), Kuukausi = <YYYY>M<MM>.
정의: ensirekisteröinti = 핀란드 최초 등록 신차(중고 수입은 별도 통계). 2026-07 합계 6,058대.
파싱: 'X yhteensä' = 브랜드 합계(그 아래 줄들의 제조사), 'Henkilöautot yhteensä' = 전체(대조용).
     'AUDI RS mallit yhteensä', 'VOLVO muut mallit yhteensä' 같은 묶음은 모델 줄로 둔다(아래에 개별 줄 없음).
     'Muut merkit yhteensä'(기타 브랜드)와 'Matkailuautot yhteensä'(캠핑카)는 make=MUUT / MATKAILUAUTOT 로 남긴다.
"""
from __future__ import annotations

import argparse
import io

import pandas as pd

from c_common import CACHE, SAMPLES, Session, check_month, find_cached, sha256, write_model_month

D = CACHE / "fi"
D.mkdir(parents=True, exist_ok=True)
URL = "https://trafi2.stat.fi/PXWeb/api/v1/fi/TraFi/Ensirekisteroinnit/050_ensirek_tau_105.px"
AREAS = ["MA1", "200", "X"]


def query(month: str) -> dict:
    return {"query": [
        {"code": "Alue", "selection": {"filter": "item", "values": AREAS}},
        {"code": "Mallisarja", "selection": {"filter": "all", "values": ["*"]}},
        {"code": "Käyttövoima", "selection": {"filter": "item", "values": ["YH"]}},
        {"code": "Kuukausi", "selection": {"filter": "item", "values": [month.replace("-", "M")]}},
    ], "response": {"format": "csv"}}


def download(month: str):
    s = Session()
    r = s.post_json(URL, query(month), timeout=300)
    r.raise_for_status()
    p = D / f"fi_traficom105_{month}.csv"
    p.write_bytes(r.content)
    return p


def parse(path) -> tuple[pd.DataFrame, float]:
    raw = path.read_bytes()
    try:
        txt = raw.decode("utf-8")
        if "Ã" in txt:
            raise UnicodeDecodeError("x", b"", 0, 1, "mojibake")
    except UnicodeDecodeError:
        txt = raw.decode("latin-1")
    d = pd.read_csv(io.StringIO(txt))
    d.columns = ["area", "ms", "fuel", "n"]
    d["n"] = pd.to_numeric(d["n"], errors="coerce").fillna(0)
    g = d.groupby("ms", sort=False)["n"].sum()
    total = float(g.get("Henkilöautot yhteensä", float("nan")))
    rows, brand = [], None
    for name, n in g.items():
        if name == "Henkilöautot yhteensä":
            continue
        if name.endswith(" yhteensä") and "mallit" not in name:
            if name == "Muut merkit yhteensä":
                rows.append(dict(make="MUUT", model="MUUT MERKIT", units=n))
            elif name == "Matkailuautot yhteensä":
                rows.append(dict(make="MATKAILUAUTOT", model="MATKAILUAUTOT", units=n))
            else:
                brand = name[: -len(" yhteensä")]
            continue
        model = name.replace(" yhteensä", "")
        rows.append(dict(make=brand, model=model, units=n))
    return pd.DataFrame(rows), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    month = check_month(a.month)
    p = find_cached([D / f"fi_traficom105_{month}.csv",
                     SAMPLES / f"fi_traficom_pxweb105_{month}.csv"])
    if p is None:
        if a.offline:
            raise SystemExit(f"{month}: 캐시 없음(--offline)")
        p = download(month)
    d, total = parse(p)
    s = float(d["units"].sum())
    meta = dict(source_id="FI_TRAFICOM_105", file=str(p), sha256=sha256(p), areas=AREAS, file_total=total,
                sum_rows=s, total_check="ok" if abs(total - s) < 0.5 else "MISMATCH",
                definition="Henkilöautojen ensirekisteröinnit (uudet), Manner-Suomi + Ulkomaat + Tuntematon")
    out = d.assign(country="Finland", month=month)[["country", "month", "make", "model", "units"]]
    out = out[out["units"] > 0]
    print(write_model_month("fi", month, out, meta))


if __name__ == "__main__":
    main()
