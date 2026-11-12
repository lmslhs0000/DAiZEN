"""아일랜드 CSO TEM20(New Private Cars Licensed for the First Time, 제조사·모델별 월) → 그 달 대수.

사용: python collect_ie.py --month 2026-07 [--offline]

출처: CSO PxStat 표 TEM20 전체 CSV(약 12MB, 2014-01~최신)
  https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/TEM20/CSV/1.0/en
  (ws.cso.ie robots.txt 없음(404) → 제한 없음. 한 번 받아 cache/ie/에 두고, 그 파일에 그 달이 없을 때만 다시 받음.)
정의: CSO = 아일랜드 차량 등록(VRT) 기준 '처음 면허된 신차 개인 승용'. 'ZZZ111 All models' = 합계(대조), 'ZZZ001 Other' = 기타.
  SIMI(업계 협회) 월 수치와 최대 40% 다르다(europe_deep README). 정의 차이(SIMI는 등록일·딜러 신고, CSO는 면허 개시 기준으로
  보임)는 확인하지 못했고 기록만 한다. 우리 Europe 시계열 검증(validation.csv)에서 어느 쪽이 맞는지 간접 확인.
'Make and Model' 칸은 'Toyota C-HR'처럼 제조사+모델이 한 문자열 → make 칸에도 같은 문자열을 넣는다(대응기가 앞 단어로 브랜드 판정).
"""
from __future__ import annotations

import argparse

import pandas as pd

from c_common import CACHE, SAMPLES, Session, check_month, sha256, write_model_month

D = CACHE / "ie"
D.mkdir(parents=True, exist_ok=True)
URL = "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/TEM20/CSV/1.0/en"


def load(month: str, offline: bool):
    code = month.replace("-", "")
    for p in [D / "TEM20.csv", SAMPLES / "ie_cso_TEM20_2014-01_latest.csv"]:
        if p.exists():
            x = pd.read_csv(p, dtype=str)
            if (x["TLIST(M1)"] == code).any():
                return p, x
    if offline:
        raise SystemExit(f"{month}: 캐시에 그 달 없음(--offline)")
    r = Session().get(URL, timeout=300)
    r.raise_for_status()
    p = D / "TEM20.csv"
    p.write_bytes(r.content)
    x = pd.read_csv(p, dtype=str)
    if not (x["TLIST(M1)"] == code).any():
        raise SystemExit(f"{month}: CSO TEM20 에 아직 없음")
    return p, x


def simi_cached(month: str):
    for p in [D / f"simi_carsByModel_{month}.json", SAMPLES / f"ie_simi_carsByModel_{month}.json"]:
        if p.exists():
            return p
    return None


SIMI = "https://stats.simi.ie"   # robots.txt: Disallow 없음(2026-10-15 확인). 로그인 없는 공개 화면.
MONTH_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
            "November", "December"]


def download_simi(month: str):
    """공개 화면과 같은 방식: 필터(그 해·그 달)를 세션에 PATCH → Inertia 부분 요청으로 carsByModel 받기.
    2026-07 로 시험하니 표본 파일과 합계(28,989)가 같았다(2026-10-15). 요청 3번, 간격 3초 이상."""
    import html, json, re, time
    from urllib.parse import unquote
    s = Session()
    page = s.get(SIMI + "/")
    data = json.loads(html.unescape(re.search(r'data-page="([^"]+)"', page.text).group(1)))
    up_to = data["props"].get("environmentDate", "")            # 예: 30/09/2026 = 자료가 들어온 마지막 날
    last = pd.Period(pd.to_datetime(up_to, dayfirst=True), freq="M") if up_to else None
    if last is None or pd.Period(month, freq="M") > last or             pd.to_datetime(up_to, dayfirst=True) < pd.Period(month, freq="M").end_time.normalize():
        raise SystemExit(f"{month}: SIMI 자료가 아직 그 달 끝까지 없음(자료 기준일 {up_to})")
    y, m = int(month[:4]), int(month[5:7])
    mo = {"name": MONTH_EN[m - 1], "value": m}
    body = {"years": [{"name": y, "value": y}], "month_from": mo, "day_from": None, "month_to": mo, "day_to": None,
            "registration_type": {"name": "Total New Registrations", "value": "new-total"}, "sales_types": [],
            "makes": [], "models": [], "body_types": [], "transmissions": [], "engine_types": [],
            "engine_capacities": [], "colours": [], "segments": [], "counties": []}
    h = {"X-Inertia": "true", "X-Inertia-Version": data["version"], "X-Requested-With": "XMLHttpRequest",
         "X-XSRF-TOKEN": unquote(s.s.cookies.get("XSRF-TOKEN", "")), "Accept": "text/html, application/xhtml+xml"}
    time.sleep(3)
    r = s.s.patch(SIMI + "/filter/passenger", json=body, headers=h, allow_redirects=False, timeout=60)
    if r.status_code not in (200, 302, 303):
        raise SystemExit(f"{month}: SIMI 필터 설정 실패 HTTP {r.status_code}")
    time.sleep(3)
    j = s.s.get(SIMI + "/", headers={**h, "X-Inertia-Partial-Component": "Public/Passenger",
                                     "X-Inertia-Partial-Data": "carsByModel,initialFilters"}, timeout=120).json()
    f = j["props"].get("initialFilters", {})
    if (f.get("month_from") or {}).get("value") != m or (f.get("month_to") or {}).get("value") != m:
        raise SystemExit(f"{month}: SIMI 가 달 필터를 받지 않음: {f}")
    p = D / f"simi_carsByModel_{month}.json"
    p.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    return p


def from_simi(month: str, p) -> None:
    """SIMI(업계 협회) 월간 모델 표(Inertia JSON: props.carsByModel.datasets[].labels.make/model, units[0].count).
    캐시(cache/ie/simi_carsByModel_<YYYY-MM>.json)가 없으면 download_simi 로 받는다."""
    import json
    j = json.load(open(p, encoding="utf-8"))["props"]["carsByModel"]
    yi = j["years"].index(int(month[:4]))
    rows = [dict(make=d["labels"]["make"], model=d["labels"]["make"] + " " + d["labels"]["model"],
                 units=float(d["units"][yi]["count"] or 0)) for d in j["datasets"]]
    d = pd.DataFrame(rows)
    meta = dict(source_id="IE_SIMI", file=str(p), sha256=sha256(p), sum_rows=float(d["units"].sum()),
                definition="SIMI new passenger car registrations (업계). 2025-07 합계가 EEA IE 와 같음(26,738 vs 26,758)")
    out = d.assign(country="Ireland", month=month)[["country", "month", "make", "model", "units"]]
    print(write_model_month("ie", month, out[out["units"] > 0], meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--source", choices=["auto", "cso", "simi"], default="auto",
                    help="auto = SIMI 캐시가 있으면 SIMI(우리 Europe 정의·EEA 와 맞음), 없으면 CSO")
    a = ap.parse_args()
    month = check_month(a.month)
    sp = simi_cached(month)
    if sp is None and not a.offline and a.source in ("auto", "simi"):
        try:
            sp = download_simi(month)
        except SystemExit as e:
            if a.source == "simi":
                raise
            print(f"SIMI 못 받음 → CSO: {e}")
    if a.source == "simi" or (a.source == "auto" and sp is not None):
        if sp is None:
            raise SystemExit(f"{month}: SIMI 캐시 없음")
        return from_simi(month, sp)
    p, x = load(month, a.offline)
    x = x[(x["STATISTIC"] == "TEM20C01") & (x["TLIST(M1)"] == month.replace("-", ""))].copy()
    x["units"] = pd.to_numeric(x["VALUE"], errors="coerce")
    total = float(x.loc[x["C03398V04090"] == "ZZZ111", "units"].sum())
    blank = int(x["units"].isna().sum())
    d = x[x["C03398V04090"] != "ZZZ111"]
    d = d[d["units"].notna()]
    s = float(d["units"].sum())
    meta = dict(source_id="IE_CSO_TEM20", file=str(p), sha256=sha256(p), file_total=total, sum_rows=s,
                blank_cells=blank, total_check="ok" if abs(total - s) < 0.5 else "MISMATCH",
                definition="New private cars licensed for the first time (CSO, VRT); SIMI 와 정의 차이 미확인")
    out = pd.DataFrame(dict(country="Ireland", month=month, make=d["Make and Model"].values,
                            model=d["Make and Model"].values, units=d["units"].values, source_key=d["C03398V04090"].values))
    out = out[out["units"] > 0]
    print(write_model_month("ie", month, out, meta))


if __name__ == "__main__":
    main()
