"""루마니아 DGPCI(Direcția Generală Permise de Conducere și Înmatriculări) 월간 등록 묶음(.rar) → 신차 승용 제조사×모델.

사용: python collect_ro.py --month 2026-07 [--offline] [--url <rar 주소>]

입력: <YYYYMM>.rar 안의 '<YYYYMM>_Autoturisme.xlsx'(M1·M1G), 시트 'Descriere_Comerciala'
  (MARCA 'TOTAL' 줄 = 브랜드 합계, 그 아래 줄 = DENUMIRE COMERCIALA(상품명)별 'TOTAL VEHICULE NOI').
  같은 묶음의 '<YYYYMM>_Acea.xlsx' 'Vehicule de pasageri (M1, M1G)' 그 달·전년 같은 달 = 대조용.
rar 풀기: Windows 기본 tar.exe(libarchive, rar 읽기 지원). 받은 파일은 cache/ro/ 에 둔다.
받는 주소: DGPCI 누리집(www.drpciv.ro → dgpci.mai.gov.ro)은 자바스크립트 앱이라 파일 주소 규칙을 확인하지 못했다.
  data.gov.ro 에는 이 월간 묶음이 없다(검색 확인, robots: /api/ 금지·Crawl-delay 10). → 주소를 알면 --url 로 준다.
총계 점검(2026-07 파일): 상품명 줄 합 11,740 = 브랜드 TOTAL 합 = 'Tip_Extensie' 신차 합 = 상세 목록(Lista_Detaliata)
  'INSCRIERE VEHICUL NOU' 11,740(M1 11,203 + M1G 537) = Acea 시트 M1 11,740 → **파일 안에서는 불일치 없음**.
  Acea 시트의 2025-07 값 16,362 = EEA RO 2025-07 16,362(정확히 같음). 예전 '총계 불일치'는 중고 첫 등록(28,711)·
  재등록(60,826) 열이 같은 표에 있어 섞어 본 것으로 보인다(확인한 범위: 이 파일 하나).
"""
from __future__ import annotations

import argparse
import subprocess

import pandas as pd

from c_common import CACHE, SAMPLES, Session, check_month, sha256, write_model_month

D = CACHE / "ro"
D.mkdir(parents=True, exist_ok=True)
TAR = r"C:\Windows\System32\tar.exe"


def get_xlsx(month: str, offline: bool, url: str | None):
    ym = month.replace("-", "")
    x = D / f"{ym}_Autoturisme.xlsx"
    if x.exists():
        return x, D / f"{ym}_Acea.xlsx"
    rar = next((p for p in [D / f"ro_dgpci_{month}.rar", SAMPLES / f"ro_dgpci_{month}.rar"] if p.exists()), None)
    if rar is None:
        if offline or not url:
            raise SystemExit(f"{month}: rar 캐시 없음(--offline 이거나 --url 미지정)")
        r = Session().get(url, timeout=300)
        r.raise_for_status()
        rar = D / f"ro_dgpci_{month}.rar"
        rar.write_bytes(r.content)
    subprocess.run([TAR, "-xf", str(rar), "-C", str(D), f"{ym}_Autoturisme.xlsx", f"{ym}_Acea.xlsx"], check=True)
    return x, D / f"{ym}_Acea.xlsx"


def acea_m1(path) -> tuple[float | None, float | None]:
    try:
        a = pd.read_excel(path, header=None)
    except Exception:
        return None, None
    for i in range(len(a)):
        lab = str(a.iat[i, 1])
        if lab.startswith("Vehicule de pasageri"):
            return float(a.iat[i, 2]), float(a.iat[i, 3])
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--url")
    a = ap.parse_args()
    month = check_month(a.month)
    xp, ap_ = get_xlsx(month, a.offline, a.url)
    x = pd.read_excel(xp, sheet_name="Descriere_Comerciala", header=None, dtype=str)
    hdr = x.index[x[0].astype(str).str.strip() == "Vehicle_brand"][0]
    x = x.iloc[hdr + 1:, :3]
    x.columns = ["marca", "den", "n"]
    x["n"] = pd.to_numeric(x["n"], errors="coerce")
    rows, brand = [], None
    brand_tot = 0.0
    for r in x.itertuples(index=False):
        if isinstance(r.marca, str) and r.marca.strip():
            brand = " ".join(r.marca.split())
        if str(r.den).strip() == "TOTAL":
            brand_tot += 0 if pd.isna(r.n) else r.n
            continue
        if pd.isna(r.n):
            continue
        rows.append(dict(make=brand, model=str(r.den).strip(), units=float(r.n)))
    d = pd.DataFrame(rows)
    cur, ly = acea_m1(ap_)
    s = float(d["units"].sum())
    meta = dict(source_id="RO_DGPCI", file=str(xp), sha256=sha256(xp), sum_rows=s, brand_total_rows=brand_tot,
                acea_sheet_m1_month=cur, acea_sheet_m1_same_month_last_year=ly,
                total_check="ok" if cur is not None and abs(cur - s) < 0.5 and abs(brand_tot - s) < 0.5 else "MISMATCH",
                definition="Înmatriculări vehicule noi M1+M1G (DGPCI)")
    out = d.assign(country="Romania", month=month)[["country", "month", "make", "model", "units"]]
    print(write_model_month("ro", month, out[out["units"] > 0], meta))


if __name__ == "__main__":
    main()
