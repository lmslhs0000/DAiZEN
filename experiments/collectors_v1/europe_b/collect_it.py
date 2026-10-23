"""이탈리아 UNRAE 월간 'Top 50 modelli' PDF(그 달 상위 50 + 연초 누계 상위 50) → 우리 Europe 시계열 대응.

출처: https://unrae.it/dati-statistici/immatricolazioni (robots.txt 전부 허용, 2026-10-15)
  · 목록 ?page=N (한 쪽 10건, 최신 순) → 'immatricolazioni-in-italia-di-autovetture-e-fuoristrada-top-50-modelli-<월>-<연도>'
  · 그 쪽 안의 '/files/... Top 50 <Mese> <Anno>_*.pdf'
  · PDF 한 줄 = '순위 브랜드 모델 그달대수 순위 브랜드 모델 누계대수' (천 단위 '.')
월값은 PDF 의 그 달 상위 50 을 그대로 쓴다(누계 차이 불필요). 상위 50 밖 차종은 값 없음.

사용:
  python collect_it.py --month 2026-07
  python collect_it.py --range 2025-01 2026-07
결과: out/it_unrae_top50.csv(원표), out/it_unrae_ours.csv(우리 차종 월값), out/validation_it.csv(2025 = EEA IT 월값과 비교)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pdfplumber

from _common import CACHE, HERE, PoliteSession, fetch_cached
from europe_map import classify

BASE = "https://unrae.it/dati-statistici/immatricolazioni"
MESI = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre",
        "novembre", "dicembre"]
OUT = HERE / "out"
ITC = CACHE / "it"
ITC.mkdir(parents=True, exist_ok=True)
_sess: PoliteSession | None = None


def sess() -> PoliteSession:
    global _sess
    if _sess is None:
        _sess = PoliteSession()
    return _sess


def _slug(month: str) -> str:
    y, m = month.split("-")
    return f"top-50-modelli-{MESI[int(m) - 1]}-{y}"


def find_pages(months: list[str], max_pages: int = 40) -> dict[str, str]:
    """목록을 최신부터 넘기며 필요한 달의 Top 50 쪽 주소를 찾는다(목록 HTML 은 캐시하지 않음: 최신 순이라 바뀜)."""
    want = {_slug(m): m for m in months}
    found: dict[str, str] = {}
    oldest = min(months)
    for page in range(1, max_pages + 1):
        url = BASE if page == 1 else f"{BASE}?page={page}"
        html = sess().get(url).text
        for link in re.findall(r'href="(https?://(?:www\.)?unrae\.it/dati-statistici/immatricolazioni/\d+/[^"]+)"', html):
            for sl, mo in want.items():
                if link.rstrip("/").endswith(sl) and "autovetture-e-fuoristrada" in link:
                    found[mo] = link
        # 이 쪽에서 가장 오래된 달이 찾는 범위보다 이전이면 멈춤
        ms = re.findall(r"-(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|"
                        r"dicembre)-(\d{4})", html)
        if ms:
            olds = min(f"{y}-{MESI.index(m) + 1:02d}" for m, y in ms)
            if olds < oldest or len(found) == len(months):
                break
    return found


def pdf_path(month: str, page_url: str | None) -> Path | None:
    p = ITC / f"unrae_top50_{month}.pdf"
    if p.exists():
        return p
    if not page_url:
        return None
    hp = ITC / f"top50_{month}.html"
    if not hp.exists():
        hp.write_text(sess().get(page_url).text, encoding="utf-8")
    pdfs = re.findall(r'href="(https?://(?:www\.)?unrae\.it/files/[^"]+\.pdf)"', hp.read_text(encoding="utf-8"), re.I)
    pdfs = [u for u in pdfs if "top 50" in u.lower() or "top50" in u.lower()] or pdfs
    if not pdfs:
        return None
    return fetch_cached(sess(), pdfs[0], p)


NUM = r"\d{1,3}(?:\.\d{3})*"
LINE = re.compile(rf"^(\d{{1,2}}) (.+?) ({NUM}) \1 (.+?) ({NUM})$")
LINE1 = re.compile(rf"^(\d{{1,2}}) ([A-Z][A-Z0-9 .\-'+&/]*?) ({NUM})$")
TWO_WORD = ("ALFA ROMEO", "LAND ROVER", "ASTON MARTIN", "LYNK & CO", "ROLLS ROYCE", "DS AUTOMOBILES", "KG MOBILITY")


def split_brand(name: str) -> tuple[str, str]:
    for two in TWO_WORD:
        if name.startswith(two + " "):
            return two, name[len(two) + 1:]
    b, _, rest = name.partition(" ")
    return b, rest


def parse_top50(pdf: Path, month: str) -> pd.DataFrame:
    rows = []
    with pdfplumber.open(pdf) as p:
        text = "\n".join(pg.extract_text() or "" for pg in p.pages)
    for line in text.split("\n"):
        m = LINE.match(line.strip())
        if not m:
            m1 = LINE1.match(line.strip())  # 1월판: 그 달 = 누계라 한 열만 있음
            if m1:
                for kind in ("month", "ytd"):
                    rows.append({"Month": month, "kind": kind, "rank": int(m1.group(1)), "name": m1.group(2),
                                 "n": int(m1.group(3).replace(".", ""))})
            continue
        rank = int(m.group(1))
        rows.append({"Month": month, "kind": "month", "rank": rank, "name": m.group(2),
                     "n": int(m.group(3).replace(".", ""))})
        rows.append({"Month": month, "kind": "ytd", "rank": rank, "name": m.group(4),
                     "n": int(m.group(5).replace(".", ""))})
    df = pd.DataFrame(rows)
    if len(df) and (df.groupby("kind")["rank"].nunique() != 50).any():
        print(f"[IT] 경고 {month}: 순위 수 {df.groupby('kind')['rank'].nunique().to_dict()}")
    return df


def validate_vs_eea(ours: pd.DataFrame) -> pd.DataFrame:
    p = CACHE / "eea_monthly_2025_FR-IT_ft.csv"
    if not p.exists():
        return pd.DataFrame()
    from europe_map import classify_frame
    from europe_validate import LCV_LIKE
    from europe_validate import load_eea  # 전 나라 월별 + 형식(T) 분리(Panda/Grande Panda 등) 적용판
    e = load_eea(2025)
    e = e[e["country"] == "IT"]
    eg = e[e["Model"].notna()].groupby(["Brand", "Model", "Month"])["n"].sum()
    rows = []
    for r in ours[ours["Month"].str.startswith("2025")].itertuples(index=False):
        ev = eg.get((r.Brand, r.Model, r.Month))
        rows.append({"check": "it_unrae_vs_eea_IT_month", "period": r.Month, "Brand": r.Brand, "Model": r.Model,
                     "lcv_like": (r.Brand, r.Model) in LCV_LIKE, "collected": r.n, "reference": ev,
                     "ratio": (r.n / ev) if ev else None,
                     "note": "UNRAE top50 month vs EEA IT (2025 provisional); EEA 는 형식(T)으로 Panda/Grande Panda·Corolla/Cross·EQE/EQS SUV 분리"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month")
    ap.add_argument("--range", nargs=2)
    a = ap.parse_args()
    months = [str(p) for p in pd.period_range(*a.range, freq="M")] if a.range else [a.month]
    OUT.mkdir(exist_ok=True)
    missing = [m for m in months if not (ITC / f"unrae_top50_{m}.pdf").exists()]
    pages = find_pages(missing) if missing else {}
    parts = []
    for mo in months:
        p = pdf_path(mo, pages.get(mo))
        if p is None:
            print(f"[IT] {mo}: PDF 못 찾음")
            continue
        parts.append(parse_top50(p, mo))
    t = pd.concat(parts, ignore_index=True)
    cls = {}
    for nm in t["name"].unique():
        mk, md = split_brand(nm)
        cls[nm] = classify(mk, md)
    for i, col in enumerate(["Brand", "Model", "rule", "confidence"]):
        t[col] = t["name"].map(lambda n: cls[n][i])
    t.to_csv(OUT / "it_unrae_top50.csv", index=False, encoding="utf-8")
    mon = t[(t["kind"] == "month") & t["Model"].notna()]
    ours = mon.groupby(["Brand", "Model", "Month"], as_index=False)["n"].sum()
    ours.insert(0, "Country_part", "IT")
    ours.to_csv(OUT / "it_unrae_ours.csv", index=False, encoding="utf-8")
    v = validate_vs_eea(ours)
    if not v.empty:
        v.to_csv(OUT / "validation_it.csv", index=False, encoding="utf-8")
        r = v["ratio"].dropna()
        print(f"[IT] EEA 비교 {len(r)}건 median={r.median():.3f} 0.9~1.1 비율={((r > .9) & (r < 1.1)).mean():.2f} "
              f"가중={v['collected'].sum() / v['reference'].sum():.3f}")
    um = t[(t["kind"] == "month") & t["Model"].isna()]
    print(f"[IT] 달 {t['Month'].nunique()}개, 그달 상위50 대응률 {mon['n'].sum() / t.loc[t.kind == 'month', 'n'].sum():.3f}, "
          f"미대응 이름 {sorted(um['name'].unique())}")
    print(f"요청 수 {sess().n_requests if _sess else 0}")


if __name__ == "__main__":
    main()
