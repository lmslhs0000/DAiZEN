"""프랑스 PFA 월간 보도자료 PDF(상위 100 차종, 연초~당월 누계) → 월값 = 누계 차이 → 우리 Europe 시계열 대응.

출처: https://pfa-auto.fr/marche-automobile/ (robots.txt 전부 허용, 2026-10-15)
  · 기사 'marche-automobile-francais-<월>-<연도>' 안에 'Dossier-de-presse-*.pdf' 링크
  · PDF 'Top 100 VP France  Janvier à <월> <연도>' 표(VP=승용) — 누계(YTD) 대수, 원자료 AAA DATA.
  · 'Top 100 VUL France' (경상용) 도 같이 읽어 둔다(우리 Ducato·Trafic 같은 승합/상용 성격 차종 참고용).
월값: m월 누계 − (m−1)월 누계. 1월은 누계 그대로. 전달 상위 100 에 없던 차종은 월값을 만들 수 없다(공란, status=prev_missing).
세대 구분('CLIO V', 'CLIO VI')은 같은 우리 차종으로 합친다. 16자에서 잘린 이름('YARIS CRO')은 앞부분이 한 차종에만 맞을 때만 대응.

사용:
  python collect_fr.py --month 2026-07              # 그 달 PDF + 전달 PDF → 월값
  python collect_fr.py --range 2025-01 2026-07      # 여러 달(전달 PDF 를 자동으로 같이 받음)
결과: out/fr_pfa_ytd.csv(누계 원표), out/fr_pfa_monthly.csv(월값·대응), out/validation_fr.csv(2025 = EEA FR 월값과 비교)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pdfplumber

from _common import CACHE, HERE, PoliteSession, fetch_cached
from europe_map import classify

LISTING = ["https://pfa-auto.fr/marche-automobile/", "https://pfa-auto.fr/marche-automobile/?upage=2"]
MOIS = ["janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre",
        "novembre", "decembre"]
OUT = HERE / "out"
FRC = CACHE / "fr"
FRC.mkdir(parents=True, exist_ok=True)
_sess: PoliteSession | None = None


def sess() -> PoliteSession:
    global _sess
    if _sess is None:
        _sess = PoliteSession()
    return _sess


def article_url(month: str) -> str | None:
    y, m = month.split("-")
    slug = f"marche-automobile-francais-{MOIS[int(m) - 1]}-{y}/"
    for i, u in enumerate(LISTING):
        p = FRC / f"listing_{i}.html"
        if not p.exists():
            p.write_text(sess().get(u).text, encoding="utf-8")
        hit = re.findall(r'https://pfa-auto\.fr/\d{4}/\d\d/\d\d/' + re.escape(slug), p.read_text(encoding="utf-8"))
        if hit:
            return hit[0]
    # 목록에서 빠지는 달이 있다(2026-08 기사는 목록에 없고 주소로만 열림) → 다음 달 1~3일 주소를 직접 시도
    nxt = pd.Period(month, freq="M") + 1
    for d in ("01", "02", "03"):
        u = f"https://pfa-auto.fr/{nxt.year}/{nxt.month:02d}/{d}/{slug}"
        if sess().get(u).status_code == 200:
            return u
    return None


def pdf_path(month: str) -> Path | None:
    p = FRC / f"pfa_dp_{month}.pdf"
    if p.exists():
        return p
    art = article_url(month)
    if not art:
        print(f"[FR] {month}: 기사 없음(목록에 없음)")
        return None
    ap = FRC / f"article_{month}.html"
    if not ap.exists():
        ap.write_text(sess().get(art).text, encoding="utf-8")
    pdfs = re.findall(r'href="(https://pfa-auto\.fr/wp-content/uploads/[^"]+\.pdf)"', ap.read_text(encoding="utf-8"))
    pdfs = [u for u in dict.fromkeys(pdfs) if "ossier" in u or "DP" in u or "dp" in u]
    if not pdfs:
        print(f"[FR] {month}: 기사에 PDF 링크 없음 {art}")
        return None
    return fetch_cached(sess(), pdfs[0], p)


# ---------------------------------------------------------------- PDF 표 읽기

def _undouble(tok: str) -> str:
    if len(tok) >= 2 and len(tok) % 2 == 0 and tok[0::2] == tok[1::2]:
        return tok[0::2]
    return tok


def _fix_bold(line: str) -> str:
    """굵은 글씨(상위 10위)는 글자가 두 번씩 찍힌다('PPEEUUGGEEOOTT 220088') → 줄 앞의 굵은 구간만 되돌린다."""
    toks = line.split(" ")
    if len(toks) < 3 or not re.fullmatch(r"([A-Z.])\1([A-Z.])\2.*", toks[1] or ""):
        return line
    out, i = [], 0
    while i < len(toks):
        out.append(_undouble(toks[i]))
        if toks[i].endswith("%%"):
            i += 1
            break
        i += 1
    return " ".join(out + toks[i:])


ENTRY = re.compile(r"(?:^|\s)(\d{1,3}) ([A-Z][A-Z0-9 .\-'+&/]*?) (\d{1,3}(?: \d{3})*) (\d{1,2},\d)%")


def _splits(name_and_count: str):
    """'NISSAN QASHQAI 3 9 643' 같은 모호한 경우의 (이름, 대수) 후보들."""
    toks = name_and_count.split(" ")
    for k in range(1, len(toks)):
        cnt = toks[k:]
        if all(re.fullmatch(r"\d{3}", t) for t in cnt[1:]) and re.fullmatch(r"\d{1,3}", cnt[0]):
            yield " ".join(toks[:k]), int("".join(cnt))


def parse_top100(pdf: Path, title: str = "Top 100 VP France") -> pd.DataFrame:
    rows = []
    with pdfplumber.open(pdf) as p:
        for pg in p.pages:
            t = pg.extract_text() or ""
            if title not in t:
                continue
            for line in t.split("\n"):
                line = _fix_bold(line.strip())
                line = re.sub(r"(^|% )(\d{1,3})([A-Z])", r"\1\2 \3", line)  # '100VOLVO' → '100 VOLVO'
                for m in ENTRY.finditer(line):
                    rows.append({"rank": int(m.group(1)), "raw": f"{m.group(2)} {m.group(3)}",
                                 "pct": float(m.group(4).replace(",", "."))})
    if not rows:
        return pd.DataFrame(columns=["rank", "name", "ytd", "pct"])
    df = pd.DataFrame(rows).drop_duplicates("rank")
    # 이름/대수 경계: 비율(pct)과 맞는 쪽을 고른다. 전체 = 대수/비율 의 중앙값(모호하지 않은 행)
    cands = [list(_splits(r)) for r in df["raw"]]
    # 각 행에서 가장 큰 대수 후보(대부분 맞음; 'QASHQAI 3' 같은 행만 틀림)로 전체를 어림한 뒤 중앙값
    big = [(max(x[1] for x in c), p) for c, p in zip(cands, df["pct"]) if c and p > 0]
    total = pd.Series([n / p * 100 for n, p in big]).median()
    names, ytds = [], []
    for c, p in zip(cands, df["pct"]):
        best = min(c, key=lambda x: abs(x[1] / total * 100 - p)) if c else (None, None)
        names.append(best[0])
        ytds.append(best[1])
    df["name"], df["ytd"] = names, ytds
    df["total_est"] = round(total)
    return df[["rank", "name", "ytd", "pct", "total_est"]].sort_values("rank")


def split_brand(name: str) -> tuple[str, str]:
    for two in ("LAND ROVER", "ALFA ROMEO", "ASTON MARTIN", "LYNK & CO", "ROLLS ROYCE", "DS AUTOMOBILES"):
        if name.startswith(two + " "):
            return two, name[len(two) + 1:]
    b, _, rest = name.partition(" ")
    return b, rest


ROMAN = r"(?:\s(?:I|II|III|IV|V|VI|VII|VIII|IX|X))+$"


def classify_fr(name: str):
    """PFA 이름: 모델 칸이 약 9자에서 잘리고('YARIS CRO', 'C5 AIR II', 'OCTAV.IV') 세대 로마숫자가 붙는다."""
    from europe_map import _alias_table, brand_of, norm, squash, _strip_brand
    mk, md = split_brand(name)
    mk = {"M.G.": "MG", "B.M.W.": "BMW"}.get(mk, mk)
    if mk == "MINI" and re.match(r"^MINI\b", md):  # 'MINI MINI IV' = 해치백(Cooper)
        md = "COOPER"
    br = brand_of(mk)
    if br and len(md) >= 8 and br not in ("Mercedes-Benz", "BMW"):  # 이 둘은 정규식 규칙('CLASSE V' 의 V 는 세대 아님)  # 잘린 이름 먼저: 세대 표시를 떼고, 더 긴 별칭 하나의 앞부분이면 그 차종
        base = re.sub(ROMAN, "", norm(md))
        sq = squash(_strip_brand(br, base))
        tab = _alias_table(br)
        if len(sq) >= 5 and not any(pre == sq for pre, _, _ in tab):
            hits = {mm for pre, mm, _ in tab if pre.startswith(sq) and len(pre) > len(sq)}
            if len(hits) == 1:
                return br, hits.pop(), f"truncated:{sq}", "medium"
    return classify(mk, md)


# ---------------------------------------------------------------- 월값

def months_between(a: str, b: str) -> list[str]:
    r = pd.period_range(a, b, freq="M")
    return [str(p) for p in r]


def prev_month(m: str) -> str:
    return str(pd.Period(m, freq="M") - 1)


def build(months: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    need = sorted(set(months) | {prev_month(m) for m in months if not m.endswith("-01")})
    ytd = []
    for mo in need:
        p = pdf_path(mo)
        if p is None:
            continue
        for seg, title in (("VP", "Top 100 VP France"), ("VUL", "Top 100 VUL France")):
            t = parse_top100(p, title)
            t.insert(0, "segment", seg)
            t.insert(0, "Month", mo)
            ytd.append(t)
    ytd = pd.concat(ytd, ignore_index=True)
    cls = {n: classify_fr(n) for n in ytd["name"].dropna().unique()}
    ytd["Brand"] = ytd["name"].map(lambda n: cls.get(n, (None,) * 4)[0])
    ytd["Model"] = ytd["name"].map(lambda n: cls.get(n, (None,) * 4)[1])
    ytd["rule"] = ytd["name"].map(lambda n: cls.get(n, (None,) * 4)[2])
    ytd["confidence"] = ytd["name"].map(lambda n: cls.get(n, (None,) * 4)[3])
    # 월값(이름 단위, 세대 구분 유지) → 우리 차종 합산
    mon = []
    for mo in months:
        for seg in ("VP", "VUL"):
            cur = ytd[(ytd["Month"] == mo) & (ytd["segment"] == seg)].set_index("name")
            if cur.empty:
                continue
            if mo.endswith("-01"):
                prev, has_prev = None, True
            else:
                pv = ytd[(ytd["Month"] == prev_month(mo)) & (ytd["segment"] == seg)]
                has_prev = not pv.empty
                prev = pv.set_index("name")["ytd"]
            for name, r in cur.iterrows():
                if prev is None:
                    val, st = r["ytd"], "ok_january"
                elif not has_prev:
                    val, st = None, "prev_pdf_missing"
                elif name in prev.index:
                    val, st = r["ytd"] - prev[name], "ok"
                else:
                    val, st = None, "prev_missing"  # 전달 상위 100 밖 → 차이 불가
                mon.append({"Month": mo, "segment": seg, "name": name, "ytd": r["ytd"], "monthly": val,
                            "status": st, "Brand": r["Brand"], "Model": r["Model"], "rule": r["rule"],
                            "confidence": r["confidence"]})
    return ytd, pd.DataFrame(mon)


def validate_vs_eea(mon: pd.DataFrame) -> pd.DataFrame:
    """2025 달: PFA 월값(우리 차종 합) vs EEA FR 같은 달. 둘 다 '그 달 신규 등록'이지만 출처(AAA DATA vs EEA)가 다르다."""
    p = CACHE / "eea_monthly_2025_FR-IT_ft.csv"
    if not p.exists():
        return pd.DataFrame()
    from europe_map import classify_frame, refine_by_type  # noqa
    from europe_validate import load_eea  # 전 나라 월별 + 형식(T) 분리(Panda/Grande Panda 등) 적용판
    e = load_eea(2025)
    e = e[e["country"] == "FR"]
    eg = e[e["Model"].notna()].groupby(["Brand", "Model", "Month"])["n"].sum()
    # VP + VUL: 프랑스는 Clio·208 의 상용 등록(société) 판이 VUL 표에 들어가지만 EEA 에서는 M1 이라 같이 센다
    from europe_validate import LCV_LIKE
    f = mon[mon["Model"].notna() & mon["Month"].str.startswith("2025")]
    lcv = [(b, m) in LCV_LIKE for b, m in zip(f["Brand"], f["Model"])]
    f = f[~(pd.Series(lcv, index=f.index) & (f["segment"] == "VUL"))]  # 승합/상용 성격 차종은 VP(=M1 쪽)만 비교
    # 한 우리 차종에 이름이 여럿이면 모두 월값이 있을 때만 합계(일부만 있으면 비교 안 함)
    rows = []
    for (b, m, mo), g in f.groupby(["Brand", "Model", "Month"]):
        if g["monthly"].isna().any():
            continue
        ev = eg.get((b, m, mo))
        rows.append({"check": "fr_pfa_vs_eea_FR_month", "period": mo, "Brand": b, "Model": m,
                     "lcv_like": (b, m) in LCV_LIKE,
                     "collected": g["monthly"].sum(), "reference": ev,
                     "ratio": (g["monthly"].sum() / ev) if ev else None,
                     "note": "PFA VP+VUL top100 YTD diff vs EEA FR M1 (2025 provisional)"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", help="YYYY-MM")
    ap.add_argument("--range", nargs=2, metavar=("FROM", "TO"))
    a = ap.parse_args()
    months = months_between(*a.range) if a.range else [a.month]
    OUT.mkdir(exist_ok=True)
    ytd, mon = build(months)
    ytd.to_csv(OUT / "fr_pfa_ytd.csv", index=False, encoding="utf-8")
    mon.to_csv(OUT / "fr_pfa_monthly.csv", index=False, encoding="utf-8")
    agg = dict(monthly=("monthly", lambda s: s.sum() if s.notna().all() else None),
               n_names=("name", "count"), status=("status", lambda s: ",".join(sorted(set(s)))))
    vp = mon[mon["Model"].notna() & (mon["segment"] == "VP")].groupby(["Brand", "Model", "Month"]).agg(**agg)
    both = mon[mon["Model"].notna()].groupby(["Brand", "Model", "Month"]).agg(**agg)
    ours = pd.concat([vp.assign(segment="VP"), both.assign(segment="VP+VUL")]).reset_index()
    ours.insert(0, "Country_part", "FR")
    ours.to_csv(OUT / "fr_pfa_ours.csv", index=False, encoding="utf-8")
    v = validate_vs_eea(mon)
    if not v.empty:
        v.to_csv(OUT / "validation_fr.csv", index=False, encoding="utf-8")
        r = v["ratio"].dropna()
        print(f"[FR] EEA 비교 {len(r)}건 median={r.median():.3f} 0.9~1.1 비율={((r > .9) & (r < 1.1)).mean():.2f}")
    vp = mon[mon["segment"] == "VP"]
    print(f"[FR] 달 {len(months)}개, VP 이름 {vp['name'].nunique()}개, 대응률(VP 누계 기준) "
          f"{ytd.loc[(ytd.segment == 'VP') & ytd.Model.notna(), 'ytd'].sum() / ytd.loc[ytd.segment == 'VP', 'ytd'].sum():.3f}, "
          f"월값 못 만든 행 {vp['monthly'].isna().sum()}/{len(vp)}")
    print(f"요청 수 {sess().n_requests if _sess else 0}")


if __name__ == "__main__":
    main()
