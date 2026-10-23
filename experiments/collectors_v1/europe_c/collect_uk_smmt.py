"""영국 SMMT 월간 보도자료의 상위 10개 차종(전체 + BEV) → 그 달 대수(부분 목록).

사용: python collect_uk_smmt.py --month 2026-08 [--offline]

쓰임: 영국 DfT VEH0160 은 분기 발표라 분기 전 달은 영국 몫이 전부 추정이 된다. SMMT 보도자료의 상위 차종만이라도
  실제 값으로 덮는다(프랑스·이탈리아 상위 목록과 같은 방식 — 목록에 있는 차종만 덮이고 나머지는 추정).
출처: https://www.smmt.co.uk (robots.txt Disallow 없음, 2026-10-15 확인). 보도자료 주소는 매달 달라서
  post-sitemap 의 최신 묶음에서 달 이름이 든 글을 찾고, 본문에 '<Month> 1 ' 상위 목록이 있는 글을 고른다.
정의: SMMT 신차 등록(승용). 같은 차종이 전체·BEV 두 목록에 있으면 큰 값(전체 목록 값)을 쓴다.
출력: out/uk_smmt_top_<M>.csv (Brand, Model, Month, units, name, list) — 우리 Europe 표기로 대응한 줄만.
"""
from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

from c_common import CACHE, OUT, Session, check_month

D = CACHE / "uk_smmt"
D.mkdir(parents=True, exist_ok=True)
BASE = "https://www.smmt.co.uk"
MONTH_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
            "November", "December"]


def _ranked(seg: str) -> list[tuple[str, int]]:
    """'1 FORD puma 2,795 2 JAECOO 7 2,022 ...' → [(이름, 대수)] (1~10위 차례대로)."""
    out, pos = [], 0
    for k in range(1, 11):
        nxt = rf"(?= {k + 1} )" if k < 10 else r"(?=\s|$)"
        m = re.compile(rf"(?:^|\s){k} (.+?) (\d{{1,3}}(?:,\d{{3}})*){nxt}").search(seg, pos)
        if not m:
            break
        out.append((m.group(1).strip(), int(m.group(2).replace(",", ""))))
        pos = m.end()
    return out


def parse_lists(text: str, month: str) -> list[dict]:
    name = MONTH_EN[int(month[5:7]) - 1]
    rows = []
    a = re.search(rf"\b{name} 1 (.*?)(?:Year-to-date|$)", text)
    if a:
        rows += [dict(list="all", rank=i + 1, name=n, units=v) for i, (n, v) in enumerate(_ranked("1 " + a.group(1)))]
    b = re.search(rf"\b{name} BEV 1 (.*?)(?:Notes to editors|Year-to-date|$)", text)
    if b:
        rows += [dict(list="bev", rank=i + 1, name=n, units=v) for i, (n, v) in enumerate(_ranked("1 " + b.group(1)))]
    return rows


def combine_lists(rows: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    d["key"] = d["name"].str.upper()
    d = d.sort_values("units", ascending=False).drop_duplicates("key")
    return d.drop(columns="key").reset_index(drop=True)


def _text(html: str) -> str:
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t.replace("&nbsp;", " "))


SKIP = r"used|pre-registration|commercial|van|truck|bus|hgv|lorry|international|members|five-minutes|production|cv-"


def find_article(month: str, s: Session) -> str:
    """주소에 달 이름이 없는 글도 있다(예: 2026-05 'two-millionth-electric-car-registered...').
    후보 = (1) 주소에 달 이름 (2) 다음 달·그다음 달에 갱신된 승용 시장 글(주소에 market/registered). 본문에 '<Month> 1 ' 상위 목록이 있는 첫 글을 쓴다."""
    name = MONTH_EN[int(month[5:7]) - 1]
    nxt = str(pd.Period(month, freq="M") + 1)
    idx = s.get(BASE + "/sitemap_index.xml").text
    maps = [u for u in re.findall(r"<loc>([^<]+)</loc>", idx) if re.search(r"/post-sitemap\d*\.xml$", u)]
    maps.sort(key=lambda u: int(re.search(r"(\d*)\.xml$", u).group(1) or 1))
    ent = re.findall(r"<loc>([^<]+)</loc>\s*<lastmod>([^<]+)</lastmod>", s.get(maps[-1]).text)
    other_year = [y for y in re.findall(r"20\d\d", " ".join(u for u, _ in ent)) if y != month[:4]]
    named = [u for u, _ in ent if name.lower() in u.lower() and not re.search(SKIP, u)
             and not any(y in u for y in set(other_year))]
    nxt2 = str(pd.Period(month, freq="M") + 2)    # 사이트 lastmod 는 게시 뒤에 바뀌기도 한다(4월 글 lastmod = 6월 1일)
    dated = [u for u, d in ent if d[:7] in (nxt, nxt2) and not re.search(SKIP, u)
             and re.search(r"market|registered", u)]      # 승용 시장 월간 보도자료는 주소에 market/registered 가 들어 있었다
    cand = list(dict.fromkeys(named[::-1] + dated))[:12]   # 최대 12개만 열어 봄
    for u in cand:
        html = s.get(u).text
        txt = _text(html)
        if f"Top Models {month[:4]} " in txt and [r for r in parse_lists(txt, month) if r["list"] == "all"]:
            (D / f"article_{month}.html").write_text(html, encoding="utf-8")
            (D / f"article_{month}.url.txt").write_text(u + "\n", encoding="utf-8")
            return html
    raise SystemExit(f"{month}: SMMT 보도자료에서 '{name} 1 ...' 상위 목록을 찾지 못함 (후보 {len(cand)}개)")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    month = check_month(a.month)
    p = D / f"article_{month}.html"
    if p.exists():
        html = p.read_text(encoding="utf-8")
    elif a.offline:
        raise SystemExit(f"{month}: 캐시 없음(--offline)")
    else:
        html = find_article(month, Session())
    rows = parse_lists(_text(html), month)
    if len([r for r in rows if r["list"] == "all"]) != 10:
        raise SystemExit(f"{month}: 전체 상위 10개를 다 읽지 못함 ({len(rows)}줄)")
    d = combine_lists(rows)
    sys.path.insert(0, str(OUT.parent.parent / "europe_b"))
    from collect_it import split_brand
    from europe_map import classify
    res = [classify(*split_brand(n.upper())) for n in d["name"]]
    d["Brand"], d["Model"] = [r[0] for r in res], [r[1] for r in res]
    d["Month"] = month
    bad = d[d["Model"].isna()]
    if len(bad):
        print(f"대응 못 한 이름(뺌): {bad['name'].tolist()}", file=sys.stderr)
    out = OUT / f"uk_smmt_top_{month}.csv"
    d[d["Model"].notna()][["Brand", "Model", "Month", "units", "name", "list"]].to_csv(out, index=False, encoding="utf-8-sig")
    print(out)


if __name__ == "__main__":
    main()
