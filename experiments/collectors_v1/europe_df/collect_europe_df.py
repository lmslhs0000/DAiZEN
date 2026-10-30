"""유럽(EU+EFTA+UK) 상위 차종 월 판매 — Dataforce 숫자를 옮겨 싣는 기사(L'Automobile·Razão Automóvel)에서.

사용: python collect_europe_df.py --month 2026-08 [--offline]
  → out/df_<M>.csv (Brand, Model, Month, units, how=month|ytd_diff, source), out/unmapped_df_<M>.csv

- 우리 Europe 원본이 Dataforce 첫 발표와 같다(2026-05·07 상위 50이 원본과 한 대도 다르지 않음, 2026-10-15 확인).
  그래서 여기 값은 추정 없는 실제 값(reported)이다. europe_c/collect_europe.py 가 계산값을 이 값으로 바꾼다.
- 그 달 표(상위 50 또는 10)가 있으면 그 값, 없고 누계 표가 이번 달·지난달 모두 있으면 누계 차이.
- 기사는 매달 같은 곳에 실리지 않는다(2026: 1~4월은 상위 10뿐, 5·7·8월 상위 50, 6월은 상반기 누계 50).
  구할 수 있는 만큼 쓰고, 없는 차종은 나라별 계산(추정)으로 남는다.
- 사이트 예절: 두 곳 모두 robots 허용(ACI robots 없음, Razão 는 검색·관리자 경로만 금지), 2초 이상 간격, 기사·sitemap 캐시.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

import df_parse as d

HERE = Path(__file__).resolve().parent
CACHE, OUT = HERE / "cache", HERE / "out"
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "europe_b"))
SITES = {  # 이름: (sitemap 목록, 글 주소 고르는 규칙)
    "aci": ("https://www.lautomobile.aci.it/sitemap_index.xml", r"lautomobile\.aci\.it/attualita/.*(vendute-in-europa|piu-vendute.*europa)"),
    "razao": ("https://www.razaoautomovel.com/sitemap_index.xml",
              r"razaoautomovel\.com/noticias/(?!.*(eletric|hibrid|marcas|plug-in)).*(modelos|carros)-mais-vendidos.*europa|lista-50-carros-mais-vendidos-europa"),
}


def month_values(month_tab, ytd_now, ytd_prev, month):
    """그 달 표 값 우선, 없으면 누계 차이(지난달 누계도 있어야). 1월은 누계 = 그 달. 음수(누계 수정)는 버림."""
    rows = [{"Brand": k[0], "Model": k[1], "units": v, "how": "month"} for k, v in month_tab.items()]
    for k, v in ytd_now.items():
        if k in month_tab:
            continue
        if month.endswith("-01"):
            rows.append({"Brand": k[0], "Model": k[1], "units": v, "how": "ytd_jan"})
        elif k in ytd_prev and v - ytd_prev[k] >= 0:
            rows.append({"Brand": k[0], "Model": k[1], "units": v - ytd_prev[k], "how": "ytd_diff"})
    out = pd.DataFrame(rows, columns=["Brand", "Model", "units", "how"])
    out["Month"] = month
    return out


def override(std, df):
    """계산기 표준 출력(std)에서 Dataforce 값이 있는 차종을 실제 값으로 바꾸고, 계산기에 없던 차종은 더한다."""
    std = std.copy()
    key = ["Brand", "Model"]
    dd = df.set_index(key)
    hit = std.set_index(key).index.isin(dd.index)
    for i in std.index[hit]:
        r = dd.loc[(std.at[i, "Brand"], std.at[i, "Model"])]
        std.loc[i, ["Sales", "value_type", "estimated_share", "source_id", "confidence"]] = \
            [float(r["units"]), "reported", 0.0, "EU_DATAFORCE", "high"]
        std.at[i, "note"] = f"Dataforce {r['how']} ({r['source']}); 계산기 값 대신"
    have = set(map(tuple, std[key].values))
    add = df[[tuple(k) not in have for k in df[key].values]]
    if len(add):
        std = pd.concat([std, pd.DataFrame({"Country": "Europe", "Brand": add["Brand"], "Model": add["Model"], "Month": add["Month"],
                                            "Sales": add["units"].astype(float), "value_type": "reported", "estimated_share": 0.0,
                                            "source_id": "EU_DATAFORCE", "confidence": "high",
                                            "note": "Dataforce " + add["how"] + " (" + add["source"] + ")"})], ignore_index=True)
    return std


def _get(s, url, offline, name):
    p = CACHE / name
    if p.exists():
        return p.read_text(encoding="utf-8", errors="ignore")
    if offline:
        return None
    r = s.get(url)
    if r.status_code != 200:
        return None
    p.write_text(r.text, encoding="utf-8")
    return r.text


def candidates(s, months, offline):
    """두 사이트 sitemap 에서 글 주소 중 규칙에 맞고, 대상 달 다음 1~2달에 고쳐진 글."""
    want = {str(pd.Period(m, freq="M") + k) for m in months for k in (1, 2)}
    out = []
    for site, (index, pat) in SITES.items():
        if offline:
            continue
        idx = s.get(index).text
        maps = [u for u in re.findall(r"<loc>([^<]+)</loc>", idx) if re.search(r"/post-sitemap\d*\.xml$", u)]
        maps.sort(key=lambda u: int(re.search(r"(\d*)\.xml$", u).group(1) or 1))
        for sm in maps[-3:]:                             # 최근 글이 든 마지막 묶음 몇 개만
            for loc, mod in re.findall(r"<loc>([^<]+)</loc>\s*<lastmod>([^<]+)</lastmod>", s.get(sm).text):
                if re.search(pat, loc) and mod[:7] in want:
                    out.append((site, loc))
    return list(dict.fromkeys(out))


def collect(month, offline=False):
    from _common import PoliteSession
    s = None if offline else PoliteSession()
    prev = str(pd.Period(month, freq="M") - 1)
    raw = pd.read_csv(ROOT / "raw" / "data.csv", encoding="utf-8-sig", usecols=["Country", "Brand", "Model"])
    e = raw[raw.Country == "Europe"].drop_duplicates()
    idx = {(b + " " + m).lower(): (b, m) for b, m in zip(e.Brand, e.Model)}
    pages = [(site, url, _get(s, url, offline, f"{site}_{url.rstrip('/').split('/')[-1]}.html"))
             for site, url in candidates(s, [month, prev], offline)]
    pages += [(p.name.split("_")[0], p.name, p.read_text(encoding="utf-8", errors="ignore")) for p in CACHE.glob("*.html")
              if offline]
    best = {"month": ({}, ""), "ytd_now": ({}, ""), "ytd_prev": ({}, "")}
    unm = []
    m_now, m_prev = int(month[5:7]), int(prev[5:7])
    for site, url, page in pages:
        if not page:
            continue
        for kind, m, body in d.page_tables(page, int(month[:4])):    # 1월은 지난달 누계가 필요 없음(누계 = 그 달)
            slots = ([("month", month)] if kind == "month" and m == m_now else []) +                     ([("ytd_now", month)] if kind == "ytd" and m == m_now else []) +                     ([("ytd_prev", prev)] if kind == "ytd" and m == m_prev else [])
            for slot, mm in slots:
                vals = {}
                for name, cur, _ in body:
                    k = d.to_ours(name, idx)
                    if k is None:
                        unm.append({"Month": mm, "name": name, "units": cur, "source": url})
                    else:
                        vals[k] = vals.get(k, 0) + cur
                if len(vals) > len(best[slot][0]):          # 더 긴 표(상위 50 > 상위 10)
                    best[slot] = (vals, url)
    v = month_values(best["month"][0], best["ytd_now"][0], best["ytd_prev"][0], month)
    v["source"] = [best["month"][1] if h == "month" else best["ytd_now"][1] for h in v["how"]]
    OUT.mkdir(exist_ok=True)
    v[["Brand", "Model", "Month", "units", "how", "source"]].to_csv(OUT / f"df_{month}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(unm, columns=["Month", "name", "units", "source"]).drop_duplicates().to_csv(
        OUT / f"unmapped_df_{month}.csv", index=False, encoding="utf-8-sig")
    print(f"{month}: 그 달 표 {len(best['month'][0])}개, 누계 {len(best['ytd_now'][0])}/{len(best['ytd_prev'][0])}개 → "
          f"{len(v)}개 차종 ({v['how'].value_counts().to_dict()}), 이름 못 맞춤 {len(unm)}", file=sys.stderr)
    return OUT / f"df_{month}.csv"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    print(collect(a.month, a.offline))


if __name__ == "__main__":
    main()
