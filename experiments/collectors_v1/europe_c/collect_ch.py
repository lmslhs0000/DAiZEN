"""스위스(+리히텐슈타인) auto-schweiz ModellePW<연도>.xlsx → 그 달 신차 승용 제조사×모델 대수.

사용: python collect_ch.py --month 2026-07 [--offline]

출처: https://www.auto.swiss/ 첫 화면의 'ModellePW<연도>.xlsx' 링크(wp-content/uploads/<연>/<월>/; 매달 같은 이름으로 새로 올림).
  robots.txt: Disallow 빈칸(허용). 원자료 ASTRA/MOFIS.
파일 구조: 시트 12개(Jan, Febr, Mrz, ... Dez) = 1월~그 달 **누계**. 각 시트 A=Marke, B=Modell, C=대수.
  월값 = 그 달 시트 누계 − 전달 시트 누계(1월은 그대로). 'Diverse'(기타)와 'Total' 줄은 빼고, Total 은 대조에 쓴다.
  전달 누계에만 있고 이번 달 누계에 없는 모델은 0으로 보고 차이를 계산(누계는 줄지 않아야 함 → 음수면 경고).
  'Seat / Cupra' 처럼 브랜드를 묶은 줄이 있다 → 대응은 collect_europe.py(europe_map 의 CH 규칙).
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

from c_common import CACHE, EB, SAMPLES, Session, check_month, sha256, write_model_month

D = CACHE / "ch"
D.mkdir(parents=True, exist_ok=True)
HOME = "https://www.auto.swiss/"
SHEETS = ["Jan", "Febr", "Mrz", "Apr", "Mai", "Juni", "Juli", "Aug", "Sept", "Okt", "Nov", "Dez"]


def _sheet_has_month(x: pd.DataFrame, month: str) -> bool:
    """시트 머리('Januar - Juli 2026', 'Stichtag: 31. Juli 2026')가 그 달까지인지 확인."""
    txt = " ".join(str(v) for v in x.iloc[:14].values.ravel() if isinstance(v, (str, int, float)))
    return month[:4] in txt


def read_sheet(x: pd.DataFrame) -> tuple[pd.Series, float | None]:
    hdr = x.index[(x[0] == "Marke") & (x[1] == "Modell")]
    start = hdr[0] + 1 if len(hdr) else 12
    t = x.iloc[start:, [0, 1, 2]].copy()
    t.columns = ["Marke", "Modell", "n"]
    t["n"] = pd.to_numeric(t["n"], errors="coerce")
    t = t[t["Marke"].notna() & t["n"].notna()]
    t["Marke"] = t["Marke"].astype(str).str.strip()
    t["Modell"] = t["Modell"].astype(str).str.strip()
    tot = t.loc[t["Marke"] == "Total", "n"]
    t = t[~t["Marke"].isin(["Total", "Diverse"])]
    return t.groupby(["Marke", "Modell"])["n"].sum(), (float(tot.iloc[0]) if len(tot) else None)


def _key(model: str) -> str:
    """표기 흔들림 정규화: 대소문자·공백·앞자리 0 ('7x'='7X', '001'='1', '01'='1')."""
    k = re.sub(r"\s+", "", str(model)).upper()
    return k.lstrip("0") or k


def month_diff(cur: pd.Series, prev: pd.Series):
    """누계 차이 → 월값. 같은 브랜드에서 이름이 바뀌거나 갈라진 모델(예: 2026-07 BMW 'X3' 누계에서 'iX3' 분리)은
    누계가 줄어든 모델 + 이번 달 처음 나온 모델을 한 묶음으로 차이를 내고, 그 묶음 월값을 이번 달 누계 비율로 나눈다
    (value_note=allocated:reclass). 짝이 없는 음수는 0으로 두고 기록한다."""
    c = cur.reset_index()
    p = prev.reset_index()
    c["k"], p["k"] = c["Modell"].map(_key), p["Modell"].map(_key)
    cg = c.groupby(["Marke", "k"]).agg(n=("n", "sum"), model=("Modell", "last"))
    pg = p.groupby(["Marke", "k"])["n"].sum()
    j = cg.join(pg.rename("n_prev"), how="outer")
    j["model"] = j["model"].fillna(pd.Series(j.index.get_level_values(1), index=j.index))
    j[["n", "n_prev"]] = j[["n", "n_prev"]].fillna(0)
    j["units"] = j["n"] - j["n_prev"]
    j["value_note"] = ""
    fixes = []
    for brand, g in j.groupby(level=0):
        neg = g[g["units"] < 0]
        if neg.empty:
            continue
        new = g[(g["n_prev"] == 0) & (g["n"] > 0)]
        grp = pd.concat([neg, new])
        if new.empty or neg["units"].sum() > -5:  # 짝이 없거나 1~4대 정정(등록 취소 등)은 0으로
            j.loc[neg.index, "units"] = 0.0
            j.loc[neg.index, "value_note"] = "negative_diff_set_0"
            fixes.append(dict(brand=brand, models=neg["model"].tolist(), action="clip0",
                              diff=float(neg["units"].sum())))
            continue
        joint = grp["n"].sum() - grp["n_prev"].sum()
        w = grp["n"] / grp["n"].sum()
        j.loc[grp.index, "units"] = (joint * w).clip(lower=0).values
        j.loc[grp.index, "value_note"] = "allocated:reclass"
        fixes.append(dict(brand=brand, models=grp["model"].tolist(), joint_month=float(joint),
                          action="joint diff split by current YTD"))
    d = j.reset_index().rename(columns={"Marke": "make"})[["make", "model", "units", "value_note"]]
    return d, fixes


def candidates(year: int):
    return [D / f"ModellePW{year}.xlsx", EB / "cache" / f"ch_ModellePW{year}.xlsx",
            SAMPLES / f"ch_autoschweiz_models_{year}-01_{year}-09.xlsx"]


def download(year: int):
    s = Session()
    html = s.get(HOME).text
    (D / "home.html").write_text(html, encoding="utf-8")
    links = re.findall(rf'href="([^"]*ModellePW{year}\.xlsx)"', html)
    if not links:
        raise SystemExit(f"첫 화면에 ModellePW{year}.xlsx 링크 없음")
    url = links[-1] if links[-1].startswith("http") else "https://www.auto.swiss" + links[-1]
    r = s.get(url)
    r.raise_for_status()
    p = D / f"ModellePW{year}.xlsx"
    p.write_bytes(r.content)
    (D / f"ModellePW{year}.url.txt").write_text(url, encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    month = check_month(a.month)
    y, m = int(month[:4]), int(month[5:7])
    path, sheets = None, None
    for p in candidates(y):
        if p.exists():
            xs = pd.read_excel(p, sheet_name=None, header=None)
            names = list(xs)
            if len(names) >= m and _sheet_has_month(xs[names[m - 1]], month) and \
                    xs[names[m - 1]].shape[0] > 30:
                path, sheets = p, xs
                break
    if path is None:
        if a.offline:
            raise SystemExit(f"{month}: 그 달 시트가 있는 캐시 없음(--offline)")
        path = download(y)
        sheets = pd.read_excel(path, sheet_name=None, header=None)
    names = list(sheets)
    cur, tot_cur = read_sheet(sheets[names[m - 1]])
    if m > 1:
        prev, tot_prev = read_sheet(sheets[names[m - 2]])
        d, fixes = month_diff(cur, prev)
        file_total = (tot_cur - tot_prev) if (tot_cur is not None and tot_prev is not None) else None
    else:
        d = cur.reset_index().rename(columns={"Marke": "make", "Modell": "model", "n": "units"})
        d["value_note"] = ""
        fixes, file_total = [], tot_cur
    meta = dict(source_id="CH_AUTOSCHWEIZ", file=str(path), sha256=sha256(path),
                sheet_cur=names[m - 1], sheet_prev=names[m - 2] if m > 1 else None,
                file_total_month=file_total, reclass_fixes=fixes,
                note="월값 = 누계 차이; 'Diverse' 제외라 합계가 Total 차이보다 조금 작을 수 있음",
                definition="Immatrikulationen neuer Personenwagen CH+FL (ASTRA/MOFIS, auto-schweiz)")
    out = d.assign(country="Switzerland", month=month)[["country", "month", "make", "model", "units", "value_note"]]
    out = out[out["units"] > 0]
    print(write_model_month("ch", month, out, meta))


if __name__ == "__main__":
    main()
