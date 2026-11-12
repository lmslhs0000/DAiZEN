"""목록에 없는 차종 알림 — 수집처에는 판매가 있는데 우리 차종 목록(대응표)에 없는 줄을 한 표로.

collect_month.py 가 수집 뒤 부르고 raw/incoming/<M>/new_models.csv 와 report.md '목록에 없는 큰 차종' 절을 쓴다.
출처별로 찾을 수 있는 것만 모은다:
- 독일: KBA 수집기 로그의 'KBA lines with sales not in mapping'
- 한국: korea/out/unmapped_korea_<M>.csv (다나와 줄 중 대응 안 된 것)
- 유럽 전체: europe_df/out/unmapped_df_<M>.csv (Dataforce 상위 표 중 대응 안 된 것 = 유럽 전체 대수)
- 유럽 나라별: europe_c/out/unmapped_europe_<M>.csv (나라 합)
일본·미국·캐나다·중국 수집기는 우리 차종의 페이지만 찾아가서 새 차종을 알 수 없다(README 참고: 순위 기사·브랜드 발표로 사람이 확인).
추가는 사람이 한다: how_to_add 의 대응표에 한 줄을 넣으면 다음 수집부터 들어온다(새 차종은 원본에 새 계열로 쌓임).
"""
import re

import pandas as pd

COLUMNS = ["market", "source", "Month", "name", "units", "note"]
HOW = {
    "KBA": "experiments/data_audit_v1/germany_kba_mapping.csv 에 KBA 줄 → 우리 Germany 차종 한 줄",
    "Danawa": "korea/mapping.csv 에 다나와 줄 → 우리 South Korea 차종 한 줄 (build_mapping.py 참고)",
    "Dataforce": "europe_df/df_parse.py ALIAS 또는 우리 Europe 차종 이름과 같게(새 차종이면 raw 에 Europe 계열로 쌓임)",
    "Europe countries": "europe_b/europe_map.py 분류 규칙 또는 europe_a/mapping_europe_a.csv",
}


def from_germany_log(text, month):
    m = re.search(r"KBA lines with sales not in mapping[^:]*:\s*(.+)", text)
    rows = []
    if m:
        for part in m.group(1).split(";"):
            mm = re.match(r"\s*(.+?)/(.+?)=(\d+(?:\.\d+)?)\s*$", part)
            if mm:
                rows.append({"market": "Germany", "source": "KBA", "Month": month, "name": f"{mm.group(1)} {mm.group(2)}",
                             "units": float(mm.group(3)), "note": ""})
    return pd.DataFrame(rows, columns=COLUMNS)


def from_korea(df, month):
    d = df[df["status"].astype(str) != "seen_before_not_ours"] if "status" in df else df
    return pd.DataFrame({"market": "South Korea", "source": "Danawa", "Month": month,
                         "name": d["brand_name"].astype(str) + " " + d["line"].astype(str), "units": d["sales"].astype(float),
                         "note": d["status"].astype(str)}, columns=COLUMNS)


def from_dataforce(df, month):
    d = df[df["Month"].astype(str) == month] if "Month" in df else df
    return pd.DataFrame({"market": "Europe", "source": "Dataforce", "Month": month, "name": d["name"], "units": d["units"].astype(float),
                         "note": "유럽 전체(Dataforce 상위 표)"}, columns=COLUMNS)


def from_europe_countries(df, month):
    g = df.groupby("model").agg(units=("units", "sum"), note=("country", lambda s: "+".join(sorted(set(s))))).reset_index()
    return pd.DataFrame({"market": "Europe", "source": "Europe countries", "Month": month, "name": g["model"],
                         "units": g["units"], "note": g["note"]}, columns=COLUMNS)


def combine(parts, min_units=100):
    d = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame(columns=COLUMNS)
    d = d[d["units"] >= min_units].sort_values("units", ascending=False).reset_index(drop=True)
    d["how_to_add"] = d["source"].map(HOW)
    return d
