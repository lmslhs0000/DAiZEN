"""mapping.csv: europe_c 나라 출처 문자열 → 우리 Europe (Brand, Model). split_rules.csv: 배분·추정 규칙.

사용: python build_mapping.py   (out/<cc>_model_month_<YYYY-MM>.csv 중 2026-07 이하 달 전부를 모아 만든다)
분류기는 collect_europe.classify_c(= REWRITES + europe_b/europe_map.classify)와 같다.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

import collect_europe as ce
from c_common import HERE, OUT, LAST_ALLOWED_TEST_MONTH

SRC = {"se": "SE_MOBILITYSWEDEN", "ch": "CH_AUTOSCHWEIZ", "fi": "FI_TRAFICOM_105", "ie": "IE_SIMI|IE_CSO_TEM20",
       "ro": "RO_DGPCI", "lu": "LU_SNCA_DELTA", "pl": "PL_CEPIK"}


def main():
    rows = []
    for p in glob.glob(str(OUT / "*_model_month_*.csv")):
        m = re.match(r"([a-z]{2})_model_month_(\d{4}-\d{2})(.*)\.csv$", Path(p).name)
        if not m or m.group(2) > LAST_ALLOWED_TEST_MONTH:
            continue
        cc, month, suf = m.groups()
        d = pd.read_csv(p, keep_default_na=False)
        d["cc"], d["file"] = cc, Path(p).name
        rows.append(d[["cc", "make", "model", "units", "file"]])
    a = pd.concat(rows, ignore_index=True)
    g = a.groupby(["cc", "make", "model"], as_index=False).agg(units=("units", "sum"), files=("file", "nunique"))
    out = []
    lump_of = {(L["brand"], mm): L for L in ce.LUMPS for mm in L["members"]}
    for r in g.itertuples(index=False):
        b, mo, rule, conf = ce.classify_c(r.make, r.model)
        note = "출처 문자열 다듬기: " + ce._rewrite(r.model) if ce._rewrite(r.model) != " ".join(str(r.model).upper().split()) else ""
        L = lump_of.get((b, mo))
        if L and r.cc.upper() in L["lumped"]:
            note = (note + "; " if note else "") + f"lumped → {'/'.join(L['members'])} 그 달 비율로 나눔(split_rules LUMP)"
        if b is None:
            note = (note + "; " if note else "") + "unmapped:" + rule
        out.append(dict(Country="Europe", Brand=b or "", Model=mo or "", source_id=SRC[r.cc],
                        source_key=f"{r.make} | {r.model}", rule=rule, confidence=conf or "", note=note,
                        units_in_test_files=round(float(r.units), 1)))
    mp = pd.DataFrame(out).sort_values(["source_id", "units_in_test_files"], ascending=[True, False])
    mp.to_csv(HERE / "mapping.csv", index=False, encoding="utf-8-sig")
    s = mp.groupby("source_id").apply(lambda x: x.loc[x["Model"] != "", "units_in_test_files"].sum()
                                      / x["units_in_test_files"].sum())
    print("mapped share by source (test files):", s.round(4).to_dict(), "rows", len(mp))

    sr = [dict(rule_id="EST_COUNTRY_SHARE", country="(안 덮인 나라 전부)", source_id="EEA_CO2_2025P+UK_DFT+CH_AUTOSCHWEIZ",
               formula="Sales = V + V*Wu/Wc; Wc = sum_{c covered} share[s,c]*f[c,m], Wu = sum_{c not covered} share[s,c]*f[c,m]; V = 덮인 나라 합",
               inputs="europe_b/eea_shares_<연도>.csv(n_2025_adj, 완전도 보정), seasonal_factors_<연도>.csv",
               fallback="차종 몫 없음 → 브랜드 몫 → 시장 몫(confidence low). 짝(PAIR_GROUPS)은 짝 합계 몫",
               limits="몫은 전년 연간 값(신차·단종·나라별 출시 시점 반영 못 함). Wc < 0.10 이면 값 없음. 밴(LCV_LIKE)은 추정 안 함"),
          dict(rule_id="SEASONAL_FACTOR", country="모든 나라", source_id="EEA monthly 2025 + DfT VEH1153 + auto-schweiz 2025",
               formula="f[c,m] = (n[c,m]/N[c]) / (E[m]/E), 0.4~2.5 로 자름; 완전도(ACEA/EEA) 0.9~1.1 밖 나라(PL·FI)는 1",
               inputs="europe_b/cache/eea_monthly_2025_ALL_ft.csv, europe_a/cache/uk/veh1153a_raw.csv, europe_b/cache/ch_ModellePW2025.xlsx",
               fallback="1", limits="IE 7월 2.5(상한), 영국 3·9월 번호판 달 반영"),
          dict(rule_id="DE_GROUP_ALLOC", country="Germany", source_id="DE_KBA_FZ10",
               formula="KBA 한 줄(우리 Germany 계열)이 Europe 여러 계열을 담으면 EEA 2025 독일 대수 비율로 나눔",
               inputs="europe_a/out/de_links.csv (de_group 69행)", fallback="우리 Europe 2025 판매 비율 → 균등",
               limits="MINI 한 줄=Cooper+Countryman+Aceman 등. 새 차종(C-HR+, e Vitara)은 LUMP 규칙이 덮어씀"),
          dict(rule_id="CH_RECLASS", country="Switzerland", source_id="CH_AUTOSCHWEIZ",
               formula="누계 차이가 음수(5대 이상)인 모델 + 이번 달 처음 나온 같은 브랜드 모델을 한 묶음으로 차이 → 이번 달 누계 비율로 나눔",
               inputs="ModellePW<연도>.xlsx 그 달·전달 시트", fallback="짝 없거나 1~4대면 0",
               limits="2026-07 BMW X3 누계에서 iX3 분리(묶음 월값 397대)")]
    for L in ce.LUMPS:
        sr.append(dict(rule_id="LUMP_" + "_".join(re.sub(r"\W", "", m) for m in L["members"]),
                       country=",".join(L["lumped"]), source_id="(그 나라 출처)",
                       formula=f"{L['brand']} {' + '.join(L['members'])} 합을 그 달 따로 내는 나라("
                               f"{','.join(L['sep']) if L['sep'] else 'lumped 아닌 모든 나라'}) 합 비율로 나눔",
                       inputs="같은 달 다른 나라 수집 결과", fallback="EEA 2025 유럽 비율",
                       limits="따로 내는 나라가 그 달 없으면(예: 7월 영국 미발표) 비율이 SE 하나로 정해짐"))
    sr.append(dict(rule_id="UK_Q2M_FUEL", country="United Kingdom", source_id="UK_DFT_VEH0160",
                   formula="europe_a/split_rules.csv 참고(분기 → 연료별 전국 월 비율)", inputs="europe_a",
                   fallback="분기 미발표 달은 영국 몫 추정(EST_COUNTRY_SHARE)", limits="value_type=split"))
    pd.DataFrame(sr).to_csv(HERE / "split_rules.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
