"""유럽 합계 계산기: 나라별 수집 결과 → 우리 Country=Europe 차종 월 값(표준 출력 계약).

사용: python collect_europe.py --month 2026-07 [--collect] [--offline] [--shares <eea_shares_YYYY.csv>]
  --collect : europe_c 나라 수집기(se ch fi ie ro lu pl)를 먼저 실행(--offline 은 그대로 넘김).
              europe_a(독·스·네·영)·europe_b(프·이) 수집기는 각 폴더에서 따로 실행한다(출력 파일만 읽음).
  끝에 out/std_europe_<YYYY-MM>.csv 경로 한 줄을 출력.

입력(그 달 파일이 있는 나라만 '덮임'):
  DE  rules_jp_na_de/out/germany_<M>_<M>.csv → germany_hist.csv → (시험용, ≤2026-07) raw/data.csv Germany
      + data_audit_v1/kba_fz10_long(우리 Germany 에 없는 KBA 줄). Germany 계열 → Europe 계열 = europe_a/out/de_links.csv.
      KBA 한 줄이 Europe 여러 계열을 담으면(Yaris+Yaris Cross 등) EEA 2025 독일 대수 비율로 나눔(allocated).
  ES NL UK  europe_a/out/{es,nl,uk}_model_month.csv + europe_a/mapping_europe_a.csv(source_key; 없으면 europe_map 분류).
      UK 는 분기값을 월로 나눈 값(split). 그 달 UK 가 없으면(분기 발표 전) 영국 몫은 추정.
  FR  europe_b/out/fr_pfa_ours.csv(상위 100, VP+VUL, 누계 차이) — 그 달 목록에 있는 차종만 덮임.
  IT  europe_b/out/it_unrae_ours.csv(상위 50) — 그 달 목록에 있는 차종만 덮임.
  SE CH FI IE RO LU PL  europe_c/out/<cc>_model_month_<M>.csv(전 차종). PL 은 meta.complete 일 때만 사용.
추정: 차종 s 의 나라별 몫 share[s,c] = EEA 2025(+UK DfT, CH auto-schweiz, 완전도 보정) — eea_shares_<연도>.csv.
  나라별 달 계절 계수 f[c,m](2025 나라 월 분포 ÷ 유럽 월 분포; 완전도 0.9~1.1 밖이면 1)를 곱해
     덮인 무게 Wc = Σ_{c 덮임} share·f, 안 덮인 무게 Wu = Σ_{c 안 덮임} share·f
     값 = 덮인 합 + 덮인 합 × Wu / Wc,  estimated_share = (덮인 합 × Wu / Wc) / 값
  차종 몫이 없거나 전년 합 1,000대 미만(2025 말·2026 출시)이면 브랜드 몫 → 시장 전체 몫(confidence low).
  덮인 몫 Wc/(Wc+Wu) < 0.10 이면 값을 만들지 않는다(2026-05·07 시험: Wc 0.05~0.10 구간 WAPE 53~67%)(checks 에 low_coverage). 덮인 합 0 이면 Wc ≥ 0.5 일 때만 0 을 쓴다.
밴(LCV_LIKE: Ducato·Transit·Sprinter 등 우리 정의 ≠ 승용 M1)은 추정하지 않고 std 에서 빼서
  out/flagged_vans_europe_<M>.csv 에 덮인 나라 승용 합만 표시.
value_type: estimated(추정 몫>0) > allocated(DE 묶음·묶인 이름 나눔·CH 이름 바뀜) > split(영국 분기→월) > reported.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from c_common import EA, EB, HERE, OUT, RAW_DATA, ROOT, LAST_ALLOWED_TEST_MONTH, check_month

sys.path.insert(0, str(EB))
from europe_map import classify, our_models  # noqa: E402
from europe_validate import LCV_LIKE, PAIR_GROUPS  # noqa: E402

AUDIT = ROOT / "experiments" / "data_audit_v1"
DE_DIR = HERE.parent / "rules_jp_na_de" / "out"
SOURCE_ID = "EUROPE_CALC_V1"
C_COUNTRIES = {"SE": "Sweden", "CH": "Switzerland", "FI": "Finland", "IE": "Ireland", "RO": "Romania",
               "LU": "Luxembourg", "PL": "Poland"}
SRC_ID = {"DE": "DE_KBA_FZ10", "ES": "ES_DGT_MATRABA", "NL": "NL_RDW", "UK": "UK_DFT_VEH0160", "FR": "FR_PFA_TOP100",
          "IT": "IT_UNRAE_TOP50", "SE": "SE_MOBILITYSWEDEN", "CH": "CH_AUTOSCHWEIZ", "FI": "FI_TRAFICOM_105",
          "IE": "IE_SIMI|IE_CSO_TEM20", "RO": "RO_DGPCI", "LU": "LU_SNCA_DELTA", "PL": "PL_CEPIK"}
MIN_COVER, ZERO_COVER = 0.10, 0.5
MIN_SERIES_PREV_YEAR = 1000   # 전년 유럽 합이 이보다 작은 차종(새 차종 사전 등록 등)은 브랜드 몫을 쓴다

# 출처 문자열 다듬기(분류 전, 대문자 기준). europe_map 은 건드리지 않고 여기서만.
REWRITES = [
    (r"\bTESLA MOTORS\b", "TESLA"),
    (r"\b(\d)-SARJA\b", r"\1 SERIES"),                 # FI 'BMW 1-sarja'
    (r"\b([A-Z])-(SARJA|KLASS)\b", r"\1-CLASS"),        # FI 'E-sarja', SE 'E-KLASS'
    (r"\bG WAGON\b", "G-CLASS"),
    (r"\bMGS(\d)\b", r"S\1"),                           # IE 'MG MGS5' = MG S5
    (r"\bEX/XC40\b", "XC40"),                           # SE 묶음 → 아래 LUMPS 로 XC40/EX40 나눔
    (r"\bEC/C40\b", "C40"),
    (r"\bS/V60\b", "S60"),
    (r"\bS/V90\w*", "S90"),
    (r"^(TOYOTA )?URBAN$", r"\1URBAN CRUISER"),          # FI 9자 잘림
    (r"^LR / RANGE ROVER$", "RANGE ROVER"),
    (r"^DS ?8\b", "N8"),                                # CH 'DS8' = DS N°8
]
# 연료·세대 구분 없이 한 이름으로 내는 나라(lumped)와 따로 내는 나라(sep, None = lumped 아닌 모든 나라).
# lumped 나라의 두 차종 값을 합친 뒤, 그 달 sep 나라들의 합 비율로 나눔(없으면 EEA 2025 유럽 비율) → allocated.
# 확인 근거(2026-04~07 원자료 이름): SE 'VOLVO EX/XC40'; CH 'A6'·FI 'AUDI A6' 만 있음; C-HR+ 를 따로 내는 곳은
# SE('TOYOTA C-HR+')·UK(DfT) 뿐(ES DGT·NL RDW·KBA 줄은 'C-HR' 하나); e Vitara 는 ES·NL·UK·SE·RO·LU·PL 이 따로 냄.
LUMPS = [
    dict(lumped=["SE"], brand="Volvo", members=["XC40", "EX40"], sep=None),
    dict(lumped=["CH", "FI"], brand="Audi", members=["A6", "A6 e-tron"], sep=None),
    dict(lumped=["DE", "ES", "NL", "FR", "IT", "CH", "FI", "IE", "RO", "LU", "PL"], brand="Toyota",
         members=["C-HR", "C-HR+"], sep=["SE", "UK"]),
    dict(lumped=["DE", "CH", "FI", "IE"], brand="Suzuki", members=["Vitara/Grand Vitara", "e Vitara"],
         sep=["ES", "NL", "UK", "SE", "RO", "LU", "PL"]),
]


# ------------------------------------------------------------------ 공통
def _rewrite(s: str) -> str:
    import re
    t = " ".join(str(s).upper().split())
    for rx, rep in REWRITES:
        t = re.sub(rx, rep, t)
    return t


_cls_cache: dict = {}


def classify_c(make, model, fuel=None):
    k = (make, model, fuel)
    if k not in _cls_cache:
        _cls_cache[k] = classify(_rewrite(make), _rewrite(model), fuel)
    return _cls_cache[k]


def latest_file(pattern_dirs: list[Path], stem: str, before_year: int) -> Path | None:
    """eea_shares_<YYYY>.csv 중 기준 연도보다 앞선 가장 최근 것."""
    best = None
    for d in pattern_dirs:
        for p in glob.glob(str(d / f"{stem}_*.csv")):
            y = Path(p).stem.split("_")[-1]
            if y.isdigit() and int(y) < before_year and (best is None or int(y) > best[0]):
                best = (int(y), Path(p))
    return best[1] if best else None


# ------------------------------------------------------------------ 나라별 입력
def load_de(month: str, raw: pd.DataFrame, log: list) -> tuple[pd.DataFrame | None, str]:
    links = pd.read_csv(EA / "out" / "de_links.csv", keep_default_na=False)
    links = links[links["Brand"] != ""]
    src = None
    p1 = DE_DIR / f"germany_{month}_{month}.csv"
    if p1.exists():
        g, src = pd.read_csv(p1), p1.name
    else:
        ph = DE_DIR / "germany_hist.csv"
        if ph.exists():
            g = pd.read_csv(ph)
            g = g[g["Month"] == month]
            src = "germany_hist.csv" if len(g) else None
        if not src and month <= LAST_ALLOWED_TEST_MONTH:
            g = raw[(raw["Country"] == "Germany") & (raw["Month"] == month)].rename(columns={"Sales": "value"})
            src = "raw/data.csv Germany (시험용)" if len(g) else None
    if not src:
        log.append(dict(check="source", key="DE", status="missing", note="독일 입력 없음 → 독일 몫 추정"))
        return None, ""
    g = g.rename(columns={"value": "v"}) if "value" in g.columns else g.rename(columns={"Sales": "v"})
    gv = g.groupby(["Brand", "Model"])["v"].sum()
    rows = []
    for (db, dm), grp in links.groupby(["de_brand", "de_model"]):
        v = gv.get((db, dm))
        if v is None or pd.isna(v):
            continue
        mem = list(zip(grp["Brand"], grp["Model"]))
        for b, m in mem:
            rows.append(dict(Brand=b, Model=m, units=float(v), de_key=f"{db}|{dm}", n_mem=len(mem),
                             conf=grp.loc[(grp["Brand"] == b) & (grp["Model"] == m), "confidence"].iloc[0]))
    d = pd.DataFrame(rows)
    # KBA 줄 직접(우리 Germany 에 없는 줄)
    kf = pd.read_csv(EA / "out" / "de_kba_fallback.csv", keep_default_na=False)
    kl_p = AUDIT / "kba_fz10_long_2016_2026-07.csv"
    if kl_p.exists():
        kl = pd.read_csv(kl_p, usecols=["period", "kba_brand", "kba_model", "sales", "row_type"])
        kl = kl[(kl["period"] == month) & (kl["row_type"] == "model")]
        for r in kf.itertuples(index=False):
            x = kl[(kl["kba_brand"] == r.kba_brand) & (kl["kba_model"] == r.kba_model)]
            if len(x):
                rows.append(dict(Brand=r.Brand, Model=r.Model, units=float(x["sales"].sum()),
                                 de_key=f"KBA:{r.kba_brand}|{r.kba_model}", n_mem=1, conf="medium"))
        d = pd.DataFrame(rows)
    log.append(dict(check="source", key="DE", status="ok", got=float(g["v"].sum()), note=src))
    return d, src


def allocate_de(d: pd.DataFrame, shares: pd.DataFrame, eu2025: pd.Series) -> pd.DataFrame:
    """DE 묶음(한 KBA 줄 → Europe 여러 계열): EEA 2025 독일 대수 비율로 나눔."""
    de_sh = shares[shares["country"] == "DE"].set_index(["Brand", "Model"])["n_2025_adj"]
    out = []
    for key, g in d.groupby("de_key"):
        if len(g) == 1:
            r = g.iloc[0].to_dict()
            r["alloc"] = False
            out.append(r)
            continue
        w = np.array([de_sh.get((b, m), 0.0) for b, m in zip(g["Brand"], g["Model"])], float)
        if w.sum() <= 0:
            w = np.array([eu2025.get((b, m), 0.0) for b, m in zip(g["Brand"], g["Model"])], float)
        if w.sum() <= 0:
            w = np.ones(len(g))
        w = w / w.sum()
        for (i, r), wi in zip(g.iterrows(), w):
            r = r.to_dict()
            r["units"] = r["units"] * wi
            r["alloc"] = True
            out.append(r)
    o = pd.DataFrame(out)
    return o.groupby(["Brand", "Model"], as_index=False).agg(units=("units", "sum"), alloc=("alloc", "max"),
                                                             conf=("conf", "min"))


def load_ea(cc: str, month: str, log: list, unm: list):
    fn = {"ES": "es_model_month.csv", "NL": "nl_model_month.csv", "UK": "uk_model_month.csv"}[cc]
    p = EA / "out" / fn
    d = pd.read_csv(p, keep_default_na=False) if p.exists() else pd.DataFrame()
    d = d[d["month"] == month] if len(d) else d
    if d.empty:
        log.append(dict(check="source", key=cc, status="missing",
                        note=("영국 분기 미발표 → 영국 몫 추정" if cc == "UK" else "그 달 파일 없음 → 몫 추정")))
        return None
    mp = pd.read_csv(EA / "mapping_europe_a.csv", keep_default_na=False)
    mp = mp[mp["source_id"] == SRC_ID[cc]].drop_duplicates("source_key").set_index("source_key")
    if cc == "UK":
        d["source_key"] = d["make"] + " | " + d["model"] + " | " + d["dft_model"] + " | " + d["fuel"]
    else:
        d["source_key"] = d["make"] + " | " + d["model"]
    rows = []
    for r in d.itertuples(index=False):
        if r.source_key in mp.index:
            mr = mp.loc[r.source_key]
            b, m, rule, conf = mr["Brand"], mr["Model"], "map:" + mr["rule"], mr["confidence"]
        else:
            b, m, rule, conf = classify_c(r.make, r.dft_model if cc == "UK" else r.model,
                                          r.fuel if cc == "UK" else None)
            rule = "classify:" + rule
        rows.append(dict(Brand=b or "", Model=m or "", units=float(r.units), rule=rule, conf=conf or "",
                         make=r.make, model=r.model))
    x = pd.DataFrame(rows)
    _unmapped(cc, x, unm)
    log.append(dict(check="source", key=cc, status="ok", got=float(x["units"].sum()),
                    ratio=round(float(x.loc[x["Model"] != "", "units"].sum() / x["units"].sum()), 4),
                    note="mapped share" + (" (UK 분기→월 split)" if cc == "UK" else "")))
    x = x[x["Model"] != ""]
    return x.groupby(["Brand", "Model"], as_index=False).agg(units=("units", "sum"), conf=("conf", "min"))


def load_top(cc: str, month: str, log: list):
    if cc == "FR":
        p = EB / "out" / "fr_pfa_ours.csv"
        d = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["Month"])
        d = d[(d["Month"] == month) & d["status"].isin(["ok", "ok_january"])]
        # 한 차종에 'VP'(승용 표만)와 'VP+VUL'(승용+업무용 표 합) 두 줄이 있다 → 승용 차종은 VP+VUL(EEA 와 맞음),
        # VP+VUL 줄이 없으면 VP. (둘을 더하면 두 번 셈)
        d = d.sort_values("segment", key=lambda s: s.map({"VP+VUL": 0, "VP": 1}))
        d = d.drop_duplicates(["Brand", "Model"], keep="first")
        d = d.rename(columns={"monthly": "units"})
    else:
        p = EB / "out" / "it_unrae_ours.csv"
        d = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["Month"])
        d = d[d["Month"] == month].rename(columns={"n": "units"})
    if d.empty:
        log.append(dict(check="source", key=cc, status="missing", note="그 달 상위 목록 없음 → 몫 추정"))
        return None
    log.append(dict(check="source", key=cc, status="ok", got=float(d["units"].sum()),
                    note=f"상위 목록 {len(d)}개 차종만 덮임"))
    return d.groupby(["Brand", "Model"], as_index=False)["units"].sum().assign(conf="high")


def load_uk_smmt(month: str, log: list):
    p = OUT / f"uk_smmt_top_{month}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, encoding="utf-8-sig")
    log.append(dict(check="source", key="UK_SMMT", status="ok", got=float(d["units"].sum()),
                    note=f"SMMT 보도자료 상위 {len(d)}개 차종만 덮임(전체·BEV 상위 10)"))
    return d.groupby(["Brand", "Model"], as_index=False)["units"].sum().assign(conf="high")


def load_c(cc: str, month: str, log: list, unm: list):
    p = OUT / f"{cc.lower()}_model_month_{month}.csv"
    mp = OUT / f"{cc.lower()}_meta_{month}.json"
    if not p.exists():
        log.append(dict(check="source", key=cc, status="missing", note="수집 결과 없음 → 몫 추정"))
        return None, {}
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    if cc == "PL" and not meta.get("complete", False):
        log.append(dict(check="source", key=cc, status="partial_unused", got=meta.get("units_total"),
                        note="폴란드 부분 수집(도 일부) → 쓰지 않고 몫 추정"))
        return None, meta
    if cc == "IE" and meta.get("source_id") == "IE_CSO_TEM20":
        # CSO 'licensed for the first time' 는 연 합계는 EEA 와 같으나(2025: 0.965) 달 배치가 다르다
        # (2025 월 비율 0.67~3.05: 1·7월 번호판 달이 뒤 달로 밀림) → 달 값으로 쓰지 않고 아일랜드 몫 추정.
        log.append(dict(check="source", key=cc, status="timing_unused", got=meta.get("units_total"),
                        note="CSO TEM20 은 달 배치가 등록일과 다름 → 쓰지 않음(SIMI 캐시가 있으면 SIMI 사용)"))
        return None, meta
    d = pd.read_csv(p, keep_default_na=False)
    rows = []
    for r in d.itertuples(index=False):
        b, m, rule, conf = classify_c(r.make, r.model)
        rows.append(dict(Brand=b or "", Model=m or "", units=float(r.units), rule=rule, conf=conf or "",
                         make=r.make, model=r.model,
                         alloc=str(getattr(r, "value_note", "")).startswith("allocated")))
    x = pd.DataFrame(rows)
    _unmapped(cc, x, unm)
    log.append(dict(check="source", key=cc, status="ok", got=float(x["units"].sum()),
                    expected=meta.get("file_total", meta.get("file_total_month")),
                    ratio=round(float(x.loc[x["Model"] != "", "units"].sum() / max(x["units"].sum(), 1)), 4),
                    note=f"mapped share; {meta.get('source_id', '')}; total_check={meta.get('total_check', '')}"))
    x = x[x["Model"] != ""]
    return x.groupby(["Brand", "Model"], as_index=False).agg(
        units=("units", "sum"), conf=("conf", "min"), alloc=("alloc", "max")), meta


def _unmapped(cc, x, unm):
    u = x[x["Model"] == ""]
    if len(u):
        g = u.groupby(["make", "model", "rule"], as_index=False)["units"].sum()
        g.insert(0, "country", cc)
        unm.append(g)


# ------------------------------------------------------------------ 몫·계절 계수
def seasonal_factors(year: int, comp: pd.Series) -> pd.DataFrame:
    """f[c,m] = (나라 c 의 m월 / c 연간) ÷ (유럽 m월 / 유럽 연간). EEA 월 + UK VEH1153 + CH auto-schweiz."""
    cache = HERE / f"seasonal_factors_{year}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    e = pd.read_csv(EB / "cache" / f"eea_monthly_{year}_ALL_ft.csv", usecols=["MS", "Month", "n"],
                    keep_default_na=False)
    t = e.groupby(["MS", "Month"])["n"].sum().rename("n").reset_index().rename(columns={"MS": "country"})
    u = pd.read_csv(EA / "cache" / "uk" / "veh1153a_raw.csv", skiprows=4)
    u = u[(u["Geography"] == "United Kingdom") & (u["Date Interval"] == "Monthly") & (u["BodyType"] == "Cars")
          & (u["Keepership"] == "Total") & u["Date"].astype(str).str.endswith(str(year))]
    u = pd.DataFrame(dict(country="UK", Month=pd.to_datetime(u["Date"], format="%B %Y").dt.strftime("%Y-%m"),
                          n=pd.to_numeric(u["Total"], errors="coerce") * 1000))
    sys.path.insert(0, str(HERE))
    from collect_ch import read_sheet
    chs = pd.read_excel(EB / "cache" / f"ch_ModellePW{year}.xlsx", sheet_name=None, header=None)
    tots = [read_sheet(v)[1] for v in list(chs.values())[:12]]
    ch = pd.DataFrame(dict(country="CH", Month=[f"{year}-{i:02d}" for i in range(1, 13)],
                           n=np.diff([0.0] + tots)))
    a = pd.concat([t, u, ch], ignore_index=True)
    a = a[a["Month"].str.startswith(str(year))]
    eu_m = a.groupby("Month")["n"].sum()
    eu_share = eu_m / eu_m.sum()
    rows = []
    for c, g in a.groupby("country"):
        g = g.set_index("Month")["n"]
        cf = comp.get(c, 1.0)
        reliable = 0.9 <= cf <= 1.1 and g.min() > 0
        for mth, v in g.items():
            f = (v / g.sum()) / eu_share[mth] if reliable else 1.0
            rows.append(dict(country=c, month_num=int(mth[5:7]), f=round(float(np.clip(f, 0.4, 2.5)), 4),
                             reliable=reliable, completeness=cf))
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


# ------------------------------------------------------------------ 계산
def build(month: str, shares_path: Path | None = None, min_cover: float = MIN_COVER):
    check_month(month)
    raw = pd.read_csv(RAW_DATA)
    eu_list = raw[raw["Country"] == "Europe"][["Brand", "Model"]].drop_duplicates()
    eu2025 = raw[(raw["Country"] == "Europe") & raw["Month"].str.startswith("2025")].groupby(["Brand", "Model"])["Sales"].sum()
    shares_path = shares_path or latest_file([HERE, EB], "eea_shares", int(month[:4]))
    shares = pd.read_csv(shares_path)
    share_year = int(shares_path.stem.split("_")[-1])
    comp = shares.drop_duplicates("country").set_index("country")["completeness_factor"]
    sf = seasonal_factors(share_year, comp)
    fm = sf[sf["month_num"] == int(month[5:7])].set_index("country")["f"]
    log, unm = [], []

    parts = {}   # cc -> DataFrame(Brand, Model, units, conf, alloc[, lump])
    full_list = {}  # cc -> True(전 차종 목록: 없는 차종 = 0) / False(목록에 있는 차종만 덮임)
    de, _ = load_de(month, raw, log)
    if de is not None:
        parts["DE"] = allocate_de(de, shares, eu2025)
        full_list["DE"] = False      # 우리 Germany 목록에 이어진 계열만 덮임
    for cc in ["ES", "NL", "UK"]:
        x = load_ea(cc, month, log, unm)
        if x is not None:
            parts[cc] = x.assign(alloc=False)
            full_list[cc] = True
    if "UK" not in parts:   # 영국 분기 발표 전 → SMMT 보도자료 상위 차종만이라도 덮음(나머지는 추정)
        x = load_uk_smmt(month, log)
        if x is not None:
            parts["UK"] = x.assign(alloc=False)
            full_list["UK"] = False
    for cc in ["FR", "IT"]:
        x = load_top(cc, month, log)
        if x is not None:
            parts[cc] = x.assign(alloc=False)
            full_list[cc] = False
    for cc in C_COUNTRIES:
        x, meta = load_c(cc, month, log, unm)
        if x is not None:
            parts[cc] = x
            full_list[cc] = True

    # 묶인 이름 나누기: lumped 나라는 두 차종 합을, 그 달 sep 나라들의 합 비율로 다시 나눈다
    for L in LUMPS:
        b, members = L["brand"], L["members"]
        sep = L["sep"] if L["sep"] is not None else [c for c in parts if c not in L["lumped"]]
        pool = {mm: 0.0 for mm in members}
        for cc in sep:
            if cc in parts:
                x = parts[cc]
                for mm, v in x[(x["Brand"] == b) & x["Model"].isin(members)].groupby("Model")["units"].sum().items():
                    pool[mm] += v
        tot, basis = sum(pool.values()), "sep_countries_this_month"
        if tot <= 0:
            s2 = shares[(shares["Brand"] == b) & shares["Model"].isin(members)].groupby("Model")["n_2025_adj"].sum()
            pool = {mm: float(s2.get(mm, 0.0)) for mm in members}
            tot, basis = sum(pool.values()), "eea_europe_prev_year"
        w = {mm: (pool[mm] / tot if tot > 0 else (1.0 if i == 0 else 0.0)) for i, mm in enumerate(members)}
        done = []
        for cc in L["lumped"]:
            if cc not in parts:
                continue
            x = parts[cc]
            hit = (x["Brand"] == b) & x["Model"].isin(members)
            if not hit.any():
                continue
            base = float(x.loc[hit, "units"].sum())
            conf = x.loc[hit, "conf"].min()
            add = pd.DataFrame([dict(Brand=b, Model=mm, units=base * w[mm], conf=conf, alloc=True) for mm in members])
            parts[cc] = pd.concat([x[~hit], add], ignore_index=True)
            done.append(cc)
        log.append(dict(check="lump_split", key=f"{b} {' + '.join(members)}", status="ok",
                        note=json.dumps({"weights": {mm: round(v, 3) for mm, v in w.items()}, "basis": basis,
                                         "pool": {mm: round(v, 1) for mm, v in pool.items()}, "applied_to": done},
                                        ensure_ascii=False)))

    # 몫 표(차종 → 브랜드 → 시장)
    sh_s = shares.pivot_table(index=["Brand", "Model"], columns="country", values="n_2025_adj", aggfunc="sum").fillna(0)
    sh_b = shares.groupby(["Brand", "country"])["n_2025_adj"].sum().unstack().fillna(0)
    sh_m = shares.groupby("country")["n_2025_adj"].sum()
    countries = sorted(set(sh_m.index))
    covered_any = set(parts)

    idx = {cc: {(r.Brand, r.Model): r for r in x.itertuples(index=False)} for cc, x in parts.items()}
    f = pd.Series({c: fm.get(c, 1.0) for c in countries})
    down = {"high": "medium", "medium": "low", "low": "low"}
    rows, vans, skipped = [], [], []
    pair_of = {s: mem for mem in PAIR_GROUPS.values() for s in mem}
    for b, m in eu_list.itertuples(index=False):
        mem = [s for s in pair_of.get((b, m), []) if s in sh_s.index]
        if mem and sh_s.loc[mem].sum().sum() > 0:
            # 공식 자료가 한쪽으로 몰아 세는 짝(Enyaq/Enyaq Coupe 등): 짝 합계의 나라 몫을 같이 쓴다
            vec, sh_src = sh_s.loc[mem].sum(), "pair"
        elif (b, m) in sh_s.index and sh_s.loc[(b, m)].sum() >= MIN_SERIES_PREV_YEAR:
            vec, sh_src = sh_s.loc[(b, m)], "series"
        elif b in sh_b.index and sh_b.loc[b].sum() > 0:
            vec, sh_src = sh_b.loc[b], "brand"
        else:
            vec, sh_src = sh_m, "market"
        vec = vec.reindex(countries).fillna(0) / vec.sum()
        cov, val, alloc, confs, uk_part = [], 0.0, False, [], 0.0
        for cc in parts:
            r = idx[cc].get((b, m))
            if r is not None:
                cov.append(cc)
                val += float(r.units)
                alloc |= bool(r.alloc)
                if r.conf:
                    confs.append(r.conf)
                if cc == "UK" and full_list.get("UK"):   # DfT 분기값을 달로 나눈 값일 때만(SMMT 월값은 아님)
                    uk_part = float(r.units)
            elif full_list.get(cc):
                cov.append(cc)          # 전 차종 목록에 없음 = 그 나라 0대
        cov_c = [c for c in cov if c in vec.index]
        unc = [c for c in countries if c not in cov]
        wc = float((vec[cov_c] * f[cov_c]).sum())
        wu = float((vec[unc] * f[unc]).sum())
        frac = wc / (wc + wu) if wc + wu > 0 else 0.0   # 덮인 몫(계절 계수 반영, 0~1)
        if (b, m) in LCV_LIKE:
            vans.append(dict(Country="Europe", Brand=b, Model=m, Month=month, covered_m1_sum=round(val, 1),
                             covered_countries=",".join(sorted(cov)), covered_weight=round(wc, 3),
                             note="밴/상용 성격: 우리 정의(승용+상용 일부?)와 승용 M1 자료가 달라 추정하지 않음"))
            continue
        if frac < min_cover or (val == 0 and frac < ZERO_COVER):
            skipped.append(dict(check="low_coverage", key=f"{b} | {m}", got=round(val, 1), ratio=round(frac, 3),
                                status="warn", note=f"덮인 몫 {frac:.2f} — 값 만들지 않음 (shares={sh_src})"))
            continue
        partner = [s for s in pair_of.get((b, m), []) if s != (b, m)]
        if val == 0 and any(idx[cc].get(s) is not None and idx[cc][s].units > 0 for cc in parts for s in partner):
            # 짝 중 한쪽만 0: 출처가 짝을 한 이름으로 셈(예: Enyaq Coupe → Enyaq) → 0 이 아니라 '값 없음'
            skipped.append(dict(check="pair_lumped", key=f"{b} | {m}", got=0.0, ratio=round(frac, 3), status="warn",
                                note="짝 차종이 다른 이름으로 함께 세어짐 → 값 만들지 않음(짝 합계는 상대 차종 값에 들어 있음)"))
            continue
        est = val * wu / wc if wc > 0 else 0.0
        total = val + est
        es = est / total if total > 0 else 0.0
        if es > 1e-9:
            vt = "estimated"
        elif alloc:
            vt = "allocated"
        elif uk_part > 0:
            vt = "split"
        else:
            vt = "reported"
        conf = ("high" if es <= 0.20 else "medium" if es <= 0.45 else "low")
        if sh_src == "brand" or "low" in confs:
            conf = down[conf]
        if sh_src == "market":
            conf = "low"
        note = (f"covw={frac:.2f}; cov={'+'.join(sorted(cov))}; est={'+'.join(c for c in unc if vec[c] * f[c] > 0.005)}"
                f"; shares={sh_src}{share_year}" + ("; UK=quarter split" if uk_part > 0 else "")
                + ("; allocated part" if alloc else ""))
        rows.append(dict(Country="Europe", Brand=b, Model=m, Month=month, Sales=round(total, 2), value_type=vt,
                         estimated_share=round(es, 4), source_id=SOURCE_ID, confidence=conf, note=note))
    std = pd.DataFrame(rows)
    log += skipped
    log.append(dict(check="calc_total", key="std rows", got=float(std["Sales"].sum()) if len(std) else 0.0,
                    note=f"{len(std)} 차종, 밴 표시 {len(vans)}, 값 없음 {len(skipped)}; "
                         f"덮인 나라 {'+'.join(sorted(covered_any))}; 몫 {shares_path.name}"))
    unmapped = pd.concat(unm, ignore_index=True).sort_values("units", ascending=False) if unm else pd.DataFrame()
    return std, pd.DataFrame(vans), pd.DataFrame(log), unmapped


def run_collectors(month: str, offline: bool):
    for cc in C_COUNTRIES:
        cmd = [sys.executable, str(HERE / f"collect_{cc.lower()}.py"), "--month", month]
        if offline and cc != "PL":
            cmd.append("--offline")
        r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(f"[{cc}] rc={r.returncode} {(r.stdout or r.stderr).strip().splitlines()[-1:]}", file=sys.stderr)
    cmd = [sys.executable, str(HERE.parent / "europe_df" / "collect_europe_df.py"), "--month", month] + (["--offline"] if offline else [])
    r = subprocess.run(cmd, cwd=str(HERE.parent / "europe_df"), capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"[EU_DATAFORCE] rc={r.returncode} {(r.stderr or r.stdout).strip().splitlines()[-1:]}", file=sys.stderr)
    cmd = [sys.executable, str(HERE / "collect_uk_smmt.py"), "--month", month] + (["--offline"] if offline else [])
    r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"[UK_SMMT] rc={r.returncode} {(r.stdout or r.stderr).strip().splitlines()[-1:]}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--shares")
    a = ap.parse_args()
    month = check_month(a.month)
    if a.collect:
        run_collectors(month, a.offline)
    std, vans, log, unm = build(month, Path(a.shares) if a.shares else None)
    std.to_csv(OUT / f"calc_europe_{month}.csv", index=False, encoding="utf-8-sig")   # 계산기 값(비교용)
    dfp = HERE.parent / "europe_df" / "out" / f"df_{month}.csv"     # Dataforce 유럽 상위 차종 = 실제 값으로 바꿈
    if dfp.exists():
        sys.path.insert(0, str(dfp.parents[1]))
        from collect_europe_df import override
        df = pd.read_csv(dfp, encoding="utf-8-sig")
        if len(df):
            std = override(std, df)
            log = pd.concat([log, pd.DataFrame([dict(check="source", key="EU_DATAFORCE", status="ok", got=float(df["units"].sum()),
                                                     note=f"Dataforce 유럽 상위 {len(df)}개 차종 = 실제 값({df['how'].value_counts().to_dict()})")])],
                            ignore_index=True)
    p = OUT / f"std_europe_{month}.csv"
    std.to_csv(p, index=False, encoding="utf-8-sig")
    vans.to_csv(OUT / f"flagged_vans_europe_{month}.csv", index=False, encoding="utf-8-sig")
    for c in ["expected", "got", "ratio"]:
        if c not in log.columns:
            log[c] = np.nan
    log[["check", "key", "expected", "got", "ratio", "status", "note"]].to_csv(
        OUT / f"checks_europe_{month}.csv", index=False, encoding="utf-8-sig")
    if len(unm):
        unm.to_csv(OUT / f"unmapped_europe_{month}.csv", index=False, encoding="utf-8-sig")
    print(p)


if __name__ == "__main__":
    main()
