"""DAIZEN 메뉴 — 번호로 골라 점수 보기·차종 관리·실행을 한 곳에서.

사용:  uv run menu.py        (또는 .venv/Scripts/python.exe menu.py)

step0_setup.py · step1_collect.py · step2_train.py · predict.py · backtest.sh 는 그대로 있고 지금처럼 따로 실행해도 된다.
메뉴의 '실행'은 그 파일들을 대신 불러 줄 뿐이다.

새 차종 추가는 수집기를 확인한 시장(중국·한국·독일)만 대응표에 한 줄을 자동으로 넣는다(형식 검사·중복 거부·백업·기록).
일본·미국·캐나다·유럽은 수집기가 코드 안 표나 원본 목록으로 차종을 맞추므로 메뉴가 고칠 곳만 안내한다.
설정(추정 몫 30%·다시 수집 3개월·실시간 성적 기간 등)은 보기만 한다 — 바꾸면 점수가 달라지므로 메뉴에서 바꾸지 않는다.
"""
import csv
import io
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
KEY = ["Country", "Brand", "Model"]
JULY_REL = "experiments/horizon_eval_v1/production/client_package_july_ic_20261013T035849_323513Z_9f3a75/client_forecast_24m.csv"
CHAMPION = "ic_champ"
MAIN_SCORE = 25.86
PARTS = [("front", "1~3개월"), ("middle", "4~12개월"), ("back", "13~24개월"), ("main", "전체")]
# 백테스트에서 '그 시점의 신차'만 모아 신차용 방법을 채점한 기록(progress.md (fu)·(fv), 2026-10-06): (이름, 방법, 오차율 %, 건수, 비교)
NEW_CAR_GROUPS = [
    ("신차 보조 예측(이력 3~11개월)", "Chronos 짧은 이력 예측", 47.44, 765, "다른 후보: 마지막 3개월 평균 51.72%, 출시 곡선 54.12%"),
    ("학습 못 본 차(이력 12개월)", "Chronos 보조 모델", 47.92, 83, "작년 그대로 59.42%"),
]
GROUP_OF_STATUS = {"OK_SHORT_HISTORY": 0, "OK_CHRONOS_ONLY": 1}
MAPPING_FILES = {"China": "experiments/collectors_v1/china/mapping.csv",
                 "South Korea": "experiments/collectors_v1/korea/mapping.csv",
                 "Germany": "experiments/data_audit_v1/germany_kba_mapping.csv"}
AUTO_MARKETS = MAPPING_FILES
MANUAL_GUIDE = {
    "Japan": ("수집기: experiments/collectors_v1/rules_jp_na_de/collect_japan.py (JADA 등록차 순위·전경련 경차 표).\n"
              "고칠 곳: experiments/collectors_v1/rules_jp_na_de/mapping.csv 에 Country=Japan 줄.\n"
              "  예) Japan,Toyota,Yaris,JP_JADA_RANK,JADA|登録乗用車|トヨタ|ヤリス,direct+fill_window,medium,,\n"
              "  source_key 는 순위표에 적힌 메이커·차명과 글자까지 같아야 한다. 새 차종은 자동으로 감지되지 않으니 순위 기사로 확인."),
    "United States": ("수집기: experiments/collectors_v1/rules_jp_na_de/na/collect_na.py (GoodCarBadCar 미국 표).\n"
                      "고칠 곳: experiments/data_sources_v1/group_1/series.csv 에 차종 한 줄, 표 이름이 우리 이름과 다르면\n"
                      "  na/collect_na.py 의 US_ALIAS 에 (브랜드, 차종): [표 이름] 한 줄. 새 차종은 자동 감지 안 됨."),
    "Canada": ("수집기: experiments/collectors_v1/rules_jp_na_de/na/collect_na.py (GoodCarBadCar 캐나다 차종 페이지).\n"
               "고칠 곳: experiments/data_sources_v1/group_1/series.csv 와 group_1/scripts/gcbc_map.csv(차종 페이지 주소). 새 차종은 자동 감지 안 됨."),
    "Europe": ("수집기: europe_df(Dataforce 상위 50) + europe_c(나라별 표).\n"
               "Dataforce 는 원본(raw/data.csv)에 이미 있는 Europe 차종 이름과 맞춘다 → 완전히 새 차종은 원본에 Europe 줄이 생겨야 잡힌다.\n"
               "  이름만 다르면 europe_df/df_parse.py 의 ALIAS 에 '표 이름(소문자)': (브랜드, 차종) 한 줄.\n"
               "나라별 표는 europe_c/mapping.csv(Brand | Model, rule prefix:/special:) 와 europe_b/europe_map.py 분류 규칙."),
}
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-\d+$")
LOG_COLUMNS = ["added_at", "market", "file", "Brand", "Model", "source_id", "source_key", "backup"]


# ---------- 읽기 ----------

def is_package(root=ROOT):
    """전달 묶음(DELIVERY_MANIFEST.json 이 있는 폴더)인지. 작업 폴더(D:/daizen)에서는 실행·추가를 막는다."""
    return (Path(root) / "DELIVERY_MANIFEST.json").exists()


def scorecard_path(root=ROOT):
    """백테스트 점수표: 묶음 안 model_info/scorecard.csv, 없으면(작업 폴더) 가장 최근 전달 묶음 사본의 것."""
    own = Path(root) / "model_info" / "scorecard.csv"
    if own.exists():
        return own
    copies = sorted(Path(root).glob("DAIZEN_v*/model_info/scorecard.csv"), key=lambda p: p.parent.parent.name)
    return copies[-1] if copies else None


def past_errors_12m(root=ROOT):
    """12개월 기준 과거 오차율 표(model_info/past_error_12m.csv; 작업 폴더면 experiments/horizon_eval_v1/ 의 원본). 없으면 None."""
    for p in (Path(root) / "model_info" / "past_error_12m.csv", Path(root) / "experiments" / "horizon_eval_v1" / "past_error_12m.csv"):
        if p.exists():
            return pd.read_csv(p, encoding="utf-8-sig")
    return None


def latest_delivery(root=ROOT):
    """납품 목록(raw/deliveries.csv)에서 기준월이 가장 늦은 예측. 목록이 없으면 7월 납품."""
    reg_path = Path(root) / "raw" / "deliveries.csv"
    if reg_path.exists():
        reg = pd.read_csv(reg_path, encoding="utf-8-sig", dtype=str).drop_duplicates("origin", keep="last").sort_values("origin")
        if len(reg):
            p = Path(reg.iloc[-1]["path"])
            return p if p.is_absolute() else Path(root) / p
    return Path(root) / JULY_REL


def load_detail(root=ROOT):
    return pd.read_csv(latest_delivery(root), encoding="utf-8-sig", dtype={"Month": str, "origin": str})


def summary(d):
    s = d.drop_duplicates(KEY)
    return {"origin": str(d["origin"].iloc[0]), "first": str(d["Month"].min()), "last": str(d["Month"].max()),
            "series": len(s), "by_status": s["status"].value_counts().to_dict(),
            "by_status_ko": s["status_ko"].value_counts().to_dict() if "status_ko" in s else {}}


def score_table(sc, model=CHAMPION):
    """백테스트 점수표(model_info/scorecard.csv)의 챔피언 시장별·전체 점수(묶음+달마다 평균 = combined)."""
    s = sc[(sc["model"] == model) & (sc["metric"] == "combined") & sc["scope"].isin(["overall", "market"])]
    rows = {}
    for (scope, val), g in s.groupby(["scope", "scope_value"]):
        name = "전체" if scope == "overall" else val
        rows[name] = {label: round(float(g.loc[g["part"] == p, "value"].iloc[0]), 1) if (g["part"] == p).any() else float("nan")
                      for p, label in PARTS}
    t = pd.DataFrame.from_dict(rows, orient="index")[[label for _, label in PARTS]]
    markets = t.drop(index="전체", errors="ignore").sort_values("전체")
    return pd.concat([markets, t.loc[["전체"]]]) if "전체" in t.index else markets


def find_series(d, query):
    words = str(query).lower().split()
    s = d.drop_duplicates(KEY)[KEY + ["status", "status_ko", "backtest_wape"]]
    if not words:
        return s.iloc[0:0].reset_index(drop=True)
    text = (s["Country"] + " " + s["Brand"] + " " + s["Model"]).str.lower()
    mask = pd.Series(True, index=s.index)
    for w in words:
        mask &= text.str.contains(w, regex=False)
    return s[mask].sort_values(KEY).reset_index(drop=True)


def live_scores(root=ROOT):
    p = Path(root) / "raw" / "monitoring_series.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


def _num(x, fmt="{:,.0f}"):
    return "" if pd.isna(x) else fmt.format(x)


def series_card(d, live, key, past12=None):
    c, b, m = key
    g = d[(d["Country"] == c) & (d["Brand"] == b) & (d["Model"] == m)].sort_values("horizon")
    if not len(g):
        return f"{c} {b} {m}: 납품 예측에 없는 차종입니다."
    f = g.iloc[0]
    lines = [f"■ {c} / {b} {m}", f"  상태: {f.get('status_ko', f['status'])}"]
    bw = f.get("backtest_wape")
    if pd.notna(bw):
        lines.append(f"  과거 오차율: {bw:.1f}%  (백테스트 기준월 5개에서 24개월 예측 vs 실제)")
    else:
        gi = GROUP_OF_STATUS.get(f["status"])
        if gi is not None:
            name, how, wape, n, _ = NEW_CAR_GROUPS[gi]
            lines.append(f"  과거 오차율: 없음 (신차). 같은 처지({name}) 차들의 과거 오차율 {wape:.1f}% (백테스트 {n}건, {how}).")
        elif f["status"] == "OK":
            r12 = None
            if past12 is not None and len(past12):
                hit = past12[(past12["Country"] == c) & (past12["Brand"] == b) & (past12["Model"] == m)]
                r12 = hit.iloc[0] if len(hit) else None
            if r12 is not None:
                lines.append(f"  과거 오차율(12개월 기준): {r12['wape_12m']:.1f}%  ({r12['origin']} 시점에서 12개월 예측 vs 실제. "
                             "24개월 점수는 백테스트 마지막 시점(2024-07) 뒤에 나온 차라 없음; 12개월만 본 값이라 24개월 기준보다 좋게 나오기 쉬움)")
            else:
                lines.append(f"  과거 오차율: 없음 — 백테스트 마지막 시점(2024-07) 뒤에 나온 차라 채점하지 못함. 예측은 본 예측이며, 본 예측 전체의 과거 오차율은 {MAIN_SCORE:.1f}%.")
        else:
            lines.append("  과거 오차율: 없음")
    row = None
    if live is not None and len(live):
        r = live[(live["Country"] == c) & (live["Brand"] == b) & (live["Model"] == m)]
        row = r.iloc[0] if len(r) else None
    if row is not None:
        months = int(row["months_scored"])
        lines.append(f"  실시간 성적: 오차율 {_num(row['live_wape'], '{:.1f}%')} · 작년 그대로 {_num(row['naive_wape'], '{:.1f}%')}"
                     f" · 쏠림 {_num(row['bias_pct'], '{:+.1f}%')} (최근 {months}개월{', 3개월 이상 쌓이면 더 믿을 만함' if months < 3 else ''})")
    else:
        lines.append("  실시간 성적: 아직 없음 (매달 실적이 들어오면 지난 예측과 비교해 쌓임)")
    if g["prediction"].isna().all():
        lines.append(f"  예측: 없음 ({f.get('status_ko', f['status'])})")
        return "\n".join(lines)
    lines.append(f"  24개월 예측 (기준월 {f['origin']}, 80% 범위):")
    for r in g.itertuples():
        lines.append(f"    {r.Month}  {_num(r.prediction):>8}  ({_num(r.lower_80)} ~ {_num(r.upper_80)})")
    yr = g.assign(year=g["Month"].str[:4]).groupby("year")["prediction"].sum()
    lines.append("  연도별 합: " + ", ".join(f"{y}년 {_num(v)}" for y, v in yr.items()))
    return "\n".join(lines)


def latest_new_models(root=ROOT):
    """가장 최근 수집 달의 '목록에 없는 차종' 후보(raw/incoming/<달>/new_models.csv), 판매 많은 순."""
    files = sorted((Path(root) / "raw" / "incoming").glob("*/new_models.csv"), key=lambda p: p.parent.name)
    if not files:
        return None, pd.DataFrame(columns=["market", "source", "Month", "name", "units", "note", "how_to_add"])
    nm = pd.read_csv(files[-1], encoding="utf-8-sig").sort_values("units", ascending=False).reset_index(drop=True)
    return files[-1].parent.name, nm


def no_forecast(d, root=ROOT):
    """예측이 없는 차종과 이유: 납품에서 예측이 빈 차종 + 원본에 없어 제외한 차종(raw/vehicle_status.csv)."""
    s = d.groupby(KEY, as_index=False).agg(empty=("prediction", lambda x: x.isna().all()), status=("status", "first"),
                                             reason=("status_ko", "first"))
    out = s[s["empty"]][KEY + ["status", "reason"]]
    vs = Path(root) / "raw" / "vehicle_status.csv"
    if vs.exists():
        v = pd.read_csv(vs, encoding="utf-8-sig")
        have = set(map(tuple, d[KEY].drop_duplicates().values))
        v = v[[tuple(k) not in have for k in v[KEY].values]][KEY + ["status", "reason"]]
        out = pd.concat([out, v], ignore_index=True)
    return out.sort_values(KEY).reset_index(drop=True)


# ---------- 새 차종 추가 ----------

def _need(brand, model):
    brand, model = str(brand).strip(), str(model).strip()
    if not brand or not model:
        raise ValueError("브랜드와 차종 이름을 모두 적어야 합니다.")
    return brand, model


def _note():
    return f"menu.py 추가 {datetime.now():%Y-%m-%d}"


def china_row(brand, model, slugs):
    """Gasgoo 차종 페이지 주소 끝 이름(예: model-y-1881). 여러 줄을 더할 땐 '+' 로 잇는다."""
    brand, model = _need(brand, model)
    parts = [s.strip() for s in str(slugs).split("+") if s.strip()]
    if not parts or not all(SLUG.match(p) for p in parts):
        raise ValueError("Gasgoo 이름 형식이 아닙니다. 예: model-y-1881 (영문 소문자·숫자·'-', 끝은 '-숫자'). 여러 개는 a-1+b-2")
    return {"Country": "China", "Brand": brand, "Model": model, "source_id": "CN_GASGOO_MODEL",
            "source_key": "+".join(parts), "rule": "sum", "confidence": "medium", "note": _note()}


def korea_candidates(root=ROOT):
    """가장 최근 한국 수집에서 다나와 줄 중 우리 목록에 없는 것(우리 것이 아니라고 이미 본 줄은 뺌), 판매 많은 순."""
    files = sorted((Path(root) / "experiments" / "collectors_v1" / "korea" / "out").glob("unmapped_korea_*.csv"))
    if not files:
        return pd.DataFrame(columns=["Month", "nat", "brand_id", "brand_name", "line", "sales", "status"])
    u = pd.read_csv(files[-1], encoding="utf-8-sig")
    u = u[u["status"].astype(str) != "seen_before_not_ours"]
    return u.sort_values("sales", ascending=False).reset_index(drop=True)


def korea_row(brand, model, cand):
    brand, model = _need(brand, model)
    source_id = "KR_DANAWA_DOMESTIC" if str(cand["nat"]) == "domestic" else "KR_DANAWA_IMPORT"
    return {"Country": "South Korea", "Brand": brand, "Model": model, "source_id": source_id,
            "source_key": f"{int(float(cand['brand_id']))}:{cand['line']}", "rule": "sum_lines_absent0",
            "confidence": "medium", "note": _note(), "priority": "1", "sales_last12": ""}


def germany_kba_brands(root=ROOT):
    m = pd.read_csv(Path(root) / MAPPING_FILES["Germany"], encoding="utf-8-sig", dtype=str)
    return sorted(set(m["kba_brand"].dropna()), key=len, reverse=True)


def split_kba_name(name, brands):
    """'ALFA ROMEO TONALE NEW' → ('ALFA ROMEO', 'TONALE NEW'): 알려진 KBA 브랜드 중 가장 긴 앞부분."""
    n = re.sub(r"\s+", " ", str(name)).strip().upper()
    for b in sorted(brands, key=len, reverse=True):
        if n.startswith(b.upper() + " "):
            return b, n[len(b) + 1:].strip()
    return None, n


def germany_row(brand, model, kba_brand, kba_line):
    brand, model = _need(brand, model)
    kba_brand, kba_line = str(kba_brand).strip().upper(), str(kba_line).strip().upper()
    if not kba_brand or not kba_line:
        raise ValueError("KBA 브랜드와 차종 줄 이름이 필요합니다(KBA 표 그대로, 대문자).")
    return {"Brand": brand, "Model": model, "kba_brand": kba_brand, "kba_model_series": kba_line, "requested_kba_names": kba_line,
            "match_type": "manual", "group_id": "", "group_our_series": f"{brand} {model}", "group_kba_lines": f"{kba_brand}/{kba_line}",
            "first_month": "", "last_month": "", "n_months": "", "total_ours": ""}


def _read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def add_mapping_row(root, market, row):
    """대응표 끝에 한 줄만 덧붙인다(기존 줄·인코딩·줄바꿈 그대로). 같은 차종·같은 출처 키가 있으면 거부. 쓰기 전 백업, 기록 남김."""
    root = Path(root)
    if market not in MAPPING_FILES:
        raise ValueError(f"{market} 는 메뉴에서 자동으로 추가하지 않습니다. 안내를 보고 직접 고치세요.")
    path = root / MAPPING_FILES[market]
    header, rows = _read_rows(path)
    extra = set(row) - set(header)
    if extra:
        raise ValueError(f"대응표에 없는 열: {sorted(extra)}")
    keycols = [k for k in KEY if k in header]
    same = lambda r: all(str(r.get(k, "")).strip().lower() == str(row.get(k, "")).strip().lower() for k in keycols)
    if any(same(r) for r in rows):
        raise ValueError(f"이미 대응표에 있는 차종입니다: {' / '.join(str(row[k]) for k in keycols)}")
    if market == "Germany":
        used = {(r["kba_brand"], a.strip()) for r in rows for a in str(r["kba_model_series"]).split(" | ")}
        if (row["kba_brand"], row["kba_model_series"]) in used:
            raise ValueError(f"이 KBA 줄은 이미 다른 차종에 쓰이고 있습니다: {row['kba_brand']}/{row['kba_model_series']}")
    elif any(r.get("source_key", "") == row["source_key"] for r in rows):
        raise ValueError(f"이 출처 키는 이미 다른 차종에 쓰이고 있습니다: {row['source_key']}")
    raw = path.read_bytes()
    backups = root / "raw" / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    backup = backups / f"mapping_{path.parent.name}_{path.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    backup.write_bytes(raw)
    nl = "\r\n" if b"\r\n" in raw else "\n"
    buf = io.StringIO()
    csv.writer(buf, lineterminator=nl).writerow([row.get(c, "") for c in header])
    sep = b"" if not raw or raw.endswith(b"\n") else nl.encode()
    path.write_bytes(raw + sep + buf.getvalue().encode("utf-8"))
    log = root / "raw" / "mapping_additions.csv"
    entry = pd.DataFrame([{"added_at": datetime.now().isoformat(timespec="seconds"), "market": market, "file": MAPPING_FILES[market],
                           "Brand": row["Brand"], "Model": row["Model"], "source_id": row.get("source_id", "DE_KBA_FZ10"),
                           "source_key": row.get("source_key", f"{row.get('kba_brand')}/{row.get('kba_model_series')}"), "backup": str(backup)}],
                         columns=LOG_COLUMNS)
    entry.to_csv(log, mode="a", header=not log.exists(), index=False, encoding="utf-8-sig")
    return backup


def known_brands(root, market):
    p = Path(root) / "raw" / "data.csv"
    if not p.exists():
        return []
    r = pd.read_csv(p, encoding="utf-8-sig", usecols=["Country", "Brand"])
    return sorted(set(r.loc[r["Country"] == market, "Brand"]))


# ---------- 화면 ----------

def _ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return "0"


def _pause():
    _ask("\n(엔터: 돌아가기) ")


def _print_table(df, n=None):
    if df is None or not len(df):
        print("  (없음)")
        return
    with pd.option_context("display.max_rows", n or 500, "display.max_columns", 20, "display.width", 200):
        print(df.head(n).to_string(index=False) if n else df.to_string(index=False))


def _header(d):
    s = summary(d)
    ok = sum(v for k, v in s["by_status"].items() if str(k).startswith("OK"))
    print("\n" + "=" * 60)
    print(f" DAIZEN  |  예측 기준월 {s['origin']}  ({s['first']} ~ {s['last']})")
    print(f"          차종 {s['series']:,}개 · 예측 있음 {ok:,}개 · 백테스트 점수 {MAIN_SCORE}%")
    print("=" * 60)


def view_scores(root, d):
    while True:
        print("\n[1. 점수 보기]\n  1) 전체·시장별 점수 (백테스트)\n  2) 차종 검색\n  3) 신차 무리 점수\n  0) 돌아가기")
        c = _ask("> ")
        if c == "1":
            sc = scorecard_path(root)
            if sc is None:
                print("  점수표(model_info/scorecard.csv)를 찾지 못했습니다.")
            else:
                if not is_package(root):
                    print(f"  (작업 폴더라 전달 묶음 사본의 점수표를 씁니다: {sc.relative_to(root)})")
                print("\n판매량 가중 오차율(%) — 낮을수록 좋음. 기준월 2022-07~2024-07, 24개월 예측을 실제와 비교.")
                _print_table(score_table(pd.read_csv(sc, encoding="utf-8-sig")).reset_index().rename(columns={"index": "시장"}))
            _pause()
        elif c == "2":
            q = _ask("찾을 차 (예: tesla model y / 독일이면 germany golf): ")
            hits = find_series(d, q)
            if not len(hits):
                print("  찾은 차종 없음.")
                _pause()
                continue
            if len(hits) > 1:
                _print_table(hits[KEY + ["status_ko"]].reset_index().rename(columns={"index": "번호"}), 40)
                pick = _ask("번호 (엔터: 첫 번째): ") or "0"
                if not pick.isdigit() or int(pick) >= len(hits):
                    continue
                key = tuple(hits.iloc[int(pick)][KEY])
            else:
                key = tuple(hits.iloc[0][KEY])
            print()
            print(series_card(d, live_scores(root), key, past_errors_12m(root)))
            _pause()
        elif c == "3":
            print("\n백테스트에서 '그 시점의 신차'만 모아 신차용 방법을 채점한 결과(판매량 가중 오차율):")
            for name, how, wape, n, cmp_ in NEW_CAR_GROUPS:
                print(f"  - {name}: {wape:.1f}%  ({how}, {n}건; {cmp_})")
            print(f"  - 참고: 본 예측 전체 {MAIN_SCORE:.1f}%")
            print("  신차 하나하나의 오차율은 실적이 들어오면 '차종 검색'의 실시간 성적에 쌓입니다(3개월 이상이면 더 믿을 만함).")
            _pause()
        else:
            return


def manage_cars(root, d):
    while True:
        print("\n[2. 차종 관리]\n  1) 목록에 없는 새 차종 후보 보기\n  2) 새 차종 추가\n  3) 예측 없는 차종과 이유\n  0) 돌아가기")
        c = _ask("> ")
        if c == "1":
            month, nm = latest_new_models(root)
            if month is None:
                print("  아직 후보가 없습니다(수집(step1)을 한 번 돌리면 raw/incoming/<달>/new_models.csv 가 생깁니다).")
            else:
                print(f"\n{month} 수집에서 판매가 잡혔는데 우리 목록에 없는 차종(100대 이상, 판매 많은 순):")
                _print_table(nm[["market", "source", "name", "units", "note"]], 60)
                print("일본·미국·캐나다·중국은 수집기가 우리 차종 페이지만 보므로 여기 안 뜹니다(순위 기사로 확인).")
            _pause()
        elif c == "2":
            add_car(root)
        elif c == "3":
            nf = no_forecast(d, root)
            print(f"\n예측 없는 차종 {len(nf)}개:")
            _print_table(nf.groupby("reason").size().rename("차종 수").reset_index())
            if _ask("목록 전체를 볼까요? (y/엔터) ").lower() == "y":
                _print_table(nf, 500)
            _pause()
        else:
            return


DEV_NOTICE = ("  여기는 작업 폴더(전달 묶음이 아님)라 이 기능은 막아 두었습니다 — 진짜 원본·대응표를 바꾸거나 옛 파일을 실행하게 됩니다.\n"
              "  깃헙에서 받은 묶음 폴더(DELIVERY_MANIFEST.json 이 있는 곳)에서 uv run menu.py 로 쓰세요.")


def add_car(root):
    if not is_package(root):
        print("\n" + DEV_NOTICE)
        _pause()
        return
    markets = ["China", "South Korea", "Germany", "Japan", "United States", "Canada", "Europe"]
    print("\n어느 시장에 추가할까요?")
    for i, m in enumerate(markets, start=1):
        print(f"  {i}) {m}" + ("  (자동 추가)" if m in AUTO_MARKETS else "  (안내만)"))
    c = _ask("> ")
    if not c.isdigit() or not 1 <= int(c) <= len(markets):
        return
    market = markets[int(c) - 1]
    if market not in AUTO_MARKETS:
        print("\n" + MANUAL_GUIDE[market])
        _pause()
        return
    brands = known_brands(root, market)
    try:
        if market == "South Korea":
            cand = korea_candidates(root)
            if not len(cand):
                print("  다나와 미대응 후보가 없습니다(수집(step1)을 먼저 돌리세요).")
                _pause()
                return
            _print_table(cand[["brand_name", "line", "sales", "nat"]].head(30).reset_index().rename(columns={"index": "번호"}))
            p = _ask("추가할 줄 번호: ")
            if not p.isdigit() or int(p) >= len(cand):
                return
            pick = cand.iloc[int(p)].to_dict()
            brand = _ask(f"우리 표 브랜드 이름 (한국 기존 예: {', '.join(brands[:8])} …): ")
            model = _ask("우리 표 차종 이름 (영문 권장, 예: EV5): ")
            row = korea_row(brand, model, pick)
        elif market == "China":
            print("Gasgoo(autonews.gasgoo.com) 차종 페이지 주소 끝 이름을 적습니다. 예: model-y-1881")
            brand = _ask(f"브랜드 (중국 기존 예: {', '.join(brands[:8])} …): ")
            model = _ask("차종: ")
            row = china_row(brand, model, _ask("Gasgoo 이름 (여러 개면 a-1+b-2): "))
        else:
            kb = germany_kba_brands(root)
            month, nm = latest_new_models(root)
            ger = nm[nm["market"] == "Germany"].reset_index(drop=True) if month else nm.iloc[0:0]
            if len(ger):
                _print_table(ger[["name", "units"]].reset_index().rename(columns={"index": "번호"}), 30)
                p = _ask("후보 번호 (엔터: 직접 입력): ")
                kba_brand, kba_line = split_kba_name(ger.iloc[int(p)]["name"], kb) if p.isdigit() and int(p) < len(ger) else (None, "")
            else:
                kba_brand, kba_line = None, ""
            kba_brand = _ask(f"KBA 브랜드 [{kba_brand or ''}]: ") or kba_brand
            kba_line = _ask(f"KBA 차종 줄 [{kba_line}]: ") or kba_line
            brand = _ask(f"우리 표 브랜드 (독일 기존 예: {', '.join(brands[:8])} …): ")
            model = _ask("우리 표 차종: ")
            row = germany_row(brand, model, kba_brand, kba_line)
        print("\n추가할 줄:")
        for k, v in row.items():
            if v != "":
                print(f"  {k}: {v}")
        if _ask("이대로 추가할까요? (y/엔터=취소) ").lower() != "y":
            print("  취소했습니다.")
            return
        backup = add_mapping_row(root, market, row)
        print(f"  추가했습니다. 백업: {backup}")
        print("  다음 수집(step1)부터 이 차종 실적이 원본에 새 줄로 쌓입니다. 이력 3개월부터 보조 예측, 13개월부터 본 예측.")
        print("  기록: raw/mapping_additions.csv")
    except ValueError as e:
        print(f"  추가하지 않았습니다: {e}")
    _pause()


def show_monitoring(root):
    md = root / "raw" / "monitoring.md"
    if not md.exists():
        print("  아직 성적이 없습니다(수집(step1)으로 새 달 실적이 들어오면 raw/monitoring.md 가 생깁니다).")
    else:
        print("\n" + md.read_text(encoding="utf-8", errors="replace"))
    _pause()


def find_bash():
    for p in [r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"]:
        if Path(p).exists():
            return p
    b = shutil.which("bash")
    return b if b and "system32" not in b.lower() else None


def run_menu(root):
    if not is_package(root):
        print("\n[4. 실행]\n" + DEV_NOTICE)
        _pause()
        return
    py = sys.executable
    runs = {"0": ("처음 설치 (step0_setup.py)", [py, "-X", "utf8", "step0_setup.py"], "10~30분, 처음 한 번, 인터넷 필요"),
            "1": ("수집 (step1_collect.py)", [py, "step1_collect.py"], "약 1시간, 인터넷 필요. 빠진 달을 모아 원본에 넣고 성적을 매김"),
            "2": ("재학습 (step2_train.py)", [py, "step2_train.py"], "약 4.5시간(GPU). 원본으로 처음부터 학습해 새 예측"),
            "3": ("예측 출력 (predict.py)", [py, "-X", "utf8", "predict.py"], "몇 초. outputs/forecast/ 에 forecast.json·CSV"),
            "4": ("백테스트 (backtest.sh)", ["BASH", "backtest.sh"], "6~10시간 이상(정확히 잰 적 없음). 모델을 바꿨을 때만. 점수만 나오고 예측은 안 바뀜")}
    while True:
        print("\n[4. 실행]  각 파일은 메뉴 없이 따로 실행해도 됩니다.")
        for k, (name, _, t) in runs.items():
            print(f"  {k}) {name}  — {t}")
        print("  9) 돌아가기")
        c = _ask("> ")
        if c not in runs:
            return
        name, cmd, t = runs[c]
        if cmd[0] == "BASH":
            b = find_bash()
            if not b:
                print("  Git Bash 를 찾지 못했습니다. Git for Windows 를 설치하세요.")
                _pause()
                continue
            cmd = [b] + cmd[1:]
        if _ask(f"{name} 을(를) 실행할까요? ({t})  y/엔터=취소: ").lower() != "y":
            continue
        print(f"  실행: {' '.join(cmd)}\n")
        rc = subprocess.run(cmd, cwd=root).returncode
        print(f"\n  끝 (종료 코드 {rc}{', 정상' if rc == 0 else ', 오류 — 위 메시지를 확인하세요'})")
        _pause()


def settings_lines(root):
    sys.path.insert(0, str(Path(root) / "experiments" / "collectors_v1" / "pipeline"))
    lines = ["\n[5. 설정 보기]  (읽기 전용 — 바꾸면 점수가 달라지므로 메뉴에서 바꾸지 않습니다)"]
    try:
        import monthly
        import monitor
        lines += [f"  원본에 넣는 기준: 추정 몫 {monthly.EST_THRESHOLD:.0%} 미만이면 raw/data.csv, 넘으면 raw/filled_values.csv",
                  "  다시 수집: 채운 값은 3개월 동안 매달 다시 조사, 원본에 들어간 칸은 다시 조사하지 않음",
                  f"  차종별 실시간 성적: 지난 예측 vs 실제를 차종마다 최근 {monitor.WINDOW}개 달로 계산"
                  f" ({monitor.MIN_MONTHS}개월 이상 쌓이면 더 믿을 만함)"]
    except Exception as e:                                   # noqa: BLE001
        lines.append(f"  (운영 설정을 읽지 못함: {e})")
    lines += [f"  예측 길이 24개월, 챔피언 레시피 {CHAMPION} (백테스트 {MAIN_SCORE}%)",
              "  설명서: README.md"]
    return lines


def show_settings(root):
    print("\n".join(settings_lines(root)))
    _pause()


def main(root=ROOT):
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    root = Path(root)
    if not is_package(root):
        print("※ 작업 폴더에서 켰습니다: 보기 기능만 쓸 수 있고, 실행·새 차종 추가는 전달 묶음 폴더에서만 됩니다.")
    while True:
        try:
            d = load_detail(root)
        except FileNotFoundError:
            print("납품 예측 파일이 없습니다. 먼저 step2_train.py 로 재학습하거나 묶음을 다시 받으세요.")
            d = None
        if d is not None:
            _header(d)
        print(" 1. 점수 보기\n 2. 차종 관리\n 3. 이번 달 성적·경보\n 4. 실행 (설치·수집·재학습·예측·백테스트)\n 5. 설정 보기\n 0. 끝내기")
        c = _ask("> ")
        if c == "1" and d is not None:
            view_scores(root, d)
        elif c == "2" and d is not None:
            manage_cars(root, d)
        elif c == "3":
            show_monitoring(root)
        elif c == "4":
            run_menu(root)
        elif c == "5":
            show_settings(root)
        elif c == "0":
            return


if __name__ == "__main__":
    main()
