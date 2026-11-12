"""한 달치 새 실적 모으기 — 시장 수집기 실행 → 표준 형식 → 점검 → 빈 차종은 직전 납품 예측으로 채움.

raw/data.csv 는 건드리지 않는다. 결과는 raw/incoming/<M>/ 에만 쓴다. 반영은 merge_month.py (사용자 확인 뒤).

사용:
  python collect_month.py --month 2026-08                     # 수집기 실행(인터넷) + 묶기
  python collect_month.py --month 2026-08 --offline           # 수집기를 캐시만으로 실행
  python collect_month.py --month 2026-08 --no-run            # 수집기는 돌리지 않고 이미 있는 출력만 묶기
  python collect_month.py --month 2026-08 --dry-run           # 실행할 명령만 출력
  --markets china,japan  --forecast <직전 납품 client_forecast_24m.csv>
"""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import pandas as pd

import monthly as m

HERE = Path(__file__).resolve().parent
COLL = HERE.parent
ROOT = COLL.parents[1]
RAW = ROOT / "raw" / "data.csv"
INCOMING = ROOT / "raw" / "incoming"
DEFAULT_FORECAST = "auto"
JULY_DELIVERY = ROOT / "experiments" / "horizon_eval_v1" / "production" / "client_package_july_ic_20261013T035849_323513Z_9f3a75" / "client_forecast_24m.csv"   # 2026-07 기준 납품(ic)
PIPELINE_PACKAGES = ROOT / "experiments" / "horizon_eval_v1" / "production"


def find_forecast(month):
    """채울 달보다 앞선 기준월의 납품 예측 중 가장 최근 것 — 납품 목록(raw/deliveries.csv, deliveries.py)에 있는 것만."""
    import deliveries
    return deliveries.pick(deliveries.load(), month)


def collector_python():
    """수집기는 시스템 파이썬(requests·openpyxl·bs4·xlrd 설치됨)으로 만들어졌다. COLLECTOR_PY 로 바꿀 수 있다."""
    import os, shutil
    own = COLL / ".venv" / "Scripts" / "python.exe"     # 묶음 설치(step0_setup.py)가 만든 수집기 환경
    return os.environ.get("COLLECTOR_PY") or (str(own) if own.exists() else None) or shutil.which("python") or sys.executable


def markets(month, work, offline):
    """시장 → (실행 명령, 출력 파일, 맞추기 함수). 명령은 수집기 폴더에서 실행한다."""
    y, mo = month.split("-")
    off = ["--offline"] if offline else []
    py = collector_python()
    return {
        "china": ([py, "collect_china.py", y, str(int(mo)), "--out", str(work / "china")] + (["--no-fetch"] if offline else []),
                  COLL / "china", work / "china" / f"collected_{y}_{int(mo)}.csv", m.adapt_china),
        "germany": ([py, "collect_germany.py", "--month", month, "--out", str(work / f"germany_{month}.csv")] + off,
                    COLL / "rules_jp_na_de", work / f"germany_{month}.csv", m.adapt_generic),
        "japan": ([py, "collect_japan.py", "--month", month, "--out", str(work / f"japan_{month}.csv")] + off,
                  COLL / "rules_jp_na_de", work / f"japan_{month}.csv", m.adapt_generic),
        "north_america": ([py, "collect_na.py", "--month", month, "--country", "both", "--out", str(work / f"na_{month}.csv")] + off,
                          COLL / "rules_jp_na_de" / "na", work / f"na_{month}.csv", m.adapt_generic),
        "korea": ([py, "collect_korea.py", "--month", month] + off,
                  COLL / "korea", COLL / "korea" / "out" / f"std_korea_{month}.csv", m.adapt_std),
        "europe": ([py, str(HERE / "europe_chain.py"), "--month", month, "--germany-file", str(work / f"germany_{month}.csv")] + off,
                   COLL / "europe_c", COLL / "europe_c" / "out" / f"std_europe_{month}.csv", m.adapt_std),
    }


def china_src_prev(native_path, month):
    """중국 수집기 lines 파일의 prev_value(같은 CPCA 줄의 직전 달)를 우리 계열로 모음(대응표 sum/split 규칙 그대로)."""
    y, mo = month.split("-")
    lp = Path(native_path).parent / f"lines_{y}_{int(mo)}.csv"
    if not lp.exists():
        return pd.DataFrame(columns=m.KEY + ["src_prev"])
    L = pd.read_csv(lp, encoding="utf-8-sig")
    L = L[L["Month"] == month].set_index("slug")["prev_value"]
    mp = pd.read_csv(COLL / "china" / "mapping.csv", encoding="utf-8-sig")
    sp = pd.read_csv(COLL / "china" / "split_rules.csv", encoding="utf-8-sig").set_index("rule_id")
    rows = []
    for r in mp.itertuples():
        if not isinstance(r.source_key, str) or r.rule == "none":
            continue
        keys = [k.strip() for k in r.source_key.split("+") if k.strip()]
        vals = [L.get(k) for k in keys]
        if any(v is None for v in vals) or all(pd.isna(v) for v in vals):
            continue
        v = float(sum(x for x in vals if pd.notna(x)))
        if isinstance(r.rule, str) and r.rule.startswith("split:"):
            q = sp.loc[r.rule.split(":", 1)[1]]
            v *= float(q["share"]) if r.Model == q["series_a"] else 1 - float(q["share"])
        rows.append({"Country": r.Country, "Brand": r.Brand, "Model": r.Model, "src_prev": v})
    return pd.DataFrame(rows, columns=m.KEY + ["src_prev"])


def new_model_alerts(month, work):
    """수집처에 있는데 우리 목록에 없는 차종(new_models.py)."""
    import new_models as nm
    parts = []
    g = work / "germany.log"
    if g.exists():
        parts.append(nm.from_germany_log(g.read_text(encoding="utf-8", errors="replace"), month))
    for path, fn in [(COLL / "korea" / "out" / f"unmapped_korea_{month}.csv", nm.from_korea),
                     (COLL / "europe_df" / "out" / f"unmapped_df_{month}.csv", nm.from_dataforce),
                     (COLL / "europe_c" / "out" / f"unmapped_europe_{month}.csv", nm.from_europe_countries)]:
        if path.exists():
            d = pd.read_csv(path, encoding="utf-8-sig")
            if len(d):
                parts.append(fn(d, month))
    return nm.combine(parts)


def jump_checks(std, raw, month):
    """직전 달 대비 3배 넘게 오르내린 값(큰 차종만) — 파서 오류·대응 오류 의심."""
    prev = str(pd.Period(month, freq="M") - 1)
    p = raw[raw["Month"] == prev][m.KEY + ["Sales"]].rename(columns={"Sales": "prev"})
    j = std.merge(p, on=m.KEY, how="left")
    big = j[(j[["Sales", "prev"]].max(axis=1) > 300)
            & ((j["Sales"] > 3 * j["prev"].clip(lower=1)) | (j["prev"] > 3 * j["Sales"].clip(lower=1)))]
    return big[m.KEY + ["Month", "prev", "Sales", "value_type", "source_id"]]


def report(month, std, filled, missing, univ, raw, jumps, logs, forecast_path, pend, newm=None):
    vol = raw[raw["Month"].isin({str(pd.Period(month, freq="M") - i) for i in range(1, 13)})] \
        .groupby(m.KEY)["Sales"].sum().rename("vol12").reset_index()
    def share(df):
        return 100 * df.merge(vol, on=m.KEY)["vol12"].sum() / max(vol.merge(univ, on=m.KEY)["vol12"].sum(), 1)
    lines = [f"# {month} 수집 결과 ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"- 대상 계열(직전 12개월 판매 있음): {len(univ)}",
             f"- 수집 값: {len(std)}행 (대상 계열 판매량 기준 {share(std.merge(univ, on=m.KEY)):.1f}%)",
             f"- 예측으로 채움: {len(filled)}행 ({share(filled):.1f}%) — 출처 {forecast_path}",
             f"  - 채움 중 출처 발표 대기(분기 발표 회사 등, 나중에 다시 수집하면 실제 값으로 바뀜): {len(pend)}행 "
             f"({share(pend.merge(univ, on=m.KEY)) if len(pend) else 0:.1f}%) — pending.csv",
             f"- 값도 예측도 없음: {len(missing)}행",
             f"- 대상 밖 수집 값(새 차종 아님·12개월 판매 없음): "
             f"{len(std) - len(std.merge(univ, on=m.KEY))}행", "", "## 시장별", "",
             "판매량(직전 12개월) 기준. 원본 = 실제·나눔·배분·추정 몫 30% 미만, 채움 = 추정 몫 30% 이상 또는 직전 예측.", "",
             "| 시장 | 차종 | 원본(실제) | 채움-추정 | 채움-예측 | 값 없음 |", "|---|---:|---:|---:|---:|---:|"]
    allrows = pd.concat([std, filled], ignore_index=True)
    allrows["dest"] = [m.destination(r) for _, r in allrows.iterrows()]
    for c in sorted(univ["Country"].unique()):
        v = vol.merge(univ[univ.Country == c], on=m.KEY).merge(allrows[m.KEY + ["dest"]], on=m.KEY, how="left")
        tot = max(v["vol12"].sum(), 1)
        p = {d: 100 * v.loc[v["dest"] == d, "vol12"].sum() / tot for d in ("raw", "estimated", "forecast")}
        lines.append(f"| {c} | {len(v)} | {p['raw']:.1f}% | {p['estimated']:.1f}% | {p['forecast']:.1f}% | "
                     f"{100 * v.loc[v['dest'].isna(), 'vol12'].sum() / tot:.1f}% |")
    lines += ["", "## 값 종류", "", std["value_type"].value_counts().to_string(), "",
              "## 목록에 없는 큰 차종 (수집처엔 있고 우리 목록엔 없음, 월 100대 이상)", "",
              (newm.head(20)[["market", "source", "name", "units", "note"]].to_string(index=False) if newm is not None and len(newm) else "없음"), "",
              "추가 방법: new_models.csv 의 how_to_add 열(해당 대응표에 한 줄). 일본·미국·캐나다·중국은 새 차종을 자동으로 못 찾음(README_운영 참고).", "",
              f"## 직전 달 대비 3배 넘는 변화 ({len(jumps)}건, 확인 필요)", "", jumps.head(40).to_string(index=False), "",
              "## 수집기 실행", ""] + logs + ["", "반영: `python merge_month.py --month " + month + "` (미리보기) → `--apply`"]
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--markets", default="china,germany,japan,north_america,korea,europe")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-run", action="store_true", help="수집기 실행 없이 이미 있는 출력만 묶기")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--forecast", default=str(DEFAULT_FORECAST))
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--outdir", default=str(INCOMING), help="시험용으로 다른 곳에 쓸 때")
    ap.add_argument("--prior", nargs="*", default=[], help="원본에 아직 안 넣은 앞 달 incoming.csv (점검용으로 메모리에서만 붙임)")
    a = ap.parse_args()
    pd.Period(a.month, freq="M")
    out = Path(a.outdir).resolve() / a.month
    work = out / "native"
    table = markets(a.month, work, a.offline)
    chosen = [k for k in a.markets.split(",") if k]
    if a.dry_run:
        for k in chosen:
            cmd, cwd, path, _ = table[k]
            print(f"[{k}] cd {cwd} && {' '.join(cmd)}\n    -> {path}")
        return
    raw = pd.read_csv(a.raw, encoding="utf-8-sig")
    raw = m.with_prior(raw, [pd.read_csv(p, encoding="utf-8-sig") for p in a.prior if Path(p).exists()])
    if (raw["Month"] >= a.month).any():
        print(f"주의: 원본에 {a.month} 이후 값이 이미 있음 — merge 때 겹치면 멈춘다")
    work.mkdir(parents=True, exist_ok=True)
    frames, logs, pends, helds = [], [], [], []
    for k in chosen:
        cmd, cwd, path, adapt = table[k]
        if not a.no_run:
            r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            (work / f"{k}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
            if r.returncode != 0:
                logs.append(f"- {k}: 실패(코드 {r.returncode}) — {work / (k + '.log')}")
                continue
        if not path.exists():
            logs.append(f"- {k}: 출력 없음 {path}")
            continue
        native = pd.read_csv(path, encoding="utf-8-sig")
        s = adapt(native, a.month)
        if k == "china":   # 출처 직전 달 ≠ 원본 직전 달이면 대응·기준 어긋남 → 넣지 않음
            held = m.basis_mismatch(china_src_prev(path, a.month), raw, a.month)
            s = m.drop_keys(s, held)
            helds.append(held.assign(market=k))
            logs.append(f"- {k}: 직전 달 기준 불일치로 보류 {len(held)}계열 (held_basis_mismatch.csv)")
        frames.append(s)
        pends.append(m.pending(native, a.month))
        logs.append(f"- {k}: {len(s)}행 ({path.name})")
    std = m.combine(frames)
    univ = m.universe(raw, a.month)
    if a.forecast == "auto":
        a.forecast = find_forecast(a.month) or ""
        print(f"채움 예측: {a.forecast or '없음'}")
    fc = pd.read_csv(a.forecast, encoding="utf-8-sig") if a.forecast and Path(a.forecast).exists() else \
        pd.DataFrame(columns=m.KEY + ["Month", "prediction"])
    filled, missing = m.fill_missing(std, univ, fc, a.month, Path(a.forecast).parent.name)
    jumps = jump_checks(std, raw, a.month)
    newm = new_model_alerts(a.month, work)
    newm.to_csv(out / "new_models.csv", index=False, encoding="utf-8-sig")
    logs.append(f"- 목록에 없는 큰 차종(월 100대 이상): {len(newm)}개 → new_models.csv")
    incoming = pd.concat([std, filled], ignore_index=True)
    incoming.to_csv(out / "incoming.csv", index=False, encoding="utf-8-sig")
    if "status" in fc.columns:   # 왜 예측이 없는지(대부분 INSUFFICIENT_HISTORY = 신차)
        st = fc[fc["Month"].astype(str) == a.month][m.KEY + ["status"]].rename(columns={"status": "forecast_status"})
        missing = missing.merge(st, on=m.KEY, how="left").fillna({"forecast_status": "NOT_IN_FORECAST"})
    missing.to_csv(out / "missing.csv", index=False, encoding="utf-8-sig")
    pend = pd.concat(pends, ignore_index=True) if pends else pd.DataFrame(columns=m.KEY + ["Month", "status"])
    pend.to_csv(out / "pending.csv", index=False, encoding="utf-8-sig")
    (pd.concat(helds, ignore_index=True) if helds else pd.DataFrame(columns=m.KEY)).to_csv(
        out / "held_basis_mismatch.csv", index=False, encoding="utf-8-sig")
    jumps.to_csv(out / "checks_jump.csv", index=False, encoding="utf-8-sig")
    (out / "report.md").write_text(report(a.month, std, filled, missing, univ, raw, jumps, logs, a.forecast, pend, newm), encoding="utf-8")
    print(out / "report.md")


if __name__ == "__main__":
    main()
