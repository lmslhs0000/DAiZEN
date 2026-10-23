"""goal v2 채점 실행. 예측 run 폴더를 읽어 공통 대상 점수표·챔피언 판정·보고서를 만든다.

  score.py --runs <run ...> --report-only   점수표만 (챔피언 변경 없음)
  score.py --runs <run ...>                 첫 챔피언 결정 또는 도전 판정
  score.py --check-origin                   확인용 기준월(2025-07) 확인, 회차 기록
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import (CHECK_HORIZON, CHECK_ORIGIN, CSV_PATH, HERE, HORIZON, KEY, ORIGINS, PROJECT_ROOT, file_sha256,
                    guarded, new_run_dir, read_json, require_unchanged, stack_outputs, write_csv, write_json)
import baseline
from naive import seasonal_naive
from scoring import (GROUPS, MAIN, PART_NAMES, SCORE_COLUMNS, abc_groups, bootstrap_tolerances, cell,
                     choose_initial_champion, coverage, decide, learnable_keys, part_errors, score_model)

BASELINE = "seasonal_naive"
CHAMPION_FILE = HERE / "champion.json"
CHECK_LOG = HERE / "check_origin_log.csv"


def relative(path):
    return str(Path(guarded(path)).relative_to(PROJECT_ROOT))


def read_run(run_dir):
    run_dir = guarded(run_dir)
    p = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    p["Month"] = pd.PeriodIndex(p["Month"], freq="M")
    p["horizon"] = p["horizon"].astype(int)
    p["prediction"] = pd.to_numeric(p["prediction"].replace("", np.nan))
    return p


def naive_predictions(data, origins, horizon):
    frames = []
    for origin in origins:
        p, s = seasonal_naive(data, origin, horizon)
        frames.append(stack_outputs([p], [s], BASELINE, pd.Period(origin, "M"))[0])
    return pd.concat(frames, ignore_index=True)


def common_parts(parts):
    """모든 모델이 OK인 (기준월, 차종, 부분)만 남기고, 모델별 포함·제외 수를 요약한다."""
    models = parts["model"].nunique()
    ids = ["origin"] + KEY + ["part"]
    ok = parts.loc[parts["status"].eq("OK")]
    counts = ok.groupby(ids)["model"].nunique().rename("n_models").reset_index()
    keep = counts.loc[counts["n_models"].eq(models), ids]
    common = ok.merge(keep, on=ids)
    marked = parts.merge(keep.assign(in_common=True), on=ids, how="left")
    marked["in_common"] = marked["in_common"].fillna(False).astype(bool)
    summary = marked.groupby(["model", "status", "in_common"]).size().rename("rows").reset_index()
    return common, summary


def score_models(data, predictions, origins, max_horizon):
    parts = part_errors(predictions.loc[predictions["origin"].isin(origins)], data, max_horizon)
    common, summary = common_parts(parts)
    groups = pd.concat([abc_groups(data, o) for o in origins], ignore_index=True)
    scores = {m: score_model(common.loc[common["model"].eq(m)], groups) for m in sorted(parts["model"].unique())}
    eligible = pd.concat([learnable_keys(data, o) for o in origins], ignore_index=True)
    return scores, common, groups, summary, coverage(parts, eligible)


def write_outputs(out, scores, summary, groups):
    card = pd.concat([table.assign(model=name) for name, table in scores.items()], ignore_index=True)
    write_csv(out / "scorecard.csv", card[["model"] + SCORE_COLUMNS])
    write_csv(out / "cohort_summary.csv", summary)
    if groups is not None:
        write_csv(out / "abc_groups.csv", groups)


def write_results(path, scores, decisions, note):
    def fmt(value):
        return f"{value:.2f}%" if np.isfinite(value) else "-"

    def order(item):
        value = cell(item[1], "overall", "all", "main", MAIN)
        return (not np.isfinite(value), value if np.isfinite(value) else 0.0)

    lines = ["# goal v2 채점 결과", "", note, "",
             "점수는 이 채점 실행에 포함된 모델들의 공통 대상 기준이다. 다른 채점 실행의 점수와 직접 비교하지 않는다.",
             "주 점수 = (묶음 + 달마다) ÷ 2. 묶음은 앞 월·중간 분기 합·뒤 1년 합, 달마다는 모든 부분을 월별로 비교한다. "
             "부분 칸은 '묶음 / 달마다', A·B·C는 규모 그룹의 주 점수다.", "",
             "| 모델 | 주 점수 | 묶음 | 달마다 | 앞 | 중간 | 뒤 | A | B | C |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, table in sorted(scores.items(), key=order):
        mains = [cell(table, "overall", "all", "main", metric) for metric in (MAIN, "pooled_wape", "monthly_wape")]
        parts = [f"{fmt(cell(table, 'overall', 'all', p))} / {fmt(cell(table, 'overall', 'all', p, 'monthly_wape'))}"
                 if p != "front" else fmt(cell(table, "overall", "all", p)) for p in PART_NAMES]
        groups = [fmt(cell(table, "group", g, "main", MAIN)) for g in GROUPS]
        lines.append(f"| {name} | " + " | ".join([fmt(v) for v in mains] + parts + groups) + " |")
    if decisions is not None and len(decisions):
        lines += ["", "## 도전 판정", "", "| 도전자 | 채택 | 실패한 검사 |", "|---|---|---|"]
        for name, table in decisions.groupby("challenger", sort=True):
            failed = ", ".join(table.loc[~table["passed"].astype(bool), "check"]) or "-"
            lines.append(f"| {name} | {'예' if table['accepted'].iloc[0] else '아니오'} | {failed} |")
    guarded(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_backtest(data, predictions, champion, owners, champion_file, out, report_only, challengers=None):
    """challengers: 이번에 넘긴 run의 모델만 도전자로 판정한다. 이전 챔피언 run은 확인용 비교를 위해 불러올 뿐 다시 판정하지 않는다."""
    scores, common, groups, summary, counts = score_models(data, predictions, ORIGINS, HORIZON)
    write_outputs(out, scores, summary, groups)
    if report_only:
        note = "점수표만 계산했습니다(챔피언 변경 없음)."
        write_results(out / "RESULTS.md", scores, None, note)
        return note
    now = datetime.now(timezone.utc).isoformat()
    # 확인용 기준월 예측이 없는 모델은 나중에 확인용 시험을 볼 수 없으므로 챔피언이 될 수 없다.
    has_check = set(predictions.loc[predictions["origin"].eq(CHECK_ORIGIN), "model"])
    if champion is None:
        candidates = {m: s for m, s in scores.items() if m == BASELINE or m in has_check}
        name = choose_initial_champion(candidates, BASELINE, counts)
        tolerances = bootstrap_tolerances(common.loc[common["model"].eq(name)], groups)
        record = {"model": name, "run_dir": owners[name], "decided_at": now,
                  "main": cell(scores[name], "overall", "all", "main", MAIN), "reason": "initial"}
        write_json(champion_file, {"current": record, "history": [record], "main_metric": MAIN,
                                   "tolerances": {k: v["tolerance"] for k, v in tolerances.items()},
                                   "tolerance_basis": tolerances})
        note = f"첫 챔피언: {name}. 허용 폭을 고정했습니다. 다음: score.py --check-origin"
        write_results(out / "RESULTS.md", scores, None, note)
        return note
    current = champion["current"]["model"]
    decisions, accepted = [], []
    for name in sorted(scores):
        if name in (current, BASELINE) or (challengers is not None and name not in challengers):
            continue
        ok, checks = decide(scores[current], scores[name], scores[BASELINE], champion["tolerances"],
                            (counts.get(current, 0), counts.get(name, 0)))
        if name not in has_check:
            missing = {"check": "has_check_origin", "champion": np.nan, "challenger": np.nan,
                       "delta": np.nan, "allowed": np.nan, "passed": False}
            checks = pd.concat([checks, pd.DataFrame([missing])], ignore_index=True)
            ok = False
        decisions.append(checks.assign(challenger=name, accepted=ok))
        if ok:
            accepted.append((cell(scores[name], "overall", "all", "main", MAIN), name))
    table = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame(columns=["challenger", "accepted"])
    write_csv(out / "decisions.csv", table)
    if accepted:
        best_main, best = min(accepted)
        record = {"model": best, "run_dir": owners[best], "decided_at": now, "main": best_main, "reason": f"beat {current}"}
        champion["current"] = record
        champion["history"].append(record)
        write_json(champion_file, champion)
        note = f"새 챔피언: {best}. 다음: score.py --check-origin (나쁘면 사용자 확인)"
    else:
        note = f"챔피언 유지: {current}"
    write_results(out / "RESULTS.md", scores, table, note)
    return note


def user_reject(champion, champion_file, reason):
    """사용자가 거절한 현재 챔피언을 이력에서 빼 rejected에 사유와 함께 남기고, 바로 앞 챔피언으로 되돌린다.

    확인용 기준월이 나빠진 챔피언은 자동 탈락시키지 않고 사용자가 정한다(설계 2절). 거절 뒤에는 이력의 마지막이
    다시 현재 챔피언이므로 다음 도전 판정과 확인용 비교가 되돌린 챔피언 기준으로 이어진다.
    """
    if champion is None or len(champion["history"]) < 2:
        raise ValueError("되돌릴 이전 챔피언이 없습니다.")
    rejected = champion["history"].pop()
    rejected.update(rejected_at=datetime.now(timezone.utc).isoformat(), rejection_reason=reason)
    champion.setdefault("rejected", []).append(rejected)
    champion["current"] = champion["history"][-1]
    write_json(champion_file, champion)
    return f"사용자 거절: {rejected['model']} → 챔피언 {champion['current']['model']}로 되돌림"


CHECK_DRAWS = 2000


def check_noise(data, predictions, current, previous, draws=CHECK_DRAWS, seed=0):
    """(2026-10-04 사용자 결정) 확인용 기준월에서 두 모델이 모두 OK인 차종을 복원 추출해 앞(1~3)·중간(4~12) 차이
    (current − previous, %p)의 점추정과 90% 구간. 앞 = 1~3개월 WAPE, 중간 = (분기 묶음 + 월별) WAPE ÷ 2."""
    quantities = ["actual_sum", "abs_error", "abs_error_monthly"]
    tables = {}
    for name in (current, previous):
        p = predictions.loc[predictions["model"].eq(name) & predictions["origin"].astype(str).eq(CHECK_ORIGIN)]
        parts = part_errors(p, data, CHECK_HORIZON)
        parts = parts.loc[parts["status"].eq("OK")]
        tables[name] = parts.pivot_table(index=KEY, columns="part", values=quantities, aggfunc="sum")
    common = tables[current].index.intersection(tables[previous].index)
    a, b = tables[previous].loc[common], tables[current].loc[common]

    def scores(t):
        front = 100 * t[("abs_error", "front")] / t[("actual_sum", "front")]
        middle = 50 * (t[("abs_error", "middle")] + t[("abs_error_monthly", "middle")]) / t[("actual_sum", "middle")]
        return front, middle

    n = len(common)
    weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=draws)
    sa = pd.DataFrame(weights @ a.to_numpy(float), columns=a.columns)
    sb = pd.DataFrame(weights @ b.to_numpy(float), columns=b.columns)
    (fa, ma), (fb, mb) = scores(sa), scores(sb)
    (pfa, pma), (pfb, pmb) = scores(a.sum(axis=0)), scores(b.sum(axis=0))
    result = {}
    for part, diffs, point in (("front", (fb - fa).to_numpy(), pfb - pfa), ("middle", (mb - ma).to_numpy(), pmb - pma)):
        low, high = np.percentile(diffs, [5, 95])
        result[part] = (float(point), float(low), float(high))
    return result


def check_origin(data, predictions, champion, check_log, out):
    if champion is None:
        raise ValueError("챔피언이 정해진 뒤에만 확인용 기준월을 봅니다.")
    history = champion["history"]
    current = history[-1]["model"]
    previous = history[-2]["model"] if len(history) > 1 else BASELINE
    selected = predictions.loc[predictions["model"].isin({current, previous})]
    scores, _, groups, summary, _ = score_models(data, selected, [CHECK_ORIGIN], CHECK_HORIZON)
    log = pd.read_csv(guarded(check_log), encoding="utf-8-sig") if guarded(check_log).exists() else pd.DataFrame()
    row = {"look_number": len(log) + 1, "looked_at": datetime.now(timezone.utc).isoformat(),
           "current": current, "previous": previous}
    worse = False
    for part in ("front", "middle"):  # 확인용 기준월은 12개월이라 앞·중간만 있다. 종합 점수로 비교한다.
        row[f"{part}_current"] = cell(scores[current], "overall", "all", part, MAIN)
        row[f"{part}_previous"] = cell(scores[previous], "overall", "all", part, MAIN)
        worse = worse or row[f"{part}_current"] > row[f"{part}_previous"]
    row["current_worse"] = bool(worse)  # 옛 규칙(조금이라도 나쁨) 기록용
    noise = check_noise(data, selected, current, previous)
    for part, (_, low, high) in noise.items():
        row[f"{part}_low"], row[f"{part}_high"] = low, high
    # 2026-10-04 사용자 결정: 앞·중간 중 하나라도 90% 구간 아래 끝이 0보다 클 때만(잡음보다 크게) 나쁨
    row["worse_beyond_noise"] = bool(any(low > 0 for _, low, _ in noise.values()))
    write_csv(check_log, pd.concat([log, pd.DataFrame([row])], ignore_index=True))
    write_outputs(out, scores, summary, groups)
    verdict = "잡음보다 크게 나쁨 → 사용자 확인 필요" if row["worse_beyond_noise"] else "잡음 기준 통과"
    note = f"확인용 기준월 {row['look_number']}회차: {current}가 {previous}보다 {verdict}"
    write_results(out / "RESULTS.md", scores, None, note)
    return note


def main(argv=None, champion_file=CHAMPION_FILE, check_log=CHECK_LOG, output_base=HERE / "reports"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--check-origin", action="store_true")
    parser.add_argument("--user-reject", metavar="REASON", help="현재 챔피언을 사용자 결정으로 거절하고 이전 챔피언으로 되돌린다")
    args = parser.parse_args(argv)
    input_hash = file_sha256(args.csv)
    data = baseline.read_sales_csv(guarded(args.csv))
    champion = read_json(champion_file) if guarded(champion_file).exists() else None
    if args.user_reject:
        note = user_reject(champion, champion_file, args.user_reject)
        print(note)
        return note
    run_dirs = [relative(r) for r in args.runs]
    if champion:
        for record in champion["history"]:
            if record["run_dir"] and record["run_dir"] not in run_dirs:
                run_dirs.append(record["run_dir"])
    frames, owners, challengers = [], {BASELINE: ""}, set()
    given = set(run_dirs[:len(args.runs)])
    for run_dir in run_dirs:
        predictions = read_run(run_dir)
        for model in predictions["model"].unique():
            if model in owners:
                raise ValueError(f"같은 모델 이름이 두 곳에 있습니다: {model}")
            owners[model] = run_dir
            if run_dir in given:
                challengers.add(model)
        frames.append(predictions)
    frames.append(naive_predictions(data, ORIGINS + [CHECK_ORIGIN], HORIZON))
    predictions = pd.concat(frames, ignore_index=True)
    out = new_run_dir(output_base, "score")
    if args.check_origin:
        note = check_origin(data, predictions, champion, check_log, out)
    else:
        note = score_backtest(data, predictions, champion, owners, champion_file, out, args.report_only, challengers)
    require_unchanged(args.csv, input_hash)
    write_json(out / "run_summary.json", {"input_sha256": input_hash, "runs": run_dirs, "origins": ORIGINS,
                                          "check_origin": args.check_origin, "report_only": args.report_only,
                                          "note": note})
    print(note)
    print(out)
    return out


if __name__ == "__main__":
    main()
