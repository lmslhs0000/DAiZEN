"""납품판(기준월 이후 공개 실적 미사용) → 80% 범위 → 고객 묶음. 사용: package_july.py <delivery dir> <name>
2026-10-15: 기준월을 납품판에서 읽는다(7월 고정 해제, 매달 재예측용). 신뢰 등급·백테스트 WAPE 는 7월 순수판의 차종별 값,
상태 설명은 상태별 값으로 붙인다(날짜가 아니라 차종·상태 기준이라 새 기준월의 새 달에도 붙음). 7월 기준이면 결과는 예전과 같다.
2026-10-16: 7월판에 없던 상태(INCOMPLETE_HISTORY 등)도 설명을 붙인다 — 2026-08 기준 첫 재학습이 여기서 멈춘 것을 고침."""
import sys, subprocess, glob, pandas as pd
from pathlib import Path
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H)); ROOT = H.parents[1]
from common import new_run_dir, write_csv, write_json, file_sha256, HERE
RES = "experiments/horizon_eval_v1/reports/fq_champ_20261005T205205_396558Z_059117"
STATUS_KO = {"OK": "본 예측", "OK_SHORT_HISTORY": "신차 보조 예측(이력 3~11개월)", "OK_CHRONOS_ONLY": "보조 예측(학습 모델 미포함 차종)",
             "INSUFFICIENT_HISTORY": "예측 없음(이력 1~2개월)", "INCOMPLETE_HISTORY": "예측 없음(최근 12개월 중 빠진 달)",
             "UNSEEN_SERIES": "예측 없음(학습에 없던 차종)", "NO_TRAINING_ROWS": "예측 없음(학습 자료 없음)",
             "PREDICTION_ERROR": "예측 없음(계산 오류)"}


def status_ko_for(status, old):
    """상태 → 한국어 설명. 7월 순수판(old)에 있으면 그 설명, 없으면 기본 표, 그래도 없으면 '예측 없음(<상태>)'."""
    table = {**STATUS_KO, **dict(zip(old["status"], old["status_ko"]))}
    return status.map(lambda s: table.get(s, f"예측 없음({s})"))


def main(delivery_dir, name):
    D = Path(delivery_dir).resolve()
    before = set(glob.glob(str(H / "production/intervals_2026*")))
    subprocess.run([sys.executable, "-X", "utf8", str(H / "prediction_intervals.py"), "--run", RES,
                    "--short-run", "experiments/horizon_eval_v1/reports/short_chronos_20261005T212154_565197Z_0b8106", "--delivery", str(D)],
                   cwd=str(ROOT), check=True, capture_output=True)
    new = sorted(set(glob.glob(str(H / "production/intervals_2026*"))) - before)
    assert len(new) == 1, new
    I = Path(new[0])
    old = pd.read_csv(H / "production/client_package_20261006T124750_849388Z_2cf467/client_forecast_24m.csv", encoding="utf-8-sig", dtype={"Month": str})
    iv = pd.read_csv(I / "future_24m_with_intervals.csv", encoding="utf-8-sig", dtype={"Month": str, "origin": str})
    origin = iv["origin"].unique()
    assert not iv["status"].eq("OK_SHORT_PUBLIC").any() and len(origin) == 1, origin
    origin = origin[0]
    key = ["Country", "Brand", "Model", "Month"]
    ser = old.drop_duplicates(key[:3])[key[:3] + ["grade", "backtest_wape"]]
    no_eval = old.loc[old["backtest_wape"].isna(), "grade"].mode().iloc[0]      # 평가 없음(신차·이력 부족)
    out = iv[key + ["horizon", "prediction", "lower_80", "upper_80", "status"]].merge(ser, on=key[:3], how="left", validate="many_to_one")
    out["status_ko"] = status_ko_for(out["status"], old.drop_duplicates("status"))
    out["grade"] = out["grade"].fillna(no_eval)
    out = out[key + ["horizon", "prediction", "lower_80", "upper_80", "status", "status_ko", "grade", "backtest_wape"]]
    for c in ["prediction", "lower_80", "upper_80"]:
        out[c] = out[c].round(1)
    out["origin"] = origin
    run = new_run_dir(HERE / "production", name)
    write_csv(run / "client_forecast_24m.csv", out)
    write_json(run / "run_summary.json", {"model": name, "rule": "delivery (no post-origin public actuals) + 80% intervals + reliability grade from pure-July package (packaging only)",
        "origin": origin, "delivery": str(D), "delivery_sha256": file_sha256(next(D.glob("*_future_24m.csv"))),
        "intervals": str(I), "intervals_sha256": file_sha256(I / "future_24m_with_intervals.csv"),
        "grade_from": "client_package_20261006T124750_849388Z_2cf467", "interval_residual_run": RES, "uses_public_actuals_after_origin": False,
        "status_counts": out.drop_duplicates(key[:3])["status"].value_counts().to_dict()})
    print(run)
    return run


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
