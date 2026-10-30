"""DAIZEN 차종별 24개월 판매량 예측 — 전달본 실행 파일(ML2 형식 출력).

납품 목록(raw/deliveries.csv)의 가장 최근 예측(처음엔 2026-07 기준 납품, step2_train.py 를 돌린 뒤에는 그 결과)을
outputs/forecast/forecast.json · forecasft_data.csv(6열) · forecast_detail.csv(범위·등급) 로 만든다. 학습은 하지 않는다.

사용:  uv run --frozen predict.py                                  (또는 .venv/Scripts/python.exe predict.py)
       uv run --frozen predict.py --output "outputs/이번예측/forecast.json"
       uv run --frozen predict.py --package <client_package 폴더>   (특정 납품 묶음 지정)
새 달 자료로 다시 예측하려면: step1_collect.py(수집·반영) → step2_train.py(재학습, 약 5시간) → predict.py
"""
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import io
import json
from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parent
for _stream in (sys.stdout, sys.stderr):   # 콘솔 인코딩이 cp949/cp1252라도 한글 출력이 깨지지 않게
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
sys.dont_write_bytecode = True
KEY = ["Country", "Brand", "Model"]
FORECAST_FIELDS = ["Country", "Brand", "Model", "Forecast Month", "Horizon", "Predicted Sales"]


def registry():
    sys.path.insert(0, str(ROOT / "experiments" / "collectors_v1" / "pipeline"))
    import deliveries
    return deliveries.load()


def pick_delivery(reg, package=None):
    """--package 가 있으면 그 폴더의 client_forecast_24m.csv, 없으면 납품 목록에서 기준월이 가장 늦은 것."""
    if package is not None:
        p = Path(package)
        p = p if p.is_file() else p / "client_forecast_24m.csv"
        if not p.exists():
            raise FileNotFoundError(f"납품 파일이 없습니다: {p}")
        return p
    if not len(reg):
        raise FileNotFoundError("납품 목록이 비어 있습니다(raw/deliveries.csv). step2_train.py 를 먼저 돌리세요.")
    return Path(reg.sort_values("origin").iloc[-1]["path"])


def to_outputs(package_dir, output, vehicle_status):
    detail = pd.read_csv(Path(package_dir) / "client_forecast_24m.csv", encoding="utf-8-sig", dtype={"Month": str, "origin": str})
    origin = str(detail["origin"].iloc[0])
    forecasts, excluded = [], []
    for key, rows in detail.groupby(KEY, sort=True):
        rows = rows.sort_values("horizon")
        identity = dict(zip(KEY, key))
        if rows["prediction"].notna().all():
            for r in rows.itertuples():
                forecasts.append({**identity, "Forecast Month": r.Month, "Horizon": int(r.horizon),
                                  "Predicted Sales": int(Decimal(str(r.prediction)).to_integral_value(rounding=ROUND_HALF_UP))})
        else:
            excluded.append({**identity, "status": str(rows["status"].iloc[0]), "reason": str(rows["status_ko"].iloc[0])})
    if vehicle_status is not None and Path(vehicle_status).exists():
        have = {tuple(r[k] for k in KEY) for r in forecasts} | {tuple(r[k] for k in KEY) for r in excluded}
        for r in pd.read_csv(vehicle_status, encoding="utf-8-sig").to_dict("records"):
            if tuple(r[k] for k in KEY) not in have:
                excluded.append({k: r[k] for k in KEY} | {"status": str(r["status"]), "reason": str(r["reason"])})
    excluded.sort(key=lambda r: tuple(r[k] for k in KEY))
    months = sorted(detail["Month"].unique())
    result = {"origin": origin, "horizon_months": int(detail["horizon"].max()), "forecast_months": f"{months[0]}~{months[-1]}",
              "model": "ic_champ (DAIZEN v7 monthly ops)", "package": Path(package_dir).name,
              "summary": {"input_series": int(detail.groupby(KEY).ngroups), "forecasted_series": len({tuple(r[k] for k in KEY) for r in forecasts}),
                          "excluded_series": len(excluded), "forecast_rows": len(forecasts)},
              "predictions": forecasts, "excluded": excluded,
              "warnings": [{"code": "SERIES_EXCLUDED", "message": f"{len(excluded)}개 시계열은 excluded에서 사유를 확인하세요."}] if excluded else []}
    json_path = Path(output)
    csv_path = json_path.with_name("forecasft_data.csv")
    detail_path = json_path.with_name("forecast_detail.csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=FORECAST_FIELDS)
    w.writeheader()
    w.writerows(result["predictions"])
    for r in result["excluded"]:
        w.writerow({"Country": r["Country"], "Brand": r["Brand"], "Model": r["Model"], "Forecast Month": "", "Horizon": "", "Predicted Sales": r["reason"]})
    staged = []
    try:
        for path, text, enc in ((json_path, json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), "utf-8"),
                                (csv_path, buf.getvalue(), "utf-8-sig"),
                                (detail_path, (Path(package_dir) / "client_forecast_24m.csv").read_text(encoding="utf-8-sig"), "utf-8-sig")):
            with tempfile.NamedTemporaryFile("w", encoding=enc, newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as t:
                staged.append((Path(t.name), path))
                t.write(text)
        for tmp, target in staged:
            tmp.replace(target)
    finally:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
    return result


def predict(output=None, package=None):
    output = Path(output) if output is not None else ROOT / "outputs" / "forecast" / "forecast.json"
    src = pick_delivery(registry() if package is None else None, package)
    result = to_outputs(src.parent, output, ROOT / "raw" / "vehicle_status.csv")
    print(f"납품 예측: {src.parent.name} (기준월 {result['origin']}, {result['forecast_months']})")
    print(f"결과 폴더: {output.parent.resolve()}")
    print(f"JSON: {output.name} / CSV: forecasft_data.csv / 상세(범위·등급): forecast_detail.csv")
    print(f"예측 {result['summary']['forecast_rows']:,}행 / 제외 차량 {result['summary']['excluded_series']:,}개")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="가장 최근 납품 예측을 ML2 형식(forecast.json·6열 CSV)으로 출력")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--package", type=Path, help="특정 client_package 폴더(또는 client_forecast_24m.csv)")
    args = parser.parse_args(argv)
    return predict(args.output, args.package)


if __name__ == "__main__":
    result = main()
