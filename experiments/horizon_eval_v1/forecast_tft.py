"""TFT A(tft_forecast.py A 방식 시장별 선택 설정)의 기준월별 24개월 백테스트 예측. 시장×기준월 작업을 병렬·재개 실행한다.

--calendar(tft_wd, 2026-10-01 사전 고정): 루트 tft_core 대신 experiments/tft_calendar_v1의 tft_core_cal(달력 9개
known real)을 쓴다. 설정·학습 규칙은 같다.
--global(tft_global, 2026-10-01 사전 고정): experiments/tft_global_v1의 tft_core_global로 기준월마다 7개 시장 전체에
TFT 하나를 학습한다(작업 이름 <기준월>_ALL, 설정 GLOBAL_PARAMETERS).
--variant log|seed43|seed44(2026-10-01 사전 고정): experiments/tft_variant_v1의 tft_core_variant. TFT A 시장별 설정에
log1p 정규화(tft_log) 또는 다른 seed(tft_a_s43/s44)만 더한다. --device cuda로 GPU 학습(예측은 CPU).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from common import (CHECK_HORIZON, CHECK_ORIGIN, CSV_PATH, HERE, HORIZON, KEY, ORIGINS, ROOT, file_sha256, guarded,
                    new_run_dir, read_json, require_unchanged, stack_outputs, write_csv, write_json)

MODEL_NAME = "tft_a"
SELECTED = "reports/tft/run_20260922T081432_215155Z_3a3e74e049cd/selected_parameters.json"
MODEL_FILES = ("model.ckpt", "dataset_parameters.pt", "feature_schema.json", "artifact_manifest.json")
GLOBAL_PARAMETERS = {"hidden_size": 32, "epochs": 50}
NHITS_PARAMETERS = {"hidden_size": 64, "epochs": 50}  # nhits_global(2026-10-02 사전 고정 (m))
TIDE_PARAMETERS = {"hidden_size": 64, "epochs": 50}  # tide_global(2026-10-02 사전 고정 (n))
NBEATS_PARAMETERS = {"hidden_size": 256, "epochs": 50}  # nbeats_global(2026-10-02 사전 고정 (q)): 블록 폭 256
# --family NAME(2026-10-03부터 새 공동 모델은 여기 등록): 이름 → (모델 이름, 코어, 기본 설정). 코어 폴더는 CORE_FOLDERS.
FAMILIES = {"tide_vw": ("tide_vw", "tide_vw_core", {"hidden_size": 64, "epochs": 50}),
            "rnn_vw": ("rnn_vw", "rnn_vw_core", {"hidden_size": 32, "epochs": 50}),
            "nhits_lvw": ("nhits_lvw", "nhits_lvw_core", {"hidden_size": 64, "epochs": 50}),
            "nhits_brand": ("nhits_brand", "nhits_brand_core", {"hidden_size": 64, "epochs": 50}),
            # (ex) N-HiTS 12개월 직접 출력(블록 재귀), 설정은 nhits_global과 같음
            "nhits_d12": ("nhits_d12", "nhits_d12_core", {"hidden_size": 64, "epochs": 50})}
# --family NAME(--global 없이): 시장마다 따로 학습하는 모델. 시장은 자료에 있는 Country, 설정은 모든 시장에 같게.
PER_MARKET_FAMILIES = {"nhits_pm": ("nhits_pm", "nhits_pm_core", {"hidden_size": 64, "epochs": 50}),
                       "nhits_pm_vw": ("nhits_pm_vw", "nhits_pm_vw_core", {"hidden_size": 64, "epochs": 50}),
                       "deepar_pm": ("deepar_pm", "deepar_pm_core", {"hidden_size": 32, "epochs": 50})}
CORE_FOLDERS = {"tft_core_cal": "tft_calendar_v1", "nhits_d12_core": "nhits_d12_v1", "tft_core_lc": "tft_lifecycle_v1", "tft_core_global": "tft_global_v1",
                "tft_core_variant": "tft_variant_v1", "tft_core_global_cal": "tft_global_cal_v1",
                "tft_core_global_pt": "tft_global_pt_v1", "tft_core_global_age": "tft_global_age_v1",
                "deepar_core": "deepar_global_v1", "nhits_core": "nhits_global_v1",
                "tide_core": "tide_global_v1", "rnn_core": "rnn_global_v1",
                "nhits_tw_core": "nhits_tw_v1", "nbeats_core": "nbeats_global_v1",
                "tft_tw_core": "tft_global_tw_v1", "tft_e24_core": "tft_global_e24_v1",
                "nhits_r48_core": "nhits_r48_v1", "nhits_vw_core": "nhits_vw_v1",
                "tft_vw_core": "tft_global_vw_v1", "deepar_vw_core": "deepar_vw_v1",
                "tft_e24vw_core": "tft_global_e24vw_v1", "nhits_e24_core": "nhits_e24_v1",
                "tide_vw_core": "tide_vw_v1", "rnn_vw_core": "rnn_vw_v1",
                "nhits_lvw_core": "nhits_lvw_v1",
                "nhits_pm_core": "nhits_pm_v1", "nhits_pm_vw_core": "nhits_pm_vw_v1",
                "deepar_pm_core": "deepar_pm_v1",
                "nhits_brand_core": "nhits_brand_v1"}
VARIANTS = {"log": ("tft_log", {"transformation": "log1p"}), "seed43": ("tft_a_s43", {"seed": 43}),
            "seed44": ("tft_a_s44", {"seed": 44}), "seed45": ("tft_a_s45", {"seed": 45}),
            "seed46": ("tft_a_s46", {"seed": 46}), "seed47": ("tft_a_s47", {"seed": 47}),
            "seed48": ("tft_a_s48", {"seed": 48})}  # (eg)


def load_parameters(path=SELECTED, method="A"):
    """원래 TFT 실행에서 method(A 또는 B) 방식으로 고른 시장별 설정."""
    chosen = {r["Country"]: r["parameters"] for r in read_json(path) if r["method"] == method and r["status"] == "OK"}
    if len(chosen) != 7:
        raise ValueError(f"{method} 방식 선택 설정이 7개 시장이 아닙니다: {sorted(chosen)}")
    return chosen


def tft_module(core="tft_core"):
    """루트 tft_core 또는 실험 폴더의 변형 코어(tft_core_cal, tft_core_global)를 불러온다."""
    if core in CORE_FOLDERS:
        folder = str(ROOT / "experiments" / CORE_FOLDERS[core])
        if folder not in sys.path:
            sys.path.insert(0, folder)
    elif core != "tft_core":
        raise ValueError(f"알 수 없는 TFT 코어: {core}")
    return importlib.import_module(core)


def _scope(history, country):
    """작업 범위: 공동 모델(ALL)은 모든 시장, 아니면 그 시장만."""
    return history if country == "ALL" else history.loc[history["Country"].eq(country)]


def run_job(job):
    """한 기준월·한 시장: 기준월까지 자료로 학습 → 저장 → 재귀 예측. done.json이 있으면 건너뛴다."""
    import torch
    import baseline

    tft_core = tft_module(job.get("core", "tft_core"))

    torch.set_num_threads(1)
    out = guarded(job["part_dir"])
    if (out / "done.json").exists():
        return str(out)
    out.mkdir(parents=True, exist_ok=True)
    # done.json이 없으면 이전 시도가 중간에 멈춘 것이므로 그 시도의 미완성 모델 파일만 지운다.
    for leftover in MODEL_FILES:
        (out / leftover).unlink(missing_ok=True)
    origin = pd.Period(job["origin"], freq="M")
    data = baseline.read_sales_csv(guarded(job["csv"]))
    history = data.loc[data["Month"].le(origin)]
    market = _scope(history, job["country"])
    bundle, log = tft_core.fit_market(history, job["country"], job["parameters"], origin,
                                      device=job.get("device", "cpu"), model_path=out / "model.ckpt")
    # 10/3: 학습 장치와 상관없이 예측은 CPU 모델로 한다(재로드 검사와 같은 장치; 코어가 GPU에 두어도 여기서 옮김)
    bundle["model"].to("cpu")
    bundle["model"].eval()
    p, s = tft_core.forecast_recursive(bundle, market, origin, job["horizon"])
    predictions, statuses = stack_outputs([p], [s], job.get("model_name", MODEL_NAME), origin)
    write_csv(out / "predictions.csv", predictions)
    write_csv(out / "statuses.csv", statuses)
    write_csv(out / "training_log.csv", log)
    write_json(out / "done.json", {"origin": str(origin), "country": job["country"], "parameters": job["parameters"]})
    return str(out)


def check_reload(csv_path, part_dir, horizon, core="tft_core"):
    """저장한 TFT 모델·전처리를 다시 불러 같은 예측이 나오는지 확인한다.

    작업 프로세스(run_job)처럼 torch 스레드 1개로 다시 예측한다. 스레드 수가 다르면 float32 합산 순서가 달라져
    24개월 재귀에서 끝자리(상대 1e-5 수준)가 달라진다(2026-10-01 tft_wd에서 확인). 끝나면 원래 스레드 수로 돌린다.
    """
    import torch
    import baseline

    tft_core = tft_module(core)

    part_dir = guarded(part_dir)
    saved = pd.read_csv(part_dir / "predictions.csv", encoding="utf-8-sig", dtype={"Model": str}, keep_default_na=False)
    origin = pd.Period(saved["origin"].iloc[0], freq="M")
    country = read_json(part_dir / "done.json")["country"]
    data = baseline.read_sales_csv(guarded(csv_path))
    market = _scope(data.loc[data["Month"].le(origin)], country)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        bundle = tft_core.load_market_model(part_dir / "model.ckpt", device="cpu")
        again, _ = tft_core.forecast_recursive(bundle, market, origin, horizon)
    finally:
        torch.set_num_threads(threads)
    order = KEY + ["horizon"]
    np.testing.assert_allclose(again.sort_values(order)["prediction"].to_numpy(float),
                               saved.sort_values(order)["prediction"].astype(float).to_numpy(), rtol=1e-7, atol=1e-6)


def _read_parts(parts, name):
    return pd.concat([pd.read_csv(p / name, encoding="utf-8-sig", dtype={"Model": str}, keep_default_na=False)
                      for p in parts], ignore_index=True)


def run(csv_path=CSV_PATH, output_base=HERE / "reports", workers=4, origins=ORIGINS, check=True,
        parameters=None, resume=None, method="A", calendar=False, global_model=False, variant=None, device="cpu",
        powertrain=False, seed=None, age=False, deepar=False, nhits=False, tide=False, rnn=False, nhits_tw=False, nbeats=False, tft_tw=False, e24=False, nhits_r48=False, nhits_vw=False, tft_vw=False, deepar_vw=False, e24vw=False, nhits_e24=False, family=None, lifecycle=False,
        markets=None):
    per_market = family in PER_MARKET_FAMILIES
    if lifecycle and (calendar or global_model or variant or per_market or method != "A"):
        raise ValueError("--lifecycle은 TFT A(시장별, 변형 없음)와만 함께 씁니다.")
    if family is not None and not ((family in FAMILIES and global_model) or (per_market and not global_model)):
        raise ValueError(f"--family: 공동 모델 {sorted(FAMILIES)}는 --global과 함께, "
                         f"시장별 모델 {sorted(PER_MARKET_FAMILIES)}는 --global 없이 씁니다.")
    if (nhits_tw or nbeats or tft_tw or e24 or nhits_r48 or nhits_vw or tft_vw or deepar_vw or e24vw
            or nhits_e24) and not global_model:
        raise ValueError("--nhits-tw·--nbeats·--tft-tw·--e24·--nhits-r48은 --global과 함께만 씁니다.")
    if rnn and not global_model:
        raise ValueError("--rnn은 --global과 함께만 씁니다(rnn_global: 7개 시장 공동 LSTM + 동력원).")
    if tide and not global_model:
        raise ValueError("--tide는 --global과 함께만 씁니다(tide_global: 7개 시장 공동 TiDE + 동력원).")
    if nhits and not global_model:
        raise ValueError("--nhits는 --global과 함께만 씁니다(nhits_global: 7개 시장 공동 N-HiTS + 동력원).")
    if deepar and not global_model:
        raise ValueError("--deepar는 --global과 함께만 씁니다(deepar_global: 7개 시장 공동 DeepAR + 동력원).")
    if age and not (global_model and powertrain):
        raise ValueError("--age는 --global --powertrain과 함께만 씁니다(tft_global_pta).")
    if seed is not None and not (global_model and (powertrain or deepar or nhits or tide or rnn or nhits_tw or nbeats or tft_tw or e24 or nhits_r48 or nhits_vw or tft_vw or deepar_vw or e24vw or nhits_e24
                                                   or family is not None) or (family in PER_MARKET_FAMILIES and not global_model)):
        raise ValueError("--seed는 --global --powertrain(tft_global_pt)에만 씁니다. TFT A 시드는 --variant를 쓰세요.")
    input_hash = file_sha256(csv_path)
    if global_model:
        # 공동 모델은 시장별 설정이 아니라 사전 고정한 설정 하나를 모든 시장에 쓴다.
        family_defaults = FAMILIES[family][2] if family is not None else None
        parameters = {"ALL": parameters or family_defaults or (NHITS_PARAMETERS if (nhits or nhits_tw or nhits_r48 or nhits_vw or nhits_e24) else TIDE_PARAMETERS if tide
                                          else NBEATS_PARAMETERS if nbeats else GLOBAL_PARAMETERS)}
    if per_market:
        # 시장별 모델: 자료에 있는 시장마다 같은 사전 고정 설정(또는 넘겨받은 설정 하나)을 쓴다.
        import baseline
        markets = sorted(baseline.read_sales_csv(guarded(csv_path))["Country"].unique())
        chosen = parameters or PER_MARKET_FAMILIES[family][2]
        parameters = {market: dict(chosen) for market in markets}
    parameters = parameters or load_parameters(method=method)
    model_name = MODEL_NAME if method == "A" else f"tft_{method.lower()}"
    core = "tft_core_global" if global_model else "tft_core_cal" if calendar else "tft_core"
    if calendar:
        model_name = "tft_wd" if method == "A" else f"{model_name}_wd"
    if lifecycle:
        # --lifecycle: TFT A + 수명 공변량 3개 known real(tft_a_lc, 2026-10-05 사전 고정 (eu))
        model_name, core = "tft_a_lc", "tft_core_lc"
    if global_model:
        # --global --calendar: 공동 TFT + 달력 9개(tft_global_cal, 2026-10-01 사전 고정)
        model_name = "tft_global_cal" if calendar else "tft_global"
        core = "tft_core_global_cal" if calendar else "tft_core_global"
        if powertrain:
            # --global --powertrain: 공동 TFT + 차종별 동력원 고정 입력(tft_global_pt, 2026-10-01 사전 고정)
            model_name, core = "tft_global_pt", "tft_core_global_pt"
            if age:
                # --age: 차종 나이 known real 추가(tft_global_pta, 2026-10-02 사전 고정 (c))
                model_name, core = "tft_global_pta", "tft_core_global_age"
        if deepar:
            # --global --deepar: 7개 시장 공동 DeepAR + 동력원, 음이항 손실(deepar_global, 2026-10-02 사전 고정 (k))
            model_name, core = "deepar_global", "deepar_core"
        if nhits:
            # --global --nhits: 7개 시장 공동 N-HiTS + 동력원(nhits_global, 2026-10-02 사전 고정 (m))
            model_name, core = "nhits_global", "nhits_core"
        if tide:
            # --global --tide: 7개 시장 공동 TiDE + 동력원(tide_global, 2026-10-02 사전 고정 (n))
            model_name, core = "tide_global", "tide_core"
        if rnn:
            # --global --rnn: 7개 시장 공동 LSTM(중앙값) + 동력원(rnn_global, 2026-10-02 사전 고정 (o)); 설정 GLOBAL_PARAMETERS
            model_name, core = "rnn_global", "rnn_core"
        if nhits_tw:
            # --global --nhits-tw: N-HiTS + Tweedie 손실(nhits_tw, 2026-10-02 사전 고정 (p))
            model_name, core = "nhits_tw", "nhits_tw_core"
        if nbeats:
            # --global --nbeats: N-BEATS, 판매 이력만 입력(nbeats_global, 2026-10-02 사전 고정 (q))
            model_name, core = "nbeats_global", "nbeats_core"
        if tft_tw:
            # --global --tft-tw: 공동 TFT + 동력원 + Tweedie 손실(tft_global_tw, 2026-10-02 사전 고정 (r)); 설정 GLOBAL_PARAMETERS
            model_name, core = "tft_global_tw", "tft_tw_core"
        if e24:
            # --global --e24: 공동 TFT + 동력원, 입력 12~24개월 가변(tft_global_e24, 2026-10-02 사전 고정 (s))
            model_name, core = "tft_global_e24", "tft_e24_core"
        if nhits_r48:
            # --global --nhits-r48: N-HiTS, 최근 48개월 정답월만 학습(nhits_r48, 2026-10-02 사전 고정 (u))
            model_name, core = "nhits_r48", "nhits_r48_core"
        if nhits_vw:
            # --global --nhits-vw: N-HiTS, 학습 가중치 = 차종 scale(nhits_vw, 2026-10-02 사전 고정 (v))
            model_name, core = "nhits_vw", "nhits_vw_core"
        if tft_vw:
            # --global --tft-vw: 공동 TFT + 동력원, 학습 가중치 = 차종 scale(tft_global_vw, 2026-10-02 사전 고정 (w))
            model_name, core = "tft_global_vw", "tft_vw_core"
        if deepar_vw:
            # --global --deepar-vw: DeepAR, 학습 가중치 = 차종 scale(deepar_vw, 2026-10-02 사전 고정 (x)); 설정 GLOBAL_PARAMETERS
            model_name, core = "deepar_vw", "deepar_vw_core"
        if e24vw:
            # --global --e24vw: 공동 TFT 입력 12~24개월 + 차종 scale 가중(tft_global_e24vw, 2026-10-03 사전 고정 (y))
            model_name, core = "tft_global_e24vw", "tft_e24vw_core"
        if nhits_e24:
            # --global --nhits-e24: N-HiTS 입력 24개월 고정(nhits_e24, 2026-10-03 사전 고정 (af))
            model_name, core = "nhits_e24", "nhits_e24_core"
        if family is not None:
            model_name, core = FAMILIES[family][:2]
        if seed is not None:
            # --seed N: tft_global_pt3(2026-10-01)·deepar3(2026-10-02)의 seed 43·44 재료 → <모델>_s<N>
            model_name = f"{model_name}_s{seed}"
            parameters = {"ALL": {**parameters["ALL"], "seed": seed}}
    if per_market:
        model_name, core = PER_MARKET_FAMILIES[family][:2]
        if seed is not None:
            # (cz) 시장별 모델의 seed 변형: 모든 시장 설정에 seed를 넣고 이름은 <모델>_s<N>
            model_name = f"{model_name}_s{seed}"
            parameters = {country: {**values, "seed": seed} for country, values in parameters.items()}
    if variant:
        # 변형은 시장별 설정에 변형 설정만 더한다(나머지 학습 규칙은 TFT A와 같다).
        model_name, extra = VARIANTS[variant]
        parameters = {country: {**values, **extra} for country, values in parameters.items()}
        core = "tft_core_variant"
    if markets:
        # (eu) --markets: 목록 시장만 실행(나머지 시장 설정은 쓰지 않음)
        unknown = sorted(set(markets) - set(parameters))
        if unknown:
            raise ValueError(f"설정에 없는 시장: {unknown}")
        parameters = {country: parameters[country] for country in markets}
    run_dir = guarded(resume) if resume else new_run_dir(output_base, model_name)
    plan = [(o, HORIZON) for o in origins] + ([(CHECK_ORIGIN, CHECK_HORIZON)] if check else [])
    jobs = [{"csv": str(guarded(csv_path)), "origin": o, "horizon": h, "country": c, "parameters": parameters[c],
             "model_name": model_name, "core": core, "device": device,
             "part_dir": str(run_dir / "parts" / f"{o}_{c}")}
            for o, h in plan for c in sorted(parameters)]
    if workers == 0:
        for job in jobs:
            print(run_job(job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for finished in pool.map(run_job, jobs):
                print(finished, flush=True)
    parts = [Path(job["part_dir"]) for job in jobs]
    predictions, statuses = _read_parts(parts, "predictions.csv"), _read_parts(parts, "statuses.csv")
    check_reload(csv_path, parts[0], jobs[0]["horizon"], core)
    require_unchanged(csv_path, input_hash)
    write_csv(run_dir / "predictions.csv", predictions)
    write_csv(run_dir / "statuses.csv", statuses)
    write_csv(run_dir / "training_log.csv", _read_parts(parts, "training_log.csv"))
    write_json(run_dir / "run_summary.json", {
        "model": model_name, "method": method, "core": core, "device": device, "input_sha256": input_hash,
        "origins": list(origins),
        "check_origin": CHECK_ORIGIN if check else None, "parameters": parameters, "jobs": len(jobs),
        "markets": sorted(parameters) if markets else None,
        "status_counts": statuses.groupby(["origin", "prediction_status"]).size().rename("n").reset_index().to_dict("records"),
        "reload_check": "PASS"})
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--output-base", default=str(HERE / "reports"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--origins", nargs="*", default=ORIGINS)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--method", default="A", choices=["A", "B"])
    parser.add_argument("--calendar", action="store_true", help="달력 9개 known real을 쓰는 tft_core_cal(tft_wd)")
    parser.add_argument("--global", dest="global_model", action="store_true",
                        help="기준월마다 7개 시장 공동 TFT 하나(tft_global)")
    parser.add_argument("--variant", choices=sorted(VARIANTS), help="tft_core_variant의 log1p 또는 다른 seed")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="학습 장치(예측은 항상 CPU)")
    parser.add_argument("--powertrain", action="store_true", help="--global과 함께: 동력원 고정 입력(tft_global_pt)")
    parser.add_argument("--nhits-tw", dest="nhits_tw", action="store_true", help="--global과 함께: N-HiTS + Tweedie")
    parser.add_argument("--family", choices=sorted({**FAMILIES, **PER_MARKET_FAMILIES}),
                        help="등록된 모델: 공동(FAMILIES)은 --global과 함께, 시장별(PER_MARKET_FAMILIES)은 --global 없이")
    parser.add_argument("--nhits-e24", dest="nhits_e24", action="store_true", help="--global과 함께: N-HiTS 입력 24개월")
    parser.add_argument("--e24vw", action="store_true", help="--global과 함께: 공동 TFT 12~24개월 + scale 가중")
    parser.add_argument("--deepar-vw", dest="deepar_vw", action="store_true", help="--global과 함께: DeepAR 차종 scale 가중")
    parser.add_argument("--tft-vw", dest="tft_vw", action="store_true", help="--global과 함께: 공동 TFT 차종 scale 가중")
    parser.add_argument("--nhits-vw", dest="nhits_vw", action="store_true", help="--global과 함께: N-HiTS 차종 scale 가중")
    parser.add_argument("--nhits-r48", dest="nhits_r48", action="store_true", help="--global과 함께: N-HiTS 최근 48개월 학습")
    parser.add_argument("--e24", action="store_true", help="--global과 함께: 공동 TFT, 입력 12~24개월")
    parser.add_argument("--tft-tw", dest="tft_tw", action="store_true", help="--global과 함께: 공동 TFT + Tweedie")
    parser.add_argument("--nbeats", action="store_true", help="--global과 함께: N-BEATS(판매 이력만)")
    parser.add_argument("--rnn", action="store_true", help="--global과 함께: 공동 LSTM + 동력원(rnn_global)")
    parser.add_argument("--tide", action="store_true", help="--global과 함께: 공동 TiDE + 동력원(tide_global)")
    parser.add_argument("--nhits", action="store_true", help="--global과 함께: 공동 N-HiTS + 동력원(nhits_global)")
    parser.add_argument("--deepar", action="store_true", help="--global과 함께: 공동 DeepAR + 동력원(deepar_global)")
    parser.add_argument("--age", action="store_true", help="--global --powertrain과 함께: 차종 나이 입력(tft_global_pta)")
    parser.add_argument("--seed", type=int, help="--global --powertrain과 함께: 학습 seed(기본 42) → tft_global_pt_s<N>")
    parser.add_argument("--lifecycle", action="store_true", help="(eu) TFT A + 수명 공변량 3개(tft_core_lc, tft_a_lc)")
    parser.add_argument("--markets", nargs="+", help="(eu) 이 시장들만 실행")
    args = parser.parse_args(argv)
    print(run(args.csv, args.output_base, args.workers, args.origins, not args.no_check, resume=args.resume,
              method=args.method, calendar=args.calendar, global_model=args.global_model, variant=args.variant,
              device=args.device, powertrain=args.powertrain, seed=args.seed, age=args.age,
              deepar=args.deepar, nhits=args.nhits, tide=args.tide, rnn=args.rnn,
              nhits_tw=args.nhits_tw, nbeats=args.nbeats, tft_tw=args.tft_tw, e24=args.e24,
              nhits_r48=args.nhits_r48, nhits_vw=args.nhits_vw,
              tft_vw=args.tft_vw, deepar_vw=args.deepar_vw,
              e24vw=args.e24vw, nhits_e24=args.nhits_e24,
              family=args.family, lifecycle=args.lifecycle, markets=args.markets))


if __name__ == "__main__":
    main()
