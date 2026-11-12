"""Chronos-2 zero-shot 백테스트 예측(2026-10-03 사전 고정 (ar)).

experiments/chronos_v1/chronos_core.py를 쓴다. 기준월마다 ETS와 같은 이력 규칙으로 차종 이력을 고르고, 학습 없이
사전학습 모델 amazon/chronos-2로 24개월(확인용 기준월은 12개월)을 한 번에 0.5 분위수로 예측해 0 미만은 0으로 자른다.
chronos가 있는 그 폴더 전용 환경으로 실행한다(루트 환경은 바꾸지 않음).
--joint-market(chronos2_xl, 2026-10-03 사전 고정 (as)): 시장마다 (Brand, Model) 순 차종을 최대 100개씩 묶어
cross-learning으로 함께 예측한다(시장 사이는 섞지 않음).
--timesfm(timesfm, 2026-10-03 사전 고정 (at)): 같은 방식으로 experiments/timesfm_v1/timesfm_core(TimesFM 2.5)를 쓴다.
timesfm이 있는 experiments/timesfm_v1/.venv로 실행한다.
--sundial(sundial, 2026-10-03 사전 고정 (cm)): experiments/sundial_v1/sundial_core(Sundial zero-shot, 고정 리비전·검토한
원격 코드, 같은 길이끼리 묶어 20개 표본 중앙값). experiments/sundial_v1/.venv로 실행한다.
--finetune(chronos2_ft, 2026-10-03 사전 고정 (au)): 기준월마다 그 기준월까지의 OK 차종 이력으로 Chronos-2를 미세조정
(experiments/chronos_ft_v1/chronos_ft_core, 라이브러리 기본 설정)한 뒤 --joint-market 방식으로 예측한다. GPU 환경
experiments/chronos_ft_v1/.venv로 `--device cuda`와 함께 실행한다. --ft-steps는 검산용으로만 줄인다.
--entrant(chronos2_xl_ent, 2026-10-03 사전 고정 (bb)): --joint-market에 그 시장의 신차 비중 2개(experiments/entrant_v1)를
이력과 같은 달의 과거 공변량으로 더한다(정의 안 되는 달은 NaN, 미래 공변량 없음, 묶음당 차종 100대).
--per-market(chronos2_ftpm, 2026-10-03 사전 고정 (bd)): --finetune을 기준월마다 시장별로 따로 한다(그 시장 차종 이력만으로
미세조정, 그 시장 모델로 예측). 체크포인트는 models/<기준월>/<시장>/finetuned-ckpt.
--covariates eb(chronos2_ft_eb, 2026-10-03 사전 고정 (bj)): --finetune에 과거 공변량 4개(신차 비중 2 + 브랜드 흐름 2,
그 차종 시장·브랜드의 그 달 기준값, NaN 허용)를 학습과 예측에 함께 넣는다. 예측은 시장 묶음(차종 100대).
--covariates lcf(chronos2_ft_lcf, 2026-10-04 사전 고정 (do)): 수명 특징 3개를 미래에도 아는 공변량으로 미세조정하고
(--lifecycle과 같은 과거·미래 값) 시장 묶음으로 예측한다.

--from-run <미세조정 run> --cross-market(chronos2_ft_xm, 2026-10-04 사전 고정 (dj)): 새 학습 없이 그 run의 기준월별
체크포인트(models/<기준월>/finetuned-ckpt)를 다시 쓰고, cross-learning 묶음만 바꾼다 — 같은 (Brand, Model)이 2개 시장
이상이면 한 묶음, 나머지는 시장마다 (Brand, Model) 순 100개씩. experiments/chronos_ft_v1/.venv로 실행한다.

--from-run <미세조정 run> --lifecycle(chronos2_ft_lc, 2026-10-04 사전 고정 (dn)): 같은 체크포인트 재사용, 시장 묶음 100개,
수명 특징 3개(catboost_v1/lifecycle_features_g)를 과거 공변량(달 m의 값은 기준 시점 m−1)과 미래 공변량(예측 달의 값은
기준 시점 = 기준월)으로 넣는다.

사용: experiments/chronos_v1/.venv/Scripts/python.exe -X utf8 -B experiments/horizon_eval_v1/forecast_chronos.py
"""

import argparse
import importlib.util
import sys
import json

import numpy as np
import pandas as pd

from common import (CHECK_HORIZON, CHECK_ORIGIN, CSV_PATH, HERE, HORIZON, KEY, ORIGINS, ROOT, file_sha256, guarded,
                    new_run_dir, require_unchanged, stack_outputs, write_csv, write_json)
import baseline

MODEL_NAME = "chronos2"
JOINT_RULE = "per Country, sorted (Brand, Model), chunks of 100"
CROSS_MARKET_RULE = "same (Brand, Model) in 2+ markets as one group; others per Country, sorted (Brand, Model), chunks of 100"
GROUP_SIZE = 100
BRAND_RULE = "per Country: each brand with 2+ series one group; single-series brands chunks of 100"
CORE_FILES = {"chronos2": ROOT / "experiments" / "chronos_v1" / "chronos_core.py",
              "timesfm": ROOT / "experiments" / "timesfm_v1" / "timesfm_core.py",
              "sundial": ROOT / "experiments" / "sundial_v1" / "sundial_core.py",
              "patchtst_fm": ROOT / "experiments" / "patchtst_fm_v1" / "patchtst_fm_core.py"}
core = None


def use_core(name):
    """chronos2 또는 timesfm 코어를 파일 경로로 불러 전역 core로 둔다(각자 전용 환경에만 라이브러리가 있음)."""
    global core
    spec = importlib.util.spec_from_file_location(f"{name}_core", CORE_FILES[name])
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


ENTRANT_FILE = ROOT / "experiments" / "entrant_v1" / "entrant_features.py"
BRAND_FILE = ROOT / "experiments" / "brand_momentum_v1" / "brand_features.py"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entrant_covariates(history, keys, histories, origin, kind="entrant"):
    """차종마다 이력의 각 달(기준월에서 거꾸로 len개월)의 과거 공변량. 기준월 이하 자료만 쓴다.
    kind "entrant"((bb)): 그 시장의 신차 비중 2개. kind "eb"((bj)): + 그 브랜드의 브랜드 흐름 2개."""
    ef = _module("entrant_features", ENTRANT_FILE)
    stats = ef.entrant_stats(history)
    origin = pd.Period(origin, freq="M")
    longest = max(len(values) for values in histories)
    months = pd.period_range(origin - longest + 1, origin, freq="M")
    tables, names = {}, list(ef.COLUMNS)
    for country in dict.fromkeys(key[0] for key in keys):
        frame = ef.entrant_features([country] * len(months), months, stats)
        tables[country] = {name: frame[name].to_numpy(float) for name in ef.COLUMNS}
    brands = {}
    if kind == "eb":
        bf = _module("brand_features", BRAND_FILE)
        totals = bf.brand_totals(history)
        names += list(bf.COLUMNS)
        for country, brand in dict.fromkeys((key[0], key[1]) for key in keys):
            frame = bf.brand_features([country] * len(months), [brand] * len(months), months, totals)
            brands[(country, brand)] = {name: frame[name].to_numpy(float) for name in bf.COLUMNS}
    result = []
    for key, values in zip(keys, histories):
        start = longest - len(values)
        row = {name: tables[key[0]][name][start:] for name in ef.COLUMNS}
        if kind == "eb":
            row.update({name: column[start:] for name, column in brands[(key[0], key[1])].items()})
        result.append(row)
    return result, names


LIFECYCLE_FILE = ROOT / "experiments" / "catboost_v1" / "lifecycle_features_g.py"
LIFECYCLE_NAMES = ["lc_months_to_replacement", "lc_months_since_newgen", "lc_months_to_newgen"]
# (em) 공변량 묶음: g = (dn) 그대로, g3 = 중국 사건표 v2 우선 + 가격 인하 뒤 개월(catboost_v1/lifecycle_features_g3)
LIFECYCLE_SETS = {
    "g": (LIFECYCLE_FILE, LIFECYCLE_NAMES, ["lifecycle_events_l2.csv", "lifecycle_events_global.csv"]),
    "g3": (LIFECYCLE_FILE.parent / "lifecycle_features_g3.py", LIFECYCLE_NAMES + ["lc_months_since_price_reset"],
           ["lifecycle_events_l2.csv", "lifecycle_events_global.csv", "lifecycle_events_china_v2.csv",
            "lifecycle_targets_china_v3.csv"]),
    # (ew) gf = g 3개 + 부분 변경 2개(facelift_features, 사건표는 g3 합치기)
    "gf": (LIFECYCLE_FILE, LIFECYCLE_NAMES + ["lc_months_since_facelift", "lc_months_to_facelift"],
           ["lifecycle_events_l2.csv", "lifecycle_events_global.csv", "lifecycle_events_china_v2.csv"]),
    # (fp) gl = g 3개 + 중국 신차 출시 압력(같은 브랜드 (m−12, m] 신차 수, 공개된 것만)
    "gl": (LIFECYCLE_FILE, LIFECYCLE_NAMES + ["launch_pressure"],
           ["lifecycle_events_l2.csv", "lifecycle_events_global.csv"]),
    # (fj) ga = g 3개 + 차종 나이(마지막 연속 구간 안 순번, 0~120)
    "ga": (LIFECYCLE_FILE, LIFECYCLE_NAMES + ["age_months"],
           ["lifecycle_events_l2.csv", "lifecycle_events_global.csv"]),
}
FACELIFT_FILE = HERE / "facelift_features.py"
AGE_CAP = 120
LAUNCH_FILE = HERE / "launch_features.py"
LAUNCH_TABLE = None
LOG1P = False  # (gc) True면 log1p 이력으로 예측하고 expm1로 되돌린다(미세조정도 log1p 이력)  # (fp) 시험용으로만 다른 표 경로를 넣는다(없으면 launch_features.TABLE)


def from_run_name(source_model, cross_market, lifecycle_set="g", with_eb=False, brand_groups=False, lifecycle=True):
    """(en) --from-run 결과 이름 = 체크포인트 출처 모델 이름(seed 포함) + _xm / _lc / _lc3 / _lceb((er)).
    (fs) 브랜드 묶음이면 _bg(공변량 없음) / _lcbg(수명 공변량 g)."""
    if brand_groups:
        return source_model + ("_lcbg" if lifecycle else "_bg")
    if cross_market:
        return source_model + "_xm"
    suffix = {"g3": "_lc3", "gf": "_lcf2", "ga": "_lcga", "gl": "_lcl"}.get(lifecycle_set, "_lc")
    return source_model + suffix + ("eb" if with_eb else "")


def lifecycle_eb_covariates(history, keys, histories, origin, horizon):
    """(er) 수명 공변량(과거·미래) + eb 4개(신차 비중·브랜드 흐름, 과거만). 기준월 이하 자료만 쓴다."""
    past, future = lifecycle_covariates(keys, histories, origin, horizon)
    eb, _ = entrant_covariates(history, keys, histories, origin, "eb")
    return [{**lc, **extra} for lc, extra in zip(past, eb)], future


def lifecycle_covariates(keys, histories, origin, horizon, version="g"):
    """(dn) 차종마다 (과거 공변량, 미래 공변량). 과거 달 m = 기준 시점 m−1, 예측 달 = 기준 시점 기준월.
    version = LIFECYCLE_SETS 이름((em) g3: 공변량 4개)."""
    path, names, _ = LIFECYCLE_SETS[version]
    lf = _module(path.stem, path)
    ff = _module("facelift_features", FACELIFT_FILE) if version == "gf" else None
    if version == "gl":
        lm = _module("launch_features", LAUNCH_FILE)
        launch_table = lm.load_table(LAUNCH_TABLE)
    origin = pd.Period(origin, freq="M")
    past, future = [], []
    for (country, brand, model), values in zip(keys, histories):
        months = pd.period_range(origin - len(values) + 1, origin, freq="M")
        before = lf.lifecycle_features([country] * len(months), [brand] * len(months), [model] * len(months),
                                       months, months - 1)
        if ff is not None:
            before = pd.concat([before, ff.facelift_features([country] * len(months), [brand] * len(months),
                                                             [model] * len(months), months, months - 1)], axis=1)
        targets = pd.period_range(origin + 1, origin + horizon, freq="M")
        ahead = lf.lifecycle_features([country] * horizon, [brand] * horizon, [model] * horizon, targets,
                                      [origin] * horizon)
        if ff is not None:
            ahead = pd.concat([ahead, ff.facelift_features([country] * horizon, [brand] * horizon, [model] * horizon,
                                                           targets, [origin] * horizon)], axis=1)
        if version == "gl":
            before = before.assign(launch_pressure=lm.launch_pressure([brand] * len(months), months, months - 1,
                                                                    launch_table).to_numpy())
            ahead = ahead.assign(launch_pressure=lm.launch_pressure([brand] * horizon, targets, [origin] * horizon,
                                                                  launch_table).to_numpy())
        if version == "ga":
            before = before.assign(age_months=np.minimum(np.arange(len(values), dtype=float), AGE_CAP))
            ahead = ahead.assign(age_months=np.minimum(np.arange(len(values), len(values) + horizon, dtype=float), AGE_CAP))
        past.append({name: before[name].to_numpy(float) for name in names})
        future.append({name: ahead[name].to_numpy(float) for name in names})
    return past, future


def nameplate_groups(keys):
    """(dj) 차종 순번 묶음: 2개 시장 이상에 있는 (Brand, Model)은 이름마다 한 묶음(이름 순), 나머지는 시장마다
    (들어온 순서대로) GROUP_SIZE개씩."""
    by_name = {}
    for i, (_, brand, model) in enumerate(keys):
        by_name.setdefault((brand, model), []).append(i)
    groups = [members for _, members in sorted(by_name.items()) if len(members) >= 2]
    shared = {i for members in groups for i in members}
    by_country = {}
    for i, key in enumerate(keys):
        if i not in shared:
            by_country.setdefault(key[0], []).append(i)
    for members in by_country.values():
        groups += [members[start:start + GROUP_SIZE] for start in range(0, len(members), GROUP_SIZE)]
    return groups


def brand_groups(keys):
    """(fs) 차종 순번 묶음: 시장마다(들어온 순서) 차종이 2개 이상인 브랜드는 브랜드마다 한 묶음(GROUP_SIZE개씩 자름),
    차종이 하나뿐인 브랜드의 차종은 그 시장 안에서 들어온 순서대로 GROUP_SIZE개씩."""
    by_country = {}
    for i, (country, brand, _) in enumerate(keys):
        by_country.setdefault(country, {}).setdefault(brand, []).append(i)
    groups = []
    for brands in by_country.values():
        singles = []
        for members in brands.values():
            if len(members) >= 2:
                groups += [members[start:start + GROUP_SIZE] for start in range(0, len(members), GROUP_SIZE)]
            else:
                singles += members
        groups += [singles[start:start + GROUP_SIZE] for start in range(0, len(singles), GROUP_SIZE)]
    return groups


def forecast_brand_groups(pipeline, keys, histories, horizon, past=None, ahead=None):
    """(fs) brand_groups 묶음마다 cross-learning으로 예측(past가 있으면 수명 공변량과 함께)해 원래 순서로 돌려준다."""
    values = np.full((len(keys), horizon), np.nan)
    for members in brand_groups(keys):
        if past is None:
            values[members] = core.forecast_many(pipeline, [histories[i] for i in members], horizon,
                                                 batch_size=len(members), cross_learning=True)
        else:
            values[members] = core.forecast_joint_cov(pipeline, [histories[i] for i in members],
                                                      [past[i] for i in members], horizon,
                                                      future=[ahead[i] for i in members])
    return values


def forecast_cross_market(pipeline, keys, histories, horizon):
    """(dj) nameplate_groups 묶음마다 cross-learning으로 예측해 원래 차종 순서로 돌려준다."""
    values = np.full((len(keys), horizon), np.nan)
    for members in nameplate_groups(keys):
        values[members] = core.forecast_many(pipeline, [histories[i] for i in members], horizon,
                                             batch_size=len(members), cross_learning=True)
    return values


def forecast_origin(pipeline, data, origin, horizon, joint=False, entrant=False, covariates=None, cross_market=False,
                    lifecycle=False, with_eb=False, brand=False):
    """한 기준월: 차종별 이력 규칙 판정 후 OK 차종을 한꺼번에 예측. (예측 표, 상태 표, OK 이력 목록)."""
    history = data.loc[data["Month"].le(pd.Period(origin, freq="M"))]
    keys, histories, states = [], [], []
    for key, rows in history.groupby(KEY, sort=True):
        prepared = core.prepare(rows[KEY + ["Month", "Sales"]].reset_index(drop=True), origin, key=list(key))
        states.append({**dict(zip(KEY, key)), "prediction_status": prepared["status"], "reason": prepared["reason"]})
        if prepared["status"] == "OK":
            keys.append(key)
            histories.append(prepared["values"])
    raw_histories = histories
    if LOG1P:
        histories = [np.log1p(np.asarray(h, dtype=float)) for h in histories]  # (gc)
    rows = []
    if keys:
        if cross_market:
            values = forecast_cross_market(pipeline, keys, histories, horizon)
        elif brand and not lifecycle:
            values = forecast_brand_groups(pipeline, keys, histories, horizon)  # (fs)
        elif lifecycle:
            if with_eb:
                past, ahead = lifecycle_eb_covariates(history, keys, histories, origin, horizon)
            else:
                past, ahead = lifecycle_covariates(keys, histories, origin, horizon,
                                                   version=lifecycle if isinstance(lifecycle, str) else "g")
            countries = [key[0] for key in keys]
            if brand:
                values = forecast_brand_groups(pipeline, keys, histories, horizon, past, ahead)  # (fs)
            else:
                values = np.concatenate([core.forecast_joint_cov(
                    pipeline, [h for h, c in zip(histories, countries) if c == country],
                    [v for v, c in zip(past, countries) if c == country], horizon,
                    future=[v for v, c in zip(ahead, countries) if c == country]) for country in dict.fromkeys(countries)])
        elif joint and (entrant or covariates):
            covariates, _ = entrant_covariates(history, keys, histories, origin, covariates or "entrant")
            countries = [key[0] for key in keys]
            values = np.concatenate([core.forecast_joint_cov(
                pipeline, [h for h, c in zip(histories, countries) if c == country],
                [v for v, c in zip(covariates, countries) if c == country], horizon)
                for country in dict.fromkeys(countries)])
        elif joint:
            countries = [key[0] for key in keys]
            values = np.concatenate([core.forecast_joint(pipeline, [h for h, c in zip(histories, countries) if c == country],
                                                         horizon) for country in dict.fromkeys(countries)])
        else:
            values = core.forecast_many(pipeline, histories, horizon)
        if LOG1P:
            values = np.maximum(0.0, np.expm1(np.asarray(values, dtype=float)))  # (gc) 원래 척도의 중앙값
        start = pd.Period(origin, freq="M")
        rows = [{**dict(zip(KEY, key)), "Month": start + h, "horizon": h, "prediction": float(v)}
                for key, series in zip(keys, values) for h, v in enumerate(series, start=1)]
    return pd.DataFrame(rows, columns=KEY + ["Month", "horizon", "prediction"]), pd.DataFrame(states), raw_histories


def load_finetune_module(timesfm=False):
    """미세조정 코어: Chronos-2(chronos_ft_v1) 또는 (ho) TimesFM 2.5(timesfm_v1/timesfm_ft_core.py)."""
    path = (ROOT / "experiments" / "timesfm_v1" / "timesfm_ft_core.py" if timesfm
            else ROOT / "experiments" / "chronos_ft_v1" / "chronos_ft_core.py")
    if timesfm and str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))  # timesfm_ft_core가 같은 폴더의 timesfm_core를 import
    spec = importlib.util.spec_from_file_location(
        "timesfm_ft_core" if timesfm else "chronos_ft_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(csv_path=CSV_PATH, output_base=HERE / "reports", origins=ORIGINS, check=True, joint=False, timesfm=False,
        finetune=False, device="cpu", ft_steps=None, entrant=False, per_market=False, covariates=None, seed=None,
        sundial=False, lora=False, from_run=None, cross_market=False, lifecycle=False, ft_lr=None, suffix=None, lifecycle_set="g",
        with_eb=False, brand=False, patchtst=False, point="median", log1p=False):
    global LOG1P
    if with_eb and not (lifecycle and from_run and lifecycle_set == "g"):
        raise ValueError("--with-eb는 --from-run --lifecycle(기본 묶음)과만 함께 씁니다.")
    if lifecycle_set not in LIFECYCLE_SETS or (lifecycle_set != "g" and not (lifecycle and from_run)):
        raise ValueError("--lifecycle-set g3는 --from-run --lifecycle과만 함께 씁니다.")
    if timesfm and (joint or (finetune and (per_market or covariates or lora or log1p or seed is not None))):
        raise ValueError("--timesfm은 --joint-market과 함께 쓰지 않고, --finetune은 기본 설정으로만 씁니다((ho)).")
    if entrant and (not joint or finetune or timesfm):
        raise ValueError("--entrant는 --joint-market과만 함께 씁니다.")
    if per_market and not finetune:
        raise ValueError("--per-market은 --finetune과 함께 씁니다.")
    if covariates and (not finetune or per_market or entrant):
        raise ValueError("--covariates는 --finetune(전 시장)과만 함께 씁니다.")
    if seed is not None and not finetune:
        raise ValueError("--seed는 --finetune과 함께 씁니다.")
    if lora and (not finetune or per_market or covariates):
        raise ValueError("--lora는 --finetune(전 시장, 공변량 없음)과만 함께 씁니다.")
    if sundial and (joint or finetune or timesfm or entrant or per_market or covariates):
        raise ValueError("--sundial은 다른 변형과 함께 쓰지 않습니다.")
    if brand and (not from_run or cross_market or with_eb or lifecycle_set != "g"):
        raise ValueError("--brand-groups는 --from-run(공변량 없음) 또는 --from-run --lifecycle(기본 묶음)과만 함께 씁니다.")
    if log1p and (not (finetune or from_run) or per_market or covariates or lora or point != "median" or patchtst):
        raise ValueError("--log1p는 --finetune(공변량 없음) 또는 --from-run과만 함께 씁니다.")
    LOG1P = bool(log1p)
    mean_point = point == "mean"
    if mean_point and (not from_run or cross_market or brand or with_eb or lifecycle_set != "g"):
        raise ValueError("--point mean은 --from-run(공변량 없음) 또는 --from-run --lifecycle(기본 묶음)과만 함께 씁니다.")
    if (cross_market or lifecycle or brand or mean_point) != bool(from_run) or (cross_market and lifecycle) or (from_run and (finetune or joint or timesfm or entrant or per_market
                                                        or covariates or seed is not None or sundial or lora)):
        raise ValueError("--from-run은 --cross-market 또는 --lifecycle 중 하나와만 함께 씁니다.")
    if patchtst and (joint or finetune or timesfm or entrant or per_market or covariates or sundial or from_run):
        raise ValueError("--patchtst-fm은 다른 변형과 함께 쓰지 않습니다.")
    zero_shot = "timesfm" if timesfm else "sundial" if sundial else "patchtst_fm" if patchtst else None
    joint = (joint or finetune) and not timesfm  # (ho) TimesFM 미세조정판은 차종별 예측(forecast_many) 그대로
    use_core(zero_shot or "chronos2")
    ft = load_finetune_module(timesfm) if finetune or from_run else None
    load = (lambda: core.load_model(device=device)) if patchtst else core.load_model if zero_shot else (
        lambda: core.load_pipeline(device=device))  # (fw) PatchTST-FM은 장치를 받음
    input_hash = file_sha256(csv_path)
    data = baseline.read_sales_csv(guarded(csv_path))
    name = (zero_shot + "_ft" if zero_shot and finetune else zero_shot if zero_shot else MODEL_NAME + "_ftpm" if per_market
            else MODEL_NAME + "_ft_" + covariates if covariates else MODEL_NAME + "_ft" if finetune else MODEL_NAME + "_xl_ent" if entrant
            else MODEL_NAME + "_xl" if joint else MODEL_NAME)
    if lora:
        name += "lora"  # (cn) chronos2_ftlora
    if from_run:
        source = guarded(from_run)
        source_summary = json.loads((source / "run_summary.json").read_text(encoding="utf-8"))
        source_model = source_summary["model"]
        if bool(source_summary.get("log1p", False)) != bool(log1p):
            raise ValueError("--log1p 여부가 체크포인트 출처 run과 다릅니다(같은 척도로만 다시 씀).")
        if mean_point and not lifecycle:
            name = source_model + "_mean"  # (ga) 시장 100개 묶음, 공변량 없음
            joint = True
        else:
            name = from_run_name(source_model, cross_market, lifecycle_set, with_eb, brand, bool(lifecycle))
            name += "_mean" if mean_point else ""
        core.POINT = point  # (dj)·(dn)·(em)·(en)·(er)
    if seed is not None and seed != 42:
        name += f"_s{seed}"  # (cg) seed만 바꾼 미세조정판
    if log1p and finetune:
        name += "_log"  # (gc) chronos2_ft_log
    if suffix:
        if not finetune:
            raise ValueError("--suffix는 --finetune과 함께 씁니다.")
        name += f"_{suffix}"  # (dr) 미세조정 설정 묶음(학습률·단계 수)
    run_dir = new_run_dir(output_base, name)
    lcf = covariates == "lcf"  # (do) 수명 특징을 미래에도 아는 공변량으로 미세조정·예측
    if lcf:
        covariates, lifecycle = None, True
    if lifecycle and lifecycle_set != "g":
        lifecycle = lifecycle_set  # (em) forecast_origin에 공변량 묶음 이름으로 넘김
    pipeline = load()
    plan = [(o, HORIZON) for o in origins] + ([(CHECK_ORIGIN, CHECK_HORIZON)] if check else [])
    frames_p, frames_s, first, checkpoints = [], [], None, {}
    for origin, horizon in plan:
        model = pipeline
        if per_market:
            parts_p, parts_s, histories = [], [], []
            for country in sorted(data["Country"].unique()):
                market = data.loc[data["Country"].eq(country)]
                _, train_histories = ft.training_inputs(market, origin)
                model, checkpoints[f"{origin}/{country}"] = ft.finetune(
                    pipeline, train_histories, run_dir / "models" / origin / country, device=device, num_steps=ft_steps,
                    seed=seed)
                print(f"{name} {origin} {country}: 미세조정 {len(train_histories)}차종", flush=True)
                part_p, part_s, part_h = forecast_origin(model, market, origin, horizon, joint, entrant)
                if first is None and part_h:
                    first = (origin, horizon, part_p, country)
                parts_p.append(part_p)
                parts_s.append(part_s)
                histories += part_h
                del model
            p, s = pd.concat(parts_p, ignore_index=True), pd.concat(parts_s, ignore_index=True)
        else:
            if finetune:
                train_keys, train_histories = ft.training_inputs(data, origin)
                if LOG1P:
                    train_histories = [np.log1p(h) for h in train_histories]  # (gc)
                train_covariates = None
                if lcf:
                    train_covariates, _ = lifecycle_covariates(train_keys, train_histories, origin, horizon)
                if covariates:
                    observed = data.loc[data["Month"].le(pd.Period(origin, freq="M"))]
                    train_covariates, _ = entrant_covariates(observed, train_keys, train_histories, origin, covariates)
                if timesfm:  # (ho)
                    model, checkpoints[origin] = ft.finetune(pipeline, train_histories, run_dir / "models" / origin,
                                                             device=device, num_steps=ft_steps)
                else:
                    model, checkpoints[origin] = ft.finetune(pipeline, train_histories, run_dir / "models" / origin,
                                                             device=device, num_steps=ft_steps, covariates=train_covariates, learning_rate=ft_lr,
                                                             seed=seed, lora=lora, known=LIFECYCLE_NAMES if lcf else None)
                print(f"{name} {origin}: 미세조정 {len(train_histories)}차종", flush=True)
            if from_run:
                checkpoints[origin] = source / "models" / origin / "finetuned-ckpt"
                model = ft.load_finetuned(checkpoints[origin], device=device)
            p, s, histories = forecast_origin(model, data, origin, horizon, joint, entrant, covariates, cross_market,
                                              lifecycle, with_eb, brand)
            if first is None and histories:
                first = (origin, horizon, p, None)
        p, s = stack_outputs([p], [s], name, pd.Period(origin, freq="M"))
        frames_p.append(p)
        frames_s.append(s)
        print(f"{name} {origin}: OK {int(s['prediction_status'].eq('OK').sum())} / {len(s)}", flush=True)
    origin, horizon, expected, country = first
    if per_market:
        reloaded = ft.load_finetuned(checkpoints[f"{origin}/{country}"], device=device)
        again, _, _ = forecast_origin(reloaded, data.loc[data["Country"].eq(country)], origin, horizon, joint, entrant)
    else:
        reloaded = ft.load_finetuned(checkpoints[origin], device=device) if finetune or from_run else load()
        again, _, _ = forecast_origin(reloaded, data, origin, horizon, joint, entrant, covariates, cross_market, lifecycle,
                                      with_eb, brand)
    np.testing.assert_allclose(again["prediction"].to_numpy(float), expected["prediction"].to_numpy(float),
                               rtol=1e-6, atol=1e-5)
    predictions, statuses = pd.concat(frames_p, ignore_index=True), pd.concat(frames_s, ignore_index=True)
    require_unchanged(csv_path, input_hash)
    write_csv(run_dir / "predictions.csv", predictions)
    write_csv(run_dir / "statuses.csv", statuses)
    write_json(run_dir / "run_summary.json", {
        "model": name, "input_sha256": input_hash, "origins": list(origins),
        "check_origin": CHECK_ORIGIN if check else None,
        "core": f"experiments/{zero_shot}_v1/{zero_shot}_core.py" if zero_shot else "experiments/chronos_v1/chronos_core.py",
        "finetune_core": ("experiments/timesfm_v1/timesfm_ft_core.py" if timesfm else "experiments/chronos_ft_v1/chronos_ft_core.py")
        if finetune else None,
        **core.model_fingerprint(), "quantile": 0.5,
        "forecast_config": getattr(core, "FORECAST_CONFIG", None),
        "point": point, "log1p": bool(log1p),
        "cross_learning": BRAND_RULE if brand else CROSS_MARKET_RULE if cross_market else JOINT_RULE if joint or lifecycle else False, "device": device,
        "lifecycle_covariates": LIFECYCLE_SETS[lifecycle_set][1] if lifecycle else None,
        "lifecycle_set": lifecycle_set if lifecycle else None,
        "past_only_eb_covariates": bool(with_eb),
        "lifecycle_source_sha256": {p.name: file_sha256(p) for p in [LIFECYCLE_SETS[lifecycle_set][0]] + [
            LIFECYCLE_FILE.parent / f for f in LIFECYCLE_SETS[lifecycle_set][2]] + (
            [FACELIFT_FILE, LIFECYCLE_FILE.parent / "lifecycle_features_g3.py"] if lifecycle_set == "gf" else []) + (
            [LAUNCH_FILE, ROOT / "experiments" / "china_lifecycle_v2" / "research" / "launch_dates_merged.csv"]
            if lifecycle_set == "gl" else [])}
        if lifecycle else None,
        "finetuned_checkpoint_source": str(source) if from_run else None,
        "finetuned_checkpoint_sha256": {o: file_sha256(path / "model.safetensors") for o, path in checkpoints.items()}
        if from_run else None,
        "finetune": dict(ft.settings_for(ft_lr), num_steps=ft_steps or ft.FINETUNE["num_steps"],
                         seed=ft.FINETUNE["seed"] if seed is None else seed, **(ft.LORA if lora else {})) if finetune else None,
        "checkpoints": {o: str(path) if from_run else str(path.relative_to(run_dir)) for o, path in checkpoints.items()},
        "past_covariates": (["entrant_share_24m", "entrant_share_12m", "brand_change_3m", "brand_change_12m"]
                            if covariates == "eb" else ["entrant_share_24m", "entrant_share_12m"] if entrant else []),
        "finetune_scope": "per Country" if per_market else "all markets" if finetune else None,
        "history_rule": "experiments/ets_v1/ets_core.prepare_history",
        "status_counts": statuses.groupby(["origin", "prediction_status"]).size().rename("n").reset_index().to_dict("records"),
        "reload_check": "PASS"})
    print(run_dir)
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--output-base", default=str(HERE / "reports"))
    parser.add_argument("--origins", nargs="*", default=ORIGINS)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--joint-market", action="store_true", help="시장 안 100개 묶음 cross-learning(chronos2_xl)")
    parser.add_argument("--timesfm", action="store_true", help="Chronos-2 대신 TimesFM 2.5(timesfm_v1 환경)")
    parser.add_argument("--finetune", action="store_true", help="기준월별 미세조정 후 시장 묶음 예측(chronos2_ft)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--ft-steps", type=int, default=None, help="검산용으로만 미세조정 단계 수를 줄인다")
    parser.add_argument("--entrant", action="store_true", help="--joint-market + 신차 비중 과거 공변량(chronos2_xl_ent)")
    parser.add_argument("--per-market", action="store_true", help="--finetune을 시장별로(chronos2_ftpm)")
    parser.add_argument("--covariates", choices=["eb", "lcf"],
                        help="--finetune + 과거 공변량(eb = 신차 비중 + 브랜드 흐름), lcf = (do) 수명 특징 3개를 미래에도 아는 공변량으로")
    parser.add_argument("--seed", type=int, default=None, help="--finetune의 seed만 바꿈((cg) chronos2_ft_s43 등)")
    parser.add_argument("--sundial", action="store_true", help="Chronos-2 대신 Sundial zero-shot(sundial_v1 환경)")
    parser.add_argument("--lora", action="store_true", help="--finetune을 LoRA(라이브러리 기본 설정, 학습률 1e-5)로((cn) chronos2_ftlora)")
    parser.add_argument("--from-run", help="(dj) 미세조정 run 폴더: 새 학습 없이 기준월별 체크포인트를 다시 씀")
    parser.add_argument("--cross-market", action="store_true", help="(dj) 같은 (Brand, Model) 여러 시장을 한 묶음으로(chronos2_ft_xm)")
    parser.add_argument("--ft-lr", type=float, default=None, help="(dr) 미세조정 학습률(없으면 라이브러리 기본 1e-6)")
    parser.add_argument("--suffix", help="(dr) 미세조정 설정 묶음 이름(모델 이름 뒤에 붙음, 예: lr5·k3)")
    parser.add_argument("--lifecycle", action="store_true", help="(dn) 수명 특징 3개를 과거·미래 공변량으로(chronos2_ft_lc)")
    parser.add_argument("--lifecycle-set", default="g", choices=sorted(LIFECYCLE_SETS),
                        help="(em) g3: 중국 사건표 v2 + 가격 인하 뒤 개월(공변량 4개, chronos2_ft_lc3)")
    parser.add_argument("--with-eb", action="store_true",
                        help="(er) --from-run --lifecycle에 eb 4개(신차 비중·브랜드 흐름)를 과거 공변량으로 더함")
    parser.add_argument("--log1p", action="store_true", help="(gc) log1p 척도로 미세조정·예측(이름 _log)")
    parser.add_argument("--point", default="median", choices=["median", "mean"],
                        help="(ga) mean = 학습 분위수 평균을 점예측으로(--from-run 전용, 이름 _mean)")
    parser.add_argument("--patchtst-fm", action="store_true",
                        help="(fw) Chronos-2 대신 PatchTST-FM-r2 zero-shot(patchtst_fm_v1 환경, --device cuda)")
    parser.add_argument("--brand-groups", action="store_true",
                        help="(fs) --from-run 예측 묶음을 시장 안 브랜드별로(chronos2_ft_bg, --lifecycle이면 chronos2_ft_lcbg)")
    args = parser.parse_args(argv)
    run(args.csv, args.output_base, args.origins, not args.no_check, args.joint_market, args.timesfm, args.finetune,
        args.device, args.ft_steps, args.entrant, args.per_market, args.covariates, args.seed, args.sundial, args.lora,
        args.from_run, args.cross_market, args.lifecycle, args.ft_lr, args.suffix, args.lifecycle_set, args.with_eb,
        args.brand_groups, args.patchtst_fm, args.point, args.log1p)


if __name__ == "__main__":
    main()
