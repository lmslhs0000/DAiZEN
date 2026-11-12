"""Chronos-2 기준월별 미세조정(2026-10-03 사전 고정 (au)).

기준월까지의 7개 시장 OK 차종 이력(experiments/chronos_v1/chronos_core.prepare = ETS 이력 규칙)으로 amazon/chronos-2를
라이브러리 기본 설정(FINETUNE)대로 full fine-tune 한 복사본을 만든다. 원래 모델은 바꾸지 않는다. 예측은 chronos_core의
forecast_joint(시장 안 100개 묶음 cross-learning)를 그대로 쓴다. 미세조정 결과는 output_dir/finetuned-ckpt에 저장된다.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("chronos_core_for_ft", ROOT / "experiments" / "chronos_v1" / "chronos_core.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

from chronos import Chronos2Pipeline  # noqa: E402

KEY = ["Country", "Brand", "Model"]
FINETUNE = {"learning_rate": 1e-6, "num_steps": 1000, "batch_size": 256, "prediction_length": 24, "seed": 42,
            "finetune_mode": "full"}


def training_inputs(data, origin):
    """기준월까지 자료에서 ETS 이력 규칙을 통과한 차종의 (키 목록, 이력 배열 목록). 기준월 이후 행은 보지 않는다."""
    history = data.loc[data["Month"].le(pd.Period(origin, freq="M"))]
    keys, histories = [], []
    for key, rows in history.groupby(KEY, sort=True):
        prepared = cc.prepare(rows[KEY + ["Month", "Sales"]].reset_index(drop=True), origin, key=list(key))
        if prepared["status"] == "OK":
            keys.append(key)
            histories.append(np.asarray(prepared["values"], dtype=float))
    return keys, histories


def _inputs(histories, covariates, known=None):
    """미세조정 입력: 공변량이 없으면 이력 텐서 목록, 있으면 {target, past_covariates} 사전 목록((bj)).
    (do) known: 미래에도 아는 공변량 이름 — 라이브러리 preprocess.from_list_of_dicts(known_covariates_names)로 준비."""
    if covariates is None:
        return [torch.tensor(values, dtype=torch.float32) for values in histories]
    names = sorted(covariates[0]) if covariates else []
    inputs = []
    for values, extra in zip(histories, covariates):
        if sorted(extra) != names or any(len(extra[name]) != len(values) for name in names):
            raise ValueError("공변량 이름이 다르거나 길이가 이력과 다릅니다.")
        inputs.append({"target": torch.tensor(values, dtype=torch.float32),
                       "past_covariates": {name: torch.tensor(np.asarray(extra[name], dtype=np.float32)) for name in names}})
    if known:
        if not set(known) <= set(names):
            raise ValueError("미래에도 아는 공변량 이름이 과거 공변량에 없습니다.")
        from chronos.chronos2.preprocess import from_list_of_dicts
        plain = [{"target": np.asarray(item["target"], dtype=np.float32),
                  "past_covariates": {name: np.asarray(value, dtype=np.float32) for name, value in item["past_covariates"].items()}}
                 for item in inputs]
        return from_list_of_dicts(plain, prediction_length=FINETUNE["prediction_length"], known_covariates_names=list(known))
    return inputs


# (cn) LoRA 미세조정: 라이브러리 기본 LoraConfig, 학습률은 라이브러리 설명이 LoRA에 권하는 1e-5
LORA = {"finetune_mode": "lora", "learning_rate": 1e-5}


def settings_for(learning_rate=None):
    """(dr) 학습률만 바꾼 미세조정 설정(없으면 라이브러리 기본 FINETUNE 그대로)."""
    return dict(FINETUNE, **({} if learning_rate is None else {"learning_rate": learning_rate}))


def finetune(pipeline, histories, output_dir, *, device="cuda", num_steps=None, covariates=None, seed=None, lora=False,
             known=None, learning_rate=None):
    """pipeline의 복사본을 histories(와 과거 공변량)로 미세조정한다. (새 pipeline, 저장된 체크포인트 폴더)를 반환한다.
    seed((cg))를 주면 그 값만 바꾸고 나머지 설정은 FINETUNE 그대로다. lora=True((cn))면 LORA 두 값만 바꾼다."""
    settings = dict(settings_for(learning_rate), num_steps=num_steps or FINETUNE["num_steps"],
                    seed=FINETUNE["seed"] if seed is None else seed, **(LORA if lora else {}))
    output_dir = Path(output_dir)
    inputs = _inputs(histories, covariates, known)
    tuned = pipeline.fit(
        inputs,
        prediction_length=settings["prediction_length"], finetune_mode=settings["finetune_mode"],
        learning_rate=settings["learning_rate"], num_steps=settings["num_steps"], batch_size=settings["batch_size"],
        output_dir=str(output_dir), seed=settings["seed"], use_cpu=device == "cpu", report_to=[], save_strategy="no")
    del tuned  # 학습에 쓴 메모리 모델은 버린다: train 모드로 남고, GPU에서는 bf16 autocast로 감싼 forward가 남는다
    path = output_dir / "finetuned-ckpt"
    if not path.is_dir():
        raise FileNotFoundError(f"미세조정 체크포인트가 없습니다: {path}")
    # 저장된 체크포인트(순수 float32)를 새로 불러 예측에 쓴다 — 납품·재현에 쓰는 것과 같은 모델
    return load_finetuned(path, device=device), path


def load_finetuned(path, *, device="cuda"):
    return Chronos2Pipeline.from_pretrained(str(path), device_map=device)
