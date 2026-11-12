"""차종별 Chronos-2 zero-shot 예측(2026-10-03 사전 고정 (ar)).

이력 규칙은 ETS와 같다(experiments/ets_v1/ets_core.prepare_history: 기준월까지 마지막 연속 구간, 최소 12개월).
사전학습 모델 amazon/chronos-2를 학습 없이 쓴다. 입력은 판매 대수 하나(공변량 없음), cross_learning=False라 차종끼리
정보를 나누지 않는다. horizon개월을 한 번에 0.5 분위수로 예측하고 0 미만은 0으로 자른다. 모델 가중치는
HF_HOME=D:/daizen/.cache/huggingface에 둔다(프로젝트 밖에 쓰지 않음).
"""

import hashlib
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from chronos import Chronos2Pipeline  # noqa: E402

_spec = importlib.util.spec_from_file_location("ets_core_for_chronos", ROOT / "experiments" / "ets_v1" / "ets_core.py")
ets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ets)

MODEL_ID = "amazon/chronos-2"
QUANTILE = 0.5
JOINT_GROUP_SIZE = 100  # (as) cross-learning 묶음 크기: Chronos-2 문서 권장값(사전학습 묶음 크기)
POINT = "median"  # (ga) "mean"이면 학습 분위수 전체로 평균(분위수 함수 적분)을 점예측으로 쓴다

_qm_spec = importlib.util.spec_from_file_location("quantile_mean", ROOT / "experiments" / "horizon_eval_v1" / "quantile_mean.py")
_qm = importlib.util.module_from_spec(_qm_spec)
_qm_spec.loader.exec_module(_qm)


def _point(pipeline, inputs, horizon, batch_size, cross_learning):
    """(ga) POINT에 따라 0.5 분위수 또는 학습 분위수 평균. (차종 수, horizon) 원시값."""
    if POINT == "mean":
        levels = list(pipeline.quantiles)
        quantiles, _ = pipeline.predict_quantiles(inputs, prediction_length=horizon, quantile_levels=levels,
                                                  batch_size=batch_size, cross_learning=cross_learning)
        return np.stack([_qm.mean_from_quantiles(levels, q[0].to(torch.float64).numpy()) for q in quantiles])
    quantiles, _ = pipeline.predict_quantiles(inputs, prediction_length=horizon, quantile_levels=[QUANTILE],
                                              batch_size=batch_size, cross_learning=cross_learning)
    return np.stack([q[0, :, 0].to(torch.float64).numpy() for q in quantiles])


def load_pipeline(device="cpu"):
    torch.manual_seed(0)
    return Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=device)


def prepare(rows, origin, *, key=None):
    return ets.prepare_history(rows, origin, key=key)


def forecast_many(pipeline, histories, horizon, batch_size=256, cross_learning=False):
    """histories(1차원 판매 배열 목록)마다 horizon개월 0.5 분위수 예측. (차종 수, horizon) 배열, 0 하한.
    cross_learning=True면 들어온 순서대로 batch_size개씩 끊은 묶음 안에서 정보를 나눈다."""
    inputs = [torch.tensor(np.asarray(values, dtype=np.float32)) for values in histories]
    raw = _point(pipeline, inputs, horizon, batch_size, cross_learning)
    if raw.shape != (len(histories), horizon) or not np.isfinite(raw).all():
        raise ValueError("Chronos-2 예측이 유한하지 않습니다.")
    return np.maximum(0.0, raw)


def forecast_joint(pipeline, histories, horizon):
    """(as) 한 시장 안 차종(호출 쪽에서 정렬)을 앞에서부터 JOINT_GROUP_SIZE개씩 묶어 cross-learning으로 예측한다."""
    return forecast_many(pipeline, histories, horizon, batch_size=JOINT_GROUP_SIZE, cross_learning=True)


def covariate_batch_size(n_covariates):
    """(bb) 라이브러리는 공변량도 한 계열로 세므로, 묶음당 차종 수가 JOINT_GROUP_SIZE가 되게 배치 크기를 키운다."""
    return JOINT_GROUP_SIZE * (1 + n_covariates)


def forecast_joint_cov(pipeline, histories, covariates, horizon, future=None):
    """(bb) forecast_joint와 같되 차종마다 과거 공변량(이름 → 이력과 같은 길이 배열, NaN 허용)을 함께 넣는다.
    (dn) future: 차종마다 미리 아는 미래 공변량(이름 → 예측 길이 배열, 이름은 과거 공변량의 일부)."""
    names = sorted(covariates[0]) if covariates else []
    inputs = []
    for i, (values, extra) in enumerate(zip(histories, covariates)):
        if sorted(extra) != names or any(len(extra[name]) != len(values) for name in names):
            raise ValueError("공변량 이름이 다르거나 길이가 이력과 다릅니다.")
        item = {"target": torch.tensor(np.asarray(values, dtype=np.float32)),
                "past_covariates": {name: torch.tensor(np.asarray(extra[name], dtype=np.float32)) for name in names}}
        if future is not None:
            ahead = future[i]
            if not set(ahead) <= set(names) or any(len(ahead[name]) != horizon for name in ahead):
                raise ValueError("미래 공변량 이름이 과거 공변량에 없거나 길이가 예측 길이와 다릅니다.")
            item["future_covariates"] = {name: torch.tensor(np.asarray(ahead[name], dtype=np.float32)) for name in sorted(ahead)}
        inputs.append(item)
    raw = _point(pipeline, inputs, horizon, covariate_batch_size(len(names)), True)
    if raw.shape != (len(histories), horizon) or not np.isfinite(raw).all():
        raise ValueError("Chronos-2 예측이 유한하지 않습니다.")
    return np.maximum(0.0, raw)


def model_fingerprint():
    """받아 둔 가중치 파일의 해시와 snapshot 폴더 이름(revision)."""
    folder = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots"
    weights = sorted(folder.glob("*/model.safetensors"))
    if len(weights) != 1:
        raise ValueError(f"가중치 파일이 하나가 아닙니다: {weights}")
    return {"model_id": MODEL_ID, "revision": weights[0].parent.name,
            "model_sha256": hashlib.sha256(weights[0].read_bytes()).hexdigest()}
