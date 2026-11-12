"""차종별 TimesFM 2.5 zero-shot 예측(2026-10-03 사전 고정 (at)).

이력 규칙은 ETS와 같다(experiments/ets_v1/ets_core.prepare_history: 기준월까지 마지막 연속 구간, 최소 12개월).
사전학습 모델 google/timesfm-2.5-200m-pytorch(Apache-2.0)를 학습 없이 쓴다. 3.0 가중치는 비상업 전용 라이선스라 쓰지 않는다.
설정은 README 권장값(FORECAST_CONFIG). horizon개월을 한 번에 예측하고 0.5 분위수 칸을 0 하한으로 쓴다.
모델 가중치는 HF_HOME=D:/daizen/.cache/huggingface에 둔다(프로젝트 밖에 쓰지 않음).
"""

import hashlib
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import timesfm  # noqa: E402
import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("ets_core_for_timesfm", ROOT / "experiments" / "ets_v1" / "ets_core.py")
ets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ets)

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
MEDIAN_COLUMN = 5  # quantile_forecast 마지막 축: [평균, 0.1, 0.2, ..., 0.9] → 0.5는 다섯째 분위(인덱스 5)
FORECAST_CONFIG = {"max_context": 128, "max_horizon": 24, "normalize_inputs": True, "use_continuous_quantile_head": True,
                   "force_flip_invariance": True, "infer_is_positive": True, "fix_quantile_crossing": True,
                   "per_core_batch_size": 64}


def load_model():
    torch.manual_seed(0)
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID, torch_compile=False)
    model.compile(timesfm.ForecastConfig(**FORECAST_CONFIG))
    return model


def prepare(rows, origin, *, key=None):
    return ets.prepare_history(rows, origin, key=key)


def forecast_many(model, histories, horizon):
    """histories(1차원 판매 배열 목록)마다 horizon개월 0.5 분위수 예측. (차종 수, horizon) 배열, 0 하한."""
    if horizon > FORECAST_CONFIG["max_horizon"]:
        raise ValueError("horizon이 max_horizon보다 깁니다.")
    _, quantiles = model.forecast(horizon=horizon, inputs=[np.asarray(values, dtype=float) for values in histories])
    raw = np.asarray(quantiles, dtype=float)[:, :horizon, MEDIAN_COLUMN]
    if raw.shape != (len(histories), horizon) or not np.isfinite(raw).all():
        raise ValueError("TimesFM 예측이 유한하지 않습니다.")
    return np.maximum(0.0, raw)


def model_fingerprint():
    """받아 둔 가중치 파일의 해시와 snapshot 폴더 이름(revision)."""
    folder = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots"
    weights = sorted(folder.glob("*/model.safetensors"))
    if len(weights) != 1:
        raise ValueError(f"가중치 파일이 하나가 아닙니다: {weights}")
    return {"model_id": MODEL_ID, "revision": weights[0].parent.name,
            "model_sha256": hashlib.sha256(weights[0].read_bytes()).hexdigest()}
