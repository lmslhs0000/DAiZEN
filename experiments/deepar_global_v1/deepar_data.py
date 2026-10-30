"""TFT의 12개월 입력·1개월 정답 창과 학습 시 고정하는 전처리를 만든다.

deepar_global(2026-10-02 사전 고정 (k)): tft_global_pt_v1/tft_data_global_pt.py 사본. 바꾼 곳은 정규화 하나 —
GroupNormalizer(center=False): 차종별 평균으로 나누기만 한다(DeepAR 음이항 손실의 요구). 아래는 원래 설명.
tft_global_pt 변형(2026-10-01 사전 고정): tft_global_v1/tft_data_global.py 사본에 차종별 동력원 코드 1개를
static_reals로 더했다(학습 = train_end 해, 예측 = 기준월 해 기준; UNKNOWN = −1). 아래는 원래 설명.
tft_global 변형: 루트 tft_data.py 사본. 7개 시장을 한 데이터셋으로 묶고 시계열 키를
[Country, Brand, Model]로 쓴다(같은 Brand·Model이라도 시장이 다르면 다른 시계열·다른 정규화). 시장 필터와
추론의 '한 시장만' 검사를 없앴다. 나머지(12+1 창, 공백 처리, 정규화 적합 범위, placeholder)는 루트와 같다.
"""

from copy import deepcopy
import importlib.util
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
import torch
from torch.utils.data import ConcatDataset, DataLoader

from baseline import SERIES_KEY


_spec = importlib.util.spec_from_file_location(
    "powertrain_features", Path(__file__).resolve().parents[1] / "powertrain_v1" / "powertrain_features.py")
ptf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ptf)
GROUP_IDS = ["Country", "Brand", "Model"]


def powertrain_static(brands, models, year):
    """그 해 기준 동력원 코드. 범주를 모르면(UNKNOWN) −1."""
    code, _ = ptf.powertrain_codes(brands, models, [int(year)] * len(brands))
    return np.where(np.isnan(code), -1.0, code)


def _with_powertrain(frame, year):
    frame = frame.copy()
    frame["powertrain_code"] = powertrain_static(frame["Brand"], frame["Model"], year)
    return frame
HISTORY_LENGTH = 12
FIRST_MONTH = pd.Period("2016-01", freq="M")
LAST_MONTH = pd.Period("2026-07", freq="M")


def _month(value):
    """문자열 또는 월 Period를 검사한다."""
    if isinstance(value, pd.Period) and value.freqstr == "M":
        return value
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{1,2}", value) is None:
        raise ValueError("월은 YYYY-MM 형식이어야 합니다.")
    return pd.to_datetime(value, format="%Y-%m").to_period("M")


def _input_frame(data):
    """원본의 키와 수치를 바꾸지 않고 필요한 열만 복사·검사한다."""
    columns = SERIES_KEY + ["Month", "Sales"]
    if not set(columns).issubset(data.columns):
        raise ValueError("TFT 입력에는 Country, Brand, Model, Month, Sales가 필요합니다.")
    frame = data[columns].copy()
    for column in SERIES_KEY:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"식별값이 비어 있습니다: {column}")
        if not frame[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"식별값은 원래 문자열이어야 합니다: {column}")
        frame[column] = frame[column].astype(object)
    if not isinstance(frame["Month"].dtype, pd.PeriodDtype) or frame["Month"].dtype.freq.name != "ME":
        frame["Month"] = pd.PeriodIndex([_month(value) for value in frame["Month"]], freq="M")
    if frame["Month"].isna().any():
        raise ValueError("Month가 비어 있습니다.")
    if frame.duplicated(SERIES_KEY + ["Month"]).any():
        raise ValueError("TFT 입력에 차종·월 중복이 있습니다.")
    try:
        frame["Sales"] = pd.to_numeric(frame["Sales"], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Sales는 수치 또는 결측값이어야 합니다.") from error
    present = frame["Sales"].dropna()
    if not np.isfinite(present).all() or present.lt(0).any():
        raise ValueError("Sales에 음수 또는 무한대가 있습니다.")
    frame = frame.sort_values(SERIES_KEY + ["Month"]).reset_index(drop=True)
    frame["time_idx"] = frame["Month"].astype("int64") - FIRST_MONTH.ordinal
    frame["month"] = frame["Month"].dt.month.map(lambda value: f"{value:02}").astype(object)
    return frame


def _valid_segments(observed):
    """공란이나 달력 공백을 건너뛰지 않고 13개월 이상 연속 구간을 남긴다."""
    segments = {}
    windows = []
    for key, vehicle in observed.groupby(SERIES_KEY, sort=True):
        valid = vehicle.loc[vehicle["Sales"].notna()].copy()
        segment_numbers = valid["time_idx"].diff().ne(1).cumsum()
        eligible = []
        for _, segment in valid.groupby(segment_numbers, sort=True):
            if len(segment) >= HISTORY_LENGTH + 1:
                eligible.append(segment.reset_index(drop=True))
                windows.append(segment.iloc[HISTORY_LENGTH:][SERIES_KEY + ["Month"]])
        if eligible:
            segments[key] = eligible
    if not segments:
        raise ValueError("NO_TRAINING_ROWS")
    window_index = pd.concat(windows, ignore_index=True).sort_values(SERIES_KEY + ["Month"]).reset_index(drop=True)
    return segments, window_index


def _fit_preprocessing(observations):
    """적격 차종의 원래 월 관측을 한 번씩 사용하여 범주·정규화를 적합한다."""
    encoders = {}
    encoded = observations.copy()
    for column in GROUP_IDS:
        encoder = NaNLabelEncoder(add_nan=False).fit(observations[column].to_numpy())
        encoders[column] = encoder
        # PF 내부 그룹 인코더도 같은 매핑을 미리 고정한다.
        encoders[f"__group_id__{column}"] = deepcopy(encoder)
        encoded[column] = encoder.transform(observations[column].to_numpy())
    calendar_months = np.array([f"{month:02}" for month in range(1, 13)], dtype=object)
    encoders["month"] = NaNLabelEncoder(add_nan=False).fit(calendar_months)
    normalizer = GroupNormalizer(
        method="standard", groups=GROUP_IDS.copy(), center=False,
        scale_by_group=False, transformation=None,
    )
    # TimeSeriesDataSet은 범주 인코딩 이후 GroupNormalizer를 호출한다.
    normalizer.fit(encoded["Sales"], encoded)
    return encoders, normalizer


def prepare_training(data, train_end):
    """7개 시장 전체의 학습 데이터셋, 고정 전처리, 대상월 및 기록을 반환한다."""
    train_end = _month(train_end)
    # 학습 전처리도 train_end 상한을 따른다. 미래 행은 여기에 들어오지 않는다.
    selected = data.copy()
    selected["Month"] = pd.PeriodIndex([_month(value) for value in selected["Month"]], freq="M")
    selected = selected.loc[selected["Month"].between(FIRST_MONTH, min(train_end, LAST_MONTH))]
    observed = _with_powertrain(_input_frame(selected), train_end.year)
    segments, window_index = _valid_segments(observed)
    seen_series = [list(key) for key in segments]
    seen_index = pd.MultiIndex.from_tuples(list(segments), names=SERIES_KEY)
    eligible = pd.MultiIndex.from_frame(observed[SERIES_KEY]).isin(seen_index)
    observations = observed.loc[eligible & observed["Sales"].notna()].reset_index(drop=True)
    encoders, normalizer = _fit_preprocessing(observations)

    # 각 차종의 첫 연속 구간을 모아 모든 적격 범주를 가진 기준 dataset을 만든다.
    first_segments = pd.concat([parts[0] for parts in segments.values()], ignore_index=True)
    prototype = TimeSeriesDataSet(
        first_segments, time_idx="time_idx", target="Sales", group_ids=GROUP_IDS.copy(),
        min_encoder_length=12, max_encoder_length=12,
        min_prediction_length=1, max_prediction_length=1,
        static_categoricals=GROUP_IDS.copy(), static_reals=["powertrain_code"],
        time_varying_known_categoricals=["month"], time_varying_known_reals=[],
        time_varying_unknown_categoricals=[], time_varying_unknown_reals=["Sales"],
        lags={}, add_relative_time_idx=False, add_target_scales=False,
        add_encoder_length=False, randomize_length=False, allow_missing_timesteps=False,
        target_normalizer=normalizer, categorical_encoders=encoders, predict_mode=False,
    )
    parameters = prototype.get_parameters()
    datasets = [prototype]
    for position in range(1, max(len(parts) for parts in segments.values())):
        next_segments = pd.concat([parts[position] for parts in segments.values() if len(parts) > position], ignore_index=True)
        # 공백 뒤 구간도 같은 원래 그룹과 전처리를 유지한다. 공백 채우기는 하지 않는다.
        datasets.append(TimeSeriesDataSet.from_parameters(
            parameters, next_segments, predict=False, stop_randomization=True,
        ))

    normalization = []
    for key, group in observations.groupby(SERIES_KEY, sort=True):
        encoded_key = tuple(int(encoders[column].classes_[value]) for column, value in zip(GROUP_IDS, key))
        record = normalizer.norm_.loc[encoded_key]
        normalization.append({
            **dict(zip(SERIES_KEY, key)), "observations": len(group),
            "first_month": str(group["Month"].min()), "last_month": str(group["Month"].max()),
            "center": float(record["center"]), "scale": float(record["scale"]),
        })
    schema = {
        "country": "ALL", "countries": sorted(observations["Country"].unique().tolist()),
        "train_end": str(train_end), "seen_series": seen_series,
        "group_ids": GROUP_IDS.copy(), "history_length": 12, "prediction_length": 1,
        "target": "Sales", "static_categoricals": GROUP_IDS.copy(),
        "time_varying_known_categoricals": ["month"], "time_varying_known_reals": [],
        "time_varying_unknown_reals": ["Sales"], "static_reals": ["powertrain_code"],
        "powertrain_table_sha256": __import__("hashlib").sha256(ptf.TABLE.read_bytes()).hexdigest(),
        "categorical_levels": {column: list(encoders[column].classes_) for column in [*GROUP_IDS, "month"]},
        "time_idx_origin": str(FIRST_MONTH), "training_rows": len(window_index),
        "training_target_start": str(window_index["Month"].min()),
        "training_target_end": str(window_index["Month"].max()),
        "normalizer_observation_count": len(observations), "normalization": normalization,
        "normalizer_fit_scope": "unique_valid_original_observations_of_eligible_series_through_train_end",
        "normalizer": {"method": "standard", "groups": GROUP_IDS.copy(), "center": False,
                       "scale_by_group": False, "transformation": None},
        "allow_missing_timesteps": False, "add_relative_time_idx": False,
        "add_target_scales": False, "add_encoder_length": False, "randomize_length": False,
        "training_shuffle": True, "drop_last": False, "num_workers": 0,
        "continuous_dtype": "float32", "categorical_dtype": "int64",
    }
    return {"dataset": prototype, "datasets": datasets, "parameters": parameters,
            "schema": schema, "window_index": window_index}


def make_training_loader(prepared, batch_size=64, seed=42):
    """완성된 12+1 창만 고정 시드로 섞고 마지막 작은 배치도 학습한다."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size는 양의 정수여야 합니다.")
    generator = torch.Generator().manual_seed(seed)
    prototype = prepared["dataset"]
    if len(prepared["datasets"]) == 1:
        return prototype.to_dataloader(
            train=True, batch_size=batch_size, shuffle=True, drop_last=False,
            num_workers=0, generator=generator,
        )
    collate_fn = prototype.to_dataloader(train=False, batch_size=batch_size, num_workers=0).collate_fn
    return DataLoader(
        ConcatDataset(prepared["datasets"]), batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=0, generator=generator, collate_fn=collate_fn,
    )


def make_inference_dataset(parameters, history_frame, origin, placeholder=0.0):
    """차종별 최근 12개월에 추론 전용 정답 칸을 더하고 고정 전처리를 적용한다."""
    origin = _month(origin)
    if not np.isfinite(placeholder):
        raise ValueError("추론 placeholder는 유한값이어야 합니다.")
    history = _with_powertrain(_input_frame(history_frame), origin.year)
    if history.empty:
        raise ValueError("추론에는 유효한 이력이 필요합니다.")
    expected_months = pd.period_range(origin - 11, origin, freq="M")
    future_rows = []
    encoders = parameters["categorical_encoders"]
    learned_pairs = parameters["target_normalizer"].norm_.index
    for key, vehicle in history.groupby(SERIES_KEY, sort=True):
        if len(vehicle) != 12 or not pd.PeriodIndex(vehicle["Month"]).equals(expected_months):
            raise ValueError(f"INCOMPLETE_HISTORY: 최근 연속 12개월이 아닙니다: {key}")
        if vehicle["Sales"].isna().any():
            raise ValueError(f"INCOMPLETE_HISTORY: 이력 판매량이 비어 있습니다: {key}")
        try:
            encoded_key = tuple(int(encoders[column].classes_[value]) for column, value in zip(GROUP_IDS, key))
        except KeyError as error:
            raise ValueError(f"UNSEEN_SERIES: {key}") from error
        if encoded_key not in learned_pairs:
            raise ValueError(f"UNSEEN_SERIES: {key}")
        future_rows.append({**dict(zip(SERIES_KEY, key)), "Month": origin + 1, "Sales": float(placeholder),
                            "time_idx": (origin + 1).ordinal - FIRST_MONTH.ordinal,
                            "month": f"{(origin + 1).month:02}"})
    inference = pd.concat([history, _with_powertrain(pd.DataFrame(future_rows), origin.year)], ignore_index=True)
    return TimeSeriesDataSet.from_parameters(parameters, inference, predict=True, stop_randomization=True)
