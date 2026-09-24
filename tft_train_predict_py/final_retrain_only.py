import json
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE

from config import (
    DATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    MODEL_METADATA_PATH,
    GROUP_COLS,
    REQUIRED_COLS,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    LAGS,
    LEARNING_RATE,
    HIDDEN_SIZE,
    ATTENTION_HEAD_SIZE,
    DROPOUT,
    HIDDEN_CONTINUOUS_SIZE,
    GRADIENT_CLIP_VAL,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
)

# 이번 calibration 결과에서 이미 확인된 값
BEST_EPOCHS = 16
CALIBRATION_BEST_VAL_MAE = 218.0018


def load_and_prepare_data(path):
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = df["sales"].astype(float)

    base_period = df["date"].min().to_period("M")
    df["time_idx"] = (
        (df["date"].dt.year - base_period.year) * 12
        + df["date"].dt.month
        - base_period.month
    ).astype(int)
    df["month"] = df["date"].dt.month.astype(str)

    dup = df.duplicated(subset=GROUP_COLS + ["time_idx"]).sum()
    if dup > 0:
        raise ValueError(f"동일 차종/월 중복 행이 {dup}개 있습니다.")

    df = df.sort_values(GROUP_COLS + ["time_idx"]).reset_index(drop=True)
    return df, base_period


def make_dataset(data):
    return TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="sales",
        group_ids=GROUP_COLS,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=GROUP_COLS,
        time_varying_known_categoricals=["month"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["sales"],
        lags={"sales": LAGS},
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True,
    )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(SEED, workers=True)

    print("=" * 80)
    print("전체 데이터로 최종 TFT 재학습만 실행")
    print("=" * 80)

    df, base_period = load_and_prepare_data(DATA_PATH)
    max_time_idx = int(df["time_idx"].max())

    training_full = make_dataset(df)
    full_loader = training_full.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # validation이 없는 최종 재학습에서는 ReduceLROnPlateau를 쓰지 않는다.
    # 동일 LR 값을 epoch 수만큼 반복한 리스트를 전달하면
    # PyTorch Forecasting이 validation 비의존 LambdaLR scheduler를 구성한다.
    final_model = TemporalFusionTransformer.from_dataset(
        training_full,
        learning_rate=[LEARNING_RATE] * BEST_EPOCHS,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        output_size=1,
        loss=MAE(),
        reduce_on_plateau_patience=None,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    final_trainer = pl.Trainer(
        max_epochs=BEST_EPOCHS,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    final_trainer.fit(
        final_model,
        train_dataloaders=full_loader,
    )

    if MODEL_PATH.exists():
        MODEL_PATH.unlink()

    final_trainer.save_checkpoint(str(MODEL_PATH))

    trained_groups = (
        df[GROUP_COLS]
        .drop_duplicates()
        .sort_values(GROUP_COLS)
        .to_dict("records")
    )

    metadata = {
        "data_path_at_training": str(DATA_PATH),
        "base_period": str(base_period),
        "trained_until": str(df["date"].max().to_period("M")),
        "max_time_idx_at_training": max_time_idx,
        "best_epochs": BEST_EPOCHS,
        "calibration_best_val_mae": CALIBRATION_BEST_VAL_MAE,
        "max_encoder_length": MAX_ENCODER_LENGTH,
        "max_prediction_length": MAX_PREDICTION_LENGTH,
        "lags": LAGS,
        "trained_groups": trained_groups,
        "reference_backtest_aggregate_wape": 23.28,
        "reference_active_vehicle_aggregate_wape": 23.24,
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 80)
    print("저장 완료")
    print("=" * 80)
    print("모델:", MODEL_PATH)
    print("메타데이터:", MODEL_METADATA_PATH)
    print("다음 실행: uv run python predict.py")


if __name__ == "__main__":
    main()
