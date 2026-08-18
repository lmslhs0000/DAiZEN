import json
from pathlib import Path

import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE

from config import (
    DATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    MODEL_METADATA_PATH,
    CALIBRATION_DIR,
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
    MAX_EPOCHS,
    EARLY_STOP_PATIENCE,
)


def load_and_prepare_data(path: Path):
    df = pd.read_csv(path)

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

    df = df[REQUIRED_COLS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="raise").astype(float)

    if df["sales"].isna().any():
        raise ValueError("sales에 NaN이 있습니다.")

    base_period = df["date"].min().to_period("M")

    df["time_idx"] = (
        (df["date"].dt.year - base_period.year) * 12
        + df["date"].dt.month
        - base_period.month
    ).astype(int)

    df["month"] = df["date"].dt.month.astype(str)

    duplicate_count = df.duplicated(
        subset=GROUP_COLS + ["time_idx"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"동일 market/brand/model/month 중복이 {duplicate_count}개 있습니다."
        )

    df = (
        df.sort_values(GROUP_COLS + ["time_idx"])
        .reset_index(drop=True)
    )

    return df, base_period


def make_dataset(data: pd.DataFrame):
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


def make_tft(dataset: TimeSeriesDataSet, use_plateau_scheduler: bool = True):
    return TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        output_size=1,
        loss=MAE(),
        # 전체 데이터 재학습 단계에는 validation이 없으므로 scheduler를 끈다.
        reduce_on_plateau_patience=4 if use_plateau_scheduler else None,
    )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(SEED, workers=True)

    print("=" * 80)
    print("1. 데이터 로드")
    print("=" * 80)

    df, base_period = load_and_prepare_data(DATA_PATH)

    print("행 수:", len(df))
    print("차종 그룹 수:", df[GROUP_COLS].drop_duplicates().shape[0])
    print("기간:", df["date"].min().date(), "~", df["date"].max().date())
    print("CUDA 사용 가능:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    max_time_idx = int(df["time_idx"].max())
    calibration_cutoff = max_time_idx - MAX_PREDICTION_LENGTH

    if calibration_cutoff <= int(df["time_idx"].min()):
        raise ValueError("Calibration을 만들 만큼 데이터가 충분하지 않습니다.")

    train_cal = df[df["time_idx"] <= calibration_cutoff].copy()
    training_cal = make_dataset(train_cal)

    train_groups = train_cal[GROUP_COLS].drop_duplicates()
    validation_source = df.merge(
        train_groups,
        on=GROUP_COLS,
        how="inner",
    )

    validation_cal = TimeSeriesDataSet.from_dataset(
        training_cal,
        validation_source,
        min_prediction_idx=calibration_cutoff + 1,
        predict=True,
        stop_randomization=True,
    )

    train_loader = training_cal.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    val_loader = validation_cal.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print()
    print("=" * 80)
    print("2. 최근 6개월 Validation으로 적정 Epoch 찾기")
    print("=" * 80)
    print("Calibration train 마지막:", train_cal["date"].max().date())
    print("Calibration validation:",
          df[df["time_idx"] > calibration_cutoff]["date"].min().date(),
          "~",
          df["date"].max().date())
    print("Train samples:", len(training_cal))
    print("Validation models:", len(validation_cal))

    calibration_ckpt = CALIBRATION_DIR / "best.ckpt"
    if calibration_ckpt.exists():
        calibration_ckpt.unlink()

    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=EARLY_STOP_PATIENCE,
        mode="min",
        verbose=False,
    )

    checkpoint = ModelCheckpoint(
        dirpath=str(CALIBRATION_DIR),
        filename="best",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        enable_version_counter=False,
    )

    calibration_model = make_tft(
        training_cal,
        use_plateau_scheduler=True,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[early_stop, checkpoint],
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        calibration_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    if not checkpoint.best_model_path:
        raise RuntimeError("Best checkpoint가 생성되지 않았습니다.")

    best_info = torch.load(
        checkpoint.best_model_path,
        map_location="cpu",
        weights_only=False,
    )

    best_epochs = int(best_info["epoch"]) + 1
    best_val_mae = float(checkpoint.best_model_score.detach().cpu())

    print()
    print("Best calibration checkpoint:", checkpoint.best_model_path)
    print("Best validation MAE:", round(best_val_mae, 4))
    print("최종 전체 재학습 Epoch:", best_epochs)

    print()
    print("=" * 80)
    print("3. 전체 데이터로 최종 TFT 재학습")
    print("=" * 80)

    pl.seed_everything(SEED, workers=True)

    training_full = make_dataset(df)
    full_loader = training_full.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    final_model = make_tft(
        training_full,
        use_plateau_scheduler=False,
    )

    final_trainer = pl.Trainer(
        max_epochs=best_epochs,
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

    # PyTorch Forecasting의 TFT checkpoint에는 모델 가중치/하이퍼파라미터뿐 아니라
    # dataset_parameters도 함께 저장된다. predict.py는 이 ckpt만으로
    # 동일한 encoder/scaler 설정을 재구성한다.
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
        "best_epochs": best_epochs,
        "calibration_best_val_mae": best_val_mae,
        "max_encoder_length": MAX_ENCODER_LENGTH,
        "max_prediction_length": MAX_PREDICTION_LENGTH,
        "lags": LAGS,
        "trained_groups": trained_groups,
        "reference_backtest_aggregate_wape": 23.28,
        "reference_active_vehicle_aggregate_wape": 23.24,
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
