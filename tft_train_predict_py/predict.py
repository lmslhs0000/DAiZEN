import json

import pandas as pd
import torch

from pytorch_forecasting import TemporalFusionTransformer

from config import (
    DATA_PATH,
    MODEL_PATH,
    MODEL_METADATA_PATH,
    OUTPUT_DIR,
    FORECAST_6M_PATH,
    FORECAST_136_PATH,
    GROUP_COLS,
    REQUIRED_COLS,
    MAX_PREDICTION_LENGTH,
    ACTIVE_WINDOW,
    BATCH_SIZE,
    NUM_WORKERS,
)


def load_and_prepare_data(path, base_period):

    df = pd.read_csv(path)

    missing_cols = [
        col
        for col in REQUIRED_COLS
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"필수 컬럼이 없습니다: {missing_cols}"
        )

    df = df[REQUIRED_COLS].copy()

    df["date"] = pd.to_datetime(df["date"])

    df["sales"] = pd.to_numeric(
        df["sales"],
        errors="raise",
    ).astype(float)

    if df["sales"].isna().any():
        raise ValueError(
            "sales에 NaN이 있습니다."
        )

    df["time_idx"] = (
        (df["date"].dt.year - base_period.year) * 12
        + df["date"].dt.month
        - base_period.month
    ).astype(int)

    df["month"] = (
        df["date"]
        .dt.month
        .astype(str)
    )

    duplicate_count = (
        df.duplicated(
            subset=GROUP_COLS + ["time_idx"]
        )
        .sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "동일 market/brand/model/month "
            f"중복이 {duplicate_count}개 있습니다."
        )

    df = (
        df
        .sort_values(
            GROUP_COLS + ["time_idx"]
        )
        .reset_index(drop=True)
    )

    return df


def get_active_groups(
    df,
    trained_groups,
):

    last_time_idx = int(
        df["time_idx"].max()
    )

    recent_start = (
        last_time_idx
        - ACTIVE_WINDOW
        + 1
    )

    recent = df[
        (df["time_idx"] >= recent_start)
        &
        (df["time_idx"] <= last_time_idx)
    ].copy()

    active = (
        recent
        .groupby(
            GROUP_COLS,
            as_index=False,
        )["sales"]
        .sum()
        .rename(
            columns={
                "sales": "recent_sales_sum"
            }
        )
    )

    active = active[
        active["recent_sales_sum"] > 0
    ].copy()

    active = active.merge(
        trained_groups,
        on=GROUP_COLS,
        how="inner",
    )

    active_groups = (
        active[GROUP_COLS]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return (
        active_groups,
        last_time_idx,
    )


def make_future_rows(
    active_groups,
    last_time_idx,
    base_period,
):

    rows = []

    for row in active_groups.itertuples(
        index=False
    ):

        group_values = dict(
            zip(
                GROUP_COLS,
                row,
            )
        )

        for h in range(
            1,
            MAX_PREDICTION_LENGTH + 1,
        ):

            time_idx = (
                last_time_idx
                + h
            )

            period = (
                base_period
                + time_idx
            )

            rows.append(
                {
                    "date":
                        period.to_timestamp(),

                    **group_values,

                    "sales":
                        0.0,

                    "time_idx":
                        time_idx,

                    "month":
                        str(period.month),
                }
            )

    return pd.DataFrame(rows)


def make_forecast_dict(
    forecast_136,
):

    forecast_dict = {}

    for _, row in (
        forecast_136.iterrows()
    ):

        key = (
            row["market"],
            row["brand"],
            row["model"],
        )

        if key not in forecast_dict:
            forecast_dict[key] = {}

        horizon = int(
            row["horizon"]
        )

        predicted_sales = int(
            row["predicted_sales"]
        )

        forecast_dict[
            key
        ][
            horizon
        ] = predicted_sales

    return forecast_dict


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델이 없습니다: {MODEL_PATH}\n"
            "먼저 train.py를 실행하세요."
        )

    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"메타데이터가 없습니다: "
            f"{MODEL_METADATA_PATH}\n"
            "먼저 train.py를 실행하세요."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        MODEL_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    base_period = pd.Period(
        metadata["base_period"],
        freq="M",
    )

    trained_groups = pd.DataFrame(
        metadata["trained_groups"]
    )

    print("모델 불러오는 중...")

    model = (
        TemporalFusionTransformer
        .load_from_checkpoint(
            str(MODEL_PATH),
            weights_only=False,
            map_location="cpu",
        )
    )

    model.eval()
    model.freeze()

    print(
        "모델 로드 완료:",
        MODEL_PATH,
    )

    print(
        "CUDA 사용 가능:",
        torch.cuda.is_available(),
    )

    df = load_and_prepare_data(
        DATA_PATH,
        base_period,
    )

    (
        active_groups,
        last_time_idx,
    ) = get_active_groups(
        df,
        trained_groups,
    )

    last_period = (
        base_period
        + last_time_idx
    )

    print(
        "입력 데이터 마지막 월:",
        last_period,
    )

    print(
        f"최근 {ACTIVE_WINDOW}개월 "
        "판매 기록이 있는 차량:",
        len(active_groups),
    )

    if active_groups.empty:
        raise RuntimeError(
            "예측 가능한 차량이 없습니다."
        )

    active_history = df.merge(
        active_groups,
        on=GROUP_COLS,
        how="inner",
    )

    future_df = make_future_rows(
        active_groups,
        last_time_idx,
        base_period,
    )

    prediction_data = pd.concat(
        [
            active_history,
            future_df,
        ],
        ignore_index=True,
    )

    prediction_data = (
        prediction_data
        .sort_values(
            GROUP_COLS + ["time_idx"]
        )
        .reset_index(drop=True)
    )

    accelerator = (
        "gpu"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("미래 판매량 예측 중...")

    prediction_result = model.predict(
        prediction_data,

        mode="prediction",

        return_index=True,

        batch_size=BATCH_SIZE,

        num_workers=NUM_WORKERS,

        trainer_kwargs={
            "accelerator":
                accelerator,

            "devices":
                1,

            "logger":
                False,

            "enable_progress_bar":
                False,
        },
    )

    pred = (
        prediction_result
        .output
        .detach()
        .cpu()
        .float()
        .numpy()
    )

    if pred.ndim == 3:
        pred = pred.squeeze(-1)

    index_df = (
        prediction_result
        .index
        .reset_index(drop=True)
    )

    if len(index_df) != pred.shape[0]:
        raise RuntimeError(
            "Prediction index와 "
            "prediction 행 수가 다릅니다."
        )

    print(
        "예측 성공 차량:",
        pred.shape[0],
    )

    print(
        "조건 미충족으로 제외된 후보:",
        len(active_groups)
        - pred.shape[0],
    )

    forecast_rows = []

    for i in range(
        pred.shape[0]
    ):

        first_prediction_idx = int(
            index_df.iloc[i]["time_idx"]
        )

        market = (
            index_df.iloc[i]["market"]
        )

        brand = (
            index_df.iloc[i]["brand"]
        )

        model_name = (
            index_df.iloc[i]["model"]
        )

        for h in range(
            MAX_PREDICTION_LENGTH
        ):

            predicted_sales = max(
                0.0,
                float(pred[i, h]),
            )

            predicted_sales = int(
                round(predicted_sales)
            )

            prediction_time_idx = (
                first_prediction_idx
                + h
            )

            prediction_period = (
                base_period
                + prediction_time_idx
            )

            forecast_rows.append(
                {
                    "market":
                        market,

                    "brand":
                        brand,

                    "model":
                        model_name,

                    "date":
                        prediction_period
                        .to_timestamp(),

                    "horizon":
                        h + 1,

                    "predicted_sales":
                        predicted_sales,
                }
            )

    forecast_df = pd.DataFrame(
        forecast_rows
    )

    forecast_df = (
        forecast_df
        .sort_values(
            GROUP_COLS + ["horizon"]
        )
        .reset_index(drop=True)
    )

    forecast_136 = (
        forecast_df[
            forecast_df[
                "horizon"
            ].isin(
                [1, 3, 6]
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    forecast_df.to_csv(
        FORECAST_6M_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    forecast_136.to_csv(
        FORECAST_136_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    forecast_dict = make_forecast_dict(
        forecast_136
    )

    print()
    print(
        forecast_136
        .head(30)
        .to_string(index=False)
    )

    print()

    print(
        "전체 6개월 CSV:",
        FORECAST_6M_PATH,
    )

    print(
        "1/3/6개월 CSV:",
        FORECAST_136_PATH,
    )

    print(
        "Dictionary 차량 수:",
        len(forecast_dict),
    )

    print()

    for i, (
        key,
        value,
    ) in enumerate(
        forecast_dict.items()
    ):

        print(
            key,
            "->",
            value,
        )

        if i >= 4:
            break

    return forecast_dict


if __name__ == "__main__":

    result = main()