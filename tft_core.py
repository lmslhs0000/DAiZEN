import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Callback, Trainer, seed_everything
import lightning.pytorch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from baseline import SERIES_KEY


PREDICTION_COLUMNS = SERIES_KEY + ["Month", "horizon", "prediction"]
STATUS_COLUMNS = SERIES_KEY + ["prediction_status", "reason"]
TRAINING_LOG_COLUMNS = ["Country", "train_end", "hidden_size", "requested_epochs", "epoch", "actual_epochs",
                        "train_loss", "training_loss", "learning_rate", "device", "epoch_seconds", "elapsed_seconds", "status"]


class FixedLearningRateTFT(TemporalFusionTransformer):


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0)


class TrainingLog(Callback):
    def __init__(self, country, train_end, hidden_size, epochs, device, progress=None):
        self.base = {"Country": country, "train_end": str(train_end), "hidden_size": hidden_size,
                     "requested_epochs": epochs, "device": device}
        self.progress = progress
        self.rows = []
        self.started = time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_started = time.perf_counter()
        self.loss_sum = 0.0
        self.sample_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        value = float(loss.detach().cpu())
        if not np.isfinite(value):
            raise ValueError("TFT 학습 손실이 유한하지 않습니다.")
        count = int(batch[0]["encoder_lengths"].shape[0])
        self.loss_sum += value * count
        self.sample_count += count

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.sample_count:
            raise ValueError("NO_TRAINING_ROWS")
        rates = [float(group["lr"]) for optimizer in trainer.optimizers for group in optimizer.param_groups]
        if not rates or any(rate != 0.001 for rate in rates) or trainer.lr_scheduler_configs:
            raise ValueError("고정 학습률 계약을 위반했습니다.")
        now = time.perf_counter()
        row = {**self.base, "epoch": int(trainer.current_epoch) + 1,
               "train_loss": self.loss_sum / self.sample_count, "learning_rate": rates[0],
               "epoch_seconds": now - self.epoch_started, "elapsed_seconds": now - self.started, "status": "OK"}
        row["training_loss"] = row["train_loss"]
        row["actual_epochs"] = row["epoch"]
        self.rows.append(row)
        if self.progress:
            self.progress(f"TFT {self.base['Country']} train_end={self.base['train_end']} "
                          f"hidden={self.base['hidden_size']} epoch={row['epoch']}/{self.base['requested_epochs']} "
                          f"loss={row['train_loss']:.6f} lr={row['learning_rate']} "
                          f"elapsed={row['elapsed_seconds']:.1f}s")


def fit_market(data, country, parameters, train_end, *, device="cpu", model_path=None, progress=None):

    from tft_data import prepare_training, make_training_loader

    train_end = _month(train_end)
    if set(parameters) - {"hidden_size", "epochs"}:
        raise ValueError("TFT 후보 설정은 hidden_size와 epochs만 허용합니다.")
    hidden_size, epochs = parameters.get("hidden_size", 16), parameters.get("epochs", 20)
    for name, value in (("hidden_size", hidden_size), ("epochs", epochs)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name}은 양의 정수여야 합니다.")
    if hidden_size not in (16, 32):
        raise ValueError("TFT hidden_size 후보는 16 또는 32입니다.")
    device = str(device)
    if device not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("장치는 cpu 또는 cuda:0이어야 합니다.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("요청한 CUDA 장치를 사용할 수 없습니다.")

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    seed_everything(42, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    if progress:
        progress(f"TFT {country} train_end={train_end} hidden={hidden_size} epochs={epochs}: 학습 창 준비")
    prepared = prepare_training(data, country, train_end)
    loader = make_training_loader(prepared, batch_size=64, seed=42)
    model = FixedLearningRateTFT.from_dataset(
        prepared["dataset"], hidden_size=hidden_size, hidden_continuous_size=8,
        attention_head_size=1, lstm_layers=1, dropout=0.1, output_size=1,
        loss=QuantileLoss(quantiles=[0.5]), optimizer="adam", learning_rate=0.001,
        weight_decay=0.0, reduce_on_plateau_patience=None, logging_metrics=torch.nn.ModuleList([]),
        log_interval=-1, log_val_interval=-1,
    )
    recorder = TrainingLog(country, train_end, hidden_size, epochs, device, progress)
    trainer = Trainer(
        max_epochs=epochs, accelerator="gpu" if device.startswith("cuda") else "cpu", devices=1,
        precision="32-true", gradient_clip_val=0.1, callbacks=[recorder], logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,

        num_sanity_val_steps=0, limit_val_batches=0, deterministic="warn" if device.startswith("cuda") else True,
    )
    trainer.fit(model, train_dataloaders=loader)
    if len(recorder.rows) != epochs:
        raise RuntimeError(f"요청 epoch {epochs}와 실제 epoch {len(recorder.rows)}가 다릅니다.")
    model.to(device)
    model.eval()
    schema = copy.deepcopy(prepared["schema"])
    schema.update({
        "algorithm": "TFT", "parameters": {"hidden_size": hidden_size, "epochs": epochs,
            "hidden_continuous_size": 8, "attention_head_size": 1, "lstm_layers": 1,
            "dropout": 0.1, "output_size": 1, "optimizer": "Adam", "learning_rate": 0.001,
            "weight_decay": 0.0, "gradient_clip_val": 0.1, "precision": "32-true", "seed": 42,
            "batch_size": 64, "num_workers": 0, "shuffle": True, "drop_last": False,
            "quantiles": [0.5], "early_stopping": False, "lr_scheduler": None},
        "actual_epochs": len(recorder.rows), "device": device, "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": "warn" if device.startswith("cuda") else True,
        "reproducibility_note": "Seed and loader seed 42; CUDA upsample_linear1d backward may be nondeterministic" if device.startswith("cuda") else "CPU deterministic algorithms enabled",
        "training_loss": "QuantileLoss(q=0.5): 2 * pinball loss, original Sales units",
        "inverse_transform": "TemporalFusionTransformer.forward -> transform_output, exactly once",
        "global_step": int(trainer.global_step), "training_seconds": recorder.rows[-1]["elapsed_seconds"],
    })
    bundle = {"model": model, "parameters": copy.deepcopy(prepared["parameters"]), "schema": schema}
    training_log = pd.DataFrame(recorder.rows, columns=TRAINING_LOG_COLUMNS)
    if model_path is not None:
        save_market_model(bundle, model_path)

    model._trainer = None
    return bundle, training_log


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_market_model(bundle, path):

    path = Path(path)
    if path.suffix.lower() != ".ckpt":
        raise ValueError("TFT 모델은 .ckpt 파일로 저장해야 합니다.")
    names = [path.name, "dataset_parameters.pt", "feature_schema.json", "artifact_manifest.json"]
    if any(path.with_name(name).exists() for name in names):
        raise FileExistsError("기존 모델이나 전처리 파일은 덮어쓰지 않습니다.")
    path.parent.mkdir(parents=True, exist_ok=True)
    model, schema = bundle["model"], bundle["schema"]
    checkpoint = {"epoch": schema["actual_epochs"] - 1, "global_step": schema["global_step"],
                  "pytorch-lightning_version": lightning.pytorch.__version__,
                  "state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)}
    model.on_save_checkpoint(checkpoint)
    with tempfile.TemporaryDirectory(prefix=".tft_staging_", dir=path.parent) as folder:
        staging = Path(folder)
        torch.save(checkpoint, staging / path.name)
        torch.save(bundle["parameters"], staging / "dataset_parameters.pt")
        (staging / "feature_schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest = {"format_version": 1, "algorithm": "TFT", "files": {
            name: _sha256(staging / name) for name in names[:-1]}}
        (staging / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        created = []
        try:
            for name in names:
                destination = path.with_name(name)
                with destination.open("xb") as target:
                    created.append(destination)
                    target.write((staging / name).read_bytes())
        except BaseException:
            for destination in created:
                destination.unlink(missing_ok=True)
            raise


def load_market_model(path, device="cpu"):

    path = Path(path)
    manifest = json.loads(path.with_name("artifact_manifest.json").read_text(encoding="utf-8"))
    for name in (path.name, "dataset_parameters.pt", "feature_schema.json"):
        if manifest.get("files", {}).get(name) != _sha256(path.with_name(name)):
            raise ValueError(f"TFT 저장 파일 해시가 일치하지 않습니다: {name}")
    schema = json.loads(path.with_name("feature_schema.json").read_text(encoding="utf-8"))

    parameters = torch.load(path.with_name("dataset_parameters.pt"), map_location="cpu", weights_only=False)
    model = FixedLearningRateTFT.load_from_checkpoint(path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return {"model": model, "parameters": parameters, "schema": schema}


def _month(value):
    if isinstance(value, pd.Period) and value.freqstr == "M":
        return value
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{1,2}", value) is None:
        raise ValueError("월은 YYYY-MM 형식이어야 합니다.")
    return pd.to_datetime(value, format="%Y-%m").to_period("M")


def _history_frame(history, origin):
    frame = history[SERIES_KEY + ["Month", "Sales"]].copy()
    frame["Month"] = frame["Month"].map(_month)
    if frame["Month"].gt(origin).any():
        raise ValueError("기준월 이후 미래 실제값을 예측 이력에 넣을 수 없습니다.")
    if frame.duplicated(SERIES_KEY + ["Month"]).any():
        raise ValueError("예측 이력에 차종·월 중복이 있습니다.")
    return frame


def predict_one_step(bundle, history, origin, placeholder=0.0):

    import torch
    from tft_data import make_inference_dataset

    origin = _month(origin)
    if _month(bundle["schema"]["train_end"]) > origin:
        raise ValueError("모델의 학습 종료월이 관측 기준월보다 뒤입니다.")
    frame = _history_frame(history, origin)
    seen = {tuple(key) for key in bundle["schema"]["seen_series"]}
    requested = {tuple(key) for key in frame[SERIES_KEY].drop_duplicates().itertuples(index=False, name=None)}
    if not requested.issubset(seen):
        raise ValueError("UNSEEN_SERIES: 해당 시장 모델에 없는 시계열입니다.")
    frame = frame.loc[frame["Month"].ge(origin - 11)].copy()
    dataset = make_inference_dataset(bundle["parameters"], frame, origin, placeholder=placeholder)
    loader = dataset.to_dataloader(train=False, batch_size=64, shuffle=False, drop_last=False, num_workers=0)
    model = bundle["model"]
    model.eval()
    device = next(model.parameters()).device
    rows = []
    with torch.inference_mode():
        for inputs, _ in loader:
            index = dataset.x_to_index(inputs)
            moved = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

            output = model(moved)["prediction"]
            values = output.detach().cpu().numpy().reshape(-1)
            if len(values) != len(index):
                raise ValueError("TFT의 1개월·1분위수 출력 행 수가 요청 차종 수와 다릅니다.")
            for position, item in enumerate(index.to_dict("records")):
                rows.append({"Country": bundle["schema"]["country"], "Brand": item["Brand"],
                             "Model": item["Model"], "Month": origin + 1,
                             "prediction": float(values[position])})
    return pd.DataFrame(rows, columns=SERIES_KEY + ["Month", "prediction"])


def forecast_recursive(bundle, history, origin, horizon, schema=None):

    origin = _month(origin)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("예측 개월 수는 양의 정수여야 합니다.")
    schema = bundle["schema"] if schema is None else schema
    if schema != bundle["schema"]:
        raise ValueError("모델과 별도의 다른 스키마를 사용할 수 없습니다.")
    if _month(schema["train_end"]) > origin:
        raise ValueError("모델의 학습 종료월이 관측 기준월보다 뒤입니다.")
    frame = _history_frame(history, origin)
    required = pd.period_range(origin - 11, origin, freq="M")
    seen = {tuple(key) for key in schema["seen_series"]}
    statuses = {}
    buffers = {}
    predictions = {}
    for key, vehicle in frame.groupby(SERIES_KEY, sort=True):
        monthly = vehicle.set_index("Month")["Sales"]
        reasons = []
        status = "OK"
        if monthly.index.min() > required[0]:
            status = "INSUFFICIENT_HISTORY"
            reasons.append("기준월까지 최소 12개월의 관측 기간이 필요합니다.")
        else:
            missing = required.difference(monthly.index)
            blank = required.intersection(monthly.index)
            blank = blank[monthly.reindex(blank).isna()]
            if len(missing) or len(blank):
                status = "INCOMPLETE_HISTORY"
                if len(missing):
                    reasons.append(f"이력 누락: {missing.astype(str).tolist()}")
                if len(blank):
                    reasons.append(f"이력 공란: {blank.astype(str).tolist()}")
        if key not in seen:
            if status == "OK":
                status = "UNSEEN_SERIES"
            reasons.append("해당 시장 모델의 학습 행에 없던 정확한 시계열 키입니다.")
        statuses[key] = {**dict(zip(SERIES_KEY, key)), "prediction_status": status, "reason": "; ".join(reasons)}
        if status == "OK":
            values = monthly.loc[required].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"예측 이력에 유효하지 않은 판매량이 있습니다: {key}")
            buffers[key] = values.copy()
            predictions[key] = []

    for step in range(horizon):
        if not buffers:
            break
        moving_origin = origin + step
        calendar = pd.period_range(moving_origin - 11, moving_origin, freq="M")
        batch = pd.concat([
            pd.DataFrame({**dict(zip(SERIES_KEY, key)), "Month": calendar, "Sales": values})
            for key, values in buffers.items()
        ], ignore_index=True)
        try:
            next_month = predict_one_step(bundle, batch, moving_origin)
            if next_month.duplicated(SERIES_KEY).any():
                raise ValueError("TFT의 1개월 예측에 중복 차종이 있습니다.")
            raw = {tuple(row[column] for column in SERIES_KEY): float(row["prediction"])
                   for row in next_month.to_dict("records")}
            if set(raw) != set(buffers):
                raise ValueError("TFT의 예측 차종과 요청 차종이 다릅니다.")
        except Exception as error:
            for key in buffers:
                statuses[key].update(prediction_status="PREDICTION_ERROR",
                                     reason=f"{moving_origin + 1} 예측 실패: {type(error).__name__}: {error}")
            break
        for key in list(buffers):
            value = raw[key]
            if not np.isfinite(value):
                statuses[key].update(prediction_status="PREDICTION_ERROR", reason=f"{moving_origin + 1} 예측값이 유한하지 않습니다.")
                del buffers[key]
                continue
            value = max(0.0, value)
            predictions[key].append(value)
            buffers[key] = np.append(buffers[key][1:], value)

    rows = []
    for key, values in predictions.items():
        if statuses[key]["prediction_status"] != "OK":
            continue
        for step, value in enumerate(values, start=1):
            rows.append({**dict(zip(SERIES_KEY, key)), "Month": origin + step, "horizon": step, "prediction": value})
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS), pd.DataFrame(list(statuses.values()), columns=STATUS_COLUMNS)
