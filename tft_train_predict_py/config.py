from pathlib import Path

# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "raw" / "korean.csv"

MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "current_best_tft.ckpt"
MODEL_METADATA_PATH = MODEL_DIR / "current_best_tft_metadata.json"
CALIBRATION_DIR = MODEL_DIR / "calibration"

OUTPUT_DIR = BASE_DIR / "output"
FORECAST_6M_PATH = OUTPUT_DIR / "future_forecast_6months.csv"
FORECAST_136_PATH = OUTPUT_DIR / "future_forecast_1_3_6months.csv"

# =========================================================
# Data / forecasting setup
# =========================================================
GROUP_COLS = ["market", "brand", "model"]
REQUIRED_COLS = ["date", "market", "brand", "model", "sales"]

MAX_ENCODER_LENGTH = 24
MAX_PREDICTION_LENGTH = 6
LAGS = [1, 3, 6, 12]

# 최근 3개월 중 판매 기록이 한 번이라도 있으면 예측 후보
ACTIVE_WINDOW = 3

# =========================================================
# Current Best TFT hyperparameters
# =========================================================
LEARNING_RATE = 0.0012066015102032692
HIDDEN_SIZE = 13
ATTENTION_HEAD_SIZE = 2
DROPOUT = 0.18011189995840438
HIDDEN_CONTINUOUS_SIZE = 11
GRADIENT_CLIP_VAL = 0.11134427689247173

# =========================================================
# Training
# =========================================================
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 0
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 5
