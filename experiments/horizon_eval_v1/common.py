"""horizon_eval_v1 공통 상수, 경로 가드, 해시, 저장 함수."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import PROJECT_ROOT, resolve_recorded_path  # noqa: E402

KEY = ["Country", "Brand", "Model"]
ORIGINS = ["2022-07", "2023-01", "2023-07", "2024-01", "2024-07"]
CHECK_ORIGIN = "2025-07"
HORIZON = 24
CHECK_HORIZON = 12
# DAIZEN_CSV: 매달 재예측 때 원본+채움 학습 파일(raw/data_with_fills.csv)을 쓰려면 지정
CSV_PATH = Path(os.environ["DAIZEN_CSV"]) if os.environ.get("DAIZEN_CSV") else PROJECT_ROOT / "raw" / "data.csv"
PREDICTION_COLUMNS = ["model", "origin"] + KEY + ["Month", "horizon", "prediction"]
STATUS_COLUMNS = ["model", "origin"] + KEY + ["prediction_status", "reason"]


def guarded(path):
    """모든 입력·출력 경로를 프로젝트 폴더 안으로 강제한다. 밖을 가리키면 ValueError."""
    return resolve_recorded_path(str(path), PROJECT_ROOT)


def file_sha256(path):
    with guarded(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def require_unchanged(path, expected_sha256):
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"실행 중 입력 파일이 바뀌었습니다: {path}")


def new_run_dir(base, prefix):
    # Windows 시계 해상도가 낮아 같은 시각 이름이 겹칠 수 있으므로 짧은 무작위 접미사를 붙인다.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = guarded(Path(base) / f"{prefix}_{stamp}_{uuid4().hex[:6]}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_csv(path, table):
    table.to_csv(guarded(path), index=False, encoding="utf-8-sig")


def write_json(path, value):
    guarded(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path):
    return json.loads(guarded(path).read_text(encoding="utf-8-sig"))


def stack_outputs(predictions, statuses, model_name, origin):
    """모델 이름·기준월 열을 붙여 예측표와 상태표를 각각 하나로 합친다."""
    base_p = KEY + ["Month", "horizon", "prediction"]
    base_s = KEY + ["prediction_status", "reason"]
    p = pd.concat([f[base_p] for f in predictions], ignore_index=True) if predictions else pd.DataFrame(columns=base_p)
    s = pd.concat([f[base_s] for f in statuses], ignore_index=True) if statuses else pd.DataFrame(columns=base_s)
    for table in (p, s):
        table.insert(0, "origin", str(origin))
        table.insert(0, "model", model_name)
    return p[PREDICTION_COLUMNS], s[STATUS_COLUMNS]
