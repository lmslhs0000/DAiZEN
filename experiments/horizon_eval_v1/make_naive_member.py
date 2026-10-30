"""naive_member(작년 그대로, 챔피언 조립 nv 단계의 기준 재료)를 지금 원본으로 다시 만든다.
처음 run(reports/naive_member_20260930T174153_431763Z_8de7c0)과 같은 규칙: naive.seasonal_naive, 기준월 5개 24개월 + 확인용 2025-07 12개월."""
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

import common  # noqa: E402,F401  (프로젝트 루트를 sys.path에)
import baseline  # noqa: E402
from common import (CHECK_HORIZON, CHECK_ORIGIN, CSV_PATH, HERE, HORIZON, ORIGINS, PREDICTION_COLUMNS, STATUS_COLUMNS,  # noqa: E402
                    file_sha256, new_run_dir, require_unchanged, write_csv, write_json)
from naive import seasonal_naive  # noqa: E402

import argparse  # noqa: E402
_ap = argparse.ArgumentParser()
_ap.add_argument("--origins", nargs="*", help="(hz) 이 기준월들만(24개월), 확인용 없음; 기본은 ORIGINS + 확인용")
_ap.add_argument("--name", default="naive_member")
_args = _ap.parse_args()
PLAN = [(o, HORIZON) for o in _args.origins] if _args.origins else [(o, HORIZON) for o in ORIGINS] + [(CHECK_ORIGIN, CHECK_HORIZON)]
h = file_sha256(CSV_PATH)
data = baseline.read_sales_csv(CSV_PATH)
data["Month"] = pd.PeriodIndex(data["Month"], freq="M")
ps, ss = [], []
for origin, horizon in PLAN:
    p, s = seasonal_naive(data, origin, horizon)
    ps.append(p.assign(model=_args.name, origin=origin))
    ss.append(s.assign(model=_args.name, origin=origin))
p, s = pd.concat(ps, ignore_index=True), pd.concat(ss, ignore_index=True)
p["Month"] = p["Month"].astype(str)
require_unchanged(CSV_PATH, h)
d = new_run_dir(HERE / "reports", _args.name)
write_csv(d / "predictions.csv", p[PREDICTION_COLUMNS])
write_csv(d / "statuses.csv", s[STATUS_COLUMNS])
write_json(d / "run_summary.json", {"model": _args.name, "rule": "naive.seasonal_naive(작년 그대로), 앙상블 재료용 사본",
                                     "origins": [o for o, _ in PLAN], "input_sha256": h, "trained_models": 0})
print(d)
