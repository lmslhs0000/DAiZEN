"""실제로 납품한 예측 목록(raw/deliveries.csv: origin, path, added_at).

빈칸 채우기(collect_month)와 매달 성적 확인(monitor)은 이 목록의 예측만 쓴다 — 실험·시험 중에 생긴 다른 예측 묶음을 잘못 집지 않게.
목록 파일이 없으면 7월 납품(production/client_package_july_ic_20261013…/client_forecast_24m.csv = v2 전달본의 ic 판) 하나로 시작한다.
(주의: handover_20261012/client_package 는 하루 전 hq3 판이라 쓰지 않는다 — 2026-10-16 확인.)
run_pipeline.sh production 이 끝나면 새 납품을 자동으로 더한다:  python deliveries.py add <client_package 폴더 또는 csv>
같은 기준월이 여러 번 더해지면 마지막 것을 쓴다.
"""
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REGISTRY = ROOT / "raw" / "deliveries.csv"
JULY = ROOT / "experiments" / "horizon_eval_v1" / "production" / "client_package_july_ic_20261013T035849_323513Z_9f3a75" / "client_forecast_24m.csv"   # 2026-07 기준 실제 납품(ic, v2 전달본과 같음)


def _origin(path):
    return str(pd.read_csv(path, encoding="utf-8-sig", usecols=["origin"], nrows=1, dtype=str)["origin"].iloc[0])


def _rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def load(registry=REGISTRY, default=JULY):
    """기준월마다 한 줄(마지막으로 더한 것). path 는 절대 경로로 돌려준다."""
    registry = Path(registry)
    if registry.exists():
        reg = pd.read_csv(registry, encoding="utf-8-sig", dtype=str)
    elif Path(default).exists():
        reg = pd.DataFrame([{"origin": _origin(default), "path": _rel(default), "added_at": "default"}])
    else:
        reg = pd.DataFrame(columns=["origin", "path", "added_at"])
    reg = reg.drop_duplicates("origin", keep="last").sort_values("origin").reset_index(drop=True)
    reg["path"] = [str(p if Path(p).is_absolute() else ROOT / p) for p in reg["path"]]
    return reg


def add(registry, path, default=JULY):
    path = Path(path)
    if path.is_dir():
        path = path / "client_forecast_24m.csv"
    registry = Path(registry)
    reg = pd.read_csv(registry, encoding="utf-8-sig", dtype=str) if registry.exists() else \
        load(registry, default).assign(path=lambda d: [_rel(p) for p in d["path"]])
    row = pd.DataFrame([{"origin": _origin(path), "path": _rel(path), "added_at": datetime.now().isoformat(timespec="seconds")}])
    pd.concat([reg, row], ignore_index=True).to_csv(registry, index=False, encoding="utf-8-sig")


def pick(reg, month):
    """채울 달보다 앞선 기준월 중 가장 늦은 납품 예측. 없으면 None."""
    r = reg[reg["origin"].astype(str) < month]
    return None if not len(r) else r.sort_values("origin").iloc[-1]["path"]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "add":
        add(REGISTRY, sys.argv[2])
        print(load().to_string(index=False))
    else:
        print(load().to_string(index=False))
