"""(wholesale) 접두사마다 같은 자료(SHA)의 '기준월 5개 run'과 '확인용 기준월만 run' 중 가장 새것을 찾아 join_runs로 합친다.
사용: join_all.py <sha8> <prefix...>"""
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import join_runs  # noqa: E402

FIVE = {"2022-07", "2023-01", "2023-07", "2024-01", "2024-07"}
sha, prefixes = sys.argv[1], sys.argv[2:]
for prefix in prefixes:
    main = extra = None
    for d in sorted(glob.glob(str(H / "reports" / f"{prefix}_20*")), key=os.path.getmtime, reverse=True):
        d = Path(d)
        try:
            meta = json.loads((d / "run_summary.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        if not str(meta.get("input_sha256", "")).startswith(sha):
            continue
        origins = set(pd.read_csv(d / "predictions.csv", usecols=["origin"], dtype=str)["origin"])
        if origins == FIVE and main is None:
            main = d
        elif origins == {"2025-07"} and extra is None:
            extra = d
        if main and extra:
            break
    if not (main and extra):
        print("MISSING", prefix, main, extra)
        sys.exit(1)
    out = join_runs.join(main, extra, H / "reports")
    print("JOINED", prefix, out.name)
