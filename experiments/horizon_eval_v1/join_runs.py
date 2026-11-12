"""(wholesale 10/11) 같은 모델의 두 run(기준월 5개 백테스트 + 확인용 기준월만 따로 돌린 run)을 하나로 합친다.
모델 이름·입력 자료(input_sha256)가 같아야 하고 기준월이 겹치면 안 된다. 새 run 이름은 모델 이름으로 시작한다.

사용: join_runs.py --main <5개 기준월 run> --extra <확인용 run> [--output-base reports]
"""
import argparse
import json

import pandas as pd

from common import HERE, PREDICTION_COLUMNS, STATUS_COLUMNS, guarded, new_run_dir, write_csv, write_json


def _load(run_dir):
    d = guarded(run_dir)
    p = pd.read_csv(d / "predictions.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    s = pd.read_csv(d / "statuses.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    meta = json.loads((d / "run_summary.json").read_text(encoding="utf-8"))
    return p, s, meta


def join(main_run, extra_run, output_base=HERE / "reports"):
    p1, s1, m1 = _load(main_run)
    p2, s2, m2 = _load(extra_run)
    names = set(p1["model"]) | set(p2["model"])
    if len(names) != 1:
        raise ValueError(f"모델 이름이 다름: {sorted(names)}")
    if m1.get("input_sha256") != m2.get("input_sha256"):
        raise ValueError("입력 자료(input_sha256)가 다름")
    overlap = set(p1["origin"]) & set(p2["origin"])
    if overlap:
        raise ValueError(f"기준월 겹침: {sorted(overlap)}")
    name = names.pop()
    run_dir = new_run_dir(output_base, name)
    write_csv(run_dir / "predictions.csv", pd.concat([p1, p2], ignore_index=True)[PREDICTION_COLUMNS])
    write_csv(run_dir / "statuses.csv", pd.concat([s1, s2], ignore_index=True)[STATUS_COLUMNS])
    write_json(run_dir / "run_summary.json", {
        "model": name, "rule": "join_runs: main run + extra origins run", "sources": [str(main_run), str(extra_run)],
        "origins": sorted(set(p1["origin"]) | set(p2["origin"])), "input_sha256": m1.get("input_sha256"),
        "main_summary": m1, "extra_summary": m2, "trained_models": 0})
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", required=True)
    parser.add_argument("--extra", required=True)
    parser.add_argument("--output-base", default=str(HERE / "reports"))
    args = parser.parse_args(argv)
    out = join(args.main, args.extra, args.output_base)
    print(out)
    return out


if __name__ == "__main__":
    main()
