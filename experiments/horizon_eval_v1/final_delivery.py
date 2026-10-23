"""최종 납품 조립(2026-10-06): 본 납품 파일에서 예측이 빈 차종을 보조 예측으로 채운 합친 판을 만든다.

- 이력 3~11개월 신차(INSUFFICIENT_HISTORY 중 short_chronos 예측이 있는 것) → 상태 OK_SHORT_HISTORY ((fu) 백테스트 47.4%)
- 학습 모델이 못 본 차종(UNSEEN_SERIES) → production chronos2_ft 예측, 상태 OK_CHRONOS_ONLY ((fv) 백테스트 47.9%)
- 본 납품의 OK 행은 값·순서 그대로, 보조 예측이 없는 빈 행(이력 1~2개월)은 빈 채로.
결과 폴더에 보조 파일(supplement_future_24m.csv)과 합친 판(<name>_future_24m.csv)을 함께 쓴다.

사용: final_delivery.py --delivery <본 납품 폴더> --short-run <production short_chronos> --chronos-run <production chronos2_ft>
      --name final [--output-base production]
"""

import argparse
from pathlib import Path

import pandas as pd

from common import HERE, KEY, file_sha256, guarded, new_run_dir, write_csv, write_json
from score import read_run
from short_history import build_supplement, fill_from_supplement


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", required=True)
    parser.add_argument("--short-run", required=True)
    parser.add_argument("--chronos-run", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-base", default=str(HERE / "production"))
    args = parser.parse_args(argv)
    path = next(Path(guarded(args.delivery)).glob("*_future_24m.csv"))
    delivery = pd.read_csv(path, encoding="utf-8-sig", dtype={"origin": str, "train_end": str, "Month": str})
    short = build_supplement(delivery, read_run(args.short_run), "short_chronos")
    unseen = build_supplement(delivery.loc[delivery["status"].eq("UNSEEN_SERIES")], read_run(args.chronos_run),
                              "chronos2_ft", status="OK_CHRONOS_ONLY")
    overlap = set(map(tuple, short[KEY].values)) & set(map(tuple, unseen[KEY].values))
    if overlap:
        raise ValueError(f"두 보조 예측이 같은 차종을 채웁니다: {sorted(overlap)[:5]}")
    run_dir = new_run_dir(args.output_base, f"delivery_{args.name}")
    supplement = pd.concat([short, unseen], ignore_index=True).sort_values(KEY + ["horizon"]).assign(run_id=run_dir.name)
    full = fill_from_supplement(delivery, supplement, args.name).assign(run_id=run_dir.name)
    ok = delivery["status"].eq("OK").to_numpy()
    if len(full) != len(delivery) or not (full.loc[ok, "prediction"].to_numpy() == delivery.loc[ok, "prediction"].to_numpy()).all():
        raise ValueError("본 납품의 OK 행이 바뀌었습니다.")
    if (full["prediction"].dropna() < 0).any():
        raise ValueError("음수 예측이 있습니다.")
    write_csv(run_dir / "supplement_future_24m.csv", supplement)
    write_csv(run_dir / f"{args.name}_future_24m.csv", full)
    write_json(run_dir / "run_summary.json", {
        "model": args.name, "rule": "main delivery + OK_SHORT_HISTORY (short_chronos) + OK_CHRONOS_ONLY (chronos2_ft for UNSEEN_SERIES)",
        "base_delivery": str(args.delivery), "base_sha256": file_sha256(path), "short_run": str(args.short_run),
        "chronos_run": str(args.chronos_run),
        "series_by_status": full.drop_duplicates(KEY)["status"].value_counts().to_dict()})
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
