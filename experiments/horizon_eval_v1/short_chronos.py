"""(fu) chronos_short: chronos2_ft 체크포인트(기준월별, 새 학습 없음)로 짧은 이력(마지막 연속 구간 3~11개월) 차종을
시장마다 (Brand, Model) 순 100개 묶음 cross-learning으로 예측(0.5 분위수, 0 하한). 대상 규칙은 short_history와 같다.

사용: experiments/chronos_ft_v1/.venv/Scripts/python.exe -X utf8 -B experiments/horizon_eval_v1/short_chronos.py
      --from-run <chronos2_ft run> [--device cuda] [--origins ...]
"""

import argparse
import json

import numpy as np
import pandas as pd

from common import (CHECK_ORIGIN, CSV_PATH, HERE, KEY, ORIGINS, PREDICTION_COLUMNS, STATUS_COLUMNS, file_sha256, guarded,
                    new_run_dir, require_unchanged, write_csv, write_json)
import baseline
import forecast_chronos as fc
from short_history import short_histories

NAME = "short_chronos"


def forecast_origin(pipeline, data, origin, horizon):
    found = short_histories(data, origin)
    keys = sorted(found)
    rows = []
    for country in dict.fromkeys(k[0] for k in keys):
        members = [k for k in keys if k[0] == country]
        values = fc.core.forecast_joint(pipeline, [found[k] for k in members], horizon)
        start = pd.Period(origin, freq="M")
        rows += [{**dict(zip(KEY, key)), "Month": start + h, "horizon": h, "prediction": float(v)}
                 for key, series in zip(members, values) for h, v in enumerate(series, start=1)]
    return pd.DataFrame(rows, columns=KEY + ["Month", "horizon", "prediction"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--output-base", default=str(HERE / "reports"))
    parser.add_argument("--origins", nargs="*", default=ORIGINS + [CHECK_ORIGIN])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--from-run")
    parser.add_argument("--zero-shot", action="store_true", help="(fz) 체크포인트 없이 사전학습 amazon/chronos-2(short_chronos_zs)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--checkpoint-origin", help="(gv) 모든 기준월에 이 기준월의 체크포인트를 씀(공개 실적을 덧붙인 납품용)")
    args = parser.parse_args(argv)
    if args.zero_shot == bool(args.from_run):
        raise ValueError("--from-run과 --zero-shot 중 하나만 줍니다.")
    fc.use_core("chronos2")
    ft = fc.load_finetune_module()
    source = guarded(args.from_run) if args.from_run else None
    name = NAME + "_zs" if args.zero_shot else NAME
    load = (lambda origin: fc.core.load_pipeline(device=args.device)) if args.zero_shot else (
        lambda origin: ft.load_finetuned(checkpoints[origin], device=args.device))
    input_hash = file_sha256(args.csv)
    data = baseline.read_sales_csv(guarded(args.csv))
    frames, checkpoints, first = [], {}, None
    for origin in args.origins:
        if source is not None:
            checkpoints[origin] = source / "models" / (args.checkpoint_origin or origin) / "finetuned-ckpt"
        model = load(origin)
        p = forecast_origin(model, data, origin, args.horizon).assign(model=name, origin=str(origin))
        if first is None and len(p):
            first = (origin, p)
        frames.append(p)
        print(f"{name} {origin}: {p[KEY].drop_duplicates().shape[0]}차종", flush=True)
    if first is not None:
        again = forecast_origin(load(first[0]), data, first[0], args.horizon)
        np.testing.assert_allclose(again["prediction"].to_numpy(float), first[1]["prediction"].to_numpy(float),
                                   rtol=1e-6, atol=1e-5)
    predictions = pd.concat(frames, ignore_index=True)[PREDICTION_COLUMNS]
    require_unchanged(args.csv, input_hash)
    run_dir = new_run_dir(args.output_base, name)
    statuses = predictions.drop_duplicates(["origin"] + KEY).assign(prediction_status="OK", reason="")
    write_csv(run_dir / "predictions.csv", predictions)
    write_csv(run_dir / "statuses.csv", statuses.reindex(columns=STATUS_COLUMNS))
    write_json(run_dir / "run_summary.json", {
        "model": name, "rule": "(fu) short-history (3-11 consecutive months ending at origin), "
        + ("pretrained amazon/chronos-2 (zero-shot, fz), " if args.zero_shot else "chronos2_ft checkpoint, ")
        + "per Country sorted (Brand, Model) chunks of 100 cross-learning, quantile 0.5, floor 0",
        "input_sha256": input_hash, "origins": [str(o) for o in args.origins], "horizon": args.horizon,
        "device": args.device, "finetuned_checkpoint_source": str(source) if source else None,
        "checkpoint_origin": args.checkpoint_origin,
        "finetuned_checkpoint_sha256": {o: file_sha256(p / "model.safetensors") for o, p in checkpoints.items()},
        **fc.core.model_fingerprint(), "trained_models": 0, "reload_check": "PASS"})
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
