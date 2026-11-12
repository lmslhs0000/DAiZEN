import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd

from baseline import SERIES_KEY, read_sales_csv
from xgboost_core import (build_training_frame, build_series_weights, fit_market, forecast_recursive,
                         load_market_model, save_market_model)
from xgboost_evaluation import evaluate_forecast


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SELECTION = ROOT / "experiments/log_tuning_v1/reports/xgboost_log_search/run_20260924T071518_398018Z_c184bd022cdf/selected_parameters.json"
FOLDS = [("A_validation_1", "2024-07"), ("A_validation_2", "2025-01"),
         ("A_validation_3", "2025-07")]
HORIZON = 6
VARIANTS = ("row_weight", "series_weight")
BASELINE_RUN = ROOT / "experiments/direct_forecast_v1/reports/run_20260927T041632_141090Z"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                          encoding="utf-8")


def write_csv(path, frame):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def load_data(path):
    data = read_sales_csv(path)
    exact = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    exact["Month"] = pd.PeriodIndex(exact.Month.str.strip(), freq="M")
    exact["Sales"] = exact.Sales.str.strip().map(lambda text: float(text) if text else float("nan"))
    return data.drop(columns="Sales").merge(exact[SERIES_KEY + ["Month", "Sales"]],
                                            on=SERIES_KEY + ["Month"], validate="one_to_one")


def runtime_versions():
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "xgboost")}}


def build_fixed_cohort(data, universe, training_frames, folds=FOLDS, horizon=HORIZON):
    """Use already-built h=1 training rows and actual availability, never model scores."""
    groups = {key: rows.set_index("Month").Sales.astype(float)
              for key, rows in data.groupby(SERIES_KEY, sort=True)}
    trained = {split: set(frame[SERIES_KEY].itertuples(index=False, name=None))
               for split, frame in training_frames.items()}
    records = []
    for key in universe[SERIES_KEY].itertuples(index=False, name=None):
        monthly = groups.get(key, pd.Series(dtype=float))
        record = dict(zip(SERIES_KEY, key))
        reasons = []
        for split, origin_text in folds:
            origin = pd.Period(origin_text, freq="M")
            observed = monthly if monthly.empty else monthly.loc[
                (monthly.index >= pd.Period("2016-01", "M")) & (monthly.index <= origin)]
            required = pd.period_range(origin - 11, origin, freq="M")
            actual = monthly.reindex(pd.period_range(origin + 1, periods=horizon, freq="M"))
            fold_reasons = []
            if observed.empty or observed.index.min() > required[0]:
                fold_reasons.append("INSUFFICIENT_HISTORY")
            elif observed.reindex(required).isna().any():
                fold_reasons.append("INCOMPLETE_HISTORY")
            if key not in trained[split]:
                fold_reasons.append("UNSEEN_SERIES")
            if actual.isna().any():
                fold_reasons.append("MISSING_ACTUAL")
            elif float(actual.sum()) <= 0:
                fold_reasons.append("ZERO_ACTUAL_SUM")
            record[f"{split}_reason"] = ";".join(fold_reasons)
            reasons.extend(f"{split}:{reason}" for reason in fold_reasons)
        record.update(is_common=not reasons, reason=";".join(reasons))
        records.append(record)
    return pd.DataFrame(records)


def reject_prediction_errors(frame):
    for column in ("status", "prediction_status"):
        if column in frame and frame[column].isin(["PREDICTION_ERROR", "NO_TRAINING_ROWS", "NO_VALIDATION_SCORE"]).any():
            raise ValueError("PREDICTION_ERROR: execution failure cannot reduce the fixed cohort")


def summary_record(frame):
    count = len(frame)
    if not count:
        return {"cohort_series": 0, "status": "NO_VALIDATION_SCORE",
                "row_weight_mean_WAPE": None, "series_weight_mean_WAPE": None,
                "row_weight_median_WAPE": None, "series_weight_median_WAPE": None,
                "delta_WAPE_pp": None, "improved_series": 0, "worsened_series": 0, "tied_series": 0}
    difference = frame.series_weight_WAPE - frame.row_weight_WAPE
    return {"cohort_series": count, "status": "OK",
            "row_weight_mean_WAPE": float(frame.row_weight_WAPE.mean()),
            "series_weight_mean_WAPE": float(frame.series_weight_WAPE.mean()),
            "row_weight_median_WAPE": float(frame.row_weight_WAPE.median()),
            "series_weight_median_WAPE": float(frame.series_weight_WAPE.median()),
            "delta_WAPE_pp": float(difference.mean()),
            "improved_series": int((difference < -1e-9).sum()),
            "worsened_series": int((difference > 1e-9).sum()), "tied_series": int((difference.abs() <= 1e-9).sum())}


def compare_metrics(metrics, cohort, folds=FOLDS):
    reject_prediction_errors(metrics)
    if metrics.duplicated(["variant", "split"] + SERIES_KEY).any():
        raise ValueError("Duplicate variant/fold/series metric")
    common = cohort.loc[cohort.is_common, SERIES_KEY]
    if common.empty:
        raise ValueError("NO_VALIDATION_SCORE: fixed cohort is empty")
    split_names = [split for split, _ in folds]
    expected = common.merge(pd.DataFrame({"split": split_names}), how="cross").merge(
        pd.DataFrame({"variant": VARIANTS}), how="cross")
    checked = expected.merge(metrics, on=["variant", "split"] + SERIES_KEY,
                             how="left", validate="one_to_one", indicator=True)
    if (checked._merge != "both").any():
        raise ValueError("Fixed cohort coverage is incomplete")
    if not (checked.status.eq("OK") & np.isfinite(checked.WAPE) & checked.WAPE.ge(0)).all():
        raise ValueError("Every fixed cohort score must be valid")
    series = cohort.copy()
    for variant in VARIANTS:
        selected = metrics.loc[metrics.variant.eq(variant) & metrics.split.isin(split_names)].copy()
        selected["valid"] = selected.status.eq("OK") & np.isfinite(selected.WAPE) & selected.WAPE.ge(0)
        aggregate = selected.groupby(SERIES_KEY, sort=True).agg(
            mean=("WAPE", "mean"), valid_folds=("valid", "sum"), rows=("split", "nunique")).reset_index()
        aggregate["mean"] = aggregate["mean"].where(aggregate.valid_folds.eq(len(folds)) & aggregate.rows.eq(len(folds)))
        aggregate = aggregate.rename(columns={"mean": f"{variant}_WAPE", "valid_folds": f"{variant}_valid_folds"})
        series = series.merge(aggregate.drop(columns="rows"), on=SERIES_KEY, how="left", validate="one_to_one")
    series["delta_WAPE_pp"] = (series.series_weight_WAPE - series.row_weight_WAPE).where(series.is_common)
    valid = series.loc[series.is_common]
    overall = pd.DataFrame([summary_record(valid)])
    market = pd.DataFrame([{"Country": country, **summary_record(rows)}
                           for country, rows in valid.groupby("Country", sort=True)])
    fold_records = []
    for split, origin in folds:
        selected = checked.loc[checked.split.eq(split)]
        paired = selected.pivot(index=SERIES_KEY, columns="variant", values="WAPE").reset_index()
        paired = paired.rename(columns={name: f"{name}_WAPE" for name in VARIANTS})
        fold_records.append({"split": split, "origin": origin, **summary_record(paired)})
    return series, overall, market, pd.DataFrame(fold_records)


def compare_horizons(predictions, cohort, folds=FOLDS, horizon=HORIZON):
    common = predictions.merge(cohort.loc[cohort.is_common, SERIES_KEY], on=SERIES_KEY,
                               how="inner", validate="many_to_one")
    rows = []
    for identifiers, group in common.groupby(SERIES_KEY + ["horizon"], sort=True):
        row = dict(zip(SERIES_KEY + ["horizon"], identifiers))
        for variant in VARIANTS:
            values = group.loc[group.variant.eq(variant)]
            if len(values) != len(folds) or values.split.nunique() != len(folds):
                raise ValueError("Horizon comparison coverage is incomplete")
            denominator = float(values.actual.sum())
            row[f"{variant}_WAPE"] = 100 * float(values.absolute_error.sum()) / denominator if denominator else np.nan
        row.update(actual_sum=denominator, status="OK" if denominator > 0 else "ZERO_ACTUAL_SUM")
        row["delta_WAPE_pp"] = row["series_weight_WAPE"] - row["row_weight_WAPE"]
        rows.append(row)
    series = pd.DataFrame(rows)
    summaries = []
    for step in range(1, horizon + 1):
        selected = series.loc[series.horizon.eq(step)]
        valid = selected.loc[selected.status.eq("OK")]
        summaries.append({"horizon": step, "zero_actual_sum_series": int(selected.status.eq("ZERO_ACTUAL_SUM").sum()),
                          "aggregation": "per-series WAPE at this horizon across all three validation folds",
                          **summary_record(valid)})
    return series, pd.DataFrame(summaries)


def complete_statuses(statuses, universe):
    result = universe.merge(statuses, on=SERIES_KEY, how="left", validate="one_to_one")
    missing = result.prediction_status.isna()
    result.loc[missing, "prediction_status"] = "INSUFFICIENT_HISTORY"
    result.loc[missing, "reason"] = "No observations at or before this origin"
    reject_prediction_errors(result)
    return result


def verify_predictions_equal(left, right, context):
    keys = SERIES_KEY + ["Month", "horizon"]
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[keys], right[keys], check_dtype=False)
    np.testing.assert_array_equal(left.prediction.to_numpy(float), right.prediction.to_numpy(float), err_msg=context)


def read_saved_predictions(path):
    """Python float parsing preserves the previous CSV's round-trip values and textual model keys."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for name in ("actual", "prediction", "absolute_error"):
        frame[name] = frame[name].map(lambda value: float(value) if value else np.nan)
    frame["horizon"] = frame.horizon.astype(int)
    return frame


def verify_baseline_replay(actual, expected):
    keys = ["split"] + SERIES_KEY + ["Month", "horizon"]
    left, right = actual.copy(), expected.copy()
    for frame in (left, right):
        for name in ("Month", "train_end", "origin"):
            frame[name] = frame[name].astype(str)
        if frame.duplicated(keys).any():
            raise ValueError("Baseline replay has duplicate keys")
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[keys], right[keys], check_dtype=False)
    for name in ("actual", "prediction", "absolute_error"):
        np.testing.assert_array_equal(left[name].to_numpy(float), right[name].to_numpy(float),
                                      err_msg=f"Baseline replay differs: {name}")
    for name in ("algorithm", "method", "train_end", "origin", "prediction_status", "status",
                 "reason", "supplementary_reasons"):
        pd.testing.assert_series_equal(left[name].fillna("").astype(str), right[name].fillna("").astype(str),
                                       check_dtype=False)


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if not Path(path).is_file() or file_sha256(path) != expected:
            raise ValueError(f"Preserved file checksum changed: {path}")


def validate_baseline_source(baseline_run, input_sha, selection_sha, countries, folds, horizon, n_jobs, selected):
    """Validate completed source, identical conditions, and every reused model/schema before fitting."""
    source = Path(baseline_run).resolve()
    summary = json.loads((source / "run_summary.json").read_text(encoding="utf-8"))
    contract = json.loads((source / "experiment_contract.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("run_id") != contract.get("run_id"):
        raise ValueError("Baseline source run must be complete with matching run_id")
    if summary.get("input_sha256") != input_sha or contract.get("input_sha256") != input_sha:
        raise ValueError("Baseline input checksum does not match current input")
    if contract.get("selection_sha256") != selection_sha:
        raise ValueError("Baseline selected parameter checksum does not match")
    verify_hashes({str(source / "input.csv"): input_sha,
                   str(source / "selected_parameters.json"): selection_sha})
    expected_folds = [{"split": split, "origin": origin, "train_end": origin, "horizon": horizon}
                      for split, origin in folds]
    for name, value in {"countries": countries, "folds": expected_folds, "n_jobs": n_jobs,
                        "device": "cpu", "seed": 42, "target_transform": "log1p",
                        "versions": runtime_versions()}.items():
        if contract.get(name) != value:
            raise ValueError(f"Baseline frozen condition mismatch: {name}")
    verify_hashes(contract.get("code_hashes", {}))
    prediction_sha = summary.get("artifact_hashes", {}).get("predictions.csv")
    if not prediction_sha or file_sha256(source / "predictions.csv") != prediction_sha:
        raise ValueError("Baseline predictions checksum mismatch")
    manifest = json.loads((source / "model_manifest.json").read_text(encoding="utf-8"))
    records = [row for row in manifest if row.get("variant") == "recursive"]
    expected_keys = {(country, split) for country in countries for split, _ in folds}
    by_key = {(row["Country"], row["split"]): row for row in records}
    if len(records) != len(expected_keys) or len(by_key) != len(records) or set(by_key) != expected_keys:
        raise ValueError("Baseline recursive market/fold coverage is incomplete or duplicated")
    origins = dict(folds)
    for (country, split), row in by_key.items():
        verify_hashes({row["model_path"]: row["model_sha256"], row["schema_path"]: row["schema_sha256"]})
        if Path(row["schema_path"]).resolve() != Path(row["model_path"]).resolve().with_name("feature_schema.json"):
            raise ValueError("Baseline model/schema paths do not form the saved bundle")
        schema = json.loads(Path(row["schema_path"]).read_text(encoding="utf-8"))
        if (row.get("train_end") != origins[split] or row.get("horizon") != 1
                or schema.get("train_end") != origins[split] or schema.get("country") != country
                or schema.get("target_transform") != "log1p" or schema.get("prediction_inverse") != "expm1"
                or schema.get("parameters") != row.get("parameters")
                or schema.get("training_rows") != row.get("training_rows")):
            raise ValueError(f"Baseline metadata mismatch: {country}/{split}")
        if schema.get("weighting", {}).get("method", "row") != "row":
            raise ValueError("Baseline must use equal row weights")
        expected_parameters = {**selected[country]["parameters"], "device": "cpu", "n_jobs": n_jobs,
                               "random_state": 42, "objective": "reg:squarederror", "tree_method": "hist",
                               "enable_categorical": True, "subsample": 1.0, "colsample_bytree": 1.0,
                               "early_stopping_rounds": None}
        if any(schema["parameters"].get(name) != value for name, value in expected_parameters.items()):
            raise ValueError(f"Baseline selected parameters mismatch: {country}/{split}")
    saved = read_saved_predictions(source / "predictions.csv")
    saved = saved.loc[saved.variant.eq("recursive")].copy()
    if set(saved.split) != {split for split, _ in folds} or set(saved.Country) != set(countries):
        raise ValueError("Baseline full prediction coverage has unexpected folds or markets")
    return {"run_id": summary["run_id"], "records": by_key, "predictions": saved,
            "contract": contract, "source": source}


def protected_hashes(input_path, selection_path, baseline_run):
    """Preserve root implementation/service artifacts and prior experiment sources/results."""
    paths = set(ROOT.glob("*.py")) | {ROOT / "AGENTS.md", ROOT / "pyproject.toml", ROOT / "uv.lock",
                                      Path(input_path), Path(selection_path), ROOT / "raw/vehicle_status.csv"}
    paths.update(path for path in (ROOT / "models").rglob("*") if path.is_file())
    for folder in (ROOT / "experiments/direct_forecast_v1", ROOT / "experiments/log_tuning_v1"):
        paths.update(folder.glob("*.py"))
    source = Path(baseline_run)
    paths.update(path for path in source.rglob("*") if path.is_file())
    manifests = [source / "model_manifest.json", Path(selection_path).with_name("selected_model_manifest.json")]
    refreshes = sorted((ROOT / "reports/model_refresh").glob("run_*/run_summary.json"))
    if refreshes:
        latest = refreshes[-1].parent
        manifests.extend([latest / "xgboost/model_manifest.json", latest / "tft/model_manifest.json"])
    for manifest in manifests:
        if not manifest.is_file():
            continue
        paths.add(manifest)
        for record in json.loads(manifest.read_text(encoding="utf-8")):
            for name, value in record.items():
                if name.endswith("_path") and isinstance(value, str) and Path(value).is_file():
                    paths.add(Path(value))
            paths.update(Path(path) for path in record.get("artifact_hashes", {}) if Path(path).is_file())
    return {str(path.resolve()): file_sha256(path) for path in sorted(paths) if path.is_file()}


def check_training_schema(training, country, schema):
    rows = training.loc[training.Country.eq(country)].sort_values(SERIES_KEY + ["Month"])
    expected = rows[SERIES_KEY].drop_duplicates().values.tolist()
    if schema["training_rows"] != len(rows) or schema["seen_series"] != expected:
        raise ValueError("Reused baseline training rows/series disagree with current valid training frame")
    for column in ("Brand", "Model"):
        if schema["categorical_levels"][column] != sorted(rows[column].unique().tolist()):
            raise ValueError("Reused baseline categories disagree with current training frame")
    for name, value in (("training_target_start", rows.Month.min()), ("training_target_end", rows.Month.max())):
        if schema[name] != str(value):
            raise ValueError("Reused baseline training period disagrees with current frame")


def fit_save_reload(training, country, parameters, origin, split, model_dir, n_jobs,
                    history, fit_number, fit_count, horizon):
    start = time.perf_counter()
    print(f"[{fit_number}/{fit_count}] {origin} {country} series_weight fitting", flush=True)
    model, schema = fit_market(training, country, parameters, origin, n_jobs=n_jobs, weighting="series")
    elapsed = time.perf_counter() - start
    model_path = model_dir / "model.ubj"
    save_market_model(model, schema, model_path)
    reloaded, loaded_schema = load_market_model(model_path)
    before, before_status = forecast_recursive(model, history, origin, horizon, schema)
    after, after_status = forecast_recursive(reloaded, history, origin, horizon, loaded_schema)
    reject_prediction_errors(before_status)
    reject_prediction_errors(after_status)
    verify_predictions_equal(before, after, "saved/reloaded full recursive predictions")
    pd.testing.assert_frame_equal(before_status, after_status)
    if schema != loaded_schema:
        raise ValueError("Reloaded schema differs from saved training metadata")
    schema_path = model_path.with_name("feature_schema.json")
    record = {"variant": "series_weight", "Country": country, "country": country, "split": split,
              "horizon": 1, "forecast_horizon": horizon, "train_end": origin,
              "reused": False, "source_run_id": None, "model_path": str(model_path), "schema_path": str(schema_path),
              "model_sha256": file_sha256(model_path), "schema_sha256": file_sha256(schema_path),
              "training_rows": int(schema["training_rows"]), "training_series": len(schema["seen_series"]),
              "weighting": schema["weighting"], "fit_seconds": elapsed, "reload_check": "passed",
              "reload_horizon": horizon, "parameters": schema["parameters"]}
    print(f"[{fit_number}/{fit_count}] saved; train_rows={len(training.loc[training.Country.eq(country)])} "
          f"fit_seconds={elapsed:.2f}", flush=True)
    return after, after_status, record


def verify_full_coverage(predictions, metrics, universe, folds, horizon):
    splits = pd.DataFrame({"split": [split for split, _ in folds]})
    expected = universe.merge(splits, how="cross").merge(pd.DataFrame({"variant": VARIANTS}), how="cross")
    keys = ["variant", "split"] + SERIES_KEY
    for label, actual, grid, columns in (
            ("metrics", metrics, expected, keys),
            ("predictions", predictions, expected.merge(pd.DataFrame({"horizon": range(1, horizon + 1)}), how="cross"),
             keys + ["horizon"])):
        if actual.duplicated(columns).any():
            raise ValueError(f"Full universe {label} has duplicate keys")
        coverage = grid[columns].merge(actual[columns], on=columns, how="outer", indicator=True, validate="one_to_one")
        if not coverage._merge.eq("both").all():
            raise ValueError(f"Full universe {label} coverage is incomplete")


def run_experiment(input_path, n_jobs=4, *, selection_path=SELECTION, baseline_run=BASELINE_RUN,
                   output_root=HERE / "reports", model_root=HERE / "models", folds=FOLDS, horizon=HORIZON):
    started = time.perf_counter()
    run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {"run_id": run_id, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "actual_fit_count": 0, "reused_model_count": 0, "method": "A", "horizon": horizon}
    try:
        source_hashes = protected_hashes(input_path, selection_path, baseline_run)
        shutil.copyfile(input_path, run_dir / "input.csv")
        shutil.copyfile(selection_path, run_dir / "selected_parameters.json")
        for source, frozen in ((input_path, "input.csv"), (selection_path, "selected_parameters.json")):
            if file_sha256(source) != file_sha256(run_dir / frozen):
                raise ValueError("Input or selected parameters changed while freezing")
        code_paths = list(HERE.glob("*.py")) + [HERE / "AGENTS.md", HERE / "docs/PLAN.md"]
        code_hashes = {str(path.resolve()): file_sha256(path) for path in sorted(code_paths) if path.is_file()}
        data = load_data(run_dir / "input.csv")
        universe = data[SERIES_KEY].drop_duplicates().sort_values(SERIES_KEY).reset_index(drop=True)
        countries = sorted(universe.Country.unique())
        last_actual = max(pd.Period(origin, "M") + horizon for _, origin in folds)
        working = data.loc[data.Month.between(pd.Period("2016-01", "M"), last_actual)].copy()
        selections = json.loads((run_dir / "selected_parameters.json").read_text(encoding="utf-8"))
        selected = {row["Country"]: row for row in selections if row["method"] == "A"}
        if len(selected) != sum(row["method"] == "A" for row in selections):
            raise ValueError("Duplicate A market parameters")
        if any(country not in selected or selected[country]["status"] != "OK" for country in countries):
            raise ValueError("Every market needs existing valid A parameters")
        baseline = validate_baseline_source(baseline_run, file_sha256(run_dir / "input.csv"),
                                            file_sha256(run_dir / "selected_parameters.json"), countries,
                                            folds, horizon, n_jobs, selected)
        training_frames, weight_tables = {}, []
        for split, origin in folds:
            print(f"Preparing fixed training frame: {split} / {origin}", flush=True)
            preparation_start = time.perf_counter()
            training, _ = build_training_frame(working, origin)
            training_frames[split] = training
            print(f"Prepared {split}: {len(training)} rows / {len(training[SERIES_KEY].drop_duplicates())} series; "
                  f"seconds={time.perf_counter() - preparation_start:.2f}", flush=True)
            for country in countries:
                rows = training.loc[training.Country.eq(country)]
                if rows.empty:
                    raise ValueError(f"NO_TRAINING_ROWS: {country}/{split}")
                _, table = build_series_weights(rows)
                table["split"], table["train_end"] = split, origin
                weight_tables.append(table)
                schema = json.loads(Path(baseline["records"][(country, split)]["schema_path"]).read_text(encoding="utf-8"))
                check_training_schema(training, country, schema)
        cohort = build_fixed_cohort(working, universe, training_frames, folds, horizon)
        if not cohort.is_common.any():
            raise ValueError("NO_VALIDATION_SCORE: fixed cohort is empty")
        weights = pd.concat(weight_tables, ignore_index=True)
        write_csv(run_dir / "fixed_cohort.csv", cohort)
        write_csv(run_dir / "training_weights.csv", weights)
        fit_count = len(countries) * len(folds)
        contract = {"run_id": run_id, "input_path": str(Path(input_path).resolve()),
                    "input_sha256": file_sha256(run_dir / "input.csv"), "input_rows": len(data),
                    "selection_source": str(Path(selection_path).resolve()),
                    "selection_sha256": file_sha256(run_dir / "selected_parameters.json"),
                    "baseline_source_run_id": baseline["run_id"], "baseline_source_run": str(baseline["source"]),
                    "baseline_manifest_sha256": file_sha256(baseline["source"] / "model_manifest.json"),
                    "baseline_predictions_sha256": file_sha256(baseline["source"] / "predictions.csv"),
                    "folds": [{"split": split, "origin": origin, "train_end": origin, "horizon": horizon} for split, origin in folds],
                    "variants": list(VARIANTS), "fit_count_expected": fit_count, "reuse_count_expected": fit_count,
                    "countries": countries, "universe_series": len(universe), "cohort_series": int(cohort.is_common.sum()),
                    "evaluation_last_actual": str(last_actual),
                    "ignored_working_rows_after_actual_limit": int((data.Month > last_actual).sum()),
                    "fixed_cohort_sha256": file_sha256(run_dir / "fixed_cohort.csv"),
                    "training_weights_sha256": file_sha256(run_dir / "training_weights.csv"),
                    "n_jobs": n_jobs, "device": "cpu", "seed": 42, "target_transform": "log1p",
                    "comparison_tie_tolerance_WAPE_pp": 1e-9,
                    "weighting_formula": "N / (K * N_i)",
                    "weighting_population": "All valid training rows in each market/fold, independent of evaluation cohort",
                    "cohort_policy": "Existing h1 training rows, recent 12 contiguous observations, all 6 actuals and positive actual sum in all A folds",
                    "selection_policy": "Inherited prior A parameters; no search, test, B, or future fit",
                    "horizon_metric": "For each vehicle and horizon, sum errors and actuals across three validation folds; zero denominators remain undefined",
                    "baseline_replay_tolerance": {"rtol": 0, "atol": 0},
                    "source_hashes": source_hashes, "code_hashes": code_hashes, "versions": runtime_versions()}
        write_json(run_dir / "experiment_contract.json", contract)
        contract_sha = file_sha256(run_dir / "experiment_contract.json")
        summary.update(input_sha256=contract["input_sha256"], input_rows=len(data), universe_series=len(universe),
                       expected_fit_count=fit_count, expected_reused_model_count=fit_count,
                       cohort_series=contract["cohort_series"], baseline_source_run_id=baseline["run_id"])
        write_json(run_dir / "run_summary.json", summary)
        print(f"Run: {run_dir}\nFixed cohort: {contract['cohort_series']}/{len(universe)}; new fits: {fit_count}", flush=True)
        all_predictions, all_metrics, manifest, replay_checks = [], [], [], []
        # All saved baselines are replayed across the full universe before any new fit.
        for split, origin in folds:
            history = working.loc[working.Month <= pd.Period(origin, "M")]
            for country in countries:
                print(f"Replay {split} / {country} from saved row_weight model", flush=True)
                source_record = baseline["records"][(country, split)]
                model, schema = load_market_model(source_record["model_path"])
                country_universe = universe.loc[universe.Country.eq(country)].copy()
                predictions, statuses = forecast_recursive(model, history.loc[history.Country.eq(country)], origin, horizon, schema)
                statuses = complete_statuses(statuses, country_universe)
                monthly, metrics = evaluate_forecast(predictions, statuses, working, country_universe,
                                                     method="A", split=split, train_end=origin, origin=origin, horizon=horizon)
                reject_prediction_errors(metrics)
                previous = baseline["predictions"].loc[baseline["predictions"].Country.eq(country)
                                                        & baseline["predictions"].split.eq(split)]
                verify_baseline_replay(monthly, previous)
                print(f"Replay passed: {country} / {split}; {len(country_universe)} series / {len(monthly)} monthly rows", flush=True)
                replay_checks.append({"Country": country, "split": split, "series": len(country_universe),
                                      "prediction_rows": len(monthly), "horizon": horizon,
                                      "all_predictions_exact_equal": True, "all_statuses_exact_equal": True})
                monthly.insert(0, "variant", "row_weight")
                metrics.insert(0, "variant", "row_weight")
                all_predictions.append(monthly)
                all_metrics.append(metrics)
                manifest.append({**source_record, "variant": "row_weight", "source_variant": "recursive",
                                 "source_run_id": baseline["run_id"], "reused": True, "fit_seconds": 0,
                                 "forecast_horizon": horizon, "baseline_replay_check": "passed"})
                summary["reused_model_count"] = len(manifest)
                write_json(run_dir / "model_manifest.json", manifest)
                write_json(run_dir / "run_summary.json", summary)
        write_csv(run_dir / "baseline_replay_checks.csv", pd.DataFrame(replay_checks))
        summary["baseline_replay_exact_equal"] = True
        write_json(run_dir / "run_summary.json", summary)
        for split, origin in folds:
            history = working.loc[working.Month <= pd.Period(origin, "M")]
            for country in countries:
                country_universe = universe.loc[universe.Country.eq(country)].copy()
                base = Path(model_root).resolve() / run_id / country / split / "series_weight"
                predictions, statuses, record = fit_save_reload(
                    training_frames[split], country, selected[country]["parameters"], origin, split, base, n_jobs,
                    history.loc[history.Country.eq(country)], summary["actual_fit_count"] + 1, fit_count, horizon)
                if record["parameters"] != baseline["records"][(country, split)]["parameters"]:
                    raise ValueError("Weighted fit changed parameters beyond sample weighting")
                manifest.append(record)
                summary["actual_fit_count"] += 1
                statuses = complete_statuses(statuses, country_universe)
                monthly, metrics = evaluate_forecast(predictions, statuses, working, country_universe,
                                                     method="A", split=split, train_end=origin, origin=origin, horizon=horizon)
                reject_prediction_errors(metrics)
                monthly.insert(0, "variant", "series_weight")
                metrics.insert(0, "variant", "series_weight")
                all_predictions.append(monthly)
                all_metrics.append(metrics)
                summary["last_completed_market_fold"] = {"Country": country, "split": split}
                write_json(run_dir / "model_manifest.json", manifest)
                write_json(run_dir / "run_summary.json", summary)
        predictions = pd.concat(all_predictions, ignore_index=True)
        metrics = pd.concat(all_metrics, ignore_index=True)
        verify_full_coverage(predictions, metrics, universe, folds, horizon)
        series, overall, market, fold_comparison = compare_metrics(metrics, cohort, folds)
        horizon_series, horizon_comparison = compare_horizons(predictions, cohort, folds, horizon)
        status_columns = ["variant", "algorithm", "method", "split", "train_end", "origin"] + SERIES_KEY + [
            "expected_months", "observed_months", "status", "prediction_status", "reason", "supplementary_reasons"]
        outputs = {"predictions.csv": predictions, "metrics.csv": metrics, "series_status.csv": metrics[status_columns].copy(),
                   "series_comparison.csv": series, "overall_comparison.csv": overall, "market_comparison.csv": market,
                   "fold_comparison.csv": fold_comparison, "horizon_series_comparison.csv": horizon_series,
                   "horizon_comparison.csv": horizon_comparison}
        for name, table in outputs.items():
            table.insert(0, "run_id", run_id)
            write_csv(run_dir / name, table)
        if summary["actual_fit_count"] != fit_count or summary["reused_model_count"] != fit_count:
            raise ValueError("Unexpected fresh/reused model counts")
        verify_hashes(source_hashes)
        verify_hashes(code_hashes)
        verify_hashes({str(run_dir / "experiment_contract.json"): contract_sha,
                       str(run_dir / "fixed_cohort.csv"): contract["fixed_cohort_sha256"],
                       str(run_dir / "training_weights.csv"): contract["training_weights_sha256"],
                       str(run_dir / "input.csv"): contract["input_sha256"],
                       str(run_dir / "selected_parameters.json"): contract["selection_sha256"]})
        artifact_names = list(outputs) + ["model_manifest.json", "baseline_replay_checks.csv", "experiment_contract.json",
                                          "fixed_cohort.csv", "training_weights.csv"]
        summary.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.perf_counter() - started, prediction_rows=len(predictions), metric_rows=len(metrics),
                       source_hashes_unchanged=True, code_hashes_unchanged=True, reload_checks_passed=fit_count,
                       baseline_replay_exact_equal=True,
                       results=overall.drop(columns="run_id").where(overall.notna(), None).to_dict(orient="records"),
                       artifact_hashes={name: file_sha256(run_dir / name) for name in artifact_names})
        write_json(run_dir / "run_summary.json", summary)
        print(f"Complete: {run_dir}\n{overall.to_string(index=False)}", flush=True)
        return run_dir
    except BaseException as error:
        summary.update(status="failed", completed_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.perf_counter() - started, error=f"{type(error).__name__}: {error}")
        write_json(run_dir / "run_summary.json", summary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare equal-series weighted log XGBoost with saved equal-row models")
    parser.add_argument("--input", type=Path, default=ROOT / "raw/data.csv")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    run_experiment(args.input.resolve(), args.n_jobs)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()


