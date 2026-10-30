"""Fit the frozen additive ETS candidates to one raw monthly sales series."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import re
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


SERIES_KEY = ["Country", "Brand", "Model"]
CANDIDATES = (
    ("ANN", None, False, None),
    ("AAdN", "add", True, None),
    ("ANA", None, False, "add"),
    ("AAdA", "add", True, "add"),
)
MAXITER = 2000


def _month(value):
    if isinstance(value, pd.Period) and value.freqstr == "M":
        return value
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}", value):
        raise ValueError("Month must be YYYY-MM")
    return pd.Period(value, freq="M")


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _versions():
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "statsmodels")}}


def prepare_history(rows, origin, *, key=None):
    origin = _month(origin)
    required = SERIES_KEY + ["Month", "Sales"]
    if any(name not in rows for name in required):
        raise ValueError("Single-series history columns are missing")
    frame = rows[required].copy()
    frame["Month"] = pd.PeriodIndex(frame["Month"], freq="M")
    frame = frame.loc[frame.Month.between(pd.Period("2016-01", "M"), origin)].copy()
    frame = frame.sort_values("Month").reset_index(drop=True)
    if frame.Month.duplicated().any():
        raise ValueError("Duplicate historical month")
    if key is not None and (len(key) != 3 or any(not isinstance(v, str) or not v for v in key)):
        raise ValueError("Invalid series key")
    observed_keys = frame[SERIES_KEY].drop_duplicates().values.tolist()
    if len(observed_keys) > 1:
        raise ValueError("History contains more than one series")
    if observed_keys:
        if any(not isinstance(v, str) or not v for v in observed_keys[0]):
            raise ValueError("Invalid series identifier")
        if key is not None and list(key) != observed_keys[0]:
            raise ValueError("History does not match requested key")
        key = observed_keys[0]
    elif key is None:
        available = rows[SERIES_KEY].drop_duplicates().values.tolist()
        key = available[0] if len(available) == 1 else None
    frame["Sales"] = pd.to_numeric(frame.Sales, errors="raise").astype(float)
    present = frame.Sales.dropna().to_numpy()
    if not np.isfinite(present).all() or (present < 0).any():
        raise ValueError("Historical sales must be finite and nonnegative; blanks remain missing")
    history = {month: float(value) for month, value in zip(frame.Month, frame.Sales)}
    segment = []
    current = origin
    while current in history and np.isfinite(history[current]):
        segment.append((current, history[current]))
        current -= 1
    segment.reverse()
    values = np.asarray([value for _, value in segment], dtype=float)
    unused = frame.loc[frame.Month < segment[0][0]] if segment else frame
    pairs = [[str(month), None if pd.isna(value) else float(value)] for month, value in zip(frame.Month, frame.Sales)]
    metadata = {
        "algorithm": "ETS", "series_key": list(key) if key is not None else None,
        "origin": str(origin), "train_end": str(origin),
        "training_start": str(segment[0][0]) if segment else None,
        "training_end": str(segment[-1][0]) if segment else None,
        "training_rows": len(segment), "latest_segment_months": len(segment),
        "source_rows_before_origin": len(frame),
        "source_start": str(frame.Month.min()) if len(frame) else None,
        "unused_prior_rows": len(unused),
        "unused_prior_start": str(unused.Month.min()) if len(unused) else None,
        "training_history_sha256": _digest(pairs),
        "training_values_sha256": _digest([[str(month), value] for month, value in segment]),
        "hash_encoding": "UTF-8 canonical JSON ordered [YYYY-MM,float-or-null] pairs; compact separators",
        "selected_candidate": None, "selected_aicc": None, "constant_value": None,
        "selection_reason": None, "selected_parameters": {}, "forecast_state": None,
        "fit_attempt_count": 0, "candidate_count": 4 if len(segment) >= 24 else 2 if len(segment) >= 12 else 0,
        "initialization_method": "estimated", "fit_maxiter": MAXITER,
        "forecast_rule": "Native fixed-parameter ETS state forecast; max(0,raw) only at returned output",
        "versions": _versions(),
    }
    if frame.empty or frame.Month.min() > origin - 11:
        status, reason = "INSUFFICIENT_HISTORY", "Fewer than twelve months of observation span through origin"
    elif len(segment) < 12:
        status, reason = "INCOMPLETE_HISTORY", "Recent twelve calendar months include an absent or blank observation"
    else:
        status, reason = "OK", ""
        if np.equal(values, values[0]).all():
            metadata["constant_value"] = float(values[0])
    return {"status": status, "reason": reason, "metadata": metadata, "values": values}


def _candidate_records(nobs):
    records = []
    for order, (candidate, trend, damped, seasonal) in enumerate(CANDIDATES):
        eligible = nobs >= (24 if seasonal else 12)
        records.append({
            "candidate_id": candidate, "order": order, "error": "add", "trend": trend,
            "damped_trend": damped, "seasonal": seasonal, "seasonal_periods": 12 if seasonal else None,
            "eligible": eligible, "status": "PENDING" if eligible else "INELIGIBLE",
            "converged": None, "aicc": None, "aicc_nonfinite": None,
            "selected": False, "reason": "" if eligible else "Requires 24 contiguous months" if seasonal else "Requires 12 contiguous months",
            "warnings": [], "exception": None, "iterations": None, "warnflag": None,
            "function_calls": None, "fit_seconds": 0.0, "nobs": nobs,
            "initialization_method": "estimated", "maxiter": MAXITER,
        })
    return records


def _select_candidate(candidates):
    good = [row for row in candidates if row["eligible"] and row["status"] == "OK"
            and row["converged"] is True and row["aicc"] is not None and np.isfinite(row["aicc"])]
    if not good:
        return None
    chosen = min(good, key=lambda row: (row["aicc"], row["order"]))
    return chosen["candidate_id"]


def _forecast_state(result, record):
    states = np.asarray(result.states, dtype=float)
    parameters = dict(zip(result.model.param_names, np.asarray(result.params, dtype=float)))
    return {
        "level": float(states[-1, 0]),
        "trend": float(states[-1, 1]) if record["trend"] else 0.0,
        "phi": float(parameters["damping_trend"]) if record["damped_trend"] else 1.0,
        "seasonal": states[-12:, -1].tolist() if record["seasonal"] else [],
        "seasonal_periods": 12 if record["seasonal"] else 0,
    }


def fit_series(rows, origin, *, key=None):
    prepared = prepare_history(rows, origin, key=key)
    metadata = prepared["metadata"]
    records = _candidate_records(metadata["training_rows"])
    bundle = {"status": prepared["status"], "reason": prepared["reason"],
              "metadata": metadata, "candidates": records, "result": None}
    if bundle["status"] != "OK":
        return bundle
    values = prepared["values"]
    if metadata["constant_value"] is not None:
        metadata.update(selected_candidate="ANN", selection_reason="CONSTANT_SERIES_ANALYTIC_ANN",
                        forecast_state={"level": metadata["constant_value"], "trend": 0.0,
                                        "phi": 1.0, "seasonal": [], "seasonal_periods": 0})
        for record in records:
            if record["candidate_id"] == "ANN":
                record.update(status="CONSTANT_SERIES", selected=True,
                              reason="Analytic constant-level ANN solution; AICc is undefined and not fabricated")
            elif record["eligible"]:
                record.update(status="SKIPPED_CONSTANT", reason="Exact constant series uses analytic ANN")
        return bundle
    fitted = {}
    for record in records:
        if not record["eligible"]:
            continue
        metadata["fit_attempt_count"] += 1
        started = time.perf_counter()
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = ETSModel(values, error="add", trend=record["trend"],
                                  damped_trend=record["damped_trend"], seasonal=record["seasonal"],
                                  seasonal_periods=record["seasonal_periods"],
                                  initialization_method="estimated").fit(maxiter=MAXITER, full_output=True, disp=False)
                aicc = float(result.aicc)
            details = result.mle_retvals
            record.update(converged=bool(details.get("converged", False)),
                          iterations=int(details["iterations"]) if details.get("iterations") is not None else None,
                          warnflag=int(details["warnflag"]) if details.get("warnflag") is not None else None,
                          function_calls=int(details["fcalls"]) if details.get("fcalls") is not None else None,
                          aicc=aicc if np.isfinite(aicc) else None,
                          aicc_nonfinite=None if np.isfinite(aicc) else str(aicc))
            if not record["converged"]:
                record.update(status="NON_CONVERGED", reason="Optimizer did not report convergence")
            elif not np.isfinite(aicc):
                record.update(status="NONFINITE_AICC", reason="Training AICc is not finite")
            elif not np.isfinite(result.params).all() or not np.isfinite(result.states).all():
                record.update(status="NONFINITE_STATE", reason="Fitted parameters or states are not finite")
            else:
                record.update(status="OK", reason="")
                fitted[record["candidate_id"]] = result
        except Exception as error:
            record.update(status="FIT_ERROR", reason=f"{type(error).__name__}: {error}",
                          exception=type(error).__name__)
        finally:
            record["warnings"] = [{"category": warning.category.__name__, "message": str(warning.message)}
                                  for warning in caught]
            record["fit_seconds"] = time.perf_counter() - started
    selected = _select_candidate(records)
    if selected is None:
        bundle.update(status="PREDICTION_ERROR", reason="No eligible candidate has converged with finite training AICc and states")
        return bundle
    chosen = next(row for row in records if row["candidate_id"] == selected)
    chosen["selected"] = True
    result = fitted[selected]
    metadata.update(selected_candidate=selected, selected_aicc=chosen["aicc"],
                    selection_reason="MINIMUM_TRAINING_AICC",
                    selected_parameters={name: float(value) for name, value in zip(result.model.param_names, result.params)},
                    forecast_state=_forecast_state(result, chosen))
    bundle["result"] = result
    return bundle


def forecast_bundle(bundle, horizon):
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("Forecast horizon must be a positive integer")
    if bundle["status"] != "OK":
        raise ValueError(f"Cannot forecast bundle with status {bundle['status']}")
    constant = bundle["metadata"]["constant_value"]
    if constant is not None:
        raw = np.full(horizon, constant, dtype=float)
    else:
        raw = np.asarray(bundle["result"].forecast(steps=horizon), dtype=float)
    if raw.shape != (horizon,) or not np.isfinite(raw).all():
        raise ValueError("PREDICTION_ERROR: ETS forecast is not finite or has the wrong shape")
    return np.maximum(0.0, raw)


def save_bundle(bundle, path):
    path = Path(path)
    if path.suffix.lower() != ".pkl":
        raise ValueError("ETS bundle path must end in .pkl")
    metadata_path = path.with_suffix(".json")
    if path.exists() or metadata_path.exists():
        raise FileExistsError("Existing ETS model or metadata cannot be overwritten")
    metadata = {field: bundle[field] for field in ("status", "reason", "metadata", "candidates")}
    payload = pickle.dumps({"metadata_sha256": _digest(metadata), "result": bundle["result"]},
                           protocol=pickle.HIGHEST_PROTOCOL)
    model_sha = hashlib.sha256(payload).hexdigest()
    sidecar = _json_bytes({"model_sha256": model_sha, "bundle": metadata}) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for destination, content in ((path, payload), (metadata_path, sidecar)):
            with destination.open("xb") as stream:
                created.append(destination)
                stream.write(content)
    except BaseException:
        for destination in created:
            destination.unlink(missing_ok=True)
        raise
    return {"model_path": str(path.resolve()), "metadata_path": str(metadata_path.resolve()),
            "model_sha256": model_sha, "metadata_sha256": hashlib.sha256(sidecar).hexdigest()}


def load_bundle(path, expected_sha256=None):
    path = Path(path)
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    content = path.read_bytes()
    actual_sha = hashlib.sha256(content).hexdigest()
    if actual_sha != sidecar["model_sha256"] or (expected_sha256 is not None and actual_sha != expected_sha256):
        raise ValueError("ETS pickle checksum mismatch; refusing to deserialize")
    stored = pickle.loads(content)
    metadata = sidecar["bundle"]
    if stored["metadata_sha256"] != _digest(metadata):
        raise ValueError("ETS model and metadata checksum binding mismatch")
    return {**metadata, "result": stored["result"]}
