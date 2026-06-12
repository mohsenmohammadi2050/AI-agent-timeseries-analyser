from __future__ import annotations

import json
import uuid
from typing import Any

import numpy as np
import pandas as pd

from app.database import get_connection
from app.ml_models.simple_forecasters import MODEL_SPECS, is_model_available, run_simple_forecast
from app.schemas import ModelInfo
from app.services.chat_store import utc_now


MODEL_REGISTRY: dict[str, str] = {
    name: spec.description for name, spec in MODEL_SPECS.items()
}


def _execute_model(
    model_name: str,
    df: pd.DataFrame,
    forecast_start: str,
    forecast_end: str,
    sample_rate_seconds: int,
) -> dict[str, Any]:
    return run_simple_forecast(
        model_name=model_name,
        df=df,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        sample_rate_seconds=sample_rate_seconds,
    )


def _clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[["timestamp", "value"]].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    return clean.dropna(subset=["timestamp", "value"]).sort_values("timestamp").reset_index(drop=True)


def _historical_before(clean: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return clean[clean["timestamp"] < cutoff].copy().reset_index(drop=True)


def _training_scope_payload(training_df: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, Any]:
    return {
        "mode": "historical_before_requested_range",
        "cutoff_timestamp": cutoff.isoformat(),
        "training_row_count": int(len(training_df)),
        "training_start_timestamp": (
            None if training_df.empty else training_df["timestamp"].min().isoformat()
        ),
        "training_end_timestamp": (
            None if training_df.empty else training_df["timestamp"].max().isoformat()
        ),
    }


def list_models() -> list[ModelInfo]:
    models: list[ModelInfo] = []
    for name, description in MODEL_REGISTRY.items():
        models.append(ModelInfo(name=name, description=description, available=is_model_available(name)))
    return models


def run_prediction_tool(
    model_name: str,
    dataset_id: str,
    df: pd.DataFrame,
    forecast_start: str,
    forecast_end: str,
    sample_rate_seconds: int = 3600,
) -> dict[str, Any]:
    if model_name not in MODEL_REGISTRY:
        return {
            "name": "prediction",
            "status": "error",
            "summary": f"Unknown model: {model_name}.",
            "data": {"available_models": list(MODEL_REGISTRY)},
        }

    forecast_start_ts = pd.to_datetime(forecast_start, errors="coerce")
    if pd.isna(forecast_start_ts):
        return {
            "name": "prediction",
            "status": "unavailable",
            "summary": "Forecast start could not be parsed, so prediction could not be generated.",
            "data": {"forecast_start": forecast_start},
        }

    clean = _clean_dataset(df)
    training_df = _historical_before(clean, forecast_start_ts)
    if training_df.empty:
        return {
            "name": "prediction",
            "status": "unavailable",
            "summary": "There is no historical data before the requested prediction range.",
            "data": {
                "model_name": model_name,
                "temporal_scope": _training_scope_payload(training_df, forecast_start_ts),
            },
        }

    try:
        normalized = _execute_model(
            model_name=model_name,
            df=training_df,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            sample_rate_seconds=sample_rate_seconds,
        )
    except Exception as exc:
        return {
            "name": "prediction",
            "status": "error",
            "summary": f"The prediction model failed: {exc}",
            "data": {"model_name": model_name},
        }

    prediction_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                id, dataset_id, model_name, start_timestamp, end_timestamp, output_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                dataset_id,
                model_name,
                forecast_start,
                forecast_end,
                json.dumps(normalized, default=str),
                utc_now(),
            ),
        )

    return {
        "name": "prediction",
        "status": "ok",
        "summary": f"Generated {len(normalized['predicted_values'])} predicted values with {model_name}.",
        "data": {
            "prediction_id": prediction_id,
            "temporal_scope": _training_scope_payload(training_df, forecast_start_ts),
            **normalized,
        },
    }


def _prediction_error_rows(
    df: pd.DataFrame,
    normalized_prediction: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pred = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(normalized_prediction.get("timestamps", []), errors="coerce"),
            "predicted": pd.to_numeric(
                pd.Series(normalized_prediction.get("predicted_values", [])),
                errors="coerce",
            ),
        }
    ).dropna(subset=["timestamp", "predicted"])
    obs = df[["timestamp", "value"]].copy()
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], errors="coerce")
    obs["observed"] = pd.to_numeric(obs["value"], errors="coerce")
    obs = obs.dropna(subset=["timestamp", "observed"])[["timestamp", "observed"]]
    merged = pred.merge(obs, on="timestamp", how="inner")
    if merged.empty:
        return [], {"matched_points": 0}

    merged["error"] = merged["observed"] - merged["predicted"]
    merged["absolute_error"] = merged["error"].abs()
    merged["absolute_percent_error"] = np.where(
        merged["observed"] != 0,
        merged["absolute_error"] / merged["observed"].abs() * 100,
        np.nan,
    )
    rows = [
        {
            "timestamp": row["timestamp"].isoformat(),
            "predicted": float(row["predicted"]),
            "observed": float(row["observed"]),
            "error": float(row["error"]),
            "absolute_error": float(row["absolute_error"]),
            "absolute_percent_error": (
                None
                if pd.isna(row["absolute_percent_error"])
                else float(row["absolute_percent_error"])
            ),
        }
        for _, row in merged.iterrows()
    ]
    ape = merged["absolute_percent_error"].dropna()
    metrics = {
        "matched_points": int(len(merged)),
        "mae": float(merged["absolute_error"].mean()),
        "max_absolute_error": float(merged["absolute_error"].max()),
        "mape": None if ape.empty else float(ape.mean()),
        "p90_absolute_error": float(merged["absolute_error"].quantile(0.9)),
    }
    return rows, metrics


def prediction_backtest_context_tool(
    model_name: str | None,
    dataset_id: str,
    df: pd.DataFrame,
    forecast_start: str | None = None,
    sample_rate_seconds: int = 3600,
) -> dict[str, Any]:
    if not model_name:
        return {
            "name": "prediction_backtest_context",
            "status": "unavailable",
            "summary": "No model name was provided, so previous prediction reliability could not be evaluated.",
            "data": {},
        }
    if model_name not in MODEL_REGISTRY:
        return {
            "name": "prediction_backtest_context",
            "status": "error",
            "summary": f"Unknown model for backtest context: {model_name}.",
            "data": {"available_models": list(MODEL_REGISTRY)},
        }
    if not forecast_start:
        return {
            "name": "prediction_backtest_context",
            "status": "unavailable",
            "summary": "No forecast start was provided, so previous-day model performance could not be evaluated.",
            "data": {},
        }

    clean = _clean_dataset(df)
    if clean.empty:
        return {
            "name": "prediction_backtest_context",
            "status": "unavailable",
            "summary": "There is no usable observed data for prediction backtesting.",
            "data": {},
        }

    forecast_start_ts = pd.to_datetime(forecast_start, errors="coerce")
    if pd.isna(forecast_start_ts):
        return {
            "name": "prediction_backtest_context",
            "status": "unavailable",
            "summary": "Forecast start could not be parsed, so previous-day model performance could not be evaluated.",
            "data": {"forecast_start": forecast_start},
        }

    end_ts = forecast_start_ts - pd.Timedelta(seconds=sample_rate_seconds)
    start_ts = forecast_start_ts - pd.Timedelta(days=1)
    observed_previous_day = clean[
        (clean["timestamp"] >= start_ts) & (clean["timestamp"] <= end_ts)
    ]
    if observed_previous_day.empty:
        return {
            "name": "prediction_backtest_context",
            "status": "unavailable",
            "summary": "There are no observed values for the day before the requested forecast start.",
            "data": {
                "forecast_start": forecast_start_ts.isoformat(),
                "previous_day_start": start_ts.isoformat(),
                "previous_day_end": end_ts.isoformat(),
                "earliest_timestamp": clean["timestamp"].min().isoformat(),
                "latest_timestamp": clean["timestamp"].max().isoformat(),
            },
        }

    training_df = _historical_before(clean, start_ts)
    if training_df.empty:
        return {
            "name": "prediction_backtest_context",
            "status": "unavailable",
            "summary": "There is no historical data before the previous-day backtest window.",
            "data": {
                "forecast_start": forecast_start_ts.isoformat(),
                "previous_day_start": start_ts.isoformat(),
                "previous_day_end": end_ts.isoformat(),
                "temporal_scope": _training_scope_payload(training_df, start_ts),
            },
        }

    start_text = start_ts.isoformat()
    end_text = end_ts.isoformat()
    try:
        prediction = _execute_model(
            model_name=model_name,
            df=training_df,
            forecast_start=start_text,
            forecast_end=end_text,
            sample_rate_seconds=sample_rate_seconds,
        )
    except Exception as exc:
        return {
            "name": "prediction_backtest_context",
            "status": "error",
            "summary": f"Could not build prediction backtest context: {exc}",
            "data": {"model_name": model_name},
        }

    rows, metrics = _prediction_error_rows(clean, prediction)
    payload = {
        "model_name": model_name,
        "dataset_id": dataset_id,
        "evaluation_type": "dynamic_previous_day",
        "forecast_start": forecast_start_ts.isoformat(),
        "start_timestamp": start_text,
        "end_timestamp": end_text,
        "sample_rate_seconds": sample_rate_seconds,
        "temporal_scope": _training_scope_payload(training_df, start_ts),
        "metrics": metrics,
        "hourly_errors": _hourly_error_context(rows),
        "examples": rows[:100],
    }
    status = "ok" if metrics.get("matched_points", 0) else "unavailable"
    summary = f"Built dynamic previous-day prediction reliability context for {model_name}."
    if status == "unavailable":
        summary = "The backtest ran, but no prediction timestamps matched observed values."
    return {
        "name": "prediction_backtest_context",
        "status": status,
        "summary": summary,
        "data": payload,
    }


def model_performance_analysis_tool(
    model_name: str | None,
    dataset_id: str,
    df: pd.DataFrame,
    evaluation_start: str | None,
    evaluation_end: str | None,
    sample_rate_seconds: int = 3600,
) -> dict[str, Any]:
    if not model_name:
        return {
            "name": "model_performance_analysis",
            "status": "unavailable",
            "summary": "No model name was provided, so model performance could not be evaluated.",
            "data": {},
        }
    if model_name not in MODEL_REGISTRY:
        return {
            "name": "model_performance_analysis",
            "status": "error",
            "summary": f"Unknown model for performance analysis: {model_name}.",
            "data": {"available_models": list(MODEL_REGISTRY)},
        }
    if not evaluation_start or not evaluation_end:
        return {
            "name": "model_performance_analysis",
            "status": "unavailable",
            "summary": "Evaluation start and end are required to evaluate model performance.",
            "data": {},
        }

    start_ts = pd.to_datetime(evaluation_start, errors="coerce")
    end_ts = pd.to_datetime(evaluation_end, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return {
            "name": "model_performance_analysis",
            "status": "unavailable",
            "summary": "Evaluation start or end could not be parsed.",
            "data": {"evaluation_start": evaluation_start, "evaluation_end": evaluation_end},
        }

    clean = _clean_dataset(df)
    observed_range = clean[(clean["timestamp"] >= start_ts) & (clean["timestamp"] <= end_ts)]
    if observed_range.empty:
        return {
            "name": "model_performance_analysis",
            "status": "unavailable",
            "summary": "No actual observed values were found for the requested evaluation range.",
            "data": {
                "evaluation_start": start_ts.isoformat(),
                "evaluation_end": end_ts.isoformat(),
            },
        }

    training_df = _historical_before(clean, start_ts)
    if training_df.empty:
        return {
            "name": "model_performance_analysis",
            "status": "unavailable",
            "summary": "There is no historical data before the requested evaluation range.",
            "data": {
                "model_name": model_name,
                "dataset_id": dataset_id,
                "evaluation_start": start_ts.isoformat(),
                "evaluation_end": end_ts.isoformat(),
                "temporal_scope": _training_scope_payload(training_df, start_ts),
            },
        }

    try:
        prediction = _execute_model(
            model_name=model_name,
            df=training_df,
            forecast_start=start_ts.isoformat(),
            forecast_end=end_ts.isoformat(),
            sample_rate_seconds=sample_rate_seconds,
        )
    except Exception as exc:
        return {
            "name": "model_performance_analysis",
            "status": "error",
            "summary": f"Could not evaluate model performance: {exc}",
            "data": {"model_name": model_name},
        }

    rows, metrics = _prediction_error_rows(clean, prediction)
    status = "ok" if metrics.get("matched_points", 0) else "unavailable"
    summary = "The model performance was evaluated against actual values in the requested range."
    if status == "ok":
        mape = metrics.get("mape")
        mape_text = "unknown" if mape is None else f"{mape:.2f}%"
        summary = (
            f"Evaluated {metrics['matched_points']} points. "
            f"MAE is {metrics['mae']:.2f} and MAPE is {mape_text}."
        )
    else:
        summary = "The model ran, but no prediction timestamps matched actual observed values."

    return {
        "name": "model_performance_analysis",
        "status": status,
        "summary": summary,
        "data": {
            "model_name": model_name,
            "dataset_id": dataset_id,
            "evaluation_start": start_ts.isoformat(),
            "evaluation_end": end_ts.isoformat(),
            "sample_rate_seconds": sample_rate_seconds,
            "temporal_scope": _training_scope_payload(training_df, start_ts),
            "metrics": metrics,
            "hourly_errors": _hourly_error_context(rows),
            "examples": rows[:100],
        },
    }


def _hourly_error_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["hour"] = frame["timestamp"].dt.hour
    grouped = frame.groupby("hour")["absolute_error"].agg(["count", "mean", "max"]).fillna(0)
    return [
        {
            "hour": int(hour),
            "count": int(row["count"]),
            "mean_absolute_error": float(row["mean"]),
            "max_absolute_error": float(row["max"]),
        }
        for hour, row in grouped.sort_index().iterrows()
    ]


def prediction_analysis_tool(
    prediction: dict[str, Any],
    hourly_context: dict[str, Any] | None = None,
    backtest_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values = prediction.get("data", {}).get("predicted_values", [])
    if not values:
        return {
            "name": "prediction_analysis",
            "status": "unavailable",
            "summary": "There are no prediction values to analyze.",
            "data": {},
        }

    series = pd.Series(values, dtype="float64")
    changes = series.diff().abs().dropna()
    largest_change = float(changes.max()) if not changes.empty else 0.0
    reliability_points = _prediction_reliability_points(
        prediction=prediction,
        hourly_context=hourly_context,
        backtest_context=backtest_context,
    )
    low_reliability_count = sum(1 for item in reliability_points if item["reliability"] == "low")
    medium_reliability_count = sum(1 for item in reliability_points if item["reliability"] == "medium")
    summary = (
        f"The prediction average is {series.mean():.2f}. "
        f"The largest step-to-step change is {largest_change:.2f}. "
        f"{low_reliability_count} predicted hours look low reliability and "
        f"{medium_reliability_count} look medium reliability based on recent hourly history."
    )
    return {
        "name": "prediction_analysis",
        "status": "ok",
        "summary": summary,
        "data": {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "min": float(series.min()),
            "max": float(series.max()),
            "largest_step_change": largest_change,
            "low_reliability_count": low_reliability_count,
            "medium_reliability_count": medium_reliability_count,
            "per_hour_reliability": reliability_points[:200],
        },
    }


def _prediction_reliability_points(
    prediction: dict[str, Any],
    hourly_context: dict[str, Any] | None,
    backtest_context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    timestamps = pd.to_datetime(prediction.get("data", {}).get("timestamps", []), errors="coerce")
    values = pd.to_numeric(pd.Series(prediction.get("data", {}).get("predicted_values", [])), errors="coerce")
    if len(timestamps) == 0 or values.empty:
        return []

    hourly_lookup = {
        int(item["hour"]): item
        for item in (hourly_context or {}).get("hourly_context", [])
        if "hour" in item
    }
    backtest_lookup = {
        int(item["hour"]): item
        for item in (backtest_context or {}).get("hourly_errors", [])
        if "hour" in item
    }

    points: list[dict[str, Any]] = []
    for ts, predicted in zip(timestamps, values):
        if pd.isna(ts) or pd.isna(predicted):
            continue
        hour = int(ts.hour)
        history = hourly_lookup.get(hour, {})
        backtest = backtest_lookup.get(hour, {})
        reasons: list[str] = []
        reliability = "high"

        hist_min = history.get("min")
        hist_max = history.get("max")
        hist_mean = history.get("mean")
        hist_std = history.get("std") or 0

        if hist_min is not None and predicted < hist_min:
            reliability = "low"
            reasons.append("prediction is below the recent observed range for this hour")
        if hist_max is not None and predicted > hist_max:
            reliability = "low"
            reasons.append("prediction is above the recent observed range for this hour")
        if hist_mean is not None and hist_std and hist_std > 0:
            distance = abs(float(predicted) - float(hist_mean)) / float(hist_std)
            if distance > 3:
                reliability = "low"
                reasons.append("prediction is more than 3 standard deviations from recent hourly average")
            elif distance > 2 and reliability != "low":
                reliability = "medium"
                reasons.append("prediction is more than 2 standard deviations from recent hourly average")

        mean_abs_error = backtest.get("mean_absolute_error")
        if mean_abs_error is not None and hist_mean not in (None, 0):
            relative_error = float(mean_abs_error) / abs(float(hist_mean))
            if relative_error > 0.3:
                reliability = "low"
                reasons.append("recent backtest error for this hour is high")
            elif relative_error > 0.15 and reliability != "low":
                reliability = "medium"
                reasons.append("recent backtest error for this hour is moderate")

        if not reasons:
            reasons.append("prediction is close to recent hourly behavior")

        points.append(
            {
                "timestamp": ts.isoformat(),
                "hour": hour,
                "predicted": float(predicted),
                "reliability": reliability,
                "reasons": reasons,
                "recent_hour_mean": None if hist_mean is None else float(hist_mean),
                "recent_hour_min": None if hist_min is None else float(hist_min),
                "recent_hour_max": None if hist_max is None else float(hist_max),
                "recent_hour_std": None if hist_std is None else float(hist_std),
                "recent_backtest_mean_absolute_error": (
                    None if mean_abs_error is None else float(mean_abs_error)
                ),
            }
        )
    return points
