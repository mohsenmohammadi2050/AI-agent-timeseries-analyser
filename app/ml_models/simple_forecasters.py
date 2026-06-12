from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


@dataclass(frozen=True)
class ForecastModelSpec:
    name: str
    label: str
    description: str
    estimator: str


MODEL_SPECS: dict[str, ForecastModelSpec] = {
    "xgboost_simple": ForecastModelSpec(
        name="xgboost_simple",
        label="Simple XGBoost",
        description="Small XGBoost regressor using time-based features.",
        estimator="xgboost",
    ),
    "linear_regression_simple": ForecastModelSpec(
        name="linear_regression_simple",
        label="Simple Linear Regression",
        description="Linear regression baseline using trend and calendar features.",
        estimator="linear_regression",
    ),
    "knn_regressor_simple": ForecastModelSpec(
        name="knn_regressor_simple",
        label="Simple KNN Regressor",
        description="K-nearest-neighbors regressor using trend and calendar features.",
        estimator="knn_regressor",
    ),
}


def is_model_available(model_name: str) -> bool:
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        return False
    if spec.estimator != "xgboost":
        return True
    try:
        import xgboost  # noqa: F401
    except Exception:
        return False
    return True


def run_simple_forecast(
    model_name: str,
    df: pd.DataFrame,
    forecast_start: str,
    forecast_end: str,
    sample_rate_seconds: int,
) -> dict[str, Any]:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_name}")

    clean = df[["timestamp", "value"]].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean = clean.dropna(subset=["timestamp", "value"]).sort_values("timestamp").reset_index(drop=True)
    if clean.empty:
        raise ValueError("The model needs at least one historical value.")

    start = pd.to_datetime(forecast_start, errors="coerce")
    end = pd.to_datetime(forecast_end, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Forecast start or end could not be parsed.")
    if end < start:
        raise ValueError("Forecast end must be after forecast start.")

    freq = pd.Timedelta(seconds=sample_rate_seconds)
    future_timestamps = pd.date_range(start=start, end=end, freq=freq)
    if future_timestamps.empty:
        raise ValueError("Forecast range produced no timestamps.")

    origin = clean["timestamp"].min()
    feature_names = [
        "elapsed_hours",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
    ]
    train_x = _time_features(clean["timestamp"], origin)
    train_y = clean["value"].to_numpy(dtype=float)
    future_x = _time_features(pd.Series(future_timestamps), origin)

    if len(clean) < 2:
        predictions = np.repeat(float(train_y[-1]), len(future_timestamps))
    else:
        estimator = _build_estimator(model_name, train_size=len(clean))
        estimator.fit(train_x, train_y)
        predictions = estimator.predict(future_x)

    predictions = np.asarray(predictions, dtype=float)
    predictions = np.maximum(predictions, 0)
    return {
        "timestamps": [item.isoformat() for item in future_timestamps],
        "predicted_values": [float(value) for value in predictions],
        "features": feature_names,
        "model_family": MODEL_SPECS[model_name].label,
        "training_rows": int(len(clean)),
    }


def _time_features(timestamps: pd.Series | pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    ts = pd.Series(pd.to_datetime(timestamps, errors="coerce"))
    elapsed_hours = (ts - origin).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    hour = ts.dt.hour.to_numpy(dtype=float)
    day_of_week = ts.dt.dayofweek.to_numpy(dtype=float)
    is_weekend = (day_of_week >= 5).astype(float)
    return np.column_stack(
        [
            elapsed_hours,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
            is_weekend,
        ]
    )


def _build_estimator(model_name: str, train_size: int):
    if model_name == "linear_regression_simple":
        return LinearRegression()
    if model_name == "knn_regressor_simple":
        return KNeighborsRegressor(n_neighbors=max(1, min(8, train_size)))
    if model_name == "xgboost_simple":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model: {model_name}")
