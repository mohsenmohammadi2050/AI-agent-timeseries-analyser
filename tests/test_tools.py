from __future__ import annotations

import pandas as pd

import app.tools.predictions as prediction_tools
from app.tools.time_series import (
    data_anomaly_warning_tool,
    data_consistency_tool,
    historical_summary_tool,
    hourly_consumption_context_tool,
    infer_frequency_seconds,
    outlier_detection_tool,
)
from app.tools.predictions import (
    list_models,
    model_performance_analysis_tool,
    prediction_analysis_tool,
    prediction_backtest_context_tool,
)


def test_infer_frequency_seconds_hourly() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
            "value": [1, 2, 3, 4],
        }
    )

    assert infer_frequency_seconds(df) == 3600


def test_data_consistency_flags_gaps_and_duplicates() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 01:00:00",
                "2026-01-01 01:00:00",
                "2026-01-01 04:00:00",
            ],
            "value": [1, 2, 2, None],
        }
    )

    result = data_consistency_tool(df)

    assert result["status"] == "warning"
    assert result["data"]["duplicate_timestamps"] == 1
    assert result["data"]["missing_values"] == 1
    assert result["data"]["gap_count"] >= 1


def test_outlier_detection_finds_spike() -> None:
    values = [10.0] * 30
    values[15] = 1000.0
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(values), freq="h"),
            "value": values,
        }
    )

    result = outlier_detection_tool(df, window=8)

    assert result["status"] == "warning"
    assert result["data"]["outlier_count"] >= 1


def test_historical_summary_returns_plain_stats() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
            "value": [1, 2, 3, 4],
        }
    )

    result = historical_summary_tool(df)

    assert result["status"] == "ok"
    assert result["data"]["mean"] == 2.5


def test_data_anomaly_warning_tool_reports_exact_timestamps() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="h"),
            "value": [100, 120, -5, 0, 0, 0, 150, 10_000_000],
        }
    )

    result = data_anomaly_warning_tool(df, jump_threshold=0.90, top_percent=0.50)

    assert result["status"] == "warning"
    assert result["data"]["negative_value_count"] == 1
    assert result["data"]["zero_sequence_count"] == 1
    assert result["data"]["jump_candidate_count"] >= 1
    assert result["data"]["negative_values"][0]["timestamp"] == "2026-01-01T02:00:00"


def test_hourly_consumption_context_returns_hour_stats() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=48, freq="h"),
            "value": list(range(48)),
        }
    )

    result = hourly_consumption_context_tool(df, days=30)

    assert result["status"] == "ok"
    assert len(result["data"]["hourly_context"]) == 24
    assert result["data"]["hourly_context"][0]["hour"] == 0


def test_prediction_analysis_uses_hourly_context_for_reliability() -> None:
    prediction = {
        "data": {
            "timestamps": ["2026-01-03 00:00:00", "2026-01-03 01:00:00"],
            "predicted_values": [1000, 12],
        }
    }
    hourly_context = {
        "hourly_context": [
            {"hour": 0, "mean": 10, "min": 5, "max": 20, "std": 2},
            {"hour": 1, "mean": 10, "min": 5, "max": 20, "std": 2},
        ]
    }

    result = prediction_analysis_tool(prediction, hourly_context=hourly_context)

    assert result["status"] == "ok"
    assert result["data"]["low_reliability_count"] == 1
    assert result["data"]["per_hour_reliability"][0]["reliability"] == "low"


def test_public_model_registry_contains_only_three_simple_models() -> None:
    names = {model.name for model in list_models()}

    assert names == {
        "xgboost_simple",
        "linear_regression_simple",
        "knn_regressor_simple",
    }


def test_prediction_backtest_requires_forecast_start() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=48, freq="h"),
            "value": list(range(48)),
        }
    )

    result = prediction_backtest_context_tool("linear_regression_simple", "dataset-1", df)

    assert result["status"] == "unavailable"
    assert "forecast start" in result["summary"].lower()


def test_model_performance_requires_range() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=48, freq="h"),
            "value": list(range(48)),
        }
    )

    result = model_performance_analysis_tool("linear_regression_simple", "dataset-1", df, None, None)

    assert result["status"] == "unavailable"
    assert "start and end" in result["summary"].lower()


def test_model_performance_uses_only_data_before_evaluation_range(monkeypatch) -> None:
    captured: dict[str, pd.DataFrame] = {}
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=96, freq="h"),
            "value": list(range(96)),
        }
    )

    def fake_execute_model(model_name, df, forecast_start, forecast_end, sample_rate_seconds):
        captured["df"] = df
        return {
            "timestamps": [item.isoformat() for item in pd.date_range(forecast_start, forecast_end, freq="h")],
            "predicted_values": [10.0] * 25,
            "features": [],
        }

    monkeypatch.setattr(prediction_tools, "_execute_model", fake_execute_model)

    result = model_performance_analysis_tool(
        "linear_regression_simple",
        "dataset-1",
        df,
        "2026-01-03 00:00:00",
        "2026-01-04 00:00:00",
    )

    assert result["status"] == "ok"
    assert captured["df"]["timestamp"].max() < pd.Timestamp("2026-01-03 00:00:00")
    assert result["data"]["metrics"]["matched_points"] == 25
    assert result["data"]["temporal_scope"]["cutoff_timestamp"] == "2026-01-03T00:00:00"


def test_prediction_backtest_uses_only_data_before_previous_day(monkeypatch) -> None:
    captured: dict[str, pd.DataFrame] = {}
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=96, freq="h"),
            "value": list(range(96)),
        }
    )

    def fake_execute_model(model_name, df, forecast_start, forecast_end, sample_rate_seconds):
        captured["df"] = df
        return {
            "timestamps": [item.isoformat() for item in pd.date_range(forecast_start, forecast_end, freq="h")],
            "predicted_values": [10.0] * 24,
            "features": [],
        }

    monkeypatch.setattr(prediction_tools, "_execute_model", fake_execute_model)

    result = prediction_backtest_context_tool(
        "linear_regression_simple",
        "dataset-1",
        df,
        forecast_start="2026-01-04 00:00:00",
    )

    assert result["status"] == "ok"
    assert captured["df"]["timestamp"].max() < pd.Timestamp("2026-01-03 00:00:00")
    assert result["data"]["metrics"]["matched_points"] == 24
    assert result["data"]["temporal_scope"]["cutoff_timestamp"] == "2026-01-03T00:00:00"
