from __future__ import annotations

from typing import Any


def prediction_interval_tool(prediction_id: str | None = None) -> dict[str, Any]:
    if not prediction_id:
        return {
            "name": "prediction_interval",
            "status": "unavailable",
            "summary": "No prediction interval source was provided for this request.",
            "data": {"configured": False},
        }
    return {
        "name": "prediction_interval",
        "status": "unavailable",
        "summary": (
            "Prediction interval retrieval is defined as a tool interface, "
            "but no real interval provider is connected yet."
        ),
        "data": {"prediction_id": prediction_id, "configured": False},
    }


def analyze_prediction_intervals(intervals: dict[str, Any]) -> dict[str, Any]:
    if intervals.get("status") == "unavailable":
        return intervals
    return {
        "name": "prediction_interval_analysis",
        "status": "ok",
        "summary": "Prediction intervals were retrieved and are ready for explanation.",
        "data": intervals,
    }

