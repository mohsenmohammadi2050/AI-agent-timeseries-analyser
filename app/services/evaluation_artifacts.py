from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import get_settings
from app.schemas import AgentMessageRequest
from app.services.dataset_store import load_dataset_frame
from app.services.chat_store import utc_now


def _artifact_dir() -> Path:
    settings = get_settings()
    path = settings.database_path.parent / "manager_prediction_evals"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tool_data(tools: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for tool in tools:
        if tool.get("name") == name:
            return tool.get("data", {})
    return {}


def _tool_was_used(tools: list[dict[str, Any]], name: str) -> bool:
    return any(tool.get("name") == name for tool in tools)


def _safe_filename_part(value: str, max_length: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return (cleaned or "unknown")[:max_length]


def _prediction_analysis_expectations() -> dict[str, Any]:
    required_tools = [
        "data_consistency",
        "data_anomaly_warnings",
        "historical_summary",
        "hourly_consumption_context",
        "prediction_backtest_context",
        "prediction",
        "prediction_analysis",
    ]
    optional_tools = ["prediction_interval"]
    return {
        "expected_intent": "prediction_analysis",
        "required_tools": required_tools,
        "optional_tools": optional_tools,
        "scoring_hint": (
            "For deterministic scoring, require expected_intent and every required tool. "
            "Do not fail only because optional tools are absent."
        ),
    }


def _actual_values_for_range(
    dataset_id: str,
    forecast_start: str,
    forecast_end: str,
) -> list[dict[str, Any]]:
    df = load_dataset_frame(dataset_id)
    start = pd.to_datetime(forecast_start, errors="coerce")
    end = pd.to_datetime(forecast_end, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return []
    actuals = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    return [
        {
            "timestamp": row["timestamp"].isoformat(),
            "actual_value": None if pd.isna(row["value"]) else float(row["value"]),
        }
        for _, row in actuals.iterrows()
    ]


def save_manager_prediction_eval_artifact(
    *,
    thread_id: str,
    run_id: str,
    request: AgentMessageRequest,
    answer: str,
    intent: str | None,
    tools: list[dict[str, Any]],
    agent_trace: list[dict[str, Any]],
) -> Path | None:
    if not all(
        [
            request.dataset_id,
            request.model_name,
            request.forecast_start,
            request.forecast_end,
        ]
    ):
        return None

    start = pd.to_datetime(request.forecast_start, errors="coerce")
    if pd.isna(start):
        return None
    if intent != "prediction_analysis" and not _tool_was_used(tools, "prediction"):
        return None

    prediction_data = _tool_data(tools, "prediction")
    prediction_values = [
        {
            "timestamp": str(timestamp),
            "predicted_value": None if pd.isna(value) else float(value),
        }
        for timestamp, value in zip(
            prediction_data.get("timestamps", []),
            prediction_data.get("predicted_values", []),
        )
    ]

    payload = {
        "artifact_type": "manager_prediction_analysis_eval_case",
        "case_date": start.date().isoformat(),
        "saved_at": utc_now(),
        "thread_id": thread_id,
        "run_id": run_id,
        "request": {
            "message": request.message,
            "dataset_id": request.dataset_id,
            "model_name": request.model_name,
            "forecast_start": request.forecast_start,
            "forecast_end": request.forecast_end,
            "sample_rate_seconds": request.sample_rate_seconds,
        },
        "manager_agent_output": {
            "intent": intent,
            "answer": answer,
            "tools": tools,
            "agent_trace": agent_trace,
        },
        "deterministic_eval_expectations": _prediction_analysis_expectations(),
        "held_out_actuals_not_shown_to_agent": _actual_values_for_range(
            request.dataset_id,
            request.forecast_start,
            request.forecast_end,
        ),
        "prediction_values": prediction_values,
        "notes": (
            "Actual values are collected after the manager agent answer is produced. "
            "They are for later evaluator-agent scoring and are not provided to the manager agent."
        ),
    }

    date_part = start.date().isoformat()
    dataset_part = _safe_filename_part(request.dataset_id)
    model_part = _safe_filename_part(request.model_name)
    path = _artifact_dir() / f"{date_part}__{dataset_part}__{model_part}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
