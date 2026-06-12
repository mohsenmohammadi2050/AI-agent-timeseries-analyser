from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from app.config import get_settings
from app.schemas import AgentMessageRequest
from app.services.dataset_store import save_uploaded_dataset
from app.services.evaluation_artifacts import save_manager_prediction_eval_artifact
from app.services.prediction_evaluator import evaluate_prediction_analysis_artifact


def _sample_prediction_tools() -> list[dict]:
    temporal_scope = {
        "mode": "historical_before_requested_range",
        "cutoff_timestamp": "2026-01-02T00:00:00",
        "row_count": 24,
    }
    return [
        {
            "name": "data_consistency",
            "status": "ok",
            "summary": "No major data consistency problems were found.",
            "data": {"temporal_scope": temporal_scope},
        },
        {
            "name": "data_anomaly_warnings",
            "status": "ok",
            "summary": "No large jumps were found.",
            "data": {"temporal_scope": temporal_scope},
        },
        {
            "name": "historical_summary",
            "status": "ok",
            "summary": "Historical average is stable.",
            "data": {"temporal_scope": temporal_scope},
        },
        {
            "name": "hourly_consumption_context",
            "status": "ok",
            "summary": "Built hourly consumption context.",
            "data": {"temporal_scope": temporal_scope},
        },
        {
            "name": "prediction_backtest_context",
            "status": "ok",
            "summary": "Built previous-day context.",
            "data": {"temporal_scope": temporal_scope},
        },
        {
            "name": "prediction",
            "status": "ok",
            "summary": "Generated two predictions.",
            "data": {
                "temporal_scope": temporal_scope,
                "timestamps": ["2026-01-02 00:00:00", "2026-01-02 01:00:00"],
                "predicted_values": [10, 30],
            },
        },
        {
            "name": "prediction_analysis",
            "status": "ok",
            "summary": "One hour looks risky.",
            "data": {
                "per_hour_reliability": [
                    {
                        "timestamp": "2026-01-02T00:00:00",
                        "reliability": "high",
                    },
                    {
                        "timestamp": "2026-01-02T01:00:00",
                        "reliability": "low",
                    },
                ]
            },
        },
    ]


def test_evaluate_prediction_analysis_artifact_saves_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.delenv("LLM_PROVIDERS_JSON", raising=False)
    get_settings.cache_clear()

    dataset = save_uploaded_dataset(
        BytesIO(
            b"timestamp,value\n"
            b"2026-01-02 00:00:00,10\n"
            b"2026-01-02 01:00:00,60\n"
        ),
        "sample.csv",
    )
    request = AgentMessageRequest(
        message="Generate a forecast and analyze reliability.",
        dataset_id=dataset.dataset_id,
        model_name="linear_regression_simple",
        forecast_start="2026-01-02 00:00:00",
        forecast_end="2026-01-02 01:00:00",
    )
    artifact_path = save_manager_prediction_eval_artifact(
        thread_id="thread-1",
        run_id="run-1",
        request=request,
        answer="The forecast is mostly reliable, but hour 01:00 has risk based on backtest history.",
        intent="prediction_analysis",
        tools=_sample_prediction_tools(),
        agent_trace=[],
    )

    report = evaluate_prediction_analysis_artifact(str(artifact_path), use_llm=False)

    assert report["score_total"] > 0
    assert report["prediction_metrics"]["matched_points"] == 2
    assert report["prediction_metrics"]["high_error_count"] == 1
    assert report["scores"]["tool_use_and_intent"]["missing_required_tools"] == []
    assert report["scores"]["temporal_fairness"]["score"] == 10
    report_path = report["evaluation_report_path"]
    saved = json.loads(Path(report_path).read_text(encoding="utf-8"))
    assert saved["source_artifact_path"] == str(artifact_path)
