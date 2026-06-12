from __future__ import annotations

import json
from io import BytesIO

from app.config import get_settings
from app.schemas import AgentMessageRequest
from app.services.dataset_store import save_uploaded_dataset
from app.services.evaluation_artifacts import save_manager_prediction_eval_artifact


def test_prediction_eval_artifact_is_unique_per_day_dataset_and_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    get_settings.cache_clear()

    csv = (
        b"timestamp,value\n"
        b"2026-01-01 00:00:00,10\n"
        b"2026-01-01 01:00:00,20\n"
    )
    dataset = save_uploaded_dataset(BytesIO(csv), "sample.csv")
    request = AgentMessageRequest(
        message="Generate a forecast and explain reliability.",
        dataset_id=dataset.dataset_id,
        model_name="linear_regression_simple",
        forecast_start="2026-01-01 00:00:00",
        forecast_end="2026-01-01 01:00:00",
        sample_rate_seconds=3600,
    )
    other_model_request = AgentMessageRequest(
        message=request.message,
        dataset_id=dataset.dataset_id,
        model_name="other_model",
        forecast_start=request.forecast_start,
        forecast_end=request.forecast_end,
        sample_rate_seconds=request.sample_rate_seconds,
    )
    tools = [
        {
            "name": "prediction",
            "status": "ok",
            "summary": "Generated two predictions.",
            "data": {
                "timestamps": ["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
                "predicted_values": [11, 19],
            },
        }
    ]

    first_path = save_manager_prediction_eval_artifact(
        thread_id="thread-1",
        run_id="run-1",
        request=request,
        answer="first answer",
        intent="prediction_analysis",
        tools=tools,
        agent_trace=[],
    )
    second_path = save_manager_prediction_eval_artifact(
        thread_id="thread-1",
        run_id="run-2",
        request=request,
        answer="second answer",
        intent="prediction_analysis",
        tools=tools,
        agent_trace=[],
    )
    other_model_path = save_manager_prediction_eval_artifact(
        thread_id="thread-1",
        run_id="run-3",
        request=other_model_request,
        answer="other model answer",
        intent="prediction_analysis",
        tools=tools,
        agent_trace=[],
    )

    assert first_path == second_path
    assert first_path != other_model_path
    assert first_path is not None
    assert other_model_path is not None
    assert first_path.name == f"2026-01-01__{dataset.dataset_id}__linear_regression_simple.json"
    assert other_model_path.name == f"2026-01-01__{dataset.dataset_id}__other_model.json"
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert payload["case_date"] == "2026-01-01"
    assert payload["run_id"] == "run-2"
    assert payload["manager_agent_output"]["answer"] == "second answer"
    assert payload["deterministic_eval_expectations"]["expected_intent"] == "prediction_analysis"
    assert "prediction" in payload["deterministic_eval_expectations"]["required_tools"]
    assert len(payload["held_out_actuals_not_shown_to_agent"]) == 2
    assert len(payload["prediction_values"]) == 2


def test_prediction_eval_artifact_skips_non_prediction_analysis(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    get_settings.cache_clear()

    request = AgentMessageRequest(
        message="How did this model perform?",
        dataset_id="dataset-1",
        model_name="linear_regression_simple",
        forecast_start="2026-01-01 00:00:00",
        forecast_end="2026-01-01 01:00:00",
    )

    path = save_manager_prediction_eval_artifact(
        thread_id="thread-1",
        run_id="run-1",
        request=request,
        answer="performance answer",
        intent="model_performance_analysis",
        tools=[{"name": "model_performance_analysis", "status": "ok", "summary": "ok", "data": {}}],
        agent_trace=[],
    )

    assert path is None
