from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.schemas import AgentMessageRequest
from app.services.evaluation_artifacts import save_manager_prediction_eval_artifact


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("API_KEYS", "test-key")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.delenv("LLM_PROVIDERS_JSON", raising=False)
    get_settings.cache_clear()
    return TestClient(create_app())


def test_health(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_interface_served(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/")

    assert response.status_code == 200
    assert "Time Series Agent" in response.text
    assert "Evaluate Latest" in response.text
    assert "Open Report" in response.text


def test_upload_and_chat_flow(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    headers = {"X-API-Key": "test-key"}
    csv = b"timestamp,value\n2026-01-01 00:00:00,1\n2026-01-01 01:00:00,2\n"

    upload = client.post(
        "/v1/datasets/upload",
        files={"file": ("sample.csv", BytesIO(csv), "text/csv")},
        headers=headers,
    )
    assert upload.status_code == 200
    dataset_id = upload.json()["dataset_id"]

    chat = client.post("/v1/chats", headers=headers)
    assert chat.status_code == 200
    thread_id = chat.json()["thread_id"]

    message = client.post(
        f"/v1/chats/{thread_id}/messages",
        json={
            "message": "Check this data for problems and explain it simply.",
            "dataset_id": dataset_id,
        },
        headers=headers,
    )

    assert message.status_code == 200
    body = message.json()
    assert body["thread_id"] == thread_id
    assert body["answer"]
    assert body["tools"]
    assert body["intent"] == "data_quality"
    assert body["agent_trace"][0]["step"] == "react_agent_start"


def test_general_question_does_not_run_dataset_tools(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    headers = {"X-API-Key": "test-key"}
    chat = client.post("/v1/chats", headers=headers)
    thread_id = chat.json()["thread_id"]

    response = client.post(
        f"/v1/chats/{thread_id}/messages",
        json={
            "message": "What is a time series outlier?",
        },
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "general_question"
    assert body["tools"] == []
    assert "time series" in body["answer"].lower()
    assert "request json" not in body["answer"].lower()
    assert "available tools" not in body["answer"].lower()
    assert "use react style" not in body["answer"].lower()
    assert any(
        step["actor"] == "manager" and step["step"] == "final_response"
        for step in body["agent_trace"]
    )


def test_prediction_evaluation_endpoint(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    headers = {"X-API-Key": "test-key"}
    upload = client.post(
        "/v1/datasets/upload",
        files={
            "file": (
                "sample.csv",
                BytesIO(
                    b"timestamp,value\n"
                    b"2026-01-02 00:00:00,10\n"
                    b"2026-01-02 01:00:00,60\n"
                ),
                "text/csv",
            )
        },
        headers=headers,
    )
    dataset_id = upload.json()["dataset_id"]
    temporal_scope = {
        "mode": "historical_before_requested_range",
        "cutoff_timestamp": "2026-01-02T00:00:00",
        "row_count": 24,
    }
    artifact_path = save_manager_prediction_eval_artifact(
        thread_id="thread-1",
        run_id="run-1",
        request=AgentMessageRequest(
            message="Generate a forecast and analyze reliability.",
            dataset_id=dataset_id,
            model_name="linear_regression_simple",
            forecast_start="2026-01-02 00:00:00",
            forecast_end="2026-01-02 01:00:00",
        ),
        answer="The forecast is mostly reliable, but hour 01:00 has risk based on backtest history.",
        intent="prediction_analysis",
        tools=[
            {"name": "data_consistency", "status": "ok", "summary": "ok", "data": {"temporal_scope": temporal_scope}},
            {"name": "data_anomaly_warnings", "status": "ok", "summary": "ok", "data": {"temporal_scope": temporal_scope}},
            {"name": "historical_summary", "status": "ok", "summary": "ok", "data": {"temporal_scope": temporal_scope}},
            {"name": "hourly_consumption_context", "status": "ok", "summary": "ok", "data": {"temporal_scope": temporal_scope}},
            {"name": "prediction_backtest_context", "status": "ok", "summary": "ok", "data": {"temporal_scope": temporal_scope}},
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
                        {"timestamp": "2026-01-02T00:00:00", "reliability": "high"},
                        {"timestamp": "2026-01-02T01:00:00", "reliability": "low"},
                    ]
                },
            },
        ],
        agent_trace=[],
    )

    response = client.post(
        "/v1/evaluations/prediction-analysis",
        json={"artifact_path": str(artifact_path), "use_llm": False},
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["score_total"] > 0
    assert body["prediction_metrics"]["matched_points"] == 2
    assert body["evaluation_report_path"].endswith("__evaluation.json")



def test_rejects_missing_api_key(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post("/v1/chats")

    assert response.status_code == 401
