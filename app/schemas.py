from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str


class DatasetSummary(BaseModel):
    dataset_id: str
    filename: str
    row_count: int
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    inferred_frequency_seconds: int | None = None
    missing_values: int = 0


class ChatCreateResponse(BaseModel):
    thread_id: str


class ChatMessage(BaseModel):
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ChatHistoryResponse(BaseModel):
    thread_id: str
    messages: list[ChatMessage]


class AgentMessageRequest(BaseModel):
    message: str = Field(min_length=1)
    dataset_id: str | None = None
    prediction_id: str | None = None
    model_name: str | None = None
    forecast_start: str | None = None
    forecast_end: str | None = None
    sample_rate_seconds: int = 3600


class ToolResult(BaseModel):
    name: str
    status: Literal["ok", "warning", "error", "unavailable"]
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class AgentTraceStep(BaseModel):
    step: str
    actor: str
    decision: str | None = None
    status: str | None = None
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class AgentMessageResponse(BaseModel):
    thread_id: str
    answer: str
    intent: str | None = None
    tools: list[ToolResult] = Field(default_factory=list)
    agent_trace: list[AgentTraceStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    eval_artifact_path: str | None = None
    run_id: str


class PredictionEvaluationRequest(BaseModel):
    artifact_path: str = Field(min_length=1)
    use_llm: bool = True


class PredictionEvaluationResponse(BaseModel):
    score_total: float
    score_max: float = 100
    prediction_metrics: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, Any] = Field(default_factory=dict)
    evaluator_agent: dict[str, Any] = Field(default_factory=dict)
    evaluation_report_path: str
    source_artifact_path: str


class ModelInfo(BaseModel):
    name: str
    description: str
    available: bool = True
