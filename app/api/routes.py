from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.agents.workflow import run_agent
from app.compat import model_dump
from app.config import get_settings
from app.database import init_db
from app.schemas import (
    AgentMessageRequest,
    AgentMessageResponse,
    AgentTraceStep,
    ChatCreateResponse,
    ChatHistoryResponse,
    DatasetSummary,
    HealthResponse,
    ModelInfo,
    PredictionEvaluationRequest,
    PredictionEvaluationResponse,
    ToolResult,
)
from app.security import require_api_key
from app.services.chat_store import add_agent_run, add_message, create_chat, get_messages
from app.services.dataset_store import get_dataset, save_uploaded_dataset
from app.services.evaluation_artifacts import save_manager_prediction_eval_artifact
from app.services.prediction_evaluator import evaluate_prediction_analysis_artifact
from app.tools.predictions import list_models


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    init_db()
    settings = get_settings()
    return HealthResponse(status="ok", app_name=settings.app_name)


@router.post(
    "/v1/datasets/upload",
    response_model=DatasetSummary,
    dependencies=[Depends(require_api_key)],
)
def upload_dataset(file: UploadFile = File(...)) -> DatasetSummary:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV uploads are supported.",
        )
    try:
        return save_uploaded_dataset(file.file, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/v1/chats",
    response_model=ChatCreateResponse,
    dependencies=[Depends(require_api_key)],
)
def create_chat_endpoint() -> ChatCreateResponse:
    return ChatCreateResponse(thread_id=create_chat())


@router.get(
    "/v1/chats/{thread_id}",
    response_model=ChatHistoryResponse,
    dependencies=[Depends(require_api_key)],
)
def get_chat(thread_id: str) -> ChatHistoryResponse:
    return ChatHistoryResponse(thread_id=thread_id, messages=get_messages(thread_id))


@router.post(
    "/v1/chats/{thread_id}/messages",
    response_model=AgentMessageResponse,
    dependencies=[Depends(require_api_key)],
)
def post_message(thread_id: str, request: AgentMessageRequest) -> AgentMessageResponse:
    if request.dataset_id and get_dataset(request.dataset_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {request.dataset_id}",
        )

    request_payload = model_dump(request)
    add_message(thread_id, "user", request.message, {"request": request_payload})
    history = get_messages(thread_id)
    state = run_agent(thread_id=thread_id, request=request, history=history)

    tools = [ToolResult(**tool) for tool in state.get("tools", [])]
    agent_trace = [AgentTraceStep(**step) for step in state.get("agent_trace", [])]
    answer = str(state.get("answer") or "The agent did not produce a final answer.")
    output_payload = {
        "answer": answer,
        "intent": state.get("intent"),
        "tools": [model_dump(tool) for tool in tools],
        "agent_trace": [model_dump(step) for step in agent_trace],
        "warnings": state.get("warnings", []),
    }
    run_id = add_agent_run(
        thread_id=thread_id,
        mode="manager",
        input_payload=request_payload,
        output_payload=output_payload,
    )
    eval_artifact_path = save_manager_prediction_eval_artifact(
        thread_id=thread_id,
        run_id=run_id,
        request=request,
        answer=output_payload["answer"],
        intent=state.get("intent"),
        tools=output_payload["tools"],
        agent_trace=output_payload["agent_trace"],
    )
    add_message(thread_id, "assistant", output_payload["answer"], {"run_id": run_id, "tools": output_payload["tools"]})

    return AgentMessageResponse(
        thread_id=thread_id,
        answer=output_payload["answer"],
        intent=state.get("intent"),
        tools=tools,
        agent_trace=agent_trace,
        warnings=state.get("warnings", []),
        eval_artifact_path=str(eval_artifact_path) if eval_artifact_path else None,
        run_id=run_id,
    )


@router.post(
    "/v1/evaluations/prediction-analysis",
    response_model=PredictionEvaluationResponse,
    dependencies=[Depends(require_api_key)],
)
def evaluate_prediction_analysis(request: PredictionEvaluationRequest) -> PredictionEvaluationResponse:
    try:
        report = evaluate_prediction_analysis_artifact(
            request.artifact_path,
            use_llm=request.use_llm,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return PredictionEvaluationResponse(
        score_total=report["score_total"],
        score_max=report["score_max"],
        prediction_metrics=report["prediction_metrics"],
        scores=report["scores"],
        evaluator_agent=report["evaluator_agent"],
        evaluation_report_path=report["evaluation_report_path"],
        source_artifact_path=report["source_artifact_path"],
    )


@router.get(
    "/v1/models",
    response_model=list[ModelInfo],
    dependencies=[Depends(require_api_key)],
)
def get_models() -> list[ModelInfo]:
    return list_models()
