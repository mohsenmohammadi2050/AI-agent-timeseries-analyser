from __future__ import annotations

import json
from typing import Any, TypedDict

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent

from app.agents.prompts import MANAGER_SYSTEM_PROMPT, REACT_TOOL_CATALOG, TOOL_GUIDANCE
from app.compat import model_dump
from app.llm.langchain_adapter import get_provider_chat_model
from app.schemas import AgentMessageRequest
from app.services.dataset_store import load_dataset_frame
from app.tools.intervals import prediction_interval_tool
from app.tools.predictions import (
    model_performance_analysis_tool,
    prediction_analysis_tool,
    prediction_backtest_context_tool,
    run_prediction_tool,
)
from app.tools.time_series import (
    data_anomaly_warning_tool,
    data_consistency_tool,
    historical_summary_tool,
    hourly_consumption_context_tool,
    outlier_detection_tool,
)


class AgentState(TypedDict, total=False):
    thread_id: str
    request: dict[str, Any]
    history: list[dict[str, Any]]
    intent: str
    tools: list[dict[str, Any]]
    tool_notes: list[dict[str, Any]]
    agent_trace: list[dict[str, Any]]
    warnings: list[str]
    answer: str


TOOL_FOCUS_BY_TOOL = {
    "data_consistency": "data_quality",
    "data_anomaly_warnings": "data_quality",
    "outlier_detection": "outlier",
    "historical_summary": "historical",
    "hourly_consumption_context": "historical",
    "model_performance_analysis": "prediction",
    "prediction_backtest_context": "prediction",
    "prediction": "prediction",
    "prediction_analysis": "prediction",
    "prediction_interval": "interval",
}


def _append_trace(
    trace: list[dict[str, Any]],
    *,
    step: str,
    summary: str,
    decision: str | None = None,
    status: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    trace.append(
        {
            "step": step,
            "actor": "manager",
            "decision": decision,
            "status": status,
            "summary": summary,
            "data": data or {},
        }
    )


def _record_tool(
    result: dict[str, Any],
    tools: list[dict[str, Any]],
    tool_notes: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> str:
    tools.append(result)
    tool_focus = TOOL_FOCUS_BY_TOOL.get(result["name"], "manager")
    tool_notes.append(
        {
            "tool_focus": tool_focus,
            "guidance": TOOL_GUIDANCE.get(tool_focus, ""),
            "tool": result["name"],
            "status": result.get("status"),
            "summary": result.get("summary"),
        }
    )
    _append_trace(
        trace,
        step=result["name"],
        status=result.get("status"),
        summary=result.get("summary", ""),
        data={"tool_focus": tool_focus},
    )
    return json.dumps(result, default=str)


def _tool_data(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for tool_result in tools:
        if tool_result.get("name") == name:
            return tool_result.get("data", {})
    return None


def _missing_dataset_tool(tool_name: str, summary: str) -> dict[str, Any]:
    return {"name": tool_name, "status": "unavailable", "summary": summary, "data": {}}


def _range_start(request: dict[str, Any]) -> pd.Timestamp | None:
    value = request.get("forecast_start")
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def _historical_before_range(df, request: dict[str, Any]):
    cutoff = _range_start(request)
    if cutoff is None:
        return df
    scoped = df.copy()
    scoped["timestamp"] = pd.to_datetime(scoped["timestamp"], errors="coerce")
    return scoped[scoped["timestamp"] < cutoff].copy()


def _add_temporal_scope(result: dict[str, Any], df, request: dict[str, Any]) -> dict[str, Any]:
    cutoff = _range_start(request)
    if cutoff is None:
        result.setdefault("data", {})["temporal_scope"] = {
            "mode": "all_available_data",
            "row_count": int(len(df)),
        }
        return result

    result.setdefault("data", {})["temporal_scope"] = {
        "mode": "historical_before_requested_range",
        "cutoff_timestamp": cutoff.isoformat(),
        "row_count": int(len(df)),
        "note": "Rows at or after the requested range start were not used by this context tool.",
    }
    return result


def _request_context(request: dict[str, Any], history: list[dict[str, Any]]) -> str:
    history_text = "\n".join(
        f"{item['role']}: {item['content']}" for item in history[-8:]
    )
    return (
        "Use ReAct style. Think, call tools when useful, observe results, then answer.\n\n"
        f"Request JSON:\n{json.dumps(request, indent=2, default=str)}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Available tools:\n{json.dumps(REACT_TOOL_CATALOG, indent=2)}\n\n"
        "Important:\n"
        "- For a conceptual/general question, answer without tools.\n"
        "- For a dataset question, inspect the data before answering.\n"
        "- If forecast_start is provided, context tools must use only data before that timestamp.\n"
        "- For a future forecast, use prediction_backtest_context before prediction_analysis when possible.\n"
        "- For historical model performance, use model_performance_analysis instead of generating a new forecast analysis.\n"
        "- Keep the final answer short, precise, and manager-friendly."
    )


def _make_react_tools(
    request: dict[str, Any],
    tools: list[dict[str, Any]],
    tool_notes: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> list[StructuredTool]:
    def dataset_frame(*, historical_only: bool = False):
        dataset_id = request.get("dataset_id")
        if not dataset_id:
            return None
        df = load_dataset_frame(dataset_id)
        if historical_only:
            return _historical_before_range(df, request)
        return df

    def record(result: dict[str, Any]) -> str:
        return _record_tool(result, tools, tool_notes, trace)

    def data_consistency() -> str:
        """Check missing values, duplicate timestamps, gaps, and irregular time steps."""
        df = dataset_frame(historical_only=True)
        if df is None:
            return record(_missing_dataset_tool("data_consistency", "No dataset was attached, so data quality could not be checked."))
        return record(_add_temporal_scope(data_consistency_tool(df), df, request))

    def data_anomaly_warnings() -> str:
        """Warn about huge jumps, negative values, and long zero-value sequences."""
        df = dataset_frame(historical_only=True)
        if df is None:
            return record(_missing_dataset_tool("data_anomaly_warnings", "No dataset was attached, so anomaly warnings could not be checked."))
        return record(_add_temporal_scope(data_anomaly_warning_tool(df), df, request))

    def outlier_detection() -> str:
        """Detect statistically unusual values."""
        df = dataset_frame(historical_only=True)
        if df is None:
            return record(_missing_dataset_tool("outlier_detection", "No dataset was attached, so outliers could not be checked."))
        return record(_add_temporal_scope(outlier_detection_tool(df), df, request))

    def historical_summary() -> str:
        """Summarize historical trend, average behavior, peaks, lows, and basic patterns."""
        df = dataset_frame(historical_only=True)
        if df is None:
            return record(_missing_dataset_tool("historical_summary", "No dataset was attached, so historical behavior could not be summarized."))
        return record(_add_temporal_scope(historical_summary_tool(df), df, request))

    def hourly_consumption_context() -> str:
        """Build last-30-days hourly consumption statistics."""
        df = dataset_frame(historical_only=True)
        if df is None:
            return record(_missing_dataset_tool("hourly_consumption_context", "No dataset was attached, so hourly context could not be built."))
        return record(_add_temporal_scope(hourly_consumption_context_tool(df, days=30), df, request))

    def prediction_backtest_context() -> str:
        """Evaluate model performance on the previous day before the requested forecast start."""
        df = dataset_frame()
        if df is None:
            return record(_missing_dataset_tool("prediction_backtest_context", "No dataset was attached, so previous-day model performance could not be evaluated."))
        return record(
            prediction_backtest_context_tool(
                model_name=request.get("model_name"),
                dataset_id=request.get("dataset_id"),
                df=df,
                forecast_start=request.get("forecast_start"),
                sample_rate_seconds=int(request.get("sample_rate_seconds", 3600)),
            )
        )

    def prediction() -> str:
        """Generate predictions for the requested range."""
        df = dataset_frame()
        if df is None:
            return record(_missing_dataset_tool("prediction", "No dataset was attached, so prediction could not be generated."))
        if not all([request.get("model_name"), request.get("forecast_start"), request.get("forecast_end")]):
            return record(
                {
                    "name": "prediction",
                    "status": "unavailable",
                    "summary": "Prediction requires model_name, forecast_start, and forecast_end.",
                    "data": {},
                }
            )
        return record(
            run_prediction_tool(
                model_name=request["model_name"],
                dataset_id=request["dataset_id"],
                df=df,
                forecast_start=request["forecast_start"],
                forecast_end=request["forecast_end"],
                sample_rate_seconds=int(request.get("sample_rate_seconds", 3600)),
            )
        )

    def prediction_analysis() -> str:
        """Analyze generated predictions and per-hour reliability."""
        prediction_result = next((item for item in tools if item.get("name") == "prediction"), None)
        if prediction_result is None:
            return record(
                {
                    "name": "prediction_analysis",
                    "status": "unavailable",
                    "summary": "Prediction analysis requires the prediction tool to run first.",
                    "data": {},
                }
            )
        return record(
            prediction_analysis_tool(
                prediction_result,
                hourly_context=_tool_data(tools, "hourly_consumption_context"),
                backtest_context=_tool_data(tools, "prediction_backtest_context"),
            )
        )

    def model_performance_analysis() -> str:
        """Evaluate ML model performance on a historical range with actual values."""
        df = dataset_frame()
        if df is None:
            return record(_missing_dataset_tool("model_performance_analysis", "No dataset was attached, so model performance could not be evaluated."))
        return record(
            model_performance_analysis_tool(
                model_name=request.get("model_name"),
                dataset_id=request.get("dataset_id"),
                df=df,
                evaluation_start=request.get("forecast_start"),
                evaluation_end=request.get("forecast_end"),
                sample_rate_seconds=int(request.get("sample_rate_seconds", 3600)),
            )
        )

    def prediction_interval() -> str:
        """Retrieve prediction interval information when available."""
        return record(prediction_interval_tool(request.get("prediction_id")))

    funcs = [
        data_consistency,
        data_anomaly_warnings,
        outlier_detection,
        historical_summary,
        hourly_consumption_context,
        prediction_backtest_context,
        prediction,
        prediction_analysis,
        model_performance_analysis,
        prediction_interval,
    ]
    return [
        StructuredTool.from_function(func=func, name=func.__name__, description=func.__doc__ or "")
        for func in funcs
    ]


def _infer_intent_from_tools(tools: list[dict[str, Any]]) -> str:
    names = {tool.get("name") for tool in tools}
    if "model_performance_analysis" in names:
        return "model_performance_analysis"
    if "prediction" in names or "prediction_analysis" in names:
        return "prediction_analysis"
    if "outlier_detection" in names:
        return "outlier_analysis"
    if "historical_summary" in names or "hourly_consumption_context" in names:
        return "historical_analysis"
    if "data_consistency" in names or "data_anomaly_warnings" in names:
        return "data_quality"
    return "general_question"


def _run_react_agent(state: AgentState) -> dict[str, Any]:
    request = state["request"]
    history = state.get("history", [])
    tools: list[dict[str, Any]] = []
    tool_notes: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    _append_trace(
        trace,
        step="react_agent_start",
        status="ok",
        summary="LangChain ReAct agent started. The LLM will decide which tools to call.",
        data={"available_tools": [item["tool"] for item in REACT_TOOL_CATALOG]},
    )
    react_tools = _make_react_tools(request, tools, tool_notes, trace)
    agent = create_agent(
        model=get_provider_chat_model(),
        tools=react_tools,
        system_prompt=MANAGER_SYSTEM_PROMPT,
    )
    result = agent.invoke({"messages": [HumanMessage(content=_request_context(request, history))]})
    messages = result.get("messages", [])
    answer = ""
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if content:
            answer = str(content)
            break
    if not answer:
        answer = "The agent finished, but did not produce a final text answer."

    _append_trace(
        trace,
        step="final_response",
        status="ok",
        summary="The manager produced the final answer after the ReAct tool loop.",
        data={"llm_used": True, "message_count": len(messages)},
    )
    return {
        "answer": answer,
        "intent": _infer_intent_from_tools(tools),
        "tools": tools,
        "tool_notes": tool_notes,
        "agent_trace": trace,
        "warnings": [tool["summary"] for tool in tools if tool.get("status") in {"warning", "error"}],
    }


def run_agent(
    thread_id: str,
    request: AgentMessageRequest,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    state: AgentState = {
        "thread_id": thread_id,
        "request": model_dump(request),
        "history": history,
    }
    return _run_react_agent(state)
