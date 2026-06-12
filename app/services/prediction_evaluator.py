from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.llm.langchain_adapter import get_provider_chat_model
from app.services.chat_store import utc_now


def _artifact_root() -> Path:
    settings = get_settings()
    path = settings.database_path.parent / "manager_prediction_evals"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _evaluation_dir() -> Path:
    path = _artifact_root() / "evaluations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_artifact_path(artifact_path: str) -> Path:
    raw = Path(artifact_path)
    path = raw if raw.is_absolute() else Path.cwd() / raw
    resolved = path.resolve()
    root = _artifact_root().resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError("Evaluation artifact path must be inside app_data/manager_prediction_evals.")
    if not resolved.exists():
        raise ValueError(f"Evaluation artifact was not found: {artifact_path}")
    if resolved.parent.name == "evaluations":
        raise ValueError("Pass a manager prediction artifact, not an evaluation report.")
    return resolved


def _prediction_actual_frame(artifact: dict[str, Any]) -> pd.DataFrame:
    predictions = pd.DataFrame(artifact.get("prediction_values") or [])
    actuals = pd.DataFrame(artifact.get("held_out_actuals_not_shown_to_agent") or [])
    if predictions.empty or actuals.empty:
        return pd.DataFrame()

    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="coerce")
    predictions["predicted_value"] = pd.to_numeric(predictions["predicted_value"], errors="coerce")
    actuals["timestamp"] = pd.to_datetime(actuals["timestamp"], errors="coerce")
    actuals["actual_value"] = pd.to_numeric(actuals["actual_value"], errors="coerce")
    merged = predictions.dropna(subset=["timestamp", "predicted_value"]).merge(
        actuals.dropna(subset=["timestamp", "actual_value"]),
        on="timestamp",
        how="inner",
    )
    if merged.empty:
        return merged
    merged["error"] = merged["actual_value"] - merged["predicted_value"]
    merged["absolute_error"] = merged["error"].abs()
    merged["absolute_percent_error"] = np.where(
        merged["actual_value"] != 0,
        merged["absolute_error"] / merged["actual_value"].abs() * 100,
        np.nan,
    )
    return merged


def _prediction_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "matched_points": 0,
            "mae": None,
            "mape": None,
            "max_absolute_error": None,
            "bias": None,
            "high_error_count": 0,
            "worst_hours": [],
        }

    ape = frame["absolute_percent_error"].dropna()
    high_error = frame["absolute_percent_error"].ge(20)
    if ape.empty:
        high_error = frame["absolute_error"].ge(frame["absolute_error"].quantile(0.75))

    worst = frame.sort_values("absolute_error", ascending=False).head(5)
    return {
        "matched_points": int(len(frame)),
        "mae": float(frame["absolute_error"].mean()),
        "mape": None if ape.empty else float(ape.mean()),
        "max_absolute_error": float(frame["absolute_error"].max()),
        "p90_absolute_error": float(frame["absolute_error"].quantile(0.9)),
        "bias": float(frame["error"].mean()),
        "high_error_count": int(high_error.fillna(False).sum()),
        "worst_hours": [
            {
                "timestamp": row["timestamp"].isoformat(),
                "predicted_value": float(row["predicted_value"]),
                "actual_value": float(row["actual_value"]),
                "absolute_error": float(row["absolute_error"]),
                "absolute_percent_error": (
                    None
                    if pd.isna(row["absolute_percent_error"])
                    else float(row["absolute_percent_error"])
                ),
            }
            for _, row in worst.iterrows()
        ],
    }


def _tool_by_name(artifact: dict[str, Any], name: str) -> dict[str, Any] | None:
    for tool in artifact.get("manager_agent_output", {}).get("tools", []):
        if tool.get("name") == name:
            return tool
    return None


def _tool_use_score(artifact: dict[str, Any]) -> dict[str, Any]:
    expectations = artifact.get("deterministic_eval_expectations") or {}
    expected_intent = expectations.get("expected_intent")
    required_tools = expectations.get("required_tools") or []
    used_tools = [
        tool.get("name")
        for tool in artifact.get("manager_agent_output", {}).get("tools", [])
        if tool.get("name")
    ]
    used_set = set(used_tools)
    missing = [name for name in required_tools if name not in used_set]
    extra = [name for name in used_tools if name not in set(required_tools + (expectations.get("optional_tools") or []))]
    actual_intent = artifact.get("manager_agent_output", {}).get("intent")

    intent_points = 5 if actual_intent == expected_intent else 0
    tool_points = 15
    if required_tools:
        tool_points = round(15 * (len(required_tools) - len(missing)) / len(required_tools), 2)

    return {
        "score": round(intent_points + tool_points, 2),
        "max_score": 20,
        "expected_intent": expected_intent,
        "actual_intent": actual_intent,
        "required_tools": required_tools,
        "used_tools": used_tools,
        "missing_required_tools": missing,
        "extra_tools": extra,
    }


def _reliability_rows(artifact: dict[str, Any]) -> pd.DataFrame:
    tool = _tool_by_name(artifact, "prediction_analysis")
    rows = ((tool or {}).get("data") or {}).get("per_hour_reliability") or []
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame.dropna(subset=["timestamp"])


def _reliability_judgment_score(artifact: dict[str, Any], merged: pd.DataFrame) -> dict[str, Any]:
    reliability = _reliability_rows(artifact)
    if merged.empty or reliability.empty:
        return {
            "score": 0,
            "max_score": 30,
            "summary": "Could not compare manager reliability labels with actual errors.",
            "high_error_count": 0,
            "flagged_risk_count": 0,
            "true_positive_risk_count": 0,
        }

    judged = merged.merge(
        reliability[["timestamp", "reliability"]],
        on="timestamp",
        how="left",
    )
    high_error = judged["absolute_percent_error"].ge(20)
    if judged["absolute_percent_error"].dropna().empty:
        high_error = judged["absolute_error"].ge(judged["absolute_error"].quantile(0.75))
    flagged = judged["reliability"].isin(["low", "medium"])

    high_count = int(high_error.fillna(False).sum())
    flagged_count = int(flagged.fillna(False).sum())
    true_positive = int((high_error.fillna(False) & flagged.fillna(False)).sum())

    if high_count == 0:
        false_warning_rate = flagged_count / max(len(judged), 1)
        score = 30 * max(0.0, 1.0 - false_warning_rate)
    else:
        recall = true_positive / high_count
        precision = true_positive / flagged_count if flagged_count else 0.0
        score = 30 * ((0.65 * recall) + (0.35 * precision))

    return {
        "score": round(score, 2),
        "max_score": 30,
        "summary": "Compared manager low/medium reliability labels with hours that later had high actual error.",
        "high_error_count": high_count,
        "flagged_risk_count": flagged_count,
        "true_positive_risk_count": true_positive,
    }


def _grounding_score(artifact: dict[str, Any]) -> dict[str, Any]:
    answer = str(artifact.get("manager_agent_output", {}).get("answer") or "").lower()
    tool_names = {
        tool.get("name")
        for tool in artifact.get("manager_agent_output", {}).get("tools", [])
        if tool.get("name")
    }
    checks = {
        "mentions_reliability_or_risk": any(word in answer for word in ("reliable", "reliability", "risk", "uncertain")),
        "mentions_prediction": any(word in answer for word in ("prediction", "forecast", "predicted")),
        "mentions_history_or_backtest": any(word in answer for word in ("history", "historical", "backtest", "previous")),
        "used_prediction_analysis_tool": "prediction_analysis" in tool_names,
        "used_backtest_context_tool": "prediction_backtest_context" in tool_names,
    }
    score = round(20 * sum(1 for ok in checks.values() if ok) / len(checks), 2)
    return {"score": score, "max_score": 20, "checks": checks}


def _temporal_fairness_score(artifact: dict[str, Any]) -> dict[str, Any]:
    request = artifact.get("request") or {}
    forecast_start = request.get("forecast_start")
    if not forecast_start:
        return {"score": 10, "max_score": 10, "summary": "No range was provided, so no temporal cutoff was required."}

    scoped_tools = []
    unscoped_tools = []
    for tool in artifact.get("manager_agent_output", {}).get("tools", []):
        name = tool.get("name")
        if name in {"prediction_analysis", "prediction_interval"}:
            continue
        scope = (tool.get("data") or {}).get("temporal_scope") or {}
        if scope.get("mode") == "historical_before_requested_range":
            scoped_tools.append(name)
        else:
            unscoped_tools.append(name)

    score = 10 if not unscoped_tools else round(10 * len(scoped_tools) / max(len(scoped_tools) + len(unscoped_tools), 1), 2)
    return {
        "score": score,
        "max_score": 10,
        "summary": "Checked whether tool outputs recorded historical-only temporal scope.",
        "scoped_tools": scoped_tools,
        "unscoped_tools": unscoped_tools,
    }


def _communication_score(artifact: dict[str, Any]) -> dict[str, Any]:
    answer = str(artifact.get("manager_agent_output", {}).get("answer") or "")
    words = answer.split()
    sentence_count = max(answer.count(".") + answer.count("!") + answer.count("?"), 1)
    avg_sentence_words = len(words) / sentence_count
    lower = answer.lower()
    checks = {
        "not_too_long": len(words) <= 220,
        "simple_sentences": avg_sentence_words <= 24,
        "has_actionable_language": any(word in lower for word in ("check", "watch", "use", "trust", "careful", "investigate", "safe", "risk")),
        "avoids_empty_answer": len(words) >= 20,
    }
    score = round(20 * sum(1 for ok in checks.values() if ok) / len(checks), 2)
    return {
        "score": score,
        "max_score": 20,
        "checks": checks,
        "word_count": len(words),
        "average_sentence_words": round(avg_sentence_words, 2),
    }


def _evaluator_agent_judgment(artifact: dict[str, Any], metrics: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "manager_answer": artifact.get("manager_agent_output", {}).get("answer"),
        "request": artifact.get("request"),
        "prediction_metrics": metrics,
        "scores": scores,
        "worst_hours": metrics.get("worst_hours", [])[:3],
    }
    prompt = (
        "Evaluate the manager agent's prediction analysis. "
        "Use simple language. Focus on whether the manager warned about real risks, "
        "was grounded in tools, and avoided overconfidence.\n\n"
        f"Evaluation data:\n{json.dumps(compact, indent=2, default=str)}\n\n"
        "Return a short verdict with good points, missed risks, and one recommendation."
    )
    try:
        model = get_provider_chat_model("evaluator")
        response = model.invoke(
            [
                SystemMessage(content="You are an evaluator agent for time-series prediction analysis."),
                HumanMessage(content=prompt),
            ]
        )
        return {"status": "ok", "verdict": str(getattr(response, "content", "") or "").strip()}
    except Exception as exc:
        return {"status": "unavailable", "verdict": f"Evaluator LLM was not used: {exc}"}


def evaluate_prediction_analysis_artifact(artifact_path: str, use_llm: bool = True) -> dict[str, Any]:
    path = _resolve_artifact_path(artifact_path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("artifact_type") != "manager_prediction_analysis_eval_case":
        raise ValueError("This artifact is not a manager prediction-analysis evaluation case.")

    merged = _prediction_actual_frame(artifact)
    metrics = _prediction_metrics(merged)
    if metrics["matched_points"] == 0:
        raise ValueError("No matching prediction and actual timestamps were found for evaluation.")

    tool_use = _tool_use_score(artifact)
    reliability = _reliability_judgment_score(artifact, merged)
    grounding = _grounding_score(artifact)
    temporal = _temporal_fairness_score(artifact)
    communication = _communication_score(artifact)
    scores = {
        "tool_use_and_intent": tool_use,
        "reliability_judgment": reliability,
        "grounding_in_tool_outputs": grounding,
        "temporal_fairness": temporal,
        "communication_quality": communication,
    }
    total = round(sum(item["score"] for item in scores.values()), 2)

    evaluator_agent = (
        _evaluator_agent_judgment(artifact, metrics, scores)
        if use_llm
        else {"status": "skipped", "verdict": "LLM evaluator was disabled for this run."}
    )
    report = {
        "report_type": "manager_prediction_analysis_dynamic_evaluation",
        "evaluated_at": utc_now(),
        "source_artifact_path": str(path),
        "evaluation_report_path": None,
        "score_total": total,
        "score_max": 100,
        "scores": scores,
        "prediction_metrics": metrics,
        "evaluator_agent": evaluator_agent,
    }
    report_path = _evaluation_dir() / f"{path.stem}__evaluation.json"
    report["evaluation_report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
