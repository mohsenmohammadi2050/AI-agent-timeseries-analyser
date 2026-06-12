from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DEFAULT_LLM_PURPOSE", "agent")
os.environ.setdefault(
    "LLM_PROVIDERS_JSON",
    '{"agent":{"provider":"mock","model":"mock-simple-explainer"}}',
)

from app.agents.prompts import MANAGER_SYSTEM_PROMPT, REACT_TOOL_CATALOG
from app.llm.langchain_adapter import get_provider_chat_model


def _noop_tool(name: str, description: str) -> StructuredTool:
    def run() -> str:
        """No-op eval tool."""
        return f"{name} called"

    return StructuredTool.from_function(func=run, name=name, description=description)


def _planned_tools(request: dict) -> tuple[str, list[str], str]:
    tools = [_noop_tool(item["tool"], item["description"]) for item in REACT_TOOL_CATALOG]
    agent = create_agent(
        model=get_provider_chat_model(),
        tools=tools,
        system_prompt=MANAGER_SYSTEM_PROMPT,
    )
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use ReAct style. Choose tools if needed.\n"
                        f"Request JSON:\n{json.dumps(request, indent=2)}\n"
                        f"Available tools:\n{json.dumps(REACT_TOOL_CATALOG, indent=2)}"
                    )
                )
            ]
        }
    )
    selected: list[str] = []
    thought = ""
    for message in result.get("messages", []):
        for call in getattr(message, "tool_calls", []) or []:
            name = call.get("name")
            if name and name not in selected:
                selected.append(name)
        content = getattr(message, "content", "")
        if content:
            thought = str(content)
    if "model_performance_analysis" in selected:
        intent = "model_performance_analysis"
    elif "prediction" in selected or "prediction_analysis" in selected:
        intent = "prediction_analysis"
    elif "outlier_detection" in selected:
        intent = "outlier_analysis"
    elif "historical_summary" in selected or "hourly_consumption_context" in selected:
        intent = "historical_analysis"
    elif "data_consistency" in selected or "data_anomaly_warnings" in selected:
        intent = "data_quality"
    else:
        intent = "general_question"
    return intent, selected, thought


def main() -> int:
    cases = json.loads((Path(__file__).parent / "cases.json").read_text(encoding="utf-8"))
    failures: list[str] = []

    for case in cases:
        intent, tools, reason = _planned_tools(case["request"])
        intent_ok = intent == case["expected_intent"]
        tools_ok = set(tools) == set(case["expected_tools"])
        status = "PASS" if intent_ok and tools_ok else "FAIL"
        print(f"{status} {case['name']}: intent={intent}, tools={tools}, reason={reason}")
        if not intent_ok or not tools_ok:
            failures.append(case["name"])

    if failures:
        print(f"\nFailed eval cases: {', '.join(failures)}")
        return 1
    print("\nAll ReAct planning evals passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
