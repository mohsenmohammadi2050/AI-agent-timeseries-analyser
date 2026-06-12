from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from app.config import LLMProviderConfig, get_settings
from app.llm.providers import _content_to_text, _extract_request_json


class ProviderChatModel(BaseChatModel):
    config: LLMProviderConfig
    bound_tools: Sequence[Any] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return f"provider-chat-{self.config.provider}"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        return ProviderChatModel(config=self.config, bound_tools=list(tools))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.config.provider == "mock":
            message = self._mock_generate(messages)
        elif self.config.provider in {"openai", "openai_compatible"}:
            message = self._openai_compatible_generate(messages)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _mock_generate(self, messages: list[BaseMessage]) -> AIMessage:
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_messages or not self.bound_tools:
            called_names = {getattr(m, "name", None) for m in tool_messages}
            available = {getattr(tool, "name", "") for tool in self.bound_tools}
            if (
                "prediction" in called_names
                and "prediction_analysis" not in called_names
                and "prediction_analysis" in available
            ):
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "prediction_analysis", "args": {}, "id": "mock-followup-1"}],
                )
            summaries = "; ".join(str(m.content)[:180] for m in tool_messages[-6:])
            content = "I used the available tools and summarized the important findings."
            if summaries:
                content += f" Tool observations: {summaries}"
            return AIMessage(content=content)

        text = "\n".join(str(getattr(m, "content", "")) for m in messages)
        request = _extract_request_json(text)
        message_text = str(request.get("message", "")).lower()
        available = {getattr(tool, "name", "") for tool in self.bound_tools}
        selected: list[str] = []

        has_dataset = bool(request.get("dataset_id"))
        has_model = bool(request.get("model_name"))
        has_range = bool(request.get("forecast_start")) and bool(request.get("forecast_end"))

        if has_dataset:
            selected.extend(["data_consistency", "data_anomaly_warnings"])
        if has_dataset and any(word in message_text for word in ("outlier", "anomaly", "spike", "unusual")):
            selected.append("outlier_detection")
        if has_dataset and any(word in message_text for word in ("history", "historical", "trend", "average", "pattern")):
            selected.extend(["historical_summary", "hourly_consumption_context"])
        if has_dataset and has_model and has_range and any(
            word in message_text for word in ("accurate", "accuracy", "performance", "perform", "mae", "mape", "error")
        ):
            selected.extend(["historical_summary", "hourly_consumption_context", "model_performance_analysis"])
        elif has_dataset and has_model and has_range and any(word in message_text for word in ("forecast", "prediction", "predict")):
            selected.extend(
                [
                    "historical_summary",
                    "hourly_consumption_context",
                    "prediction_backtest_context",
                    "prediction",
                    "prediction_interval",
                ]
            )

        calls = []
        seen: set[str] = set()
        for name in selected:
            if name in available and name not in seen:
                calls.append({"name": name, "args": {}, "id": f"mock-{len(calls) + 1}"})
                seen.add(name)
        return AIMessage(content="", tool_calls=calls)

    def _openai_compatible_generate(self, messages: list[BaseMessage]) -> AIMessage:
        api_key = os.getenv(self.config.api_key_env or "")
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {self.config.api_key_env}")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [self._message_to_dict(message) for message in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.bound_tools:
            payload["tools"] = [convert_to_openai_tool(tool) for tool in self.bound_tools]
            payload["tool_choice"] = "auto"
        if self.config.extra_body:
            payload.update(self.config.extra_body)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.headers:
            headers.update(self.config.headers)

        base_url = (self.config.base_url or "https://api.openai.com/v1").rstrip("/")
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            data: dict[str, Any] = json.loads(response.read().decode("utf-8"))

        message = (data.get("choices") or [{}])[0].get("message") or {}
        content = _content_to_text(message.get("content"))
        tool_calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            args_raw = function.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                {
                    "name": function.get("name", ""),
                    "args": args,
                    "id": call.get("id", ""),
                }
            )
        return AIMessage(content=content, tool_calls=tool_calls)

    def _message_to_dict(self, message: BaseMessage) -> dict[str, Any]:
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        if isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        if isinstance(message, AIMessage):
            payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": call.get("id"),
                        "type": "function",
                        "function": {
                            "name": call.get("name"),
                            "arguments": json.dumps(call.get("args", {})),
                        },
                    }
                    for call in message.tool_calls
                ]
            return payload
        return {"role": "user", "content": str(message.content)}


def get_provider_chat_model(purpose: str | None = None) -> ProviderChatModel:
    settings = get_settings()
    return ProviderChatModel(config=settings.get_llm_provider(purpose))
