from __future__ import annotations

import json
from typing import Any


def _extract_request_json(user_text: str) -> dict[str, Any]:
    marker = "Request JSON:"
    if marker not in user_text:
        return {"message": user_text}
    after = user_text.split(marker, 1)[1]
    for stop_marker in ("\n\nRecent conversation:", "\n\nAvailable tools:", "\nAvailable tools:"):
        if stop_marker in after:
            after = after.split(stop_marker, 1)[0]
            break
    try:
        parsed = json.loads(after.strip())
        return parsed if isinstance(parsed, dict) else {"message": user_text}
    except json.JSONDecodeError:
        return {"message": user_text}


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part.strip() for part in parts if part and part.strip())
    return str(content).strip()


def extract_openai_compatible_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("LLM provider response did not include any choices.")

    message = choices[0].get("message") or {}
    content = _content_to_text(message.get("content"))
    if content:
        return content

    for key in ("text", "output_text", "response"):
        value = _content_to_text(message.get(key) or choices[0].get(key))
        if value:
            return value

    if message.get("reasoning") or message.get("reasoning_details"):
        raise ValueError(
            "The LLM provider returned reasoning tokens but no final answer content. "
            "For OpenRouter reasoning models, try setting extra_body.reasoning.exclude to true "
            "or increase max_tokens."
        )

    raise ValueError("The LLM provider returned an empty final answer.")
