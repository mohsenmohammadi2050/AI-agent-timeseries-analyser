from __future__ import annotations

import os

import pytest

from app.config import _load_env_file, get_settings
from app.llm.langchain_adapter import get_provider_chat_model
from app.llm.providers import extract_openai_compatible_text


def test_extracts_standard_chat_content() -> None:
    data = {"choices": [{"message": {"content": "hello"}}]}

    assert extract_openai_compatible_text(data) == "hello"


def test_extracts_list_content_text_parts() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": "second"},
                    ]
                }
            }
        ]
    }

    assert extract_openai_compatible_text(data) == "first\nsecond"


def test_reasoning_without_content_raises_clear_error() -> None:
    data = {"choices": [{"message": {"content": None, "reasoning": "hidden reasoning"}}]}

    with pytest.raises(ValueError, match="reasoning tokens but no final answer"):
        extract_openai_compatible_text(data)


def test_old_default_llm_purpose_falls_back_to_agent(monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_LLM_PURPOSE", "reasoning")
    monkeypatch.setenv(
        "LLM_PROVIDERS_JSON",
        '{"agent":{"provider":"mock","model":"mock-simple-explainer"}}',
    )
    get_settings.cache_clear()

    model = get_provider_chat_model()

    assert model.config.name == "agent"


def test_load_env_file_sets_missing_values_without_overriding(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APP_NAME=From File",
                "DEFAULT_LLM_PURPOSE=agent",
                'LLM_PROVIDERS_JSON={"agent":{"provider":"mock","model":"mock-from-env-file"}}',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_NAME", "Already Set")
    monkeypatch.delenv("DEFAULT_LLM_PURPOSE", raising=False)
    monkeypatch.delenv("LLM_PROVIDERS_JSON", raising=False)

    _load_env_file(env_file)

    assert os.environ["APP_NAME"] == "Already Set"
    assert os.environ["DEFAULT_LLM_PURPOSE"] == "agent"
    assert "mock-from-env-file" in os.environ["LLM_PROVIDERS_JSON"]
