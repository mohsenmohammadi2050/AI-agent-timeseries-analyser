from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMProviderConfig:
    name: str
    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    max_tokens: int = 900
    extra_body: dict[str, Any] | None = None
    headers: dict[str, str] | None = None


@dataclass(frozen=True)
class Settings:
    app_name: str
    api_keys: tuple[str, ...]
    database_path: Path
    upload_dir: Path
    default_llm_purpose: str
    llm_providers: dict[str, LLMProviderConfig]

    def get_llm_provider(self, purpose: str | None = None) -> LLMProviderConfig:
        purpose_name = purpose or self.default_llm_purpose
        config = self.llm_providers.get(purpose_name)
        if config is not None:
            return config
        config = self.llm_providers.get("agent")
        if config is not None:
            return config
        return next(iter(self.llm_providers.values()))


def _parse_api_keys(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ("dev-api-key",)
    return tuple(key.strip() for key in raw.split(",") if key.strip())


def _parse_provider_config(raw: str | None) -> dict[str, LLMProviderConfig]:
    if not raw:
        return {
            "agent": LLMProviderConfig(
                name="agent",
                provider="mock",
                model="mock-simple-explainer",
            )
        }

    parsed: dict[str, Any] = json.loads(raw)
    providers: dict[str, LLMProviderConfig] = {}
    for purpose, item in parsed.items():
        providers[purpose] = LLMProviderConfig(
            name=purpose,
            provider=item["provider"],
            model=item["model"],
            api_key_env=item.get("api_key_env"),
            base_url=item.get("base_url"),
            temperature=float(item.get("temperature", 0.2)),
            max_tokens=int(item.get("max_tokens", 900)),
            extra_body=item.get("extra_body"),
            headers=item.get("headers"),
        )
    if "agent" not in providers:
        fallback = providers.get("general") or next(iter(providers.values()), None)
        if fallback is not None:
            providers = {"agent": replace(fallback, name="agent")}
    return providers


@lru_cache
def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[1]
    return Settings(
        app_name=os.getenv("APP_NAME", "Agentic Time Series Analysis API"),
        api_keys=_parse_api_keys(os.getenv("API_KEYS")),
        database_path=Path(os.getenv("DATABASE_PATH", root / "app_data" / "agent_api.sqlite3")),
        upload_dir=Path(os.getenv("UPLOAD_DIR", root / "app_data" / "uploads")),
        default_llm_purpose=os.getenv("DEFAULT_LLM_PURPOSE", "agent"),
        llm_providers=_parse_provider_config(os.getenv("LLM_PROVIDERS_JSON")),
    )
