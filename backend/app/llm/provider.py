"""LLM provider interface: anthropic | openai | ollama, chosen by LLM_PROVIDER in .env.

Every provider has two methods:
- complete(system, user) -> plain text
- complete_json(system, user, schema) -> validated Pydantic object (one retry on bad JSON)
Temperature is always 0.
"""

import json
import re
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)

RETRIES = 4
RETRY_WAIT_SECONDS = 3
MAX_TOKENS = 4000
OPENROUTER_URL = "https://openrouter.ai/api/v1"
TIMEOUT_SECONDS = 120


class LLMError(RuntimeError):
    """Raised when the LLM cannot be reached or keeps returning invalid output."""


def extract_json(text: str) -> object:
    """Parse JSON from a model reply, tolerating code fences and surrounding prose."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if not starts:
        raise ValueError("No JSON found in the reply.")
    start = min(starts)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end <= start:
        raise ValueError("No JSON found in the reply.")
    return json.loads(cleaned[start : end + 1])


class BaseLLM(ABC):
    """Shared logic. Subclasses only implement _chat()."""

    @abstractmethod
    def _chat(self, system: str, user: str) -> str:
        """Send one system + user message and return the reply text."""

    def _call(self, system: str, user: str) -> str:
        """_chat() with retries for rate limits / temporary errors; other errors become LLMError."""
        for attempt in range(RETRIES + 1):
            try:
                return self._chat(system, user)
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
                status = getattr(exc, "status_code", None)
                temporary = status in (408, 429) or (
                    isinstance(status, int) and status >= 500
                )
                if temporary and attempt < RETRIES:
                    time.sleep(RETRY_WAIT_SECONDS * 2**attempt)
                    continue
                raise LLMError(
                    f"LLM request failed ({type(exc).__name__}, status {status})."
                ) from None
        raise LLMError("LLM request failed.")  # not reached

    def complete(self, system: str, user: str) -> str:
        """Return a plain-text reply."""
        reply = self._call(system, user)
        return re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        """Return a reply parsed and validated as `schema`; retry once if it is invalid."""
        prompt = user
        last_error = ""
        for _attempt in range(2):
            reply = self._call(system, prompt)
            try:
                return schema.model_validate(extract_json(reply))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                prompt = (
                    f"{user}\n\nYour previous reply was not valid ({last_error[:300]}). "
                    "Reply again with ONLY valid JSON that matches the requested format."
                )
        raise LLMError(f"The model did not return valid JSON: {last_error[:200]}")


class AnthropicLLM(BaseLLM):
    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key, timeout=TIMEOUT_SECONDS)
        self._model = model

    def _chat(self, system: str, user: str) -> str:
        reply = self._client.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            b.text for b in reply.content if getattr(b, "type", "") == "text"
        )


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        extra: dict | None = None,
    ) -> None:
        import openai

        self._client = openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=TIMEOUT_SECONDS
        )
        self._model = model
        self._extra = extra or {}

    def _chat(self, system: str, user: str) -> str:
        reply = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self._extra,
        )
        return reply.choices[0].message.content or ""


class OllamaLLM(BaseLLM):
    def __init__(self, base_url: str, model: str) -> None:
        self._url = base_url.rstrip("/") + "/api/chat"
        self._model = model

    def _chat(self, system: str, user: str) -> str:
        try:
            response = httpx.post(
                self._url,
                json={
                    "model": self._model,
                    "stream": False,
                    "options": {"temperature": 0},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not reach Ollama: {exc}") from None
        return response.json()["message"]["content"]


def build_llm(settings: Settings) -> BaseLLM:
    """Create the provider named in settings."""
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        return AnthropicLLM(settings.anthropic_api_key, settings.llm_model)
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set.")
        return OpenAILLM(settings.openai_api_key, settings.llm_model)
    if settings.llm_provider == "lmstudio":
        # LM Studio's local server speaks the OpenAI protocol; the key is ignored.
        # Reasoning is switched off: a reasoning model otherwise spends its whole token
        # budget thinking (83 s and an empty reply in testing) and the agent needs JSON fast.
        return OpenAILLM(
            "lm-studio",
            settings.llm_model,
            settings.lmstudio_base_url,
            extra={"reasoning_effort": "none"},
        )
    if settings.llm_provider == "openrouter":
        if not settings.openrouter_api_key:
            raise LLMError("OPENROUTER_API_KEY is not set.")
        return OpenAILLM(
            settings.openrouter_api_key, settings.llm_model, OPENROUTER_URL
        )
    return OllamaLLM(settings.ollama_base_url, settings.llm_model)


@lru_cache
def get_llm() -> BaseLLM:
    """The configured LLM (cached)."""
    return build_llm(get_settings())
