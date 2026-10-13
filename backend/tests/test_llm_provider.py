import pytest
from pydantic import BaseModel, RootModel

from app.config import Settings
from app.llm.fake import FakeLLM
from app.llm.provider import (
    AnthropicLLM,
    LLMError,
    OllamaLLM,
    OpenAILLM,
    build_llm,
    extract_json,
)


class Item(BaseModel):
    name: str
    value: float


Items = RootModel[list[Item]]


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json("Sure! Here it is: [1, 2] hope that helps") == [1, 2]
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_complete_plain_text():
    assert FakeLLM(["  hello  "]).complete("s", "u") == "hello"


def test_complete_json_valid():
    llm = FakeLLM(['{"name": "revenue", "value": 12.5}'])
    assert llm.complete_json("s", "u", Item) == Item(name="revenue", value=12.5)
    assert len(llm.calls) == 1


def test_complete_json_retries_once_then_succeeds():
    llm = FakeLLM(["not json at all", '[{"name": "x", "value": 1}]'])
    result = llm.complete_json("s", "u", Items)
    assert result.root[0].name == "x"
    assert len(llm.calls) == 2
    assert "not valid" in llm.calls[1][1]  # the retry tells the model what went wrong


def test_complete_json_schema_failure_retries_then_raises():
    llm = FakeLLM(['{"name": "x"}', '{"name": "x"}'])
    with pytest.raises(LLMError):
        llm.complete_json("s", "u", Item)
    assert len(llm.calls) == 2  # exactly one retry


def test_build_llm_picks_provider():
    assert isinstance(
        build_llm(Settings(llm_provider="anthropic", anthropic_api_key="k")),
        AnthropicLLM,
    )
    assert isinstance(
        build_llm(Settings(llm_provider="openai", openai_api_key="k")), OpenAILLM
    )
    assert isinstance(build_llm(Settings(llm_provider="ollama")), OllamaLLM)


def test_build_llm_requires_keys():
    with pytest.raises(LLMError):
        build_llm(Settings(llm_provider="anthropic", anthropic_api_key=""))
    with pytest.raises(LLMError):
        build_llm(Settings(llm_provider="openai", openai_api_key=""))


def test_rate_limit_is_retried_then_wrapped(monkeypatch):
    from app.llm import provider
    from app.llm.provider import BaseLLM

    monkeypatch.setattr(provider.time, "sleep", lambda s: None)

    class Limited(Exception):
        status_code = 429

    class Flaky(BaseLLM):
        calls = 0

        def _chat(self, system, user):
            self.calls += 1
            if self.calls < 3:
                raise Limited()
            return "ok"

    llm = Flaky()
    assert llm.complete("s", "u") == "ok" and llm.calls == 3

    class Always(Flaky):
        def _chat(self, system, user):
            self.calls += 1
            raise Limited()

    always = Always()
    with pytest.raises(LLMError):
        always.complete("s", "u")
    assert always.calls == provider.RETRIES + 1


def test_lmstudio_needs_no_key():
    llm = build_llm(
        Settings(llm_provider="lmstudio", llm_model="google/gemma-4-12b-qat")
    )
    assert isinstance(llm, OpenAILLM)
