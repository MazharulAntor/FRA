"""A fake LLM for tests: no network, fully scripted."""

from collections.abc import Callable

from app.llm.provider import BaseLLM


class FakeLLM(BaseLLM):
    """Replies come from a list (one per call) or a function(system, user) -> str.

    Every call is recorded in `calls` as (system, user).
    """

    def __init__(self, replies: list[str] | Callable[[str, str], str]) -> None:
        self._replies = replies
        self.calls: list[tuple[str, str]] = []

    def _chat(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if callable(self._replies):
            return self._replies(system, user)
        if not self._replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        return self._replies.pop(0)
