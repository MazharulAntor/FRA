"""Test doubles."""

import hashlib
import math
import re

from app.llm.fake import FakeLLM


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: similar words -> similar vectors."""

    DIM = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.DIM
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def scripted(**replies):
    """FakeLLM that answers by TASK tag. A value may be a string, a list (one per call) or a function."""
    seen: dict[str, int] = {}

    def respond(system, user):
        task = user.split("\n", 1)[0].replace("TASK:", "").strip()
        seen[task] = seen.get(task, 0) + 1
        reply = replies.get(task)
        if reply is None:
            raise AssertionError(f"Unexpected LLM call for task '{task}'")
        if callable(reply):
            return reply(user)
        if isinstance(reply, list):
            return reply[min(seen[task], len(reply)) - 1]
        return reply

    llm = FakeLLM(respond)
    llm.seen = seen
    return llm
