"""POST /ask: run the agent on one question."""

import logging
import time

from fastapi import APIRouter

from app.agent.graph import run_agent
from app.errors import ApiError
from app.llm.provider import LLMError
from app.schemas import AskRequest, AskResponse, ChartSpec, Citation
from app.services import get_agent, get_doc_store

log = logging.getLogger(__name__)
router = APIRouter(tags=["ask"])

MAX_HISTORY = 6


def _clean_history(history: list[dict]) -> list[dict]:
    """Keep the last 6 well-formed messages."""
    clean = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
    ]
    return clean[-MAX_HISTORY:]


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise ApiError(422, "empty_question", "Please type a question.")
    info = get_doc_store().get(request.doc_id)
    if info is None:
        raise ApiError(404, "not_found", "Unknown document.")
    if info.status != "ready":
        raise ApiError(409, "not_ready", "The document is not ready yet. Please wait.")

    started = time.perf_counter()
    try:
        state = run_agent(
            get_agent(),
            request.question.strip(),
            request.doc_id,
            _clean_history(request.history),
        )
    except LLMError as exc:
        log.warning("ask failed: %s", type(exc).__name__)
        raise ApiError(
            502,
            "llm_error",
            "The language model is not available right now. Please try again.",
        ) from None

    log.info(
        "ask type=%s chunks=%d grounded=%s seconds=%.1f",
        state.get("question_type"),
        len(state.get("verified_chunks") or state.get("chunks") or []),
        state.get("grounded"),
        time.perf_counter() - started,
    )
    chart = state.get("chart")
    return AskResponse(
        answer=state["answer"],
        citations=[Citation(**c) for c in state.get("citations", [])],
        chart=ChartSpec(**chart) if chart else None,
        question_type=state.get("question_type", "lookup"),
        calculations=state.get("calculations", []),
        grounded=bool(state.get("grounded")),
    )
