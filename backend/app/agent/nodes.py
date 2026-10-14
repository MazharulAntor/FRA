"""The six agent steps: classify, retrieve, verify, extract, calculate, answer.

Each node takes the state and returns only the keys it changes.
"""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.agent import prompts
from app.agent.prompts import (
    ADVICE_REFUSAL,
    NOT_FOUND_MESSAGE,
    OUT_OF_SCOPE_REFUSAL,
    SYSTEM_PROMPT,
)
from app.agent.state import (
    CALC_TYPES,
    AgentState,
    CalcPlan,
    ClassifyOutput,
    Extraction,
    VerifyOutput,
)
from app.llm.provider import BaseLLM
from app.rag.vector_store import SearchHit
from app.schemas import ChartSeries, ChartSpec, Citation
from app.tools.calculator import run_tool

MAX_CALLS = 6
PROTECTED_TOP = 3  # top search results that verify may not drop
SNIPPET_CHARS = 200
# Numbers like 12,345.6 or 12 345 (space as thousands separator, used by e.g. Nestle).
NUM_RE = re.compile(r"\d{1,3}(?:[    ]\d{3})+(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?")
_SEPARATORS = re.compile(r"[ ,   ]")


def to_float(token: str) -> float:
    """'12,345.6' or '12 345' -> a float."""
    return float(_SEPARATORS.sub("", token))


MARKER_RE = re.compile(r"\[(?:([^\[\]]*?),\s*)?p(?:age|p)?\.?\s*(\d+)\]", re.IGNORECASE)
YEAR_RE = re.compile(r"(?:19|20)\d{2}")
PIE_WORDS = re.compile(
    r"breakdown|composition|proportion|split|mix|share of", re.IGNORECASE
)
NUMBER_WORDS = {
    w: i
    for i, w in enumerate(
        [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ],
        start=1,
    )
}


@dataclass
class AgentDeps:
    """Things the nodes need: the LLM and a search function."""

    llm: BaseLLM
    search: Callable[[str, str, int], list[SearchHit]]
    top_k: int = 6


# ---------- number helpers (plain code, no LLM) ----------


_THOUSANDS = re.compile(r"[$’']\s?0{3}|’000|'000", re.IGNORECASE)


def clean_unit(unit: str | None) -> str | None:
    """Spell out report shorthand such as $'000 (thousands of dollars) in plain words."""
    if not unit:
        return unit
    if _THOUSANDS.search(unit):
        return "thousand " + ("AUD" if "A$" in unit or "AU" in unit else "dollars")
    return unit


def numbers_in(text: str) -> list[float]:
    """All numbers written in a text (commas removed, sign ignored)."""
    return [to_float(m) for m in NUM_RE.findall(text or "")]


def parse_number(text: str) -> float | None:
    """'12,345' -> 12345.0, '(1,234)' -> -1234.0, '-5.2%' -> -5.2, no number -> None."""
    raw = (text or "").strip()
    found = NUM_RE.search(raw)
    if not found:
        return None
    value = to_float(found.group())
    negative = raw.startswith(("-", "−")) or (raw.startswith("(") and ")" in raw)
    return -value if negative else value


def _history_text(history: list[dict]) -> str:
    return " ".join(str(m.get("content", "")) for m in history)


# ---------- 1. classify ----------


CALC_WORDS = re.compile(
    r"\b(ratio|margin|percent|percentage|growth|grow|grew|cagr|average|mean|difference|"
    r"per share)\b|%",
    re.IGNORECASE,
)
TREND_WORDS = re.compile(
    r"\b(trend|over time|over the (last|past|years)|year[- ]over[- ]year)\b",
    re.IGNORECASE,
)


def _fix_type(question: str, label: str) -> str:
    """Small models often call calculations 'lookup'. Obvious wording wins."""
    if label in ("lookup", "summary"):
        if TREND_WORDS.search(question):
            return "trend"
        if CALC_WORDS.search(question):
            return "calculation"
    return label


def classify(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    prompt = prompts.classify_prompt(state["question"], state.get("history", []))
    result = deps.llm.complete_json(SYSTEM_PROMPT, prompt, ClassifyOutput)
    return {"question_type": _fix_type(state["question"], result.question_type)}


# ---------- 2. retrieve ----------


def retrieve(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    question = state["question"]
    history = state.get("history") or []
    query = question
    if history:
        rewritten = deps.llm.complete(
            SYSTEM_PROMPT, prompts.rewrite_prompt(question, history)
        )
        query = rewritten.strip().strip('"').strip() or question
    hits = deps.search(query, state["doc_id"], deps.top_k)
    chunks = [
        {
            "doc_id": h.doc_id,
            "filename": h.filename,
            "page": h.page,
            "chunk_index": h.chunk_index,
            "text": h.text,
        }
        for h in hits
    ]
    return {"query": query, "chunks": chunks}


# ---------- 3. verify ----------


def verify(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    chunks = state.get("chunks") or []
    if not chunks:
        return {"verified_chunks": [], "grounded": False}
    prompt = prompts.verify_prompt(state.get("query") or state["question"], chunks)
    result = deps.llm.complete_json(SYSTEM_PROMPT, prompt, VerifyOutput)
    chosen = {i for i in result.supporting if 0 <= i < len(chunks)}
    if chosen:
        # Small models over-filter, so the best-ranked search results are always kept.
        chosen |= set(range(min(PROTECTED_TOP, len(chunks))))
    keep = sorted(chosen)
    verified = [chunks[i] for i in keep]
    return {"verified_chunks": verified, "grounded": bool(verified)}


# ---------- 4. extract ----------


def extract(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    verified = state.get("verified_chunks") or []
    prompt = prompts.extract_prompt(
        state.get("query") or state["question"],
        verified,
        calculation=state.get("question_type") in CALC_TYPES,
    )
    parsed = deps.llm.complete_json(SYSTEM_PROMPT, prompt, Extraction)

    numbers_by_page: dict[int, set[float]] = {}
    filename_by_page: dict[int, str] = {}
    for chunk in verified:
        numbers_by_page.setdefault(chunk["page"], set()).update(
            numbers_in(chunk["text"])
        )
        filename_by_page.setdefault(chunk["page"], chunk["filename"])

    figures: list[dict] = []
    seen: set[tuple] = set()
    for fig in parsed.figures:
        number = parse_number(fig.value)
        if number is None:
            continue
        key = abs(number)
        # Guard: the value must literally appear in a verified chunk. Never trust the LLM.
        page = fig.page if key in numbers_by_page.get(fig.page, set()) else None
        if page is None:
            page = next((p for p, nums in numbers_by_page.items() if key in nums), None)
        if page is None:
            continue
        ident = (fig.name, fig.period, fig.value)
        if ident in seen:
            continue
        seen.add(ident)
        figures.append(
            {
                "name": fig.name,
                "value": fig.value,
                "unit": clean_unit(fig.unit),
                "period": fig.period,
                "page": page,
                "filename": filename_by_page[page],
            }
        )
    return {"extracted": figures}


# ---------- 5. calculate ----------


def _allowed_literals(question: str, history: list[dict]) -> set[float]:
    text = f"{question} {_history_text(history)}"
    allowed = set(numbers_in(text))
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in NUMBER_WORDS:
            allowed.add(float(NUMBER_WORDS[word]))
    return allowed


def _resolve_arg(value: Any, figures: list[dict], literals: set[float]) -> Any:
    """Turn a figure id ('f1'), a list of ids, or a number from the question into calculator input."""
    if isinstance(value, list):
        return [_resolve_arg(v, figures, literals) for v in value]
    if isinstance(value, str):
        match = next((f for f in figures if f["id"] == value.strip()), None)
        if match is None:
            raise ValueError(f"Unknown figure '{value}'.")
        number = parse_number(match["value"])
        if number is None:
            raise ValueError(f"'{match['name']}' is not a number.")
        label = (
            f"{match['name']} ({match['period']})"
            if match.get("period")
            else match["name"]
        )
        return {
            "value": number,
            "unit": match.get("unit"),
            "label": label,
            "page": match["page"],
        }
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if abs(float(value)) in literals:
            return value
        raise ValueError(
            f"The number {value} does not come from the question or the report."
        )
    raise ValueError(f"Unsupported input: {value!r}")


def calculate(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    extracted = state.get("extracted") or []
    if not extracted:
        return {"calculations": []}
    figures = [{"id": f"f{i}", **f} for i, f in enumerate(extracted, start=1)]
    question = state.get("query") or state["question"]
    plan = deps.llm.complete_json(
        SYSTEM_PROMPT, prompts.calculate_prompt(question, figures), CalcPlan
    )
    literals = _allowed_literals(state["question"], state.get("history") or [])

    results: list[dict] = []
    for call in plan.calls[:MAX_CALLS]:
        try:
            args = {k: _resolve_arg(v, figures, literals) for k, v in call.args.items()}
            done = run_tool(call.name, args)
            results.append(
                {
                    "name": done["name"],
                    "formula": done["formula"],
                    "inputs": done["inputs"],
                    "result": done["result"],
                    "unit": done["unit"],
                }
            )
        except ValueError as exc:  # includes unit mismatch and divide by zero
            results.append({"name": call.name, "inputs": call.args, "error": str(exc)})
    return {"calculations": results}


# ---------- 6. answer ----------


def _snippet(text: str, anchors: list[str]) -> str:
    """Up to 200 characters of the chunk, centred on a figure if one is found."""
    flat = " ".join(text.split())
    for anchor in anchors:
        at = flat.find(anchor)
        if at != -1:
            start = max(0, at - (SNIPPET_CHARS - len(anchor)) // 2)
            return flat[start : start + SNIPPET_CHARS].strip()
    return flat[:SNIPPET_CHARS].strip()


def _citation(state: AgentState, page: int, figures: list[dict]) -> Citation:
    chunks = [c for c in state["verified_chunks"] if c["page"] == page]
    anchors = [f["value"] for f in figures if f["page"] == page]
    chosen = next(
        (c for c in chunks if any(a in c["text"] for a in anchors)), chunks[0]
    )
    return Citation(
        doc_id=chosen["doc_id"],
        filename=chosen["filename"],
        page=page,
        snippet=_snippet(chosen["text"], anchors),
    )


def _apply_citations(
    text: str, state: AgentState, figures: list[dict]
) -> tuple[str, list[int]]:
    """Normalise [file, p.N] markers, drop invalid ones, and add one if the LLM wrote none."""
    verified = state["verified_chunks"]
    pages_available = {c["page"] for c in verified}
    filename = verified[0]["filename"]
    pages: list[int] = []

    def fix(match: re.Match) -> str:
        page = int(match.group(2))
        if page not in pages_available:
            return ""
        pages.append(page)
        return f"[{filename}, p.{page}]"

    text = MARKER_RE.sub(fix, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text).strip()
    if not pages:
        fallback = [f["page"] for f in figures if f["page"] in pages_available]
        pages = list(dict.fromkeys(fallback))[:3] or [verified[0]["page"]]
        text += " " + " ".join(f"[{filename}, p.{p}]" for p in pages)
    return text, list(dict.fromkeys(pages))


def unsupported_numbers(
    text: str,
    figures: list[dict],
    calculations: list[dict],
    citations: list[Citation],
    question: str,
    history: list[dict],
) -> list[str]:
    """Numbers in the answer that are not in the figures, calculations, snippets or question."""
    allowed: set[float] = set()

    def add(values: list[float]) -> None:
        allowed.update(round(abs(v), 6) for v in values)

    for f in figures:
        add(
            numbers_in(
                f"{f['value']} {f['name']} {f.get('period') or ''} {f.get('unit') or ''}"
            )
        )
    for c in calculations:
        if "result" in c:
            for places in range(5):
                add([round(abs(c["result"]), places)])
        stack: list[Any] = list(c.get("inputs", {}).values())
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                if isinstance(item.get("value"), (int, float)):
                    add([float(item["value"])])
                add(numbers_in(str(item.get("label", ""))))
            elif isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                add([float(item)])
    for cite in citations:
        add(numbers_in(cite.snippet))
    add(numbers_in(question))
    add(numbers_in(_history_text(history)))

    body = re.sub(r"\[[^\]]*p\.\s*\d+\]", " ", text)
    bad: list[str] = []
    for token in NUM_RE.findall(body):
        if round(to_float(token), 6) not in allowed and token not in bad:
            bad.append(token)
    return bad


def build_chart(
    state: AgentState, question_type: str, figures: list[dict]
) -> dict | None:
    """ChartSpec from extracted figures (trend, comparison, or any answer with 2+ numbers)."""
    points: list[tuple[dict, float]] = []
    seen: set[tuple] = set()
    for f in figures:
        number = parse_number(f["value"])
        key = (f["name"], f.get("period"), number)
        if number is None or key in seen:
            continue
        seen.add(key)
        points.append((f, number))
    if not points:
        return None
    top_unit = Counter((f.get("unit") or "").lower() for f, _ in points).most_common(1)[
        0
    ][0]
    points = [(f, n) for f, n in points if (f.get("unit") or "").lower() == top_unit]
    if len(points) < 2:
        return None

    question = state["question"]
    if question_type == "trend":
        chart_type = "line"
        years = [YEAR_RE.search(f.get("period") or "") for f, _ in points]
        if all(years):
            points = [
                p
                for _, p in sorted(
                    zip(years, points, strict=True), key=lambda x: x[0].group()
                )
            ]
    elif PIE_WORDS.search(question) and all(n > 0 for _, n in points):
        chart_type = "pie"
    else:
        chart_type = "bar"

    names = {f["name"] for f, _ in points}
    unit = points[0][0].get("unit")
    title = (
        (f"{points[0][0]['name']}" + (f" ({unit})" if unit else ""))
        if len(names) == 1
        else question[:80]
    )
    pages = list(dict.fromkeys(f["page"] for f, _ in points))[:5]
    spec = ChartSpec(
        chart_type=chart_type,  # type: ignore[arg-type]
        title=title,
        series=[
            ChartSeries(
                label=f["name"], value=n, period=f.get("period"), unit=f.get("unit")
            )
            for f, n in points
        ],
        citations=[_citation(state, p, figures) for p in pages],
    )
    return spec.model_dump()


def _final(answer: str, grounded: bool) -> dict[str, Any]:
    return {"answer": answer, "citations": [], "chart": None, "grounded": grounded}


def answer(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    question_type = state.get("question_type", "lookup")
    if question_type == "advice":
        return _final(
            ADVICE_REFUSAL, grounded=True
        )  # a refusal makes no unsupported claim
    if question_type == "out_of_scope":
        return _final(OUT_OF_SCOPE_REFUSAL, grounded=True)
    verified = state.get("verified_chunks") or []
    if not state.get("grounded", True) or not verified:
        return _final(NOT_FOUND_MESSAGE, grounded=False)

    figures = state.get("extracted") or []
    calculations = state.get("calculations") or []
    history = state.get("history") or []
    query = state.get("query") or state["question"]

    feedback: str | None = None
    for _attempt in range(2):  # one regeneration allowed
        draft = deps.llm.complete(
            SYSTEM_PROMPT,
            prompts.answer_prompt(
                state["question"], query, verified, figures, calculations, feedback
            ),
        )
        if not draft.strip():
            feedback = "Your previous reply was empty. Write the answer text."
            continue
        text, pages = _apply_citations(draft, state, figures)
        citations = [_citation(state, p, figures) for p in pages]
        bad = unsupported_numbers(
            text, figures, calculations, citations, state["question"], history
        )
        if not bad:
            return {
                "answer": text,
                "citations": [c.model_dump() for c in citations],
                "chart": build_chart(state, question_type, figures),
                "grounded": True,
            }
        allowed = "; ".join(
            f"{f['value']} {f.get('unit') or ''}".strip() for f in figures
        ) + "; ".join(f" {c['result']}" for c in calculations if "result" in c)
        feedback = (
            f"Your previous answer contained numbers that are not in the extracted figures or "
            f"calculation results: {', '.join(bad)}. Do NOT convert units or multiply. "
            f"Write the figures exactly as given, for example: {allowed}"
        )
    return _final(NOT_FOUND_MESSAGE, grounded=False)  # safe fallback
