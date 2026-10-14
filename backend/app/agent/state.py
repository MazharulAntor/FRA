"""Agent state and the small Pydantic models used to validate LLM JSON output."""

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator

QuestionType = Literal[
    "lookup",
    "calculation",
    "comparison",
    "trend",
    "summary",
    "advice",
    "out_of_scope",
]
REFUSAL_TYPES = ("advice", "out_of_scope")
CALC_TYPES = ("calculation", "comparison", "trend")


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]
    doc_id: str
    query: str  # standalone search query (follow-ups resolved)
    question_type: str
    chunks: list[dict]  # retrieved: doc_id, filename, page, chunk_index, text
    verified_chunks: list[dict]
    extracted: list[dict]  # figures: name, value, unit, period, page, filename
    calculations: list[dict]
    answer: str
    citations: list[dict]
    chart: dict | None
    grounded: bool
    error: str | None


class ClassifyOutput(BaseModel):
    question_type: QuestionType


class VerifyOutput(BaseModel):
    supporting: list[int] = Field(default_factory=list)


class Figure(BaseModel):
    """One number found in the report, kept exactly as written."""

    name: str
    value: str
    unit: str | None = None
    period: str | None = None
    page: int
    filename: str = ""

    @field_validator("value", mode="before")
    @classmethod
    def _value_to_text(cls, v: Any) -> Any:
        return str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v

    @field_validator("unit", "period", mode="before")
    @classmethod
    def _blank_to_none(cls, v: Any) -> Any:
        if v is None:
            return None
        text = str(v).strip()
        return text or None


class Extraction(BaseModel):
    figures: list[Figure] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _wrap_bare_list(cls, data: Any) -> Any:
        return {"figures": data} if isinstance(data, list) else data


class CalcCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class CalcPlan(BaseModel):
    calls: list[CalcCall] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _wrap_bare_list(cls, data: Any) -> Any:
        return {"calls": data} if isinstance(data, list) else data
