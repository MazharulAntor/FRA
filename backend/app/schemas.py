"""All Pydantic models shared by the API and the agent."""

from typing import Literal

from pydantic import BaseModel


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    status: Literal["processing", "ready", "failed"]
    error: str | None = None


class Citation(BaseModel):
    doc_id: str
    filename: str
    page: int
    snippet: str  # snippet <= 200 chars


class ChartSeries(BaseModel):
    label: str
    value: float
    period: str | None = None
    unit: str | None = None


class ChartSpec(BaseModel):
    chart_type: Literal["bar", "line", "pie"]
    title: str
    series: list[ChartSeries]
    citations: list[Citation]


class AskRequest(BaseModel):
    doc_id: str
    question: str
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": str}]


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    chart: ChartSpec | None = None
    question_type: str
    calculations: list[dict] = []  # name, inputs, result
    grounded: bool  # False if evidence was insufficient


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    model: str
