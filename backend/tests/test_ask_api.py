import json

import pytest
from fastapi.testclient import TestClient

from app.llm.provider import LLMError
from app.main import app
from tests.fakes import scripted

REVENUE_PAGE = "Income statement. Total revenue FY2024: 12,000 million. Total revenue FY2023: 10,000 million. "


@pytest.fixture
def client():
    return TestClient(app)


def upload(client, path):
    with open(path, "rb") as f:
        return client.post(
            "/documents", files={"file": ("Annual_Report.pdf", f, "application/pdf")}
        ).json()


@pytest.fixture
def doc_id(client, tmp_path):
    from tests.conftest import make_pdf

    pdf = tmp_path / "r.pdf"
    filler = "The board met and discussed governance. " * 12
    make_pdf(pdf, ["Governance. " + filler, REVENUE_PAGE * 4, "Leases. " + filler])
    return upload(client, pdf)["doc_id"]


def use_llm(monkeypatch, llm):
    monkeypatch.setattr("app.services.get_llm", lambda: llm)


def lookup_llm(
    answer="Total revenue in FY2024 was 12,000 million [Annual_Report.pdf, p.2].",
):
    fig = {
        "name": "Total revenue",
        "value": "12,000",
        "unit": "USD million",
        "period": "FY2024",
        "page": 2,
    }
    return scripted(
        classify=json.dumps({"question_type": "lookup"}),
        verify=json.dumps({"supporting": [0]}),
        extract=json.dumps({"figures": [fig]}),
        answer=answer,
    )


def test_ask_returns_cited_answer(client, doc_id, monkeypatch):
    use_llm(monkeypatch, lookup_llm())
    r = client.post(
        "/ask", json={"doc_id": doc_id, "question": "What was total revenue in FY2024?"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True and body["question_type"] == "lookup"
    assert (
        body["citations"][0]["page"] == 2
        and body["citations"][0]["filename"] == "Annual_Report.pdf"
    )
    assert "12,000" in body["answer"]


def test_ask_advice_refused(client, doc_id, monkeypatch):
    use_llm(monkeypatch, scripted(classify=json.dumps({"question_type": "advice"})))
    body = client.post(
        "/ask", json={"doc_id": doc_id, "question": "Should I buy?"}
    ).json()
    assert body["question_type"] == "advice" and body["citations"] == []


def test_ask_follow_up_with_history(client, doc_id, monkeypatch):
    llm_replies = scripted(
        classify=json.dumps({"question_type": "lookup"}),
        rewrite="total revenue FY2024",
        verify=json.dumps({"supporting": [0]}),
        extract=json.dumps(
            {
                "figures": [
                    {
                        "name": "Total revenue",
                        "value": "12,000",
                        "unit": "USD million",
                        "period": "FY2024",
                        "page": 2,
                    }
                ]
            }
        ),
        answer="Total revenue in FY2024 was 12,000 million [p.2].",
    )
    use_llm(monkeypatch, llm_replies)
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"bad": 1},
    ]
    r = client.post(
        "/ask", json={"doc_id": doc_id, "question": "and FY2024?", "history": history}
    )
    assert r.status_code == 200 and llm_replies.seen["rewrite"] == 1


def test_error_cases(client, doc_id, tmp_path, monkeypatch):
    use_llm(monkeypatch, lookup_llm())
    assert (
        client.post("/ask", json={"doc_id": doc_id, "question": "   "}).status_code
        == 422
    )
    assert client.post("/ask", json={"doc_id": doc_id}).status_code == 422
    r = client.post("/ask", json={"doc_id": "nope", "question": "x"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_not_ready_gives_409(client, tmp_path, monkeypatch):
    from app.schemas import DocumentInfo
    from app.services import get_doc_store

    get_doc_store().add(
        DocumentInfo(doc_id="busy", filename="b.pdf", pages=1, status="processing")
    )
    use_llm(monkeypatch, lookup_llm())
    r = client.post("/ask", json={"doc_id": "busy", "question": "x"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_ready"


def test_llm_failure_is_friendly_and_has_no_trace(client, doc_id, monkeypatch):
    def boom(system, user):
        raise LLMError("secret details sk-123")

    from app.llm.fake import FakeLLM

    use_llm(monkeypatch, FakeLLM(boom))
    r = client.post("/ask", json={"doc_id": doc_id, "question": "What was revenue?"})
    assert r.status_code == 502
    assert "sk-123" not in r.text and "Traceback" not in r.text
    assert r.json()["error"]["code"] == "llm_error"
