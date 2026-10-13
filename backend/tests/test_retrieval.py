from fastapi.testclient import TestClient

from app.main import app
from app.rag.chunker import Chunk
from app.rag.retriever import retrieve
from app.rag.vector_store import VectorStore
from app.services import get_vector_store
from tests.conftest import make_pdf
from tests.fakes import FakeEmbedder


def _store(tmp_path):
    return VectorStore(str(tmp_path / "vs"), FakeEmbedder())


def _chunk(doc, page, text, i):
    return Chunk(doc_id=doc, filename=f"{doc}.pdf", page=page, chunk_index=i, text=text)


def test_search_filters_by_doc_and_keeps_page(tmp_path):
    store = _store(tmp_path)
    store.add_chunks(
        [
            _chunk("a", 3, "total revenue and net income for the year", 0),
            _chunk("a", 7, "list of directors and board members", 1),
            _chunk("b", 1, "total revenue of the other company", 0),
        ]
    )
    hits = retrieve(store, "total revenue", "a", 5)
    assert {h.doc_id for h in hits} == {"a"}
    assert hits[0].page == 3 and hits[0].filename == "a.pdf"


def test_delete_document(tmp_path):
    store = _store(tmp_path)
    store.add_chunks([_chunk("a", 1, "revenue", 0)])
    store.delete_document("a")
    assert retrieve(store, "revenue", "a", 5) == []


def test_empty_question_returns_nothing(tmp_path):
    assert retrieve(_store(tmp_path), "  ", "a", 3) == []


def test_upload_then_search_finds_income_statement_page(tmp_path):
    pdf = tmp_path / "r.pdf"
    filler = "The board met several times and discussed governance. " * 10
    make_pdf(
        pdf,
        [
            "Corporate governance. " + filler,
            "Income statement: total revenue was 500 million and total revenue grew. "
            * 5,
            "Notes about leases and property. " + filler,
        ],
    )
    client = TestClient(app)
    with open(pdf, "rb") as f:
        doc_id = client.post(
            "/documents", files={"file": ("r.pdf", f, "application/pdf")}
        ).json()["doc_id"]
    assert client.get(f"/documents/{doc_id}").json()["status"] == "ready"
    hits = retrieve(get_vector_store(), "total revenue", doc_id, 3)
    assert hits[0].page == 2
    client.delete(f"/documents/{doc_id}")
    assert retrieve(get_vector_store(), "total revenue", doc_id, 3) == []
