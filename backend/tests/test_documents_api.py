from fastapi.testclient import TestClient

from app.main import app
from app.services import reset_services


def _upload(client, path, name="r.pdf"):
    with open(path, "rb") as f:
        return client.post("/documents", files={"file": (name, f, "application/pdf")})


def test_upload_status_list_delete(sample_pdf):
    client = TestClient(app)
    r = _upload(client, sample_pdf)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "processing" and body["pages"] == 3
    doc_id = body["doc_id"]

    info = client.get(f"/documents/{doc_id}").json()
    assert info["status"] == "ready" and info["pages"] == 3
    assert [d["doc_id"] for d in client.get("/documents").json()] == [doc_id]

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get(f"/documents/{doc_id}").status_code == 404


def test_scanned_pdf_fails_with_message(scanned_pdf):
    client = TestClient(app)
    doc_id = _upload(client, scanned_pdf).json()["doc_id"]
    info = client.get(f"/documents/{doc_id}").json()
    assert info["status"] == "failed"
    assert info["error"] == "Scanned PDFs are not supported"


def test_non_pdf_rejected():
    client = TestClient(app)
    r = client.post("/documents", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "not_pdf"
    r = client.post(
        "/documents", files={"file": ("a.pdf", b"hello", "application/pdf")}
    )
    assert r.status_code == 400


def test_too_large_rejected(sample_pdf, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")
    reset_services()
    assert _upload(TestClient(app), sample_pdf).status_code == 413


def test_interrupted_documents_are_failed_on_startup():
    from app.schemas import DocumentInfo
    from app.services import get_doc_store

    store = get_doc_store()
    store.add(DocumentInfo(doc_id="a", filename="a.pdf", pages=1, status="processing"))
    store.add(DocumentInfo(doc_id="b", filename="b.pdf", pages=1, status="ready"))
    assert store.fail_interrupted() == 1
    assert store.get("a").status == "failed" and store.get("b").status == "ready"
