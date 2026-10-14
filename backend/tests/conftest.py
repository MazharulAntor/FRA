import pymupdf
import pytest

from app.services import reset_services
from tests.fakes import FakeEmbedder


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Every test gets its own upload/chroma folders."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr("app.services.get_embedder", lambda: FakeEmbedder())
    reset_services()
    yield
    reset_services()


def make_pdf(path, pages: list[str]) -> None:
    """Write a small text PDF with one string per page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    doc.save(str(path))
    doc.close()


LONG = "Total revenue for the year was 12,345 million. " * 12


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    make_pdf(
        path,
        [
            "Page one intro. " + LONG,
            "Income statement. " + LONG,
            "Balance sheet. " + LONG,
        ],
    )
    return path


@pytest.fixture
def scanned_pdf(tmp_path):
    path = tmp_path / "scanned.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc.save(str(path))
    doc.close()
    return path
