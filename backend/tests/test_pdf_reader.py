import pytest

from app.rag.pdf_reader import ScannedPdfError, read_pdf


def test_page_numbers_are_correct(sample_pdf):
    pages = read_pdf(sample_pdf)
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "Income statement" in pages[1].text
    assert "Balance sheet" in pages[2].text


def test_scanned_pdf_rejected(scanned_pdf):
    with pytest.raises(ScannedPdfError, match="Scanned PDFs are not supported"):
        read_pdf(scanned_pdf)
