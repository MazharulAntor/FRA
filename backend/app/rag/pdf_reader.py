"""Read a PDF into per-page text, keeping 1-based page numbers."""

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF

MIN_TOTAL_CHARS = 200
SCANNED_MESSAGE = "Scanned PDFs are not supported"
ROW_TOLERANCE = 3.0  # points: words this close vertically are on the same table row
TABLE_NUMBER_RATIO = 0.2  # pages with this share of numeric words are treated as tables

_LEADERS = re.compile(r"(?:\s*[.\u2026\u00b7]){3,}")
_NUMERIC_WORD = re.compile(r"\d")


class ScannedPdfError(ValueError):
    """Raised when the PDF has (almost) no extractable text."""


@dataclass
class PageText:
    page_number: int  # 1-based
    text: str


def _clean(text: str) -> str:
    """Remove dot leaders ('Revenue ....... 12') and tidy spaces."""
    text = _LEADERS.sub(" ", text)
    text = re.sub(r"[ \t\u00a0]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _table_rows(page: pymupdf.Page) -> str:
    """Rebuild table rows so a label and its numbers end up on one line."""
    words = [w for w in page.get_text("words") if not _LEADERS.fullmatch(" " + w[4])]
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[list[tuple]] = []
    row_y = None
    for w in words:
        y = (w[1] + w[3]) / 2
        if row_y is None or y - row_y > ROW_TOLERANCE:
            rows.append([])
            row_y = y
        rows[-1].append(w)
    return "\n".join(
        " ".join(w[4] for w in sorted(r, key=lambda w: w[0])) for r in rows
    )


def _page_text(page: pymupdf.Page) -> str:
    plain = page.get_text("text")
    plain = _clean(plain)
    tokens = plain.split()
    if tokens:
        numeric = sum(1 for t in tokens if _NUMERIC_WORD.search(t))
        if numeric / len(tokens) >= TABLE_NUMBER_RATIO:
            return _clean(_table_rows(page))
    return plain


def read_pdf(path: str | Path) -> list[PageText]:
    """Return the text of every page. Raise ScannedPdfError if there is almost no text."""
    pages: list[PageText] = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            pages.append(PageText(page_number=index, text=_page_text(page)))
    if sum(len(p.text.strip()) for p in pages) < MIN_TOTAL_CHARS:
        raise ScannedPdfError(SCANNED_MESSAGE)
    return pages


def count_pages(path: str | Path) -> int:
    """Number of pages in the PDF."""
    with pymupdf.open(path) as doc:
        return doc.page_count
