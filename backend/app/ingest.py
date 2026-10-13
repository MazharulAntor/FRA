"""Background ingestion: PDF -> pages -> chunks -> vectors."""

import logging
from pathlib import Path

from app.config import get_settings
from app.rag.chunker import chunk_pages
from app.rag.pdf_reader import ScannedPdfError, read_pdf
from app.services import get_doc_store, get_vector_store

log = logging.getLogger(__name__)


def ingest_document(doc_id: str, path: Path, filename: str) -> None:
    """Process one uploaded PDF and update its status. Never raises."""
    store = get_doc_store()
    settings = get_settings()
    try:
        pages = read_pdf(path)
        store.update(doc_id, pages=len(pages))
        chunks = chunk_pages(
            pages, doc_id, filename, settings.chunk_size, settings.chunk_overlap
        )
        get_vector_store().add_chunks(chunks)
        log.info("ingested %s: %d pages, %d chunks", filename, len(pages), len(chunks))
        store.update(doc_id, status="ready", error=None)
    except ScannedPdfError as exc:
        store.update(doc_id, status="failed", error=str(exc))
    except Exception:
        log.exception("ingestion failed for %s", filename)
        store.update(doc_id, status="failed", error="Could not read this PDF.")
