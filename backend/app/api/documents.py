"""Upload, status, list and delete endpoints."""

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, UploadFile

from app.config import get_settings
from app.errors import ApiError
from app.ingest import ingest_document
from app.rag.pdf_reader import count_pages
from app.schemas import DocumentInfo
from app.services import get_doc_store, get_vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentInfo, status_code=202)
async def upload_document(
    file: UploadFile, background: BackgroundTasks
) -> DocumentInfo:
    settings = get_settings()
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise ApiError(400, "not_pdf", "Only PDF files are accepted.")
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise ApiError(
            413, "file_too_large", f"File is larger than {settings.max_upload_mb} MB."
        )
    if not data.startswith(b"%PDF"):
        raise ApiError(400, "not_pdf", "The file is not a valid PDF.")

    doc_id = uuid.uuid4().hex[:12]
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{doc_id}.pdf"
    path.write_bytes(data)

    try:
        pages = count_pages(path)
    except Exception:  # noqa: BLE001
        path.unlink(missing_ok=True)
        raise ApiError(400, "not_pdf", "The file is not a valid PDF.") from None

    info = DocumentInfo(
        doc_id=doc_id, filename=filename, pages=pages, status="processing"
    )
    get_doc_store().add(info)
    background.add_task(ingest_document, doc_id, path, filename)
    return info


@router.get("", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return get_doc_store().list()


@router.get("/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str) -> DocumentInfo:
    info = get_doc_store().get(doc_id)
    if info is None:
        raise ApiError(404, "not_found", "Unknown document.")
    return info


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str) -> None:
    store = get_doc_store()
    if store.get(doc_id) is None:
        raise ApiError(404, "not_found", "Unknown document.")
    (Path(get_settings().upload_dir) / f"{doc_id}.pdf").unlink(missing_ok=True)
    get_vector_store().delete_document(doc_id)
    store.delete(doc_id)
