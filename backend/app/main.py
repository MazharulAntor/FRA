"""FastAPI application entry point."""

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import ask, documents
from app.config import get_settings
from app.errors import register_error_handlers
from app.schemas import HealthResponse
from app.services import get_doc_store, get_embedder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def _warm_up() -> None:
    """Load the embedding model in the background so the first question is not slow."""
    try:
        get_embedder()
    except Exception:
        log.warning("embedding model warm-up failed", exc_info=True)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    get_doc_store().fail_interrupted()
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="AI-Powered Financial Analyst", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    app.include_router(documents.router)
    app.include_router(ask.router)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok", llm_provider=settings.llm_provider, model=settings.llm_model
        )

    return app


app = create_app()
