"""Shared singletons (document store, embedder, vector store) built from settings."""

import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.agent.graph import build_graph
from app.agent.nodes import AgentDeps
from app.config import get_settings
from app.llm.provider import get_llm
from app.rag.doc_store import DocumentStore
from app.rag.embeddings import Embedder
from app.rag.retriever import retrieve
from app.rag.vector_store import VectorStore


@lru_cache
def get_doc_store() -> DocumentStore:
    return DocumentStore(Path(get_settings().upload_dir) / "documents.db")


_embedder_lock = threading.Lock()


@lru_cache
def _build_embedder() -> Embedder:
    return Embedder(get_settings().embedding_model)


def get_embedder() -> Embedder:
    """The embedding model (loaded once, even if two threads ask at the same time)."""
    with _embedder_lock:
        return _build_embedder()


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore(get_settings().chroma_path, get_embedder())


@lru_cache
def get_agent() -> Any:
    """The compiled LangGraph agent wired to the configured LLM and the vector store."""
    settings = get_settings()
    store = get_vector_store()
    deps = AgentDeps(
        llm=get_llm(),
        search=lambda query, doc_id, k: retrieve(store, query, doc_id, k),
        top_k=settings.top_k,
    )
    return build_graph(deps)


def reset_services() -> None:
    """Clear cached singletons (used by tests after changing settings)."""
    get_agent.cache_clear()
    get_doc_store.cache_clear()
    get_vector_store.cache_clear()
    get_settings.cache_clear()
