"""ChromaDB wrapper. Only this class knows about Chroma, so Qdrant could replace it."""

from dataclasses import dataclass
from typing import Protocol

import chromadb

from app.rag.chunker import Chunk

COLLECTION = "reports"


class EmbedderLike(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class SearchHit:
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    score: float  # cosine similarity, higher is better


class VectorStore:
    """One Chroma collection for the app; every query filters by doc_id."""

    def __init__(self, path: str, embedder: EmbedderLike) -> None:
        self._embedder = embedder
        client = chromadb.PersistentClient(path=path)
        self._cache: dict[str, list[SearchHit]] = {}
        self._collection = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._cache.clear()
        vectors = self._embedder.embed([c.text for c in chunks])
        self._collection.add(
            ids=[f"{c.doc_id}-{c.chunk_index}" for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def search(self, query: str, doc_id: str, k: int) -> list[SearchHit]:
        vector = self._embedder.embed([query])[0]
        result = self._collection.query(
            query_embeddings=[vector], n_results=k, where={"doc_id": doc_id}
        )
        hits: list[SearchHit] = []
        for text, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            hits.append(
                SearchHit(
                    doc_id=str(meta["doc_id"]),
                    filename=str(meta["filename"]),
                    page=int(meta["page"]),
                    chunk_index=int(meta["chunk_index"]),
                    text=text,
                    score=1.0 - float(dist),
                )
            )
        return hits

    def document_chunks(self, doc_id: str) -> list[SearchHit]:
        """Every chunk of one document (cached), used for keyword search."""
        if doc_id not in self._cache:
            data = self._collection.get(where={"doc_id": doc_id})
            self._cache[doc_id] = [
                SearchHit(
                    doc_id=str(m["doc_id"]),
                    filename=str(m["filename"]),
                    page=int(m["page"]),
                    chunk_index=int(m["chunk_index"]),
                    text=t,
                    score=0.0,
                )
                for t, m in zip(data["documents"], data["metadatas"], strict=True)
            ]
        return self._cache[doc_id]

    def delete_document(self, doc_id: str) -> None:
        self._cache.clear()
        self._collection.delete(where={"doc_id": doc_id})
