"""Tiny SQLite table that remembers uploaded documents and their processing status."""

import sqlite3
from contextlib import closing
from pathlib import Path

from app.schemas import DocumentInfo

_COLS = "doc_id, filename, pages, status, error"


class DocumentStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._run(
            "CREATE TABLE IF NOT EXISTS documents ("
            "doc_id TEXT PRIMARY KEY, filename TEXT, pages INTEGER, status TEXT, error TEXT)"
        )

    def _run(self, sql: str, params: tuple = ()) -> list[tuple]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(sql, params).fetchall()
            conn.commit()
        return rows

    @staticmethod
    def _info(row: tuple) -> DocumentInfo:
        return DocumentInfo(
            doc_id=row[0], filename=row[1], pages=row[2], status=row[3], error=row[4]
        )

    def add(self, info: DocumentInfo) -> None:
        self._run(
            f"INSERT INTO documents ({_COLS}) VALUES (?,?,?,?,?)",
            (info.doc_id, info.filename, info.pages, info.status, info.error),
        )

    def update(self, doc_id: str, **fields: object) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self._run(
            f"UPDATE documents SET {cols} WHERE doc_id=?", (*fields.values(), doc_id)
        )

    def get(self, doc_id: str) -> DocumentInfo | None:
        rows = self._run(f"SELECT {_COLS} FROM documents WHERE doc_id=?", (doc_id,))
        return self._info(rows[0]) if rows else None

    def list(self) -> list[DocumentInfo]:
        return [
            self._info(r)
            for r in self._run(f"SELECT {_COLS} FROM documents ORDER BY rowid")
        ]

    def delete(self, doc_id: str) -> None:
        self._run("DELETE FROM documents WHERE doc_id=?", (doc_id,))

    def fail_interrupted(self) -> int:
        """Mark documents left in 'processing' (server stopped mid-upload) as failed."""
        stuck = [d for d in self.list() if d.status == "processing"]
        for doc in stuck:
            self.update(
                doc.doc_id,
                status="failed",
                error="Processing was interrupted. Please upload the file again.",
            )
        return len(stuck)
