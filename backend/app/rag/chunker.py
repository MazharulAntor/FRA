"""Split page text into chunks. Each page is split separately so a chunk has one page."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.pdf_reader import PageText


@dataclass
class Chunk:
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    text: str


def chunk_pages(
    pages: list[PageText],
    doc_id: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Turn pages into chunks that keep doc_id, filename, page and chunk_index."""
    chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        # Later chunks of a page start with the page's first line (usually the table title),
        # so "Total revenues 179,465" is still tied to "Statements of income".
        title = next((ln.strip() for ln in page.text.splitlines() if ln.strip()), "")[
            :80
        ]
        prefix = f"{title} | "
        room = max(200, chunk_size - len(prefix))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=room, chunk_overlap=min(chunk_overlap, room // 2)
        )
        for number, piece in enumerate(splitter.split_text(page.text)):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    filename=filename,
                    page=page.page_number,
                    chunk_index=len(chunks),
                    text=prefix + piece if number > 0 and title else piece,
                )
            )
    return chunks
