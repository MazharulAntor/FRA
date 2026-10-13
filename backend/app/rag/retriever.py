"""Hybrid retrieval: vector search + BM25 keyword search, merged with reciprocal rank fusion.

Vector search finds text with a similar meaning; keyword search finds exact words such as
"Total revenues", which matters a lot for financial statement tables.
"""

import math
import re
from collections import Counter

from app.rag.vector_store import SearchHit, VectorStore

CANDIDATES = 40  # how many results each method contributes before merging
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
STOPWORDS = frozenset(
    "a an and are as at be by compared compare did do does for from how in is it its of on "
    "or that the to was were what when which who why with year years".split()
)
_WORD = re.compile(r"[a-z0-9]+")
TABLE_RATIO = 0.2  # chunks with this share of numeric words look like financial tables


def _is_table(text: str) -> bool:
    words = text.split()
    return (
        bool(words)
        and sum(1 for w in words if any(c.isdigit() for c in w)) / len(words)
        >= TABLE_RATIO
    )


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


def _bm25_ranking(query: str, chunks: list[SearchHit]) -> list[SearchHit]:
    """Chunks ordered by BM25 score for the query (chunks with no match are left out)."""
    terms = set(_tokens(query))
    if not terms or not chunks:
        return []
    docs = [Counter(_tokens(c.text)) for c in chunks]
    lengths = [sum(d.values()) or 1 for d in docs]
    average = sum(lengths) / len(lengths)
    scored: list[tuple[float, int]] = []
    for term in terms:
        containing = sum(1 for d in docs if term in d)
        if containing:
            idf = math.log(1 + (len(docs) - containing + 0.5) / (containing + 0.5))
            for i, d in enumerate(docs):
                tf = d.get(term, 0)
                if tf:
                    norm = tf + BM25_K1 * (1 - BM25_B + BM25_B * lengths[i] / average)
                    scored.append((idf * tf * (BM25_K1 + 1) / norm, i))
    totals: dict[int, float] = {}
    for score, i in scored:
        totals[i] = totals.get(i, 0.0) + score
    return [chunks[i] for i, _ in sorted(totals.items(), key=lambda x: -x[1])]


EXPANSIONS = (
    (
        re.compile(r"current ratio", re.I),
        "total current assets total current liabilities",
    ),
    (re.compile(r"(net )?(profit )?margin", re.I), "net income total revenues"),
    (re.compile(r"debt[- ]to[- ]equity", re.I), "total debt total stockholders equity"),
    (
        re.compile(r"(eps|earnings per share)", re.I),
        "net income weighted average shares",
    ),
    (
        re.compile(r"(growth|grow|grew|change[d]?|percent)", re.I),
        "2024 2023 prior year",
    ),
)


def expand_query(question: str) -> str:
    """Add the line items a ratio needs, so search finds the right statement."""
    extra = [words for pattern, words in EXPANSIONS if pattern.search(question)]
    return f"{question} {' '.join(extra)}" if extra else question


def retrieve(store: VectorStore, question: str, doc_id: str, k: int) -> list[SearchHit]:
    """Return the top-k chunks (with page metadata) for a question in one document."""
    if not question.strip():
        return []
    question = expand_query(question)
    vector_hits = store.search(question, doc_id, max(k, CANDIDATES))
    chunks = store.document_chunks(doc_id)
    keyword_hits = _bm25_ranking(question, chunks)[:CANDIDATES]
    # Figures live in statement tables, so tables get a second keyword vote.
    table_hits = _bm25_ranking(question, [c for c in chunks if _is_table(c.text)])[
        :CANDIDATES
    ]

    fused: dict[int, float] = {}
    by_index: dict[int, SearchHit] = {}
    for ranking in (vector_hits, keyword_hits, table_hits):
        for rank, hit in enumerate(ranking):
            fused[hit.chunk_index] = fused.get(hit.chunk_index, 0.0) + 1 / (
                RRF_K + rank + 1
            )
            by_index.setdefault(hit.chunk_index, hit)
    best = sorted(fused, key=lambda i: -fused[i])[:k]
    return [
        SearchHit(
            doc_id=by_index[i].doc_id,
            filename=by_index[i].filename,
            page=by_index[i].page,
            chunk_index=i,
            text=by_index[i].text,
            score=fused[i],
        )
        for i in best
    ]
