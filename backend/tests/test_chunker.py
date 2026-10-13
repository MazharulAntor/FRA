from app.rag.chunker import chunk_pages
from app.rag.pdf_reader import PageText


def _pages():
    return [
        PageText(1, "alpha beta gamma. " * 120),
        PageText(2, "delta epsilon. " * 120),
    ]


def test_every_chunk_has_metadata_and_one_page():
    chunks = chunk_pages(_pages(), "d1", "r.pdf", 1000, 150)
    assert chunks
    for i, c in enumerate(chunks):
        assert c.doc_id == "d1" and c.filename == "r.pdf"
        assert c.page in (1, 2)
        assert c.chunk_index == i
    assert {c.page for c in chunks} == {1, 2}
    assert not any("delta" in c.text for c in chunks if c.page == 1)


def test_sizes_within_limit():
    assert all(
        len(c.text) <= 1000 for c in chunk_pages(_pages(), "d1", "r.pdf", 1000, 150)
    )


def test_overlap_works():
    chunks = chunk_pages([PageText(1, "word " * 600)], "d", "f", 200, 50)
    assert len(chunks) > 2
    assert chunks[0].text.split()[-1] in chunks[1].text


def test_empty_page_skipped():
    assert chunk_pages([PageText(1, "   ")], "d", "f") == []
