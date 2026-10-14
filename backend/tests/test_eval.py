from eval.run_eval import (
    aggregate,
    load_questions,
    metrics_markdown,
    score_question,
    write_results,
)

PAGE_NUMBERS = {45: {12345.0, 2024.0, 10000.0}, 46: {7.0}}


def chunk(page, text):
    return {
        "doc_id": "d",
        "filename": "r.pdf",
        "page": page,
        "chunk_index": 0,
        "text": text,
    }


def cite(page):
    return {"doc_id": "d", "filename": "r.pdf", "page": page, "snippet": "x"}


LOOKUP_Q = {
    "doc": "r.pdf",
    "question": "What was total revenue in FY2024?",
    "type": "lookup",
    "expected_pages": [45],
    "expected_value": "12,345",
}


def test_good_lookup():
    state = {
        "answer": "Revenue was 12,345 million in FY2024 [r.pdf, p.45].",
        "chunks": [chunk(45, "Revenue 12,345 million"), chunk(46, "Other 7")],
        "citations": [cite(45)],
        "calculations": [],
        "question_type": "lookup",
        "grounded": True,
    }
    row = score_question(LOOKUP_Q, state, PAGE_NUMBERS, 1.5)
    assert row["retrieval_hit"] is True and row["precision_at_k"] == 0.5
    assert row["citation_ok"] is True and row["hallucinated"] is False
    assert row["calc_ok"] is None and row["latency_s"] == 1.5


def test_hallucinated_number_and_wrong_citation_page():
    state = {
        "answer": "Revenue was 99,999 million [r.pdf, p.46].",
        "chunks": [chunk(45, "Revenue 12,345 million")],
        "citations": [cite(46)],
        "calculations": [],
    }
    row = score_question(LOOKUP_Q, state, PAGE_NUMBERS, 1.0)
    assert row["hallucinated"] is True
    assert row["citation_ok"] is False


def test_retrieval_miss():
    state = {
        "answer": "x",
        "chunks": [chunk(3, "nothing")],
        "citations": [],
        "calculations": [],
    }
    row = score_question(LOOKUP_Q, state, PAGE_NUMBERS, 1.0)
    assert row["retrieval_hit"] is False and row["precision_at_k"] == 0.0
    assert row["citation_ok"] is None  # no figures / no citations: not applicable


def test_calculation_within_one_percent():
    q = {
        **LOOKUP_Q,
        "type": "calculation",
        "question": "Growth FY2023 to FY2024?",
        "expected_value": "8.5",
    }
    base = {
        "answer": "Revenue grew 8.52 percent [r.pdf, p.45].",
        "chunks": [chunk(45, "revenue 12,345 and 11,376")],
        "citations": [cite(45)],
    }
    ok = score_question(
        q,
        {
            **base,
            "calculations": [{"name": "percent_change", "inputs": {}, "result": 8.52}],
        },
        PAGE_NUMBERS,
        1,
    )
    bad = score_question(
        q,
        {
            **base,
            "calculations": [{"name": "percent_change", "inputs": {}, "result": 9.5}],
        },
        PAGE_NUMBERS,
        1,
    )
    assert (
        ok["calc_ok"] is True and ok["hallucinated"] is False
    )  # 8.52 comes from the calculator
    assert bad["calc_ok"] is False


def test_aggregate_and_files(tmp_path):
    good = score_question(
        LOOKUP_Q,
        {
            "answer": "Revenue was 12,345 million [r.pdf, p.45].",
            "chunks": [chunk(45, "12,345")],
            "citations": [cite(45)],
            "calculations": [],
        },
        PAGE_NUMBERS,
        2.0,
    )
    bad = score_question(
        LOOKUP_Q,
        {
            "answer": "It was 5.",
            "chunks": [chunk(3, "none")],
            "citations": [],
            "calculations": [],
        },
        PAGE_NUMBERS,
        4.0,
    )
    metrics = {name: value for name, value, _ in aggregate([good, bad])}
    assert metrics["Retrieval precision (hit on expected page)"] == "50.0%"
    assert metrics["Hallucination rate (lower is better)"] == "50.0%"
    assert metrics["Latency median (s)"] == "3.00"
    md, csv_path = write_results([good, bad], tmp_path)
    assert "| Metric | Value |" in md.read_text(encoding="utf-8")
    assert len(csv_path.read_text(encoding="utf-8").strip().splitlines()) == 3
    assert "Questions run: 2" in metrics_markdown(aggregate([good, bad]), 2)


def test_question_file_loads_and_skips_template():
    from eval import run_eval

    shipped = run_eval.BACKEND_DIR / "eval" / "eval_questions.json"
    questions = load_questions(shipped)
    assert len(questions) >= 30
    assert all(q["doc"] and q["question"] and q["expected_pages"] for q in questions)
    assert len({q["doc"] for q in questions}) == 5
