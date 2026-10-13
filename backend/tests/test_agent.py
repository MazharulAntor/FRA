import json

from app.agent.graph import build_graph, run_agent
from app.agent.nodes import AgentDeps, parse_number, unsupported_numbers
from app.agent.prompts import ADVICE_REFUSAL, NOT_FOUND_MESSAGE
from app.rag.vector_store import SearchHit
from tests.fakes import scripted

TEXT_45 = "Income statement. Total revenue FY2023: 10,000 million. Total revenue FY2024: 12,000 million."
TEXT_46 = "Net income FY2024: 1,500 million. Net income FY2023: 1,200 million."


def hit(page, text, idx=0):
    return SearchHit("d1", "Annual_Report.pdf", page, idx, text, 0.9)


class Search:
    """Fake search that records its calls."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def __call__(self, query, doc_id, k):
        self.calls.append((query, doc_id, k))
        return self.hits


def run(llm, search, question="What was total revenue in FY2024?", history=None):
    graph = build_graph(AgentDeps(llm=llm, search=search, top_k=6))
    return run_agent(graph, question, "d1", history)


def cls(label):
    return json.dumps({"question_type": label})


VERIFY_0 = json.dumps({"supporting": [0]})
VERIFY_01 = json.dumps({"supporting": [0, 1]})


def fig(name, value, period, page=45, unit="USD million"):
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "period": period,
        "page": page,
        "filename": "Annual_Report.pdf",
    }


def extraction(*figs):
    return json.dumps({"figures": list(figs)})


# ---------- routing ----------


def test_advice_is_refused_without_retrieval():
    search = Search([hit(45, TEXT_45)])
    state = run(scripted(classify=cls("advice")), search, "Should I buy this stock?")
    assert state["answer"] == ADVICE_REFUSAL
    assert state["citations"] == [] and state["chart"] is None
    assert search.calls == []


def test_out_of_scope_is_refused():
    search = Search([])
    state = run(scripted(classify=cls("out_of_scope")), search, "What is the weather?")
    assert "outside" in state["answer"] and search.calls == []


def test_empty_retrieval_is_not_grounded():
    llm = scripted(classify=cls("lookup"))
    state = run(llm, Search([]))
    assert state["grounded"] is False and state["answer"] == NOT_FOUND_MESSAGE
    assert "verify" not in llm.seen and "extract" not in llm.seen


def test_no_supporting_chunk_is_not_grounded():
    llm = scripted(classify=cls("lookup"), verify=json.dumps({"supporting": []}))
    state = run(llm, Search([hit(3, "Board of directors list.")]))
    assert state["grounded"] is False and state["answer"] == NOT_FOUND_MESSAGE
    assert "extract" not in llm.seen


# ---------- lookup ----------


def lookup_llm(answer_text):
    return scripted(
        classify=cls("lookup"),
        verify=VERIFY_0,
        extract=extraction(fig("Total revenue", "12,000", "FY2024")),
        answer=answer_text,
    )


def test_lookup_answer_has_citation_and_no_calculator():
    llm = lookup_llm(
        "Total revenue in FY2024 was 12,000 million [Annual_Report.pdf, p.45]."
    )
    state = run(llm, Search([hit(45, TEXT_45)]))
    assert state["grounded"] is True
    assert state["question_type"] == "lookup"
    assert "calculate" not in llm.seen
    assert [c["page"] for c in state["citations"]] == [45]
    assert "12,000" in state["citations"][0]["snippet"]
    assert len(state["citations"][0]["snippet"]) <= 200
    assert state["chart"] is None  # only one number


def test_missing_citation_is_added_and_invalid_page_dropped():
    llm = lookup_llm(
        "Total revenue in FY2024 was 12,000 million [Annual_Report.pdf, p.99]."
    )
    state = run(llm, Search([hit(45, TEXT_45)]))
    assert "p.99" not in state["answer"]
    assert state["answer"].endswith("[Annual_Report.pdf, p.45]")
    assert [c["page"] for c in state["citations"]] == [45]


def test_invented_figure_is_dropped_by_guard():
    llm = scripted(
        classify=cls("lookup"),
        verify=VERIFY_0,
        extract=extraction(
            fig("Total revenue", "12,000", "FY2024"), fig("Profit", "777", "FY2024")
        ),
        answer="Revenue was 12,000 million [p.45].",
    )
    state = run(llm, Search([hit(45, TEXT_45)]))
    assert [f["value"] for f in state["extracted"]] == ["12,000"]


def test_wrong_page_in_extraction_is_corrected():
    llm = scripted(
        classify=cls("lookup"),
        verify=VERIFY_01,
        extract=extraction(fig("Net income", "1,500", "FY2024", page=45)),
        answer="Net income was 1,500 million [p.46].",
    )
    state = run(llm, Search([hit(45, TEXT_45), hit(46, TEXT_46, 1)]))
    assert state["extracted"][0]["page"] == 46


# ---------- number guard ----------


def test_number_guard_regenerates_once():
    llm = scripted(
        classify=cls("lookup"),
        verify=VERIFY_0,
        extract=extraction(fig("Total revenue", "12,000", "FY2024")),
        answer=[
            "Revenue was 13,500 million [p.45].",
            "Revenue was 12,000 million [p.45].",
        ],
    )
    state = run(llm, Search([hit(45, TEXT_45)]))
    assert llm.seen["answer"] == 2
    assert state["grounded"] is True and "12,000" in state["answer"]
    assert any(
        "13,500" in user for _, user in llm.calls
    )  # feedback was given to the model


def test_number_guard_falls_back_to_safe_message():
    llm = lookup_llm("Revenue was 13,500 million [p.45].")
    state = run(llm, Search([hit(45, TEXT_45)]))
    assert llm.seen["answer"] == 2
    assert state["answer"] == NOT_FOUND_MESSAGE and state["grounded"] is False
    assert state["citations"] == []


def test_unsupported_numbers_ignores_citation_markers_and_question_numbers():
    bad = unsupported_numbers(
        "In 2024 revenue was 12,000 [Annual_Report_2024.pdf, p.45]. Also 55.",
        [fig("Total revenue", "12,000", "FY2024")],
        [],
        [],
        "What was revenue in 2024?",
        [],
    )
    assert bad == ["55"]


def test_parse_number():
    assert parse_number("12,345") == 12345.0
    assert parse_number("(1,234)") == -1234.0
    assert parse_number("-5.2%") == -5.2
    assert parse_number("n/a") is None


# ---------- calculation ----------


def calc_llm(plan, answer_text="Revenue grew 20 percent [p.45]."):
    return scripted(
        classify=cls("calculation"),
        verify=VERIFY_0,
        extract=extraction(
            fig("Total revenue", "10,000", "FY2023"),
            fig("Total revenue", "12,000", "FY2024"),
        ),
        calculate=json.dumps(plan),
        answer=answer_text,
    )


def test_calculation_uses_calculator_and_builds_chart():
    plan = {"calls": [{"name": "percent_change", "args": {"old": "f1", "new": "f2"}}]}
    llm = calc_llm(
        plan, "Revenue grew 20 percent from 10,000 to 12,000 million [p.45]."
    )
    state = run(llm, Search([hit(45, TEXT_45)]), "By what percent did revenue grow?")
    calc = state["calculations"][0]
    assert (
        calc["name"] == "percent_change"
        and calc["result"] == 20.0
        and calc["unit"] == "%"
    )
    assert state["grounded"] is True
    chart = state["chart"]
    assert chart["chart_type"] == "bar" and len(chart["series"]) == 2
    assert chart["citations"][0]["page"] == 45


def test_calculation_error_is_reported_not_raised():
    plan = {"calls": [{"name": "ratio", "args": {"a": "f1", "b": 0}}]}
    llm = calc_llm(plan, "It could not be calculated [p.45].")
    state = run(llm, Search([hit(45, TEXT_45)]))
    assert "error" in state["calculations"][0]
    assert any("ERROR" in user for _, user in llm.calls)


def test_literal_numbers_must_come_from_question():
    plan = {
        "calls": [{"name": "cagr", "args": {"start": "f1", "end": "f2", "years": 5}}]
    }
    state = run(calc_llm(plan), Search([hit(45, TEXT_45)]), "What was the growth?")
    assert "error" in state["calculations"][0]
    plan = {
        "calls": [{"name": "cagr", "args": {"start": "f1", "end": "f2", "years": 1}}]
    }
    state = run(
        calc_llm(plan), Search([hit(45, TEXT_45)]), "What was the CAGR over 1 year?"
    )
    assert state["calculations"][0]["result"] == 20.0


def test_unit_mismatch_is_an_error():
    llm = scripted(
        classify=cls("calculation"),
        verify=VERIFY_0,
        extract=extraction(
            fig("Revenue", "10,000", "FY2023", unit="USD million"),
            fig("Revenue", "12,000", "FY2024", unit="%"),
        ),
        calculate=json.dumps(
            {"calls": [{"name": "difference", "args": {"a": "f2", "b": "f1"}}]}
        ),
        answer="They cannot be compared [p.45].",
    )
    state = run(llm, Search([hit(45, TEXT_45)]))
    assert "error" in state["calculations"][0]
    assert state["chart"] is None  # mixed units are never charted together


def test_trend_makes_sorted_line_chart():
    llm = scripted(
        classify=cls("trend"),
        verify=VERIFY_0,
        extract=extraction(
            fig("Total revenue", "12,000", "FY2024"),
            fig("Total revenue", "10,000", "FY2023"),
        ),
        calculate=json.dumps(
            {"calls": [{"name": "percent_change", "args": {"old": "f2", "new": "f1"}}]}
        ),
        answer="Revenue rose from 10,000 to 12,000 million, up 20 percent [p.45].",
    )
    state = run(llm, Search([hit(45, TEXT_45)]), "How did revenue trend?")
    chart = state["chart"]
    assert chart["chart_type"] == "line"
    assert [s["period"] for s in chart["series"]] == ["FY2023", "FY2024"]


# ---------- follow-ups ----------


def test_follow_up_rewrites_query_with_history():
    llm = scripted(
        classify=cls("lookup"),
        rewrite="total revenue FY2023",
        verify=VERIFY_0,
        extract=extraction(fig("Total revenue", "10,000", "FY2023")),
        answer="Total revenue in FY2023 was 10,000 million [p.45].",
    )
    search = Search([hit(45, TEXT_45)])
    history = [
        {"role": "user", "content": "What was total revenue in FY2024?"},
        {"role": "assistant", "content": "It was 12,000 million."},
    ]
    state = run(llm, search, "and for 2023?", history)
    assert search.calls[0][0] == "total revenue FY2023"
    assert state["grounded"] is True
    assert any("earlier" in user.lower() and "12,000" in user for _, user in llm.calls)


def test_no_history_means_no_rewrite_call():
    llm = lookup_llm("Revenue was 12,000 million [p.45].")
    run(llm, Search([hit(45, TEXT_45)]))
    assert "rewrite" not in llm.seen


def test_space_separated_thousands():
    from app.agent.nodes import numbers_in

    assert numbers_in("Sales 3 89 490 91 354 and 9 254") == [
        3.0,
        89490.0,
        91354.0,
        9254.0,
    ]
    assert parse_number("89 490") == 89490.0
    assert numbers_in("FY2024 2025 12,345") == [2024.0, 2025.0, 12345.0]


def test_empty_answer_is_regenerated():
    llm = scripted(
        classify=cls("lookup"),
        verify=VERIFY_0,
        extract=extraction(fig("Total revenue", "12,000", "FY2024")),
        answer=["   ", "Revenue was 12,000 million [p.45]."],
    )
    state = run(llm, Search([hit(45, TEXT_45)]))
    assert llm.seen["answer"] == 2 and "12,000" in state["answer"]
    both_empty = scripted(
        classify=cls("lookup"),
        verify=VERIFY_0,
        extract=extraction(fig("Total revenue", "12,000", "FY2024")),
        answer="",
    )
    state = run(both_empty, Search([hit(45, TEXT_45)]))
    assert state["answer"] == NOT_FOUND_MESSAGE and state["grounded"] is False


def test_fix_type_and_unit_cleaning():
    from app.agent.nodes import _fix_type, clean_unit

    assert (
        _fix_type("What was the net profit margin in 2024?", "lookup") == "calculation"
    )
    assert _fix_type("How did revenue trend over the years?", "lookup") == "trend"
    assert _fix_type("What was total revenue?", "lookup") == "lookup"
    assert _fix_type("Should I buy it? what is the ratio", "advice") == "advice"
    assert clean_unit("A$'000") == "thousand AUD"
    assert clean_unit("USD million") == "USD million"


def test_verify_keeps_top_ranked_chunks():
    chunks = [
        hit(1, "a", 0),
        hit(2, "b", 1),
        hit(3, "c", 2),
        hit(4, "d", 3),
        hit(5, "e", 4),
    ]
    llm = scripted(
        classify=cls("lookup"),
        verify=json.dumps({"supporting": [4]}),
        extract=extraction(),
        answer="x",
    )
    state = run(llm, Search(chunks))
    assert [c["page"] for c in state["verified_chunks"]] == [1, 2, 3, 5]
