"""Evaluation script: runs the agent on eval_questions.json and prints/writes the metrics.

Run from backend/:   python eval/run_eval.py
Needs: the PDFs named in eval_questions.json inside data/sample_reports/, and a working LLM in .env.
Writes eval/results.md (markdown table) and eval/results.csv (one row per question).
"""

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.agent.nodes import numbers_in, parse_number  # noqa: E402

CALC_TOLERANCE = 0.01  # 1 percent
MARKER_RE = re.compile(r"\[[^\]]*p\.\s*\d+\]")
CSV_COLUMNS = [
    "doc",
    "question",
    "type",
    "predicted_type",
    "expected_pages",
    "retrieved_pages",
    "cited_pages",
    "retrieval_hit",
    "precision_at_k",
    "citation_ok",
    "hallucinated",
    "calc_ok",
    "answer_ok",
    "not_found",
    "grounded",
    "latency_s",
    "answer",
]


# ---------- loading ----------


def load_questions(path: Path, include_examples: bool = False) -> list[dict]:
    """Read the question list. Entries marked _template are always skipped."""
    items = json.loads(path.read_text(encoding="utf-8"))
    return [
        q
        for q in items
        if not q.get("_template") and (include_examples or not q.get("_example"))
    ]


def answer_body(answer: str) -> str:
    """The answer text without [file, p.N] citation markers."""
    return MARKER_RE.sub(" ", answer)


# ---------- scoring one question (pure functions, unit tested) ----------


def _calc_results(calculations: list[dict]) -> set[float]:
    """Calculator results, in every rounding the answer might use (these are not on a page)."""
    return {
        round(abs(calc["result"]), places)
        for calc in calculations
        if "result" in calc
        for places in range(5)
    }


def _calc_numbers(calculations: list[dict]) -> set[float]:
    """Numbers an answer may use from calculations: the results and the inputs."""
    allowed = _calc_results(calculations)
    for calc in calculations:
        stack: list[Any] = list(calc.get("inputs", {}).values())
        while stack:
            item = stack.pop()
            if isinstance(item, dict) and isinstance(item.get("value"), (int, float)):
                allowed.add(round(abs(float(item["value"])), 6))
            elif isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                allowed.add(round(abs(float(item)), 6))
    return allowed


def stated_numbers(answer: str, question: str) -> list[float]:
    """Numbers the answer states (citation markers and numbers from the question excluded)."""
    asked = {round(n, 6) for n in numbers_in(question)}
    return [n for n in numbers_in(answer_body(answer)) if round(n, 6) not in asked]


def score_question(
    q: dict,
    state: dict,
    page_numbers: dict[int, set[float]],
    latency: float,
) -> dict:
    """Compute every per-question metric. None means 'not applicable to this question'."""
    answer = state.get("answer", "")
    chunks = state.get("chunks") or []
    citations = state.get("citations") or []
    calculations = state.get("calculations") or []
    expected_pages = set(q.get("expected_pages") or [])
    retrieved_pages = [c["page"] for c in chunks]
    cited_pages = sorted({c["page"] for c in citations})

    retrieval_hit: bool | None = None
    precision_at_k: float | None = None
    if expected_pages:
        on_target = [p for p in retrieved_pages if p in expected_pages]
        retrieval_hit = bool(on_target)
        precision_at_k = (
            len(on_target) / len(retrieved_pages) if retrieved_pages else 0.0
        )

    # Citation accuracy: every figure the answer states must be on a cited page.
    citation_ok: bool | None = None
    calc_numbers = _calc_numbers(calculations)
    calc_results = _calc_results(calculations)
    figures = [
        n
        for n in stated_numbers(answer, q["question"])
        if round(n, 6) not in calc_results
    ]
    if citations and figures:
        cited_text_numbers: set[float] = set()
        for page in cited_pages:
            cited_text_numbers |= page_numbers.get(page, set())
        citation_ok = all(round(n, 6) in cited_text_numbers for n in figures)

    # Hallucination: a number in the answer that is in neither the retrieved text nor a calculation.
    retrieved_numbers = {round(n, 6) for c in chunks for n in numbers_in(c["text"])}
    hallucinated = any(
        round(n, 6) not in retrieved_numbers and round(n, 6) not in calc_numbers
        for n in stated_numbers(answer, q["question"])
    )

    calc_ok: bool | None = None
    if q.get("type") == "calculation" and q.get("expected_value") is not None:
        expected = parse_number(str(q["expected_value"]))
        if expected is not None:
            calc_ok = any(
                "result" in c
                and abs(c["result"] - expected) <= CALC_TOLERANCE * abs(expected)
                for c in calculations
            )

    # Answer correctness: the expected figure appears in the answer text (lookup / comparison /
    # trend), or the calculator produced the expected result (calculation).
    answer_ok: bool | None = None
    expected_text = q.get("expected_value")
    if expected_text is not None and q.get("type") != "summary":
        expected_number = parse_number(str(expected_text))
        if expected_number is not None:
            if q.get("type") == "calculation":
                answer_ok = calc_ok
            else:
                wanted = {abs(expected_number)}
                wanted |= {
                    abs(parse_number(v) or 0) for v in q.get("expected_also", [])
                }
                stated = {round(n, 6) for n in numbers_in(answer_body(answer))}
                answer_ok = bool(wanted & stated)
    not_found = answer.strip().startswith("I could not find")

    return {
        "doc": q["doc"],
        "question": q["question"],
        "type": q.get("type", ""),
        "predicted_type": state.get("question_type", ""),
        "expected_pages": " ".join(str(p) for p in sorted(expected_pages)),
        "retrieved_pages": " ".join(str(p) for p in sorted(set(retrieved_pages))),
        "cited_pages": " ".join(str(p) for p in cited_pages),
        "retrieval_hit": retrieval_hit,
        "precision_at_k": precision_at_k,
        "citation_ok": citation_ok,
        "hallucinated": hallucinated,
        "calc_ok": calc_ok,
        "answer_ok": answer_ok,
        "not_found": not_found,
        "grounded": state.get("grounded"),
        "latency_s": round(latency, 2),
        "answer": answer.replace("\n", " "),
    }


def _rate(values: list[bool | None]) -> tuple[float | None, int]:
    """Share of True among the applicable (not None) values, and how many were applicable."""
    applicable = [v for v in values if v is not None]
    if not applicable:
        return None, 0
    return sum(applicable) / len(applicable), len(applicable)


def aggregate(rows: list[dict]) -> list[tuple[str, str, int]]:
    """Return the metrics as (name, value text, number of questions used)."""

    def pct(r: float | None) -> str:
        return "n/a" if r is None else f"{r * 100:.1f}%"

    hit, n_hit = _rate([r["retrieval_hit"] for r in rows])
    prec_values = [r["precision_at_k"] for r in rows if r["precision_at_k"] is not None]
    cite, n_cite = _rate([r["citation_ok"] for r in rows])
    hall, n_hall = _rate([r["hallucinated"] for r in rows])
    calc, n_calc = _rate([r["calc_ok"] for r in rows])
    correct, n_correct = _rate([r["answer_ok"] for r in rows])
    missed, n_missed = _rate([r["not_found"] for r in rows])
    latencies = sorted(r["latency_s"] for r in rows)
    p90 = (
        latencies[min(len(latencies) - 1, int(0.9 * len(latencies)))]
        if latencies
        else 0.0
    )
    return [
        ("Retrieval precision (hit on expected page)", pct(hit), n_hit),
        (
            "Retrieval precision@k (share of retrieved chunks on expected pages)",
            pct(statistics.mean(prec_values)) if prec_values else "n/a",
            len(prec_values),
        ),
        (
            "Answer correctness (expected figure in answer / calculator result)",
            pct(correct),
            n_correct,
        ),
        ('Answered "could not find" (lower is better)', pct(missed), n_missed),
        ("Citation accuracy", pct(cite), n_cite),
        ("Hallucination rate (lower is better)", pct(hall), n_hall),
        ("Calculation correctness (within 1%)", pct(calc), n_calc),
        (
            "Latency median (s)",
            f"{statistics.median(latencies):.2f}" if latencies else "n/a",
            len(latencies),
        ),
        ("Latency p90 (s)", f"{p90:.2f}", len(latencies)),
    ]


# ---------- output ----------


def metrics_markdown(metrics: list[tuple[str, str, int]], total: int) -> str:
    lines = [
        "# Evaluation results",
        "",
        f"Questions run: {total}",
        "",
        "| Metric | Value | Questions used |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {value} | {n} |" for name, value, n in metrics]
    return "\n".join(lines) + "\n"


def write_results(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, csv_path = out_dir / "results.md", out_dir / "results.csv"
    md_path.write_text(metrics_markdown(aggregate(rows), len(rows)), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return md_path, csv_path


# ---------- running the agent ----------


def _setup_work_dirs(work_dir: Path, fresh: bool) -> None:
    """Use separate upload/vector folders so evaluation never touches the app's documents."""
    if fresh and work_dir.exists():
        shutil.rmtree(work_dir)
    os.environ["UPLOAD_DIR"] = str(work_dir / "uploads")
    os.environ["CHROMA_PATH"] = str(work_dir / "chroma")


def _ingest(pdf: Path) -> str:
    """Ingest a PDF (or reuse an earlier ingestion with the same file name). Returns doc_id."""
    from app.config import get_settings
    from app.ingest import ingest_document
    from app.rag.pdf_reader import count_pages
    from app.schemas import DocumentInfo
    from app.services import get_doc_store

    store = get_doc_store()
    for doc in store.list():
        if doc.filename == pdf.name and doc.status == "ready":
            return doc.doc_id
    doc_id = uuid.uuid4().hex[:12]
    upload_dir = Path(get_settings().upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{doc_id}.pdf"
    shutil.copyfile(pdf, target)
    store.add(
        DocumentInfo(
            doc_id=doc_id,
            filename=pdf.name,
            pages=count_pages(target),
            status="processing",
        )
    )
    ingest_document(doc_id, target, pdf.name)
    info = store.get(doc_id)
    if info is None or info.status != "ready":
        raise RuntimeError(
            f"Could not ingest {pdf.name}: {info.error if info else 'unknown error'}"
        )
    return doc_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--questions", type=Path, default=BACKEND_DIR / "eval" / "eval_questions.json"
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=REPO_DIR / "data" / "sample_reports"
    )
    parser.add_argument("--out-dir", type=Path, default=BACKEND_DIR / "eval")
    parser.add_argument("--work-dir", type=Path, default=BACKEND_DIR / "eval" / ".work")
    parser.add_argument(
        "--limit", type=int, default=0, help="only run the first N questions"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="re-ingest PDFs from scratch"
    )
    parser.add_argument(
        "--include-examples",
        action="store_true",
        help="also run entries marked _example",
    )
    args = parser.parse_args(argv)

    questions = load_questions(args.questions, args.include_examples)
    if args.limit:
        questions = questions[: args.limit]
    runnable = [q for q in questions if (args.reports_dir / q["doc"]).exists()]
    for name in sorted({q["doc"] for q in questions} - {q["doc"] for q in runnable}):
        print(
            f"WARNING: {args.reports_dir / name} not found; its questions are skipped."
        )
    if not runnable:
        print(
            "No runnable questions. Put your PDFs in data/sample_reports/ and add questions "
            "for them to eval/eval_questions.json (see the _template entry)."
        )
        return 1

    _setup_work_dirs(args.work_dir, args.fresh)
    from app.agent.graph import run_agent
    from app.llm.provider import LLMError
    from app.rag.pdf_reader import read_pdf
    from app.services import get_agent

    try:
        graph = get_agent()
    except LLMError as exc:
        print(f"LLM is not configured: {exc} (check .env)")
        return 1

    doc_ids: dict[str, str] = {}
    page_numbers: dict[str, dict[int, set[float]]] = {}
    rows: list[dict] = []
    for i, q in enumerate(runnable, start=1):
        name = q["doc"]
        if name not in doc_ids:
            print(f"Ingesting {name} ...")
            doc_ids[name] = _ingest(args.reports_dir / name)
            page_numbers[name] = {
                p.page_number: set(numbers_in(p.text))
                for p in read_pdf(args.reports_dir / name)
            }
        started = time.perf_counter()
        try:
            state = run_agent(graph, q["question"], doc_ids[name], [])
        except LLMError as exc:
            print(f"[{i}/{len(runnable)}] LLM error: {exc}")
            state = {"answer": f"ERROR: {exc}", "grounded": False}
        elapsed = time.perf_counter() - started
        row = score_question(q, state, page_numbers[name], elapsed)
        rows.append(row)
        print(f"[{i}/{len(runnable)}] {elapsed:5.1f}s  {q['question'][:70]}")

    md_path, csv_path = write_results(rows, args.out_dir)
    print()
    print(metrics_markdown(aggregate(rows), len(rows)))
    print(f"Wrote {md_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
