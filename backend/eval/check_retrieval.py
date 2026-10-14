"""Retrieval-only check (no LLM, no API quota): does search find the right page and figure?

Run from backend/:   python eval/check_retrieval.py [--k 8] [--work-dir eval/.work]
Uses the PDFs already ingested by run_eval.py (or ingests them first).
"""

import argparse
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--work-dir", type=Path, default=BACKEND_DIR / "eval" / ".work")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=BACKEND_DIR.parent / "data" / "sample_reports",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.environ["UPLOAD_DIR"] = str(args.work_dir / "uploads")
    os.environ["CHROMA_PATH"] = str(args.work_dir / "chroma")
    from app.agent.nodes import numbers_in, parse_number
    from app.rag.retriever import retrieve
    from app.services import get_vector_store
    from eval.run_eval import _ingest, load_questions

    questions = [
        q
        for q in load_questions(BACKEND_DIR / "eval" / "eval_questions.json")
        if (args.reports_dir / q["doc"]).exists()
    ]
    doc_ids = {
        name: _ingest(args.reports_dir / name)
        for name in sorted({q["doc"] for q in questions})
    }

    page_hits = value_hits = lookups = 0
    for q in questions:
        hits = retrieve(get_vector_store(), q["question"], doc_ids[q["doc"]], args.k)
        pages = [h.page for h in hits]
        page_ok = bool(set(pages) & set(q["expected_pages"]))
        page_hits += page_ok
        value_ok = None
        if q["type"] == "lookup" and q.get("expected_value"):
            lookups += 1
            want = abs(parse_number(str(q["expected_value"])) or 0)
            value_ok = any(want in numbers_in(h.text) for h in hits)
            value_hits += value_ok
        if args.verbose or not page_ok or value_ok is False:
            print(
                f"{'ok  ' if page_ok else 'MISS'} value={value_ok}  {q['doc'][:14]}  {q['question'][:55]}  expected {q['expected_pages']} got {pages}"
            )
    print(f"\nright page in top {args.k}: {page_hits}/{len(questions)}")
    print(f"lookup figure present in retrieved text: {value_hits}/{lookups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
