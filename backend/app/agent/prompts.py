"""All prompt templates. Every prompt starts with a TASK tag (used by the test fake)."""

import json

SYSTEM_PROMPT = (
    "You are a careful financial report analyst. Use only the provided context. "
    "If the context does not contain the answer, say so. Never guess a number. "
    "Never give investment advice."
)

NOT_FOUND_MESSAGE = "I could not find this in the report."
ADVICE_REFUSAL = (
    "I can't give investment advice or buy/sell recommendations. "
    "I can answer factual questions about this report, such as figures, ratios and trends."
)
OUT_OF_SCOPE_REFUSAL = (
    "That question is outside what I can answer. "
    "I can answer questions about the figures and text in the uploaded report."
)

CALCULATOR_FUNCTIONS = """\
percent_change(old, new)        -> (new - old) / abs(old) * 100
ratio(a, b)                     -> a / b
margin(profit, revenue)         -> profit / revenue * 100
cagr(start, end, years)         -> compound annual growth rate, percent
difference(a, b)                -> a - b
total(values)                   -> sum of a list
average(values)                 -> mean of a list
debt_to_equity(debt, equity)    -> debt / equity
current_ratio(assets, liabilities) -> current assets / current liabilities
eps(net_income, shares)         -> net income / shares"""


def format_history(history: list[dict], limit: int = 6) -> str:
    """Last few chat messages as plain text."""
    lines = []
    for msg in history[-limit:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines) if lines else "(no earlier messages)"


def format_chunks(chunks: list[dict]) -> str:
    """Numbered chunks with their source, for the LLM context."""
    return "\n\n".join(
        f"[Chunk {i} | {c['filename']}, p.{c['page']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )


def classify_prompt(question: str, history: list[dict]) -> str:
    return f"""TASK: classify
Classify the user's latest question into exactly one label.

Labels:
- lookup: asks for one specific figure or fact from the report
- calculation: needs a ratio, growth rate, margin, difference, total or average
- comparison: compares two or more items or periods
- trend: asks how something changed over several periods
- summary: asks for an overview or explanation of a section
- advice: asks whether to buy/sell/invest, for predictions or recommendations
- out_of_scope: not about the report's content (chit-chat, other companies, live market data)

Use the earlier messages to understand follow-ups such as "and for 2023?".

Earlier messages:
{format_history(history)}

Latest question: {question}

Reply with JSON only: {{"question_type": "<label>"}}"""


def rewrite_prompt(question: str, history: list[dict]) -> str:
    return f"""TASK: rewrite
Rewrite the latest question as one standalone search query for a financial report.
Resolve words like "it", "that" or "and for 2023?" using the earlier messages.
Reply with the query text only.

Earlier messages:
{format_history(history)}

Latest question: {question}"""


def verify_prompt(question: str, chunks: list[dict]) -> str:
    return f"""TASK: verify
Question: {question}

For each chunk below decide whether it contains information needed to answer the question.
List the numbers of the chunks that do.

{format_chunks(chunks)}

Reply with JSON only: {{"supporting": [<chunk numbers>]}}"""


CALC_HINTS = """
This question needs a calculation, so extract EVERY input the calculation needs, each with its period:
- growth or change between periods: the value for BOTH periods
- margin: the profit figure AND revenue for the same period
- current ratio: total current assets AND total current liabilities
- debt to equity: total debt AND total equity
- earnings per share: net income AND number of shares
- trend: the value for every period shown
"""


def extract_prompt(question: str, chunks: list[dict], calculation: bool = False) -> str:
    hints = CALC_HINTS if calculation else ""
    return f"""TASK: extract
Question: {question}

Find the figures in the chunks that are needed to answer the question.
Rules:
- Copy each value EXACTLY as written in the text (keep commas, decimals, brackets).
- Never calculate, convert or round anything.
- "unit" is the unit stated in the text (for example "USD million", "%", "thousand"), or null.
- "period" is the year or period the figure belongs to (for example "FY2024"), or null.
- "page" and "filename" come from the chunk header.
- If a figure is not in the text, do not include it.
- Prefer the main financial statements of the entity the question names over notes or summaries.
{hints}

{format_chunks(chunks)}

Reply with JSON only:
{{"figures": [{{"name": "...", "value": "...", "unit": "...", "period": "...", "page": 0, "filename": "..."}}]}}"""


def calculate_prompt(question: str, figures: list[dict]) -> str:
    listing = "\n".join(
        f'{f["id"]}: {f["name"]} = {f["value"]} {f.get("unit") or ""} ({f.get("period") or "no period"})'
        for f in figures
    )
    return f"""TASK: calculate
Question: {question}

Choose the calculator functions needed to answer the question. You must NOT do any arithmetic yourself.
Available functions:
{CALCULATOR_FUNCTIONS}

Figures you can use (refer to them by id, for example "f1"):
{listing}

For list arguments (total, average) pass a list of ids. Use a plain number only for values
that the user wrote in the question (for example the number of years).
For a trend, call percent_change once for each consecutive pair of periods.

Reply with JSON only:
{{"calls": [{{"name": "percent_change", "args": {{"old": "f1", "new": "f2"}}}}]}}"""


def answer_prompt(
    question: str,
    query: str,
    chunks: list[dict],
    figures: list[dict],
    calculations: list[dict],
    feedback: str | None = None,
) -> str:
    figure_lines = (
        "\n".join(
            f'- {f["name"]}: {f["value"]} {f.get("unit") or ""} ({f.get("period") or "no period"}) '
            f'[{f["filename"]}, p.{f["page"]}]'
            for f in figures
        )
        or "(none)"
    )
    calc_lines = (
        "\n".join(
            f"- {c['name']}: "
            + (
                f"ERROR: {c['error']}"
                if c.get("error")
                else f"result {c['result']} {c.get('unit') or ''} ({c.get('formula')}); inputs {json.dumps(c['inputs'])}"
            )
            for c in calculations
        )
        or "(none)"
    )
    extra = f"\n\nIMPORTANT: {feedback}" if feedback else ""
    return f"""TASK: answer
Question: {question}
Standalone question: {query}

Context chunks:
{format_chunks(chunks)}

Extracted figures:
{figure_lines}

Calculation results (computed by code):
{calc_lines}

Rules:
- Use ONLY the context, extracted figures and calculation results above.
- Every number you write must be one of the extracted figures or calculation results, written exactly as given. Do not convert units or do any arithmetic.
- State the unit and period for every figure, copying the figure and its unit exactly (for example "17,999 thousand AUD"). Never convert thousands to full amounts.
- If a calculation shows an ERROR, say it could not be calculated and why.
- After each statement that uses a figure, add a citation in the form [filename, p.N] using the filename and page shown above.
- Keep the answer under 150 words unless the user asks for detail.
- No investment advice.{extra}

Write the answer now."""
