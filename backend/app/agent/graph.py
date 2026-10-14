"""LangGraph wiring.

classify --advice/out_of_scope--> answer
classify --otherwise--> retrieve --> verify
verify --not grounded--> answer
verify --ok--> extract --calculation/comparison/trend--> calculate --> answer
                       --lookup/summary--> answer
"""

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent import nodes
from app.agent.nodes import AgentDeps
from app.agent.state import CALC_TYPES, REFUSAL_TYPES, AgentState


def _after_classify(state: AgentState) -> str:
    return "answer" if state.get("question_type") in REFUSAL_TYPES else "retrieve"


def _after_verify(state: AgentState) -> str:
    return "extract" if state.get("grounded") else "answer"


def _after_extract(state: AgentState) -> str:
    return "calculate" if state.get("question_type") in CALC_TYPES else "answer"


def build_graph(deps: AgentDeps) -> Any:
    """Compile the six-step agent graph."""
    graph = StateGraph(AgentState)
    for name in ("classify", "retrieve", "verify", "extract", "calculate", "answer"):
        graph.add_node(name, partial(getattr(nodes, name), deps=deps))
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", _after_classify, ["answer", "retrieve"])
    graph.add_edge("retrieve", "verify")
    graph.add_conditional_edges("verify", _after_verify, ["extract", "answer"])
    graph.add_conditional_edges("extract", _after_extract, ["calculate", "answer"])
    graph.add_edge("calculate", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def run_agent(
    graph: Any, question: str, doc_id: str, history: list[dict] | None = None
) -> AgentState:
    """Run the graph for one question and return the final state."""
    initial: AgentState = {
        "question": question,
        "history": history or [],
        "doc_id": doc_id,
        "grounded": True,
        "extracted": [],
        "calculations": [],
        "citations": [],
        "chart": None,
        "error": None,
    }
    return graph.invoke(initial)
