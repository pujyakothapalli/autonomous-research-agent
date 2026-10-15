"""Graph assembly and control flow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from research_agent.nodes import (
    make_critic,
    make_finalizer,
    make_planner,
    make_searcher,
    make_verifier,
    make_writer,
    route_after_critique,
)
from research_agent.state import ResearchState
from research_agent.tools import SearchTool, get_search_tool


def build_graph(search: SearchTool | None = None):
    """Compile the research pipeline.

    `search` is injectable so tests and the eval harness can hold retrieval
    fixed. Defaults to live Tavily, or the mock when no key is set.
    """
    search = search or get_search_tool()

    builder = StateGraph(ResearchState)

    builder.add_node("plan", make_planner())
    builder.add_node("search", make_searcher(search))
    builder.add_node("verify", make_verifier())
    builder.add_node("write", make_writer())
    builder.add_node("critique", make_critic())
    builder.add_node("finalize", make_finalizer())

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "verify")
    builder.add_edge("verify", "write")
    builder.add_edge("write", "critique")

    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {"write": "write", "finalize": "finalize"},
    )

    builder.add_edge("finalize", END)

    return builder.compile()


def run(query: str, search: SearchTool | None = None) -> ResearchState:
    """Run one research question end to end."""
    graph = build_graph(search)
    return graph.invoke(
        {"query": query, "sources": [], "trace": [], "draft_count": 0},
        # Backstop only. The real bound is draft_count vs MAX_REVISIONS in
        # route_after_critique; this catches wiring bugs that create a cycle.
        config={"recursion_limit": 25},
    )