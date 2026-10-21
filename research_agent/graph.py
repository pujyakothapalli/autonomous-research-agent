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


def build_graph(
    search: SearchTool | None = None,
    *,
    include_critic: bool = True,
    skip_retrieval: bool = False,
):
    search = search or get_search_tool()
    builder = StateGraph(ResearchState)

    builder.add_node("verify", make_verifier())
    builder.add_node("write", make_writer())

    if skip_retrieval:
        # Sources are supplied in the input state. Used by the eval harness so
        # both arms verify and write over byte-identical retrieval.
        builder.add_edge(START, "verify")
    else:
        builder.add_node("plan", make_planner())
        builder.add_node("search", make_searcher(search))
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "search")
        builder.add_edge("search", "verify")

    builder.add_edge("verify", "write")

    if include_critic:
        builder.add_node("critique", make_critic())
        builder.add_node("finalize", make_finalizer())
        builder.add_edge("write", "critique")
        builder.add_conditional_edges(
            "critique", route_after_critique,
            {"write": "write", "finalize": "finalize"},
        )
        builder.add_edge("finalize", END)
    else:
        builder.add_edge("write", END)

    return builder.compile()


def run(
    query: str,
    search: SearchTool | None = None,
    *,
    include_critic: bool = True,
) -> ResearchState:
    """Run one research question end to end."""
    graph = build_graph(search, include_critic=include_critic)
    return graph.invoke(
        {"query": query, "sources": [], "trace": [], "draft_count": 0},
        config={"recursion_limit": 25},
    )