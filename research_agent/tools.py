"""Search backends and the model factory."""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

from langchain_anthropic import ChatAnthropic

from research_agent.state import Source


class SearchTool(Protocol):
    """Anything with this shape can be plugged into the search node."""

    def search(self, query: str, max_results: int = 3) -> list[Source]: ...


class TavilySearch:
    """Live web search.

    `include_raw_content` pulls full page text instead of Tavily's short
    snippet. Verification can only check a claim against text that actually
    contains it, so raw content matters more here than in a typical RAG setup.
    """

    def __init__(
        self,
        api_key: str,
        search_depth: str = "advanced",
        include_raw_content: bool = True,
        max_chars: int = 4000,
    ) -> None:
        from tavily import TavilyClient

        self._client = TavilyClient(api_key=api_key)
        self._search_depth = search_depth
        self._include_raw = include_raw_content
        self._max_chars = max_chars

    def search(self, query: str, max_results: int = 3) -> list[Source]:
        response = self._client.search(
            query=query,
            max_results=max_results,
            search_depth=self._search_depth,
            include_raw_content=self._include_raw,
            include_answer=False,
        )

        sources: list[Source] = []
        for result in response.get("results", []):
            text = result.get("raw_content") or result.get("content") or ""
            sources.append(
                Source(
                    url=result["url"],
                    title=result.get("title") or result["url"],
                    content=text[: self._max_chars].strip(),
                )
            )
        return sources


class MockSearch:
    """Offline stand-in so the graph runs without a Tavily key.

    Deterministic: the same query always returns the same URLs, which keeps
    tests stable. The content is fabricated — useful for wiring, useless for
    judging report quality.
    """

    def search(self, query: str, max_results: int = 3) -> list[Source]:
        digest = hashlib.sha1(query.encode()).hexdigest()[:8]
        return [
            Source(
                url=f"https://example.com/{digest}/{i}",
                title=f"Mock source {i}: {query}",
                content=(
                    f"This mock source discusses {query}. "
                    f"Any figures here are placeholders and support nothing."
                ),
            )
            for i in range(1, max_results + 1)
        ]


def get_search_tool() -> SearchTool:
    """Live search when TAVILY_API_KEY is set, mock otherwise."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    return TavilySearch(api_key) if api_key else MockSearch()


def using_mock_search() -> bool:
    return not os.getenv("TAVILY_API_KEY", "").strip()


def get_llm(max_tokens: int = 2048) -> ChatAnthropic:
    """One model, configured per call site by token budget.

    temperature=0 because this pipeline is a verification system — the same
    sources should produce the same claims run to run, or the eval numbers
    mean nothing.
    """
    return ChatAnthropic(
        model=os.getenv("RESEARCH_MODEL", "claude-haiku-4-5"),
        max_tokens=max_tokens,
        temperature=0,
    )