"""The agents. Each factory returns a callable node for the graph."""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.state import Plan, ResearchState, Review, Source, Verification
from research_agent.tools import SearchTool, get_llm


# --- Prompts ---

PLANNER_SYSTEM = """You decompose a research question into focused sub-questions.

Produce 2-5 sub-questions that together cover the main question. Vary the angle:
definitions and mechanisms, quantitative evidence, counter-evidence and criticism,
recent developments. Do not produce sub-questions that paraphrase each other.

For each, write a keyword-style search query. Search engines reward keywords and
punish full sentences, so "QLoRA vs full fine-tuning benchmark results" beats
"How does QLoRA compare to full fine-tuning?"."""

VERIFIER_SYSTEM = """You are a fact-checker. You are given numbered sources and a
research question.

Extract the factual claims the sources support. For each claim, list the URLs of
every source whose text supports it.

Rules:
- A claim must be supported by text actually present in a source. If you cannot
  point to the supporting passage, it is not a claim.
- Use ONLY the sources. Do not add facts from your own knowledge, however certain
  you are. A true fact with no source here is still unsupported.
- If sources disagree on a point, record it as a contradiction with each position
  attributed to its source, rather than picking a winner.
- Statements that seem relevant but no source supports go in unsupported_claims.
  Be willing to put things here. An empty unsupported_claims list on messy sources
  usually means you were not strict enough.
- Prefer specific claims over vague ones. "Latency dropped 40% in their benchmark"
  is more useful than "performance improved"."""

WRITER_SYSTEM = """You write research reports in Markdown from verified claims.

Structure:
## Summary
## Findings
## Contradictions and Open Questions
## Sources

Rules:
- Use ONLY the verified claims given to you. Do not add facts, context, or
  background from your own knowledge.
- Cite every factual sentence inline as [n], using the source numbers provided.
  A sentence with a fact and no citation is a defect.
- The Sources section must list EVERY source you were given, numbered exactly as
  provided, with title and URL — including sources you did not cite. Do not
  renumber, reorder, or omit entries.
- Every contradiction listed must appear in the report with both sides attributed.
- State uncertainty plainly. If the sources are thin on a point, say so.
- Do not pad. A short report backed by sources beats a long one that drifts."""

CRITIC_SYSTEM = """You review a research report against the verification notes it
was built from. You are the last check before the report is released, and a
rubber-stamp review is worse than no review.

Return "revise" if ANY of these hold:
- A factual sentence has no citation.
- A citation points to a source that does not support that claim per the notes.
- The report states something not present in the verified claims.
- A contradiction in the notes is missing from the report, or is stated without
  attributing both sides.
- A claim listed as unsupported appears in the report as fact.
- The Sources section omits a provided source, renumbers them, or lists them in
  a different order than given.
- The report hedges so heavily it asserts nothing, or asserts confidently where
  the notes show only one weak source.

Return "accept" only if none hold.

Each issue must be concrete and actionable: name the sentence or claim and what
is wrong with it. "Needs more detail" is not an issue. "The second paragraph of
Findings states a 40% improvement citing [3], but [3] is not among the sources
supporting that claim" is."""


# --- Config ---

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# --- Formatting helpers ---

def _numbered_sources(sources: list[Source]) -> str:
    return "\n\n".join(
        f"[{i}] {s.title}\nURL: {s.url}\nFound via: {s.sub_question}\n{s.content}"
        for i, s in enumerate(sources, start=1)
    )


def _url_index(sources: list[Source]) -> dict[str, int]:
    return {s.url: i for i, s in enumerate(sources, start=1)}


def _format_verification(verification: Verification, url_index: dict[str, int]) -> str:
    def cite(urls: list[str]) -> str:
        nums = sorted(url_index[u] for u in urls if u in url_index)
        return "".join(f"[{n}]" for n in nums) or "[?]"

    claims = "\n".join(f"- {c.statement} {cite(c.source_urls)}" for c in verification.claims)

    if verification.contradictions:
        contradictions = "\n".join(
            f"- {c.topic}:\n"
            + "\n".join(
                f"    - {p.claim} {cite([p.source_url])}" for p in c.positions
            )
            for c in verification.contradictions
        )
    else:
        contradictions = "- None found"

    unsupported = "\n".join(f"- {u}" for u in verification.unsupported_claims) or "- None"

    return (
        f"VERIFIED CLAIMS (the only facts you may use):\n{claims}\n\n"
        f"CONTRADICTIONS (all must appear in the report):\n{contradictions}\n\n"
        f"UNSUPPORTED (must NOT appear as fact):\n{unsupported}"
    )


# --- Nodes ---

def make_planner():
    llm = get_llm(max_tokens=1000).with_structured_output(Plan)

    def plan(state: ResearchState) -> dict:
        result: Plan = llm.invoke(
            [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=state["query"])]
        )
        return {
            "plan": result,
            "trace": [f"plan: {len(result.sub_questions)} sub-questions"],
        }

    return plan


def make_searcher(search: SearchTool):
    def run_search(state: ResearchState) -> dict:
        per_sub = _env_int("RESULTS_PER_SUBQUESTION", 3)
        cap = _env_int("MAX_SOURCES", 6)

        collected: list[Source] = []
        seen: set[str] = set()

        for sub in state["plan"].sub_questions:
            for source in search.search(sub.search_query, max_results=per_sub):
                if source.url in seen:
                    continue
                seen.add(source.url)
                collected.append(source.model_copy(update={"sub_question": sub.question}))

        collected = collected[:cap]
        return {
            "sources": collected,
            "trace": [f"search: {len(collected)} unique sources"],
        }

    return run_search


def make_verifier():
    llm = get_llm(max_tokens=3000).with_structured_output(Verification)

    def verify(state: ResearchState) -> dict:
        result: Verification = llm.invoke(
            [
                SystemMessage(content=VERIFIER_SYSTEM),
                HumanMessage(
                    content=(
                        f"Research question: {state['query']}\n\n"
                        f"Sources:\n\n{_numbered_sources(state['sources'])}"
                    )
                ),
            ]
        )
        return {
            "verification": result,
            "trace": [
                f"verify: {len(result.claims)} claims, "
                f"{len(result.contradictions)} contradictions, "
                f"{len(result.unsupported_claims)} unsupported"
            ],
        }

    return verify


def make_writer():
    llm = get_llm(max_tokens=4000)

    def write(state: ResearchState) -> dict:
        sources = state["sources"]
        notes = _format_verification(state["verification"], _url_index(sources))
        source_list = "\n".join(f"[{i}] {s.title} — {s.url}" for i, s in enumerate(sources, 1))

        feedback = ""
        review = state.get("review")
        if review and review.verdict == "revise":
            feedback = "\n\nFIX THESE ISSUES FROM THE PREVIOUS DRAFT:\n" + "\n".join(
                f"- {issue}" for issue in review.issues
            )

        message = llm.invoke(
            [
                SystemMessage(content=WRITER_SYSTEM),
                HumanMessage(
                    content=(
                        f"Research question: {state['query']}\n\n"
                        f"{notes}\n\n"
                        f"SOURCE LIST (cite by these numbers):\n{source_list}"
                        f"{feedback}"
                    )
                ),
            ]
        )

        draft_count = state.get("draft_count", 0) + 1
        return {
            "report": message.content,
            "draft_count": draft_count,
            "trace": [f"write: draft {draft_count}"],
        }

    return write


def make_critic():
    llm = get_llm(max_tokens=1500).with_structured_output(Review)

    def critique(state: ResearchState) -> dict:
        notes = _format_verification(state["verification"], _url_index(state["sources"]))
        result: Review = llm.invoke(
            [
                SystemMessage(content=CRITIC_SYSTEM),
                HumanMessage(
                    content=f"{notes}\n\nREPORT UNDER REVIEW:\n\n{state['report']}"
                ),
            ]
        )
        return {
            "review": result,
            "trace": [f"critique: {result.verdict} ({len(result.issues)} issues)"],
        }

    return critique


def make_finalizer():
    """Appends unresolved critic issues to the report instead of hiding them."""

    def finalize(state: ResearchState) -> dict:
        review = state.get("review")
        if not review or review.verdict == "accept":
            return {"trace": ["finalize: accepted clean"]}

        caveats = "\n".join(f"- {issue}" for issue in review.issues)
        report = (
            f"{state['report']}\n\n"
            f"---\n\n"
            f"## Unresolved Review Issues\n\n"
            f"The revision budget was exhausted with these problems outstanding. "
            f"Treat the affected claims with caution.\n\n{caveats}\n"
        )
        return {
            "report": report,
            "trace": [f"finalize: released with {len(review.issues)} unresolved issues"],
        }

    return finalize


# --- Routing ---

def route_after_critique(state: ResearchState) -> str:
    review = state.get("review")
    if review and review.verdict == "revise":
        if state.get("draft_count", 0) < _env_int("MAX_REVISIONS", 2) + 1:
            return "write"
    return "finalize"