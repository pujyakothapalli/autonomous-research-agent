"""LLM judge for citation support rate.

Design note. The judge does NOT decide what counts as a sentence. Sentences are
extracted deterministically in code and handed over as a numbered list; the model
only classifies each index. An earlier version let the model select, transcribe
and classify in one call, and its sentence count varied between 17 and 20 on
byte-identical input — a per-report standard deviation of 8.2 points, six times
the effect under measurement.
"""

from __future__ import annotations

import os
import re
import statistics
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_agent.state import Source

VERDICTS = ("supported", "partial", "unsupported", "uncited", "not_factual")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\[(])")
_SKIP_PREFIX = ("#", "|", ">", "---", "===", "```")
_BOLD_ONLY = re.compile(r"^\*\*.+\*\*:?$")


def extract_sentences(report: str) -> list[str]:
    """Deterministic sentence extraction. Same report in, same list out."""
    body = report.split("## Sources")[0]
    out: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith(_SKIP_PREFIX):
            continue
        if _BOLD_ONLY.match(line):       # a fully-bold line is a sub-heading
            continue
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        for s in _SENTENCE_SPLIT.split(line):
            s = s.strip()
            if len(s) >= 25:
                out.append(s)
    return out


JUDGE_SYSTEM = """You classify sentences from a research report against the
sources it was built from.

You receive numbered sources, the report for context, and a NUMBERED LIST OF
SENTENCES. Return exactly one verdict per sentence number. Do not add sentences,
do not merge them, do not skip any.

Verdicts:
  supported    - the sentence cites sources, and the cited source text states
                 this or something it follows from directly.
  partial      - the sentence cites sources, but overstates them, generalizes
                 beyond them, or adds specifics they do not contain.
  unsupported  - the sentence cites sources that do not state this, or cites a
                 different source than the one containing it.
  uncited      - the sentence asserts a fact about the world and carries no
                 citation.
  not_factual  - the sentence asserts nothing about the world. Use this for:
                   * statements about the evidence base: "the sources do not
                     address X", "sources disagree on Y", "Reference [3] states
                     Z", "the verified claims do not provide W"
                   * explicit statements of scope or uncertainty
                   * transitions and framing that carry no claim
                 A sentence describing what the SOURCES say or omit is
                 not_factual, even when it carries a citation marker.

Rules:
- Judge ONLY against the source text given. Do not use your own knowledge. A
  sentence true in the world but absent from its cited source is "unsupported".
- Be strict about "partial": a source saying "improved accuracy in one
  benchmark" does not support "improves accuracy".
- A sentence may inherit a citation from the one immediately before it only if
  it is plainly a continuation; otherwise treat it as uncited."""


class SentenceVerdict(BaseModel):
    index: int = Field(description="The sentence number exactly as given")
    verdict: Literal["supported", "partial", "unsupported", "uncited", "not_factual"]
    cited: list[int] = Field(default_factory=list, description="Source numbers cited")
    reason: str = Field(description="One short line")


class Judgment(BaseModel):
    verdicts: list[SentenceVerdict]


def _numbered_sources(sources: list[Source]) -> str:
    return "\n\n".join(
        f"[{i}] {s.title}\nURL: {s.url}\n{s.content}"
        for i, s in enumerate(sources, start=1)
    )


def get_judge_llm() -> ChatAnthropic:
    """Defaults to RESEARCH_MODEL, meaning the judge shares the writer's model
    family and its blind spots. Set JUDGE_MODEL to a different, stronger model
    for any published result."""
    return ChatAnthropic(
        model=os.getenv("JUDGE_MODEL", os.getenv("RESEARCH_MODEL", "claude-haiku-4-5")),
        max_tokens=8000,
        temperature=0,
    )


def judge_report(report: str, sources: list[Source], sentences: list[str] | None = None) -> Judgment:
    """One classification pass over a fixed sentence list."""
    sentences = extract_sentences(report) if sentences is None else sentences
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))

    llm = get_judge_llm().with_structured_output(Judgment)
    message = HumanMessage(
        content=(
            f"SOURCES:\n\n{_numbered_sources(sources)}\n\n"
            + "=" * 60
            + f"\n\nREPORT (context only):\n\n{report}\n\n"
            + "=" * 60
            + f"\n\nSENTENCES TO CLASSIFY — return exactly {len(sentences)} "
            f"verdicts, indices 1 to {len(sentences)}:\n\n{numbered}"
        )
    )

    result = llm.invoke([SystemMessage(content=JUDGE_SYSTEM), message])
    if len({v.index for v in result.verdicts}) != len(sentences):
        result = llm.invoke([SystemMessage(content=JUDGE_SYSTEM), message])
    return result


def judge_report_repeated(
    report: str, sources: list[Source], n: int | None = None
) -> list[Judgment]:
    n = n or int(os.getenv("JUDGE_REPEATS", "3"))
    sentences = extract_sentences(report)
    return [judge_report(report, sources, sentences) for _ in range(n)]


def score(sentences: list[str], judgment: Judgment) -> dict:
    """Denominator is fixed by `sentences`, not by the model."""
    by_index = {v.index: v for v in judgment.verdicts}
    counts = dict.fromkeys(VERDICTS, 0)
    missing = 0

    for i in range(1, len(sentences) + 1):
        v = by_index.get(i)
        if v is None:
            missing += 1
            counts["not_factual"] += 1
        else:
            counts[v.verdict] += 1

    factual = (
        counts["supported"] + counts["partial"] + counts["unsupported"] + counts["uncited"]
    )

    return {
        "extracted_sentences": len(sentences),
        "missing_verdicts": missing,
        "supported": counts["supported"],
        "partial": counts["partial"],
        "unsupported": counts["unsupported"],
        "uncited_factual": counts["uncited"],
        "not_factual": counts["not_factual"],
        "total_factual": factual,
        "cited_sentences": counts["supported"] + counts["partial"] + counts["unsupported"],
        # PRIMARY
        "sourced_claim_rate": counts["supported"] / factual if factual else 0.0,
        # Secondary diagnostics
        "citation_support_rate": (
            counts["supported"] / (factual - counts["uncited"])
            if factual - counts["uncited"]
            else 0.0
        ),
        "citation_coverage": (
            (factual - counts["uncited"]) / factual if factual else 0.0
        ),
    }


def score_consensus(sentences: list[str], judgments: list[Judgment]) -> dict:
    """Mean across repeated judgments, with the observed spread reported."""
    scored = [score(sentences, j) for j in judgments]
    out = {k: sum(s[k] for s in scored) / len(scored) for k in scored[0]}
    rates = [s["sourced_claim_rate"] for s in scored]
    out["sourced_claim_rate_spread"] = max(rates) - min(rates)
    out["sourced_claim_rate_stdev"] = statistics.pstdev(rates) if len(rates) > 1 else 0.0
    out["judge_repeats"] = len(scored)
    return out