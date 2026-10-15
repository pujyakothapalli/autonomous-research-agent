"""Data contracts for the research graph."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


# --- Planning ---

class SubQuestion(BaseModel):
    question: str = Field(description="A focused sub-question answering part of the main query")
    search_query: str = Field(description="Keyword-style web search query for this sub-question")


class Plan(BaseModel):
    sub_questions: list[SubQuestion] = Field(min_length=2, max_length=5)


# --- Retrieval ---

class Source(BaseModel):
    url: str
    title: str
    content: str
    sub_question: str = ""


# --- Verification ---

class Claim(BaseModel):
    statement: str = Field(description="A single factual claim drawn from the sources")
    source_urls: list[str] = Field(
        min_length=1, description="URLs of sources whose text supports this claim"
    )


class Position(BaseModel):
    claim: str = Field(description="What this source asserts")
    source_url: str


class Contradiction(BaseModel):
    topic: str = Field(description="The point of disagreement")
    positions: list[Position] = Field(min_length=2)


class Verification(BaseModel):
    claims: list[Claim]
    contradictions: list[Contradiction] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Statements no source supports; excluded from the report",
    )


# --- Review ---

class Review(BaseModel):
    verdict: Literal["accept", "revise"]
    issues: list[str] = Field(
        default_factory=list, description="Specific, fixable problems found in the draft"
    )


# --- Graph state ---

class ResearchState(TypedDict, total=False):
    query: str

    plan: Plan
    verification: Verification
    report: str
    review: Review
    draft_count: int

    sources: Annotated[list[Source], operator.add]
    trace: Annotated[list[str], operator.add]