"""Adversarial check: the critic must reject a report with known defects.

Calls the live API, so it costs a fraction of a cent per run. Skipped without
a key. This is the project's load-bearing test — if it starts passing
vacuously, the critic has stopped working.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from research_agent.nodes import make_critic
from research_agent.state import Claim, Contradiction, Position, Source, Verification

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY"
)

SOURCES = [
    Source(url="https://a.com", title="Source A", content=""),
    Source(url="https://b.com", title="Source B", content=""),
]

VERIFICATION = Verification(
    claims=[
        Claim(statement="PSI above 0.25 indicates significant drift",
              source_urls=["https://a.com"]),
        Claim(statement="The KS test is used for drift detection",
              source_urls=["https://b.com"]),
    ],
    contradictions=[
        Contradiction(
            topic="retraining cadence",
            positions=[
                Position(claim="retrain on a weekly schedule", source_url="https://a.com"),
                Position(claim="retrain only when drift is detected", source_url="https://b.com"),
            ],
        )
    ],
    unsupported_claims=["Model drift costs the industry $500B annually"],
)

BROKEN_REPORT = """## Summary
Model drift is a major problem, costing the industry $500B annually [1].

## Findings
PSI above 0.25 indicates significant drift [2].
The KS test is used for drift detection [2].
Transformer architectures are inherently immune to data drift.

## Contradictions and Open Questions
No contradictions were found in the verified claims.

## Sources
[1] Source A — https://a.com
[2] Source B — https://b.com
"""

CLEAN_REPORT = """## Summary
Statistical tests are used to detect model drift in production.

## Findings
PSI above 0.25 indicates significant drift [1].
The KS test is used for drift detection [2].

## Contradictions and Open Questions
Sources disagree on retraining cadence: one recommends a weekly schedule [1],
the other recommends retraining only when drift is detected [2].

## Sources
[1] Source A — https://a.com
[2] Source B — https://b.com
"""


def _review(report: str):
    critic = make_critic()
    return critic({"sources": SOURCES, "verification": VERIFICATION, "report": report})["review"]


def test_critic_rejects_broken_report():
    review = _review(BROKEN_REPORT)
    assert review.verdict == "revise"
    assert len(review.issues) >= 3, f"too few issues found: {review.issues}"


def test_critic_accepts_clean_report():
    """Guards the other direction: a critic that rejects everything is also broken."""
    review = _review(CLEAN_REPORT)
    assert review.verdict == "accept", f"false rejection: {review.issues}"