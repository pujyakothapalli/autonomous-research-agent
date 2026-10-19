from dotenv import load_dotenv
load_dotenv()

from research_agent.nodes import make_critic
from research_agent.state import Claim, Contradiction, Position, Source, Verification

sources = [
    Source(url="https://a.com", title="Source A", content=""),
    Source(url="https://b.com", title="Source B", content=""),
]

verification = Verification(
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

critic = make_critic()
result = critic({"sources": sources, "verification": verification, "report": BROKEN_REPORT})

print("verdict:", result["review"].verdict)
for issue in result["review"].issues:
    print("  -", issue)