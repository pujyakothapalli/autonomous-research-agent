from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


def _slug(text: str, max_len: int = 60) -> str:
    cleaned = "".join(c if c.isalnum() else "-" for c in text.lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")[:max_len].strip("-")


def _header(query: str, state: dict, model: str, mock: bool) -> str:
    review = state.get("review")
    return (
        f"<!--\n"
        f"question: {query}\n"
        f"generated: {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
        f"model: {model}\n"
        f"sources: {len(state.get('sources', []))}\n"
        f"drafts: {state.get('draft_count', 0)}\n"
        f"verdict: {review.verdict if review else 'n/a'}\n"
        f"search: {'MOCK' if mock else 'tavily'}\n"
        f"-->\n\n"
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Research a question and produce a source-verified, cited report.",
    )
    parser.add_argument("question", nargs="+", help="The research question")
    parser.add_argument(
        "-o", "--out", default="reports", help="Output directory (default: reports)"
    )
    parser.add_argument(
        "--print", dest="print_report", action="store_true", help="Print the report to stdout"
    )
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to .env and try again.")

    # Imported after load_dotenv so the factories read a populated environment.
    from research_agent.graph import run
    from research_agent.tools import using_mock_search

    query = " ".join(args.question)
    mock = using_mock_search()
    model = os.getenv("RESEARCH_MODEL", "claude-haiku-4-5")

    if mock:
        print("WARNING: no TAVILY_API_KEY — using mock search. Output is not real research.\n")

    print(f"Researching: {query}\n")

    try:
        state = run(query)
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")

    for line in state.get("trace", []):
        print(f"  {line}")

    review = state.get("review")
    if review and review.verdict == "revise":
        print(f"\n  {len(review.issues)} issue(s) unresolved at budget exhaustion:")
        for issue in review.issues:
            print(f"    - {issue}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now():%Y-%m-%d}-{_slug(query)}.md"
    out_path.write_text(
        _header(query, state, model, mock) + state["report"], encoding="utf-8"
    )

    print(f"\nReport: {out_path}")

    if args.print_report:
        print("\n" + "=" * 70 + "\n")
        print(state["report"])


if __name__ == "__main__":
    main()