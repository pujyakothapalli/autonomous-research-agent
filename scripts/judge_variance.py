"""Measure the judge's own variance on identical input.

The evaluator's noise floor bounds what the ablation can detect. If the critic's
effect is smaller than this spread, no effect can be claimed.

    python scripts/judge_variance.py
    python scripts/judge_variance.py --n 10 --question q05
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# One judge call for the draft; this script does its own repeats below.
os.environ["JUDGE_REPEATS"] = "1"

from evals.cache import CachedSearch, PlanCache  # noqa: E402
from evals.judge import extract_sentences, judge_report, score  # noqa: E402
from evals.run_eval import retrieve, run_arm  # noqa: E402
from research_agent.tools import get_search_tool  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="q05")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    spec = yaml.safe_load((ROOT / "evals" / "questions.yaml").read_text(encoding="utf-8"))
    q = next(x for x in spec["questions"] if x["id"] == args.question)

    search = CachedSearch(get_search_tool(), ROOT / "evals" / "results" / "search_cache.json")
    plans = PlanCache(ROOT / "evals" / "results" / "plan_cache.json")

    print("retrieving ...", end="", flush=True)
    sources = retrieve(q["question"], search, plans)

    print("\ndrafting ...", end="", flush=True)
    arm = run_arm(q["question"], sources, include_critic=False)
    report = arm["report"]

    sentences = extract_sentences(report)

    print("\n\nONE report")
    print("  extracted sentences (deterministic): %d" % len(sentences))
    print("  judged %d times\n" % args.n)

    rates, factuals = [], []
    for i in range(args.n):
        s = score(sentences, judge_report(report, sources, sentences))
        rates.append(s["sourced_claim_rate"])
        factuals.append(s["total_factual"])
        print(
            "  judge %2d: %5.1f%%   factual=%2d  supported=%2d partial=%d "
            "unsupported=%d uncited=%d not_factual=%2d missing=%d"
            % (
                i + 1,
                s["sourced_claim_rate"] * 100,
                s["total_factual"],
                s["supported"],
                s["partial"],
                s["unsupported"],
                s["uncited_factual"],
                s["not_factual"],
                s["missing_verdicts"],
            )
        )

    print()
    print("  sourced_claim_rate")
    print("    mean   %.1f%%" % (statistics.mean(rates) * 100))
    print("    stdev  %.1f points" % (statistics.pstdev(rates) * 100))
    print("    range  %.1f points" % ((max(rates) - min(rates)) * 100))
    print()
    print("  denominator (factual sentence count)")
    print("    values %s" % sorted(set(factuals)))
    print("    stdev  %.2f sentences" % statistics.pstdev(factuals))
    print()
    print("  Extraction is deterministic by construction. Any remaining")
    print("  denominator movement is the judge's not_factual classification.")


if __name__ == "__main__":
    main()