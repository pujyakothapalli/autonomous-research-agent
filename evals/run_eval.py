"""Critic ablation with a placebo control.

    python evals/run_eval.py
    python evals/run_eval.py --questions q01,q03

Three arms per question, all over identical retrieval:
  control  — no critic
  critic   — full graph with the revision loop
  placebo  — no critic, run a second time

Arm order matters. The placebo runs LAST so it is separated from the control by
the same amount of intervening work as the critic arm is. Run back-to-back with
the control, two identical calls agree trivially and the placebo measures
nothing.

Each report is judged several times (JUDGE_REPEATS, default 3) and the spread is
reported, because the judge is the dominant noise source in this measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from evals.cache import CachedSearch, PlanCache  # noqa: E402
from evals.judge import (  # noqa: E402
    extract_sentences,
    judge_report_repeated,
    score_consensus,
)
from research_agent.graph import build_graph  # noqa: E402
from research_agent.nodes import make_planner, make_searcher  # noqa: E402
from research_agent.state import Source  # noqa: E402
from research_agent.tools import get_search_tool, using_mock_search  # noqa: E402


# ---------------------------------------------------------------------------
# Retrieval — runs once per question, shared by all three arms
# ---------------------------------------------------------------------------

def retrieve(question: str, search: CachedSearch, plans: PlanCache) -> list[Source]:
    plan = plans.get(question)
    if plan is None:
        out = make_planner()({"query": question})
        plan = out["plan"]
        plans.put(question, plan)
        print(f"\n      {out['trace'][0]}", end="", flush=True)
    else:
        print(
            f"\n      plan: {len(plan.sub_questions)} sub-questions (cached)",
            end="",
            flush=True,
        )

    search_out = make_searcher(search)({"query": question, "plan": plan})
    print(f"\n      {search_out['trace'][0]}", end="", flush=True)
    return search_out["sources"]


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------

def run_arm(question: str, sources: list[Source], include_critic: bool) -> dict:
    graph = build_graph(include_critic=include_critic, skip_retrieval=True)
    inputs = {"query": question, "sources": sources, "trace": [], "draft_count": 0}

    state: dict = {}
    seen = 0
    for snapshot in graph.stream(
        inputs, config={"recursion_limit": 25}, stream_mode="values"
    ):
        state = snapshot
        trace = state.get("trace", [])
        for line in trace[seen:]:
            print(f"\n      {line}", end="", flush=True)
        seen = len(trace)

    print("\n      judging ...", end="", flush=True)
    sentences = extract_sentences(state["report"])
    judgments = judge_report_repeated(state["report"], state["sources"])
    metrics = score_consensus(sentences, judgments)

    review = state.get("review")
    return {
        "metrics": metrics,
        "drafts": state.get("draft_count", 0),
        "critic_intervened": state.get("draft_count", 0) > 1,
        "n_sources": len(state.get("sources", [])),
        # Stored so a report can be re-judged offline without rerunning the
        # pipeline — judge experiments then cost only judge calls.
        "source_payload": [s.model_dump() for s in state["sources"]],
        "contradictions": len(state["verification"].contradictions),
        "unsupported_dropped": len(state["verification"].unsupported_claims),
        "critic_verdict": review.verdict if review else None,
        "critic_issues": review.issues if review else [],
        "trace": state.get("trace", []),
        "report": state["report"],
        "sentences": sentences,
        "judged": [v.model_dump() for v in judgments[0].verdicts],
        "review_history": [r.model_dump() for r in state.get("reviews", [])],
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(arm: list[dict]) -> dict:
    n = len(arm)
    if not n:
        return {}

    def mean(key: str) -> float:
        return sum(r["metrics"][key] for r in arm) / n

    return {
        "questions": n,
        "sourced_claim_rate": mean("sourced_claim_rate"),
        "citation_support_rate": mean("citation_support_rate"),
        "citation_coverage": mean("citation_coverage"),
        "judge_spread": mean("sourced_claim_rate_spread"),
        "unsupported": mean("unsupported"),
        "uncited_factual": mean("uncited_factual"),
        "not_factual": mean("not_factual"),
        "total_factual": mean("total_factual"),
        "extracted_sentences": mean("extracted_sentences"),
        "missing_verdicts": mean("missing_verdicts"),
        "drafts": sum(r["drafts"] for r in arm) / n,
        "reports_with_contradictions": sum(1 for r in arm if r["contradictions"] > 0),
    }


def rate(result: dict) -> float:
    return result["metrics"]["sourced_claim_rate"]


def spread(result: dict) -> float:
    return result["metrics"]["sourced_claim_rate_spread"]


def rms(values: list[float]) -> float:
    return (sum(v * v for v in values) / len(values)) ** 0.5 if values else 0.0


RATE_KEYS = (
    "sourced_claim_rate",
    "citation_support_rate",
    "citation_coverage",
    "judge_spread",
)
COUNT_KEYS = (
    "unsupported",
    "uncited_factual",
    "not_factual",
    "total_factual",
    "extracted_sentences",
    "missing_verdicts",
    "drafts",
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Critic ablation with placebo control")
    parser.add_argument("--questions", help="Comma-separated question ids")
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        sys.exit("ANTHROPIC_API_KEY is not set.")

    if using_mock_search():
        sys.exit(
            "Refusing to run: no TAVILY_API_KEY, so search returns fabricated "
            "text. Any number produced would be meaningless."
        )

    spec = yaml.safe_load((ROOT / "questions.yaml").read_text(encoding="utf-8"))
    questions = spec["questions"]

    if args.questions:
        wanted = {q.strip() for q in args.questions.split(",")}
        questions = [q for q in questions if q["id"] in wanted]

    if not questions:
        sys.exit("No matching questions.")

    search = CachedSearch(get_search_tool(), ROOT / "results" / "search_cache.json")
    plans = PlanCache(ROOT / "results" / "plan_cache.json")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    control: list[dict] = []
    placebo: list[dict] = []
    critic: list[dict] = []
    per_question: list[dict] = []
    skipped: list[str] = []

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q['id']}: {q['question']}")
        try:
            print("  retrieving ...", end="", flush=True)
            sources = retrieve(q["question"], search, plans)

            print("\n  control ...", end="", flush=True)
            c = run_arm(q["question"], sources, include_critic=False)
            print(f"\n    sourced={rate(c):.1%}  judge spread={spread(c):.1%}")

            print("  critic  ...", end="", flush=True)
            t = run_arm(q["question"], sources, include_critic=True)
            print(
                f"\n    sourced={rate(t):.1%}  judge spread={spread(t):.1%}"
                f"  drafts={t['drafts']}  verdict={t['critic_verdict']}"
            )

            # Placebo LAST: separated from the control by the same intervening
            # work as the critic arm, so it samples under the same conditions.
            print("  placebo ...", end="", flush=True)
            p = run_arm(q["question"], sources, include_critic=False)
            print(f"\n    sourced={rate(p):.1%}  judge spread={spread(p):.1%}")

        except Exception as exc:
            print(f"\n  SKIPPED {q['id']}: {type(exc).__name__}: {exc}")
            skipped.append(q["id"])
            continue

        print(
            f"    -> treatment {rate(t) - rate(c):+.1%}"
            f"   placebo {rate(p) - rate(c):+.1%}"
        )

        control.append(c)
        placebo.append(p)
        critic.append(t)
        per_question.append(
            {
                "id": q["id"],
                "question": q["question"],
                "control": c,
                "placebo": p,
                "critic": t,
                "treatment_delta": rate(t) - rate(c),
                "placebo_delta": rate(p) - rate(c),
            }
        )

        # Checkpoint after every question: a crash costs one question, not the run.
        (out_dir / "partial.json").write_text(
            json.dumps(per_question, indent=2), encoding="utf-8"
        )

    n = len(control)
    if n == 0:
        sys.exit("No questions completed.")

    agg_c, agg_p, agg_t = aggregate(control), aggregate(placebo), aggregate(critic)

    treatment_deltas = [rate(critic[i]) - rate(control[i]) for i in range(n)]
    placebo_deltas = [rate(placebo[i]) - rate(control[i]) for i in range(n)]

    mean_treatment = sum(treatment_deltas) / n
    mean_placebo = sum(placebo_deltas) / n
    noise_floor = rms(placebo_deltas)
    net_effect = mean_treatment - mean_placebo

    # Within-report judge spread, averaged across every arm and question.
    judge_spread = sum(
        spread(r) for arm in (control, placebo, critic) for r in arm
    ) / (3 * n)

    intervened = [i for i in range(n) if critic[i]["critic_intervened"]]
    unresolved = [i for i in range(n) if critic[i]["critic_verdict"] == "revise"]

    if intervened:
        mean_treatment_int = sum(treatment_deltas[i] for i in intervened) / len(intervened)
        mean_placebo_int = sum(placebo_deltas[i] for i in intervened) / len(intervened)
        net_effect_int = mean_treatment_int - mean_placebo_int
    else:
        mean_treatment_int = mean_placebo_int = net_effect_int = None

    # --- Report ---

    print("\n" + "=" * 74)
    print(f"{'metric':<34}{'control':>12}{'placebo':>12}{'critic':>12}")
    print("-" * 74)
    for key in RATE_KEYS:
        print(f"{key:<34}{agg_c[key]:>11.1%}{agg_p[key]:>12.1%}{agg_t[key]:>12.1%}")
    for key in COUNT_KEYS:
        print(f"{key:<34}{agg_c[key]:>12.2f}{agg_p[key]:>12.2f}{agg_t[key]:>12.2f}")

    print("=" * 74)
    print(f"{'treatment delta (critic - control)':<46}{mean_treatment:>+10.1%}")
    print(f"{'placebo delta (control - control)':<46}{mean_placebo:>+10.1%}")
    print(f"{'noise floor (RMS of placebo deltas)':<46}{noise_floor:>10.1%}")
    print(f"{'judge spread within report (mean)':<46}{judge_spread:>10.1%}")
    print(f"{'NET EFFECT (treatment - placebo)':<46}{net_effect:>+10.1%}")

    if abs(net_effect) < max(noise_floor, judge_spread):
        print("\n  NET EFFECT IS BELOW THE MEASUREMENT NOISE. No effect can be claimed.")

    print("-" * 74)
    print(f"questions completed: {n}/{len(questions)}"
          + (f"   skipped: {', '.join(skipped)}" if skipped else ""))
    print(f"critic intervened on {len(intervened)}/{n} questions "
          f"({len(intervened) / n:.0%})")
    print(f"budget exhausted with issues open: {len(unresolved)}/{n}")
    if intervened:
        print(f"on the {len(intervened)} it touched: treatment {mean_treatment_int:+.1%}, "
              f"placebo {mean_placebo_int:+.1%}, net {net_effect_int:+.1%}")

    print(f"\nsearch cache hit rate: {search.hit_rate:.0%} "
          f"({search.hits} hits, {search.misses} misses)")

    # --- Persist ---

    out_path = out_dir / f"{datetime.now():%Y-%m-%d}-eval.json"
    out_path.write_text(
        json.dumps(
            {
                "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
                "research_model": os.getenv("RESEARCH_MODEL", "claude-haiku-4-5"),
                "judge_model": os.getenv(
                    "JUDGE_MODEL", os.getenv("RESEARCH_MODEL", "claude-haiku-4-5")
                ),
                "judge_repeats": int(os.getenv("JUDGE_REPEATS", "3")),
                "max_revisions": int(os.getenv("MAX_REVISIONS", "2")),
                "design": (
                    "Three arms over identical retrieval: control (no critic), "
                    "critic (full graph) and placebo (no critic, re-run). The "
                    "placebo runs last so it is separated from the control by "
                    "the same intervening work as the critic arm. Sentences are "
                    "extracted deterministically in code; the judge only "
                    "classifies, and each report is judged JUDGE_REPEATS times "
                    "with the spread reported."
                ),
                "questions_requested": len(questions),
                "questions_completed": n,
                "questions_skipped": skipped,
                "search_cache_hit_rate": search.hit_rate,
                "aggregate": {
                    "control": agg_c,
                    "placebo": agg_p,
                    "critic": agg_t,
                },
                "effect": {
                    "mean_treatment_delta": mean_treatment,
                    "mean_placebo_delta": mean_placebo,
                    "noise_floor_rms": noise_floor,
                    "mean_judge_spread_within_report": judge_spread,
                    "net_effect": net_effect,
                    "effect_below_noise": abs(net_effect)
                    < max(noise_floor, judge_spread),
                    "treatment_deltas": treatment_deltas,
                    "placebo_deltas": placebo_deltas,
                },
                "intervention": {
                    "rate": len(intervened) / n,
                    "unresolved_rate": len(unresolved) / n,
                    "mean_treatment_delta_when_intervened": mean_treatment_int,
                    "mean_placebo_delta_when_intervened": mean_placebo_int,
                    "net_effect_when_intervened": net_effect_int,
                },
                "per_question": per_question,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()