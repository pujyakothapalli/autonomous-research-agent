"""Disk-backed search cache.

Both ablation arms must see identical sources, or retrieval variance pollutes
the measurement and the number means nothing. This wraps any SearchTool and
replays recorded results, so the second arm never hits the network.

Side benefit: reruns cost no Tavily quota, which matters on the free tier.
"""

from __future__ import annotations

import json
from pathlib import Path

from research_agent.state import Source
from research_agent.tools import SearchTool


class CachedSearch:
    def __init__(self, inner: SearchTool, cache_path: str | Path) -> None:
        self._inner = inner
        self._path = Path(cache_path)
        self._cache: dict[str, list[dict]] = {}
        self.hits = 0
        self.misses = 0

        if self._path.exists():
            self._cache = json.loads(self._path.read_text(encoding="utf-8"))

    def search(self, query: str, max_results: int = 3) -> list[Source]:
        key = f"{query}::{max_results}"

        if key in self._cache:
            self.hits += 1
            return [Source(**item) for item in self._cache[key]]

        self.misses += 1
        results = self._inner.search(query, max_results=max_results)
        self._cache[key] = [s.model_dump() for s in results]
        self._flush()
        return results

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

class PlanCache:
    """Caches planner output per question so reruns are free and reproducible."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, dict] = {}
        if self._path.exists():
            self._data = json.loads(self._path.read_text(encoding="utf-8"))

    def get(self, question: str):
        from research_agent.state import Plan
        raw = self._data.get(question)
        return Plan(**raw) if raw else None

    def put(self, question: str, plan) -> None:
        self._data[question] = plan.model_dump()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")