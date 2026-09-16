"""The fixture world an incident scenario describes.

A scenario file is a frozen snapshot of one production incident: the alert that
fired, plus whatever logs, metrics, Kubernetes state, deploy history, runbooks
and remediation actions an on-call engineer would have been able to reach for.
The MCP tools in `server.py` read exclusively from here, so a scenario replays
identically every time -- which is what makes the harness numbers meaningful.

Timestamps in scenario files are relative offsets in minutes from the moment the
alert fired (`t: -12` is twelve minutes before). `World` resolves them against
`alert.fired_at` so tools can hand absolute ISO timestamps to the model.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios"
COMMON_FILE = SCENARIO_DIR / "_common.yaml"


class ScenarioError(ValueError):
    """A scenario file is missing required structure."""


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ScenarioError(f"{where}: missing required key {key!r}")
    return mapping[key]


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    category: str
    alert: dict[str, Any]
    world: dict[str, Any]
    ground_truth: dict[str, Any]
    path: Path

    @property
    def fired_at(self) -> dt.datetime:
        raw = self.alert["fired_at"]
        if isinstance(raw, dt.datetime):
            stamp = raw
        else:
            stamp = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=dt.timezone.utc)
        return stamp


def load_scenario(ref: str | Path) -> Scenario:
    """Load a scenario by id (``SC-003``), bare filename, or path."""
    path = Path(ref)
    if not path.suffix:
        candidate = SCENARIO_DIR / f"{ref}.yaml"
        path = candidate if candidate.exists() else SCENARIO_DIR / str(ref)
    if not path.exists():
        raise ScenarioError(f"no scenario file for {ref!r} (looked at {path})")

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ScenarioError(f"{path}: expected a YAML mapping at the top level")

    where = path.name
    return Scenario(
        id=_require(raw, "id", where),
        title=_require(raw, "title", where),
        category=raw.get("category", "uncategorised"),
        alert=_require(raw, "alert", where),
        world=_require(raw, "world", where),
        ground_truth=_require(raw, "ground_truth", where),
        path=path,
    )


def all_scenarios() -> list[Scenario]:
    """Every scenario file. Leading-underscore files are shared data, not scenarios."""
    return [
        load_scenario(p)
        for p in sorted(SCENARIO_DIR.glob("*.yaml"))
        if not p.name.startswith("_")
    ]


@lru_cache(maxsize=1)
def common_catalogue() -> dict[str, Any]:
    """Remediation actions and general runbooks shared by every scenario."""
    if not COMMON_FILE.exists():
        return {"actions": [], "runbooks": []}
    raw = yaml.safe_load(COMMON_FILE.read_text()) or {}
    return {"actions": raw.get("actions", []), "runbooks": raw.get("runbooks", [])}


def _merge_by_id(shared: list[dict[str, Any]], local: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shared entries first; a local entry reusing an id replaces the shared one."""
    merged = {entry["id"]: entry for entry in shared}
    merged.update({entry["id"]: entry for entry in local})
    return list(merged.values())


@dataclass
class World:
    """Read-only query surface over one scenario, used by the MCP tools."""

    scenario: Scenario
    _proposals: list[dict[str, Any]] = field(default_factory=list)

    # -- time -------------------------------------------------------------

    def at(self, offset_minutes: float) -> str:
        stamp = self.scenario.fired_at + dt.timedelta(minutes=float(offset_minutes))
        return stamp.isoformat().replace("+00:00", "Z")

    def _within(self, entry: dict[str, Any], since_minutes: float) -> bool:
        return float(entry.get("t", 0)) >= -abs(float(since_minutes))

    # -- inventory --------------------------------------------------------

    def services(self) -> list[dict[str, Any]]:
        return list(self.scenario.world.get("services", []))

    def service_names(self) -> list[str]:
        return [s["name"] for s in self.services()]

    # -- logs -------------------------------------------------------------

    def logs(
        self,
        service: str,
        pattern: str | None = None,
        level: str | None = None,
        since_minutes: float = 60,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        entries = self.scenario.world.get("logs", {}).get(service)
        if entries is None:
            raise KeyError(service)

        matcher = re.compile(pattern, re.IGNORECASE) if pattern else None
        wanted_level = level.upper() if level else None

        hits = []
        for entry in entries:
            if not self._within(entry, since_minutes):
                continue
            if wanted_level and entry.get("level", "INFO").upper() != wanted_level:
                continue
            if matcher and not matcher.search(entry.get("msg", "")):
                continue
            hits.append(
                {
                    "timestamp": self.at(entry.get("t", 0)),
                    "level": entry.get("level", "INFO"),
                    "message": entry.get("msg", ""),
                    **({"count": entry["count"]} if "count" in entry else {}),
                }
            )
        # Newest last, mirroring how a log UI reads; truncate from the front so
        # the most recent lines always survive the limit.
        return hits[-limit:]

    # -- metrics ----------------------------------------------------------

    def metric_names(self, service: str) -> list[str]:
        series = self.scenario.world.get("metrics", {}).get(service)
        if series is None:
            raise KeyError(service)
        return sorted(series.keys())

    def metrics(
        self,
        service: str,
        metric: str | None = None,
        since_minutes: float = 60,
    ) -> dict[str, Any]:
        series = self.scenario.world.get("metrics", {}).get(service)
        if series is None:
            raise KeyError(service)
        if metric is not None and metric not in series:
            raise KeyError(metric)

        chosen = {metric: series[metric]} if metric else series
        out: dict[str, Any] = {}
        for name, spec in chosen.items():
            points = [p for p in spec.get("points", []) if self._within(p, since_minutes)]
            values = [float(p["v"]) for p in points]
            out[name] = {
                "unit": spec.get("unit", ""),
                "points": [
                    {"timestamp": self.at(p["t"]), "value": p["v"]} for p in points
                ],
                "summary": {
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "latest": values[-1] if values else None,
                },
            }
        return out

    # -- kubernetes -------------------------------------------------------

    def k8s(self, kind: str, name: str, namespace: str = "prod") -> dict[str, Any]:
        for resource in self.scenario.world.get("k8s", []):
            if (
                resource.get("kind", "").lower() == kind.lower()
                and resource.get("name") == name
                and resource.get("namespace", "prod") == namespace
            ):
                out = dict(resource)
                out["events"] = [
                    {
                        "timestamp": self.at(e.get("t", 0)),
                        "type": e.get("type", "Normal"),
                        "reason": e.get("reason", ""),
                        "message": e.get("msg", ""),
                    }
                    for e in resource.get("events", [])
                ]
                return out
        raise KeyError(f"{kind}/{name} in namespace {namespace}")

    def k8s_index(self) -> list[str]:
        return [
            f"{r.get('kind')}/{r.get('name')} (ns={r.get('namespace', 'prod')})"
            for r in self.scenario.world.get("k8s", [])
        ]

    # -- deploys ----------------------------------------------------------

    def deploys(
        self, service: str | None = None, since_minutes: float = 180
    ) -> list[dict[str, Any]]:
        out = []
        for d in self.scenario.world.get("deploys", []):
            if service and d.get("service") != service:
                continue
            if not self._within(d, since_minutes):
                continue
            out.append(
                {
                    "timestamp": self.at(d.get("t", 0)),
                    "minutes_before_alert": -float(d.get("t", 0)),
                    "service": d.get("service"),
                    "version": d.get("version"),
                    "commit": d.get("commit", ""),
                    "author": d.get("author", ""),
                    "summary": d.get("summary", ""),
                }
            )
        return sorted(out, key=lambda d: d["timestamp"])

    # -- runbooks ---------------------------------------------------------

    def search_runbooks(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """Keyword-overlap search.

        Deliberately dumb: P2 replaces this with a real retriever, and keeping
        the interface stable means the agent side needs no changes when it does.
        """
        terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
        scored = []
        books = _merge_by_id(
            common_catalogue()["runbooks"], self.scenario.world.get("runbooks", [])
        )
        for book in books:
            haystack = " ".join(
                [book.get("title", ""), book.get("body", ""), " ".join(book.get("tags", []))]
            ).lower()
            tokens = set(re.findall(r"[a-z0-9]+", haystack))
            overlap = len(terms & tokens)
            if overlap:
                scored.append((overlap, book))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "id": b.get("id"),
                "title": b.get("title"),
                "body": b.get("body"),
                "match_score": score,
            }
            for score, b in scored[:limit]
        ]

    # -- remediation ------------------------------------------------------

    def actions(self) -> list[dict[str, Any]]:
        return _merge_by_id(
            common_catalogue()["actions"], self.scenario.world.get("actions", [])
        )

    def action(self, action_id: str) -> dict[str, Any]:
        for a in self.actions():
            if a.get("id") == action_id:
                return a
        raise KeyError(action_id)

    def record_proposal(self, proposal: dict[str, Any]) -> None:
        self._proposals.append(proposal)

    @property
    def proposals(self) -> list[dict[str, Any]]:
        return list(self._proposals)
