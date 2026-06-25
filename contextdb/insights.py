"""Analytics over the event log.

Turns the raw timeline into the answers the project is about:
  * how are the agents working (tool usage, latency, errors)?
  * how is the context budget being spent?
  * what context was carried in from previous turns/conversations, and how
    much did it cost?
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import EventType
from .store import EventStore


@dataclass
class ToolStat:
    name: str
    calls: int = 0
    errors: int = 0
    total_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.calls if self.calls else 0.0


@dataclass
class SessionReport:
    session_id: str
    event_count: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    carried_context_tokens: int = 0
    context_sources: dict[str, int] = field(default_factory=dict)
    tool_stats: list[ToolStat] = field(default_factory=list)
    errors: int = 0
    duration_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def context_carryover_ratio(self) -> float:
        """Share of input tokens that came from carried-in context.

        High values mean most of the prompt budget is spent re-feeding old
        context rather than the user's actual new input.
        """

        return self.carried_context_tokens / self.input_tokens if self.input_tokens else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "event_count": self.event_count,
            "by_type": self.by_type,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "carried_context_tokens": self.carried_context_tokens,
            "context_carryover_ratio": round(self.context_carryover_ratio, 3),
            "context_sources": self.context_sources,
            "errors": self.errors,
            "duration_ms": round(self.duration_ms, 2),
            "tool_stats": [
                {
                    "name": t.name,
                    "calls": t.calls,
                    "errors": t.errors,
                    "avg_ms": round(t.avg_ms, 2),
                    "error_rate": round(t.error_rate, 3),
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                }
                for t in sorted(self.tool_stats, key=lambda s: s.calls, reverse=True)
            ],
        }


class Insights:
    """Read-only analytics over an :class:`EventStore`."""

    def __init__(self, store: EventStore) -> None:
        self.store = store

    def session_report(self, session_id: str) -> SessionReport:
        events = self.store.events_for(session_id)
        report = SessionReport(session_id=session_id, event_count=len(events))
        by_type: Counter[str] = Counter()
        tools: dict[str, ToolStat] = {}
        sources: Counter[str] = Counter()
        ts_min: Optional[float] = None
        ts_max: Optional[float] = None

        for ev in events:
            by_type[ev.type.value] += 1
            report.input_tokens += ev.tokens.input_tokens
            report.output_tokens += ev.tokens.output_tokens
            if ev.error:
                report.errors += 1
            ts_min = ev.timestamp_ms if ts_min is None else min(ts_min, ev.timestamp_ms)
            ts_max = ev.timestamp_ms if ts_max is None else max(ts_max, ev.timestamp_ms)

            if ev.type == EventType.CONTEXT_INJECTION:
                report.carried_context_tokens += ev.tokens.input_tokens
                for ref in ev.context_refs:
                    sources[ref.source_kind] += ref.tokens

            if ev.type in (EventType.TOOL_CALL, EventType.FUNCTION_CALL) and not ev.metadata.get(
                "is_turn"
            ):
                stat = tools.setdefault(ev.name, ToolStat(name=ev.name))
                stat.calls += 1
                if ev.error:
                    stat.errors += 1
                if ev.duration_ms:
                    stat.total_ms += ev.duration_ms
                stat.input_tokens += ev.tokens.input_tokens
                stat.output_tokens += ev.tokens.output_tokens

        report.by_type = dict(by_type)
        report.context_sources = dict(sources)
        report.tool_stats = list(tools.values())
        if ts_min is not None and ts_max is not None:
            report.duration_ms = ts_max - ts_min
        return report

    def tool_leaderboard(self, session_id: Optional[str] = None) -> list[ToolStat]:
        """Aggregate tool usage across one session or the whole store."""

        sessions = [session_id] if session_id else self.store.session_ids()
        merged: dict[str, ToolStat] = {}
        for sid in sessions:
            for stat in self.session_report(sid).tool_stats:
                m = merged.setdefault(stat.name, ToolStat(name=stat.name))
                m.calls += stat.calls
                m.errors += stat.errors
                m.total_ms += stat.total_ms
                m.input_tokens += stat.input_tokens
                m.output_tokens += stat.output_tokens
        return sorted(merged.values(), key=lambda s: s.calls, reverse=True)

    def context_flow(self, session_id: str) -> list[dict[str, Any]]:
        """The ordered list of context injections in a session.

        Each entry shows what was carried in, from where, and the token cost —
        i.e. the literal answer to "what context was taken from last
        conversations" for this run.
        """

        flow: list[dict[str, Any]] = []
        for ev in self.store.events_for(session_id):
            if ev.type != EventType.CONTEXT_INJECTION:
                continue
            flow.append(
                {
                    "event_id": ev.id,
                    "timestamp_ms": ev.timestamp_ms,
                    "label": ev.name,
                    "total_tokens": ev.tokens.input_tokens,
                    "refs": [r.to_dict() for r in ev.context_refs],
                }
            )
        return flow

    def token_timeline(self, session_id: str) -> list[dict[str, Any]]:
        """Per-event running token total — useful for plotting budget growth."""

        running = 0
        timeline: list[dict[str, Any]] = []
        for ev in self.store.events_for(session_id):
            running += ev.tokens.total
            timeline.append(
                {
                    "event_id": ev.id,
                    "type": ev.type.value,
                    "name": ev.name,
                    "tokens": ev.tokens.total,
                    "cumulative_tokens": running,
                    "timestamp_ms": ev.timestamp_ms,
                }
            )
        return timeline

    def summary(self) -> dict[str, Any]:
        """Store-wide rollup across every session."""

        reports = [self.session_report(sid) for sid in self.store.session_ids()]
        total_in = sum(r.input_tokens for r in reports)
        total_carried = sum(r.carried_context_tokens for r in reports)
        return {
            "sessions": len(reports),
            "events": len(self.store),
            "total_input_tokens": total_in,
            "total_output_tokens": sum(r.output_tokens for r in reports),
            "total_carried_context_tokens": total_carried,
            "overall_carryover_ratio": round(total_carried / total_in, 3) if total_in else 0.0,
            "total_errors": sum(r.errors for r in reports),
        }
