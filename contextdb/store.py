"""The in-memory event store.

Thread-safe, append-only timeline of :class:`Event` objects with indexes by
session. Supports lightweight subscribers (for live dashboards/streaming) and
an optional JSON / SQLite snapshot so an in-memory run can be persisted and
re-loaded for later analysis.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict
from typing import Callable, Iterable, Optional

from .models import Event, EventType


class EventStore:
    """An append-only, thread-safe, in-memory store of agent events."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._by_session: dict[str, list[Event]] = defaultdict(list)
        self._lock = threading.RLock()
        self._subscribers: list[Callable[[Event], None]] = []

    # -- writing -----------------------------------------------------------

    def add(self, event: Event) -> Event:
        with self._lock:
            self._events.append(event)
            self._by_session[event.session_id].append(event)
        for sub in list(self._subscribers):
            try:
                sub(event)
            except Exception:  # noqa: BLE001 - a bad subscriber must not break logging
                pass
        return event

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        """Register a callback fired on every new event. Returns an unsubscribe fn."""

        self._subscribers.append(callback)

        def _unsub() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return _unsub

    # -- reading -----------------------------------------------------------

    def all(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def session_ids(self) -> list[str]:
        with self._lock:
            return list(self._by_session.keys())

    def events_for(self, session_id: str) -> list[Event]:
        with self._lock:
            return list(self._by_session.get(session_id, []))

    def get(self, event_id: str) -> Optional[Event]:
        with self._lock:
            for ev in reversed(self._events):
                if ev.id == event_id:
                    return ev
        return None

    def query(
        self,
        *,
        session_id: Optional[str] = None,
        types: Optional[Iterable[EventType]] = None,
        name: Optional[str] = None,
    ) -> list[Event]:
        type_set = set(types) if types else None
        with self._lock:
            source = (
                self._by_session.get(session_id, [])
                if session_id is not None
                else self._events
            )
            return [
                ev
                for ev in source
                if (type_set is None or ev.type in type_set)
                and (name is None or ev.name == name)
            ]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._by_session.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    # -- persistence (snapshots of the in-memory state) --------------------

    def to_json(self, path: str) -> None:
        with self._lock:
            data = [ev.to_dict() for ev in self._events]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

    def to_sqlite(self, path: str) -> None:
        """Dump the timeline to a SQLite file for ad-hoc SQL analysis."""

        conn = sqlite3.connect(path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    type TEXT,
                    name TEXT,
                    timestamp_ms REAL,
                    duration_ms REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    parent_id TEXT,
                    error TEXT,
                    payload TEXT
                )
                """
            )
            with self._lock:
                events = list(self._events)
            rows = [
                (
                    ev.id,
                    ev.session_id,
                    ev.type.value,
                    ev.name,
                    ev.timestamp_ms,
                    ev.duration_ms,
                    ev.tokens.input_tokens,
                    ev.tokens.output_tokens,
                    ev.parent_id,
                    ev.error,
                    json.dumps(ev.to_dict(), default=str),
                )
                for ev in events
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
            )
            conn.commit()
        finally:
            conn.close()
