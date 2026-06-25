"""Tests for the EventStore: querying, indexing, subscribers, snapshots."""

from __future__ import annotations

import json

import contextdb as cdb
from contextdb.control import ControlPlane
from contextdb.models import Event, EventType, TokenUsage
from contextdb.store import EventStore


def fresh() -> EventStore:
    store = EventStore()
    cdb.set_store(store)
    cdb.set_control(ControlPlane())
    return store


def _ev(session_id: str, type_=EventType.TOOL_CALL, name="t") -> Event:
    return Event(type=type_, session_id=session_id, name=name)


def test_len_and_all():
    store = fresh()
    assert len(store) == 0
    store.add(_ev("s1"))
    store.add(_ev("s1"))
    assert len(store) == 2
    assert len(store.all()) == 2


def test_events_grouped_by_session():
    store = fresh()
    store.add(_ev("s1"))
    store.add(_ev("s2"))
    store.add(_ev("s1"))
    assert len(store.events_for("s1")) == 2
    assert len(store.events_for("s2")) == 1
    assert set(store.session_ids()) == {"s1", "s2"}


def test_query_by_type_and_name():
    store = fresh()
    store.add(_ev("s1", EventType.TOOL_CALL, "search"))
    store.add(_ev("s1", EventType.TOOL_CALL, "write"))
    store.add(_ev("s1", EventType.USER_MESSAGE, "msg"))
    assert len(store.query(types=[EventType.TOOL_CALL])) == 2
    assert len(store.query(name="search")) == 1
    assert len(store.query(session_id="s1", types=[EventType.USER_MESSAGE])) == 1


def test_get_by_id():
    store = fresh()
    e = store.add(_ev("s1"))
    assert store.get(e.id) is e
    assert store.get("does-not-exist") is None


def test_subscribe_and_unsubscribe():
    store = fresh()
    seen = []
    unsub = store.subscribe(lambda ev: seen.append(ev.id))
    e1 = store.add(_ev("s1"))
    assert seen == [e1.id]
    unsub()
    store.add(_ev("s1"))
    assert seen == [e1.id]  # no new deliveries after unsubscribe


def test_bad_subscriber_does_not_break_logging():
    store = fresh()

    def boom(_ev):
        raise RuntimeError("subscriber blew up")

    store.subscribe(boom)
    # Should not raise despite the broken subscriber.
    store.add(_ev("s1"))
    assert len(store) == 1


def test_clear():
    store = fresh()
    store.add(_ev("s1"))
    store.clear()
    assert len(store) == 0
    assert store.session_ids() == []


def test_to_json_roundtrips(tmp_path):
    store = fresh()
    ev = Event(
        type=EventType.USER_MESSAGE, session_id="s1", content="hi",
        tokens=TokenUsage(input_tokens=3),
    )
    store.add(ev)
    path = tmp_path / "events.json"
    store.to_json(str(path))
    data = json.loads(path.read_text())
    assert len(data) == 1
    assert data[0]["type"] == "user_message"
    assert data[0]["tokens"]["input_tokens"] == 3
