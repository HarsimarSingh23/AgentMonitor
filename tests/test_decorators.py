"""Tests for the @log_tool / @log_function decorators."""

from __future__ import annotations

import asyncio

import contextdb as cdb
from contextdb.control import ControlPlane
from contextdb.models import EventType
from contextdb.store import EventStore


def fresh() -> EventStore:
    store = EventStore()
    cdb.set_store(store)
    cdb.set_control(ControlPlane())
    return store


def test_records_input_and_output():
    store = fresh()

    @cdb.log_tool
    def add(a, b):
        return a + b

    with cdb.session("s"):
        add(2, 3)

    ev = store.events_for("s")[0]
    assert ev.type == EventType.TOOL_CALL
    assert ev.input_data["args"] == [2, 3]
    assert ev.output_data == 5
    assert ev.duration_ms is not None


def test_record_io_false_omits_payloads():
    store = fresh()

    @cdb.log_tool(record_io=False)
    def secret(password):
        return "ok"

    with cdb.session("s"):
        secret("hunter2")

    ev = store.events_for("s")[0]
    assert ev.input_data is None
    assert ev.output_data is None
    assert ev.tokens.input_tokens == 0


def test_max_chars_truncates_large_inputs():
    store = fresh()

    @cdb.log_tool(max_chars=20)
    def big(blob):
        return "done"

    with cdb.session("s"):
        big("x" * 5000)

    ev = store.events_for("s")[0]
    arg0 = ev.input_data["args"][0]
    assert "truncated" in arg0
    assert len(arg0) < 100


def test_name_override():
    store = fresh()

    @cdb.log_tool(name="web.search")
    def search(q):
        return []

    with cdb.session("s"):
        search("hi")

    assert store.events_for("s")[0].name == "web.search"


def test_function_call_type_tag():
    store = fresh()

    @cdb.log_function
    def helper():
        return 1

    with cdb.session("s"):
        helper()

    assert store.events_for("s")[0].type == EventType.FUNCTION_CALL


def test_error_is_logged_and_reraised():
    store = fresh()

    @cdb.log_tool
    def boom():
        raise ValueError("kaboom")

    with cdb.session("s"):
        try:
            boom()
            assert False, "should raise"
        except ValueError:
            pass

    ev = store.events_for("s")[0]
    assert ev.error is not None and "kaboom" in ev.error
    assert ev.output_data is None


def test_implicit_session_when_none_active():
    store = fresh()

    @cdb.log_tool
    def standalone():
        return 1

    # No active session — should still be logged under the implicit session.
    standalone()
    assert "implicit" in store.session_ids()
    assert any(e.name == "standalone" for e in store.events_for("implicit"))


def test_async_records_output():
    store = fresh()

    @cdb.log_tool
    async def fetch(x):
        await asyncio.sleep(0)
        return x * 2

    with cdb.session("s"):
        assert asyncio.run(fetch(21)) == 42

    ev = store.events_for("s")[0]
    assert ev.output_data == 42
    assert ev.name == "fetch"


def test_metadata_preserved_on_functools_wraps():
    fresh()

    @cdb.log_tool
    def documented():
        """my docstring"""
        return 1

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "my docstring"
