"""Tests for contextdb. Run with: python -m pytest -q  (or python -m tests... )."""

from __future__ import annotations

import asyncio

import threading
import time

import contextdb as cdb
from contextdb.control import ControlPlane
from contextdb.store import EventStore
from contextdb.models import EventType


def fresh_store() -> EventStore:
    store = EventStore()
    cdb.set_store(store)
    cdb.set_control(ControlPlane())  # reset control plane between tests
    return store


def test_basic_logging_and_token_counts():
    store = fresh_store()
    with cdb.session("s1") as s:
        s.log_user_message("hello world")
        s.log_response("hi there")

    rep = cdb.insights(store).session_report("s1")
    assert rep.event_count == 2
    assert rep.input_tokens > 0
    assert rep.output_tokens > 0
    assert rep.by_type[EventType.USER_MESSAGE.value] == 1


def test_tool_decorator_records_calls_and_errors():
    store = fresh_store()

    @cdb.log_tool
    def add(a, b):
        return a + b

    @cdb.log_tool(name="boom")
    def boom():
        raise RuntimeError("nope")

    with cdb.session("s2"):
        assert add(2, 3) == 5
        try:
            boom()
        except RuntimeError:
            pass

    rep = cdb.insights(store).session_report("s2")
    by_name = {t.name: t for t in rep.tool_stats}
    assert by_name["add"].calls == 1
    assert by_name["add"].errors == 0
    assert by_name["boom"].errors == 1
    assert rep.errors == 1


def test_context_carryover_ratio():
    store = fresh_store()
    with cdb.session("s3") as s:
        s.log_context([cdb.ContextRef("previous_session", "old", "carried", tokens=100)])
        s.log_user_message("a short new message")

    rep = cdb.insights(store).session_report("s3")
    assert rep.carried_context_tokens == 100
    assert 0 < rep.context_carryover_ratio <= 1
    assert rep.context_sources["previous_session"] == 100

    flow = cdb.insights(store).context_flow("s3")
    assert len(flow) == 1
    assert flow[0]["total_tokens"] == 100


def test_turns_nest_and_are_excluded_from_tool_stats():
    store = fresh_store()

    @cdb.log_tool
    def t():
        return 1

    with cdb.session("s4") as s:
        with s.turn("turn") as parent:
            t()

    events = store.events_for("s4")
    tool_events = [e for e in events if e.type == EventType.TOOL_CALL]
    assert tool_events[0].parent_id == parent.id

    rep = cdb.insights(store).session_report("s4")
    names = {x.name for x in rep.tool_stats}
    assert "turn" not in names and "t" in names


def test_async_tool():
    store = fresh_store()

    @cdb.log_tool
    async def fetch(x):
        await asyncio.sleep(0)
        return x * 2

    with cdb.session("s5"):
        assert asyncio.run(fetch(21)) == 42

    rep = cdb.insights(store).session_report("s5")
    assert rep.tool_stats[0].name == "fetch"


def test_sqlite_snapshot(tmp_path):
    store = fresh_store()
    with cdb.session("s6") as s:
        s.log_user_message("persist me")
    path = tmp_path / "snap.sqlite"
    store.to_sqlite(str(path))
    import sqlite3

    conn = sqlite3.connect(path)
    n = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    conn.close()
    assert n == 1


def test_control_pause_blocks_then_resume():
    fresh_store()
    ctrl = cdb.get_control()

    @cdb.log_tool
    def t():
        return 1

    ctrl.pause()
    done = threading.Event()

    def worker():
        with cdb.session("p"):
            t()
        done.set()

    threading.Thread(target=worker).start()
    time.sleep(0.2)
    assert not done.is_set()  # gated while paused
    ctrl.resume()
    assert done.wait(timeout=2)


def test_control_fault_injection_and_forced_return():
    fresh_store()
    ctrl = cdb.get_control()

    @cdb.log_tool
    def boom():
        return "real"

    ctrl.patch_tool("boom", "raise", error="x")
    with cdb.session("f"):
        try:
            boom()
            assert False, "should have raised"
        except cdb.PatchedFault as e:
            assert str(e) == "x"

    ctrl.patch_tool("boom", "return", value="patched")
    with cdb.session("r"):
        assert boom() == "patched"


def test_control_breakpoint_and_context_injection():
    store = fresh_store()
    ctrl = cdb.get_control()

    @cdb.log_tool
    def hit():
        return 1

    ctrl.set_breakpoint("hit", True)

    def worker():
        with cdb.session("b"):
            hit()

    threading.Thread(target=worker).start()
    time.sleep(0.2)
    assert ctrl.state()["mode"] == "paused"
    ctrl.set_breakpoint("hit", False)
    ctrl.resume()

    ctrl.queue_context("dashboard_patch", "live note", tokens=10)
    with cdb.session("c") as s:
        with s.turn("agent_turn"):
            pass
    inj = [e for e in store.events_for("c") if e.type == EventType.CONTEXT_INJECTION]
    assert len(inj) == 1 and inj[0].context_refs[0].summary == "live note"


def test_control_abort():
    fresh_store()
    ctrl = cdb.get_control()

    @cdb.log_tool
    def t():
        return 1

    ctrl.abort()
    with cdb.session("a"):
        try:
            t()
            assert False, "should have aborted"
        except cdb.AgentAborted:
            pass
    ctrl.clear_abort()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            import inspect

            if "tmp_path" in inspect.signature(fn).parameters:
                import tempfile, pathlib

                fn(pathlib.Path(tempfile.mkdtemp()))
            else:
                fn()
            print(f"ok  {name}")
    print("all tests passed")
