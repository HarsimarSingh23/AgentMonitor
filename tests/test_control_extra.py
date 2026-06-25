"""Deeper control-plane tests: stepping, patches, context queue, dispatch."""

from __future__ import annotations

import threading
import time

import contextdb as cdb
from contextdb.control import PROCEED, RAISE, RETURN, ControlPlane
from contextdb.dashboard import _coerce_value, _dispatch_control
from contextdb.store import EventStore


def fresh() -> EventStore:
    store = EventStore()
    cdb.set_store(store)
    cdb.set_control(ControlPlane())
    return store


def test_step_advances_exactly_one_checkpoint():
    fresh()
    ctrl = cdb.get_control()
    order = []

    @cdb.log_tool
    def a():
        order.append("a")

    @cdb.log_tool
    def b():
        order.append("b")

    ctrl.pause()

    def worker():
        with cdb.session("s"):
            a()
            b()

    threading.Thread(target=worker).start()
    time.sleep(0.2)
    assert order == []  # paused: nothing ran

    ctrl.step(1)
    time.sleep(0.2)
    assert order == ["a"]  # exactly one checkpoint advanced
    assert ctrl.state()["mode"] == "paused"

    ctrl.resume()
    time.sleep(0.2)
    assert order == ["a", "b"]


def test_state_shape():
    fresh()
    st = cdb.get_control().state()
    for key in ("mode", "waiting_at", "abort", "reason", "patches", "breakpoints", "pending_context"):
        assert key in st
    assert st["mode"] == "running"


def test_clear_all_patches():
    fresh()
    ctrl = cdb.get_control()
    ctrl.patch_tool("a", RETURN, value=1)
    ctrl.patch_tool("b", RAISE, error="x")
    assert len(ctrl.state()["patches"]) == 2
    ctrl.clear_patch()  # no name -> clear all
    assert ctrl.state()["patches"] == {}


def test_intercept_proceed_when_no_patch():
    fresh()
    ctrl = cdb.get_control()
    action, payload = ctrl.intercept("nothing")
    assert action == PROCEED and payload is None


def test_context_queue_and_drain():
    fresh()
    ctrl = cdb.get_control()
    ctrl.queue_context("a", "first", tokens=1)
    ctrl.queue_context("b", "second", tokens=2)
    assert len(ctrl.state()["pending_context"]) == 2
    drained = ctrl.drain_context()
    assert len(drained) == 2
    assert ctrl.state()["pending_context"] == []  # drained
    assert ctrl.drain_context() == []  # empty second time


def test_checkpoint_honors_abort():
    fresh()
    ctrl = cdb.get_control()
    ctrl.abort()
    try:
        cdb.checkpoint("manual")
        assert False, "should raise AgentAborted"
    except cdb.AgentAborted:
        pass


def test_coerce_value_parses_json_else_string():
    assert _coerce_value('{"rows": 0}') == {"rows": 0}
    assert _coerce_value("[1, 2, 3]") == [1, 2, 3]
    assert _coerce_value("plain text") == "plain text"
    assert _coerce_value(5) == 5
    assert _coerce_value(None) is None


def test_dispatch_control_commands():
    fresh()
    ctrl = cdb.get_control()

    assert _dispatch_control(ctrl, {"cmd": "pause"})["mode"] == "paused"
    assert _dispatch_control(ctrl, {"cmd": "resume"})["mode"] == "running"

    st = _dispatch_control(
        ctrl, {"cmd": "patch_tool", "name": "db.query", "action": "return", "value": '{"rows": 0}'}
    )
    assert st["patches"]["db.query"]["value"] == {"rows": 0}

    st = _dispatch_control(ctrl, {"cmd": "breakpoint", "name": "web_search", "on": True})
    assert "web_search" in st["breakpoints"]

    st = _dispatch_control(
        ctrl, {"cmd": "queue_context", "source_kind": "ops", "summary": "db degraded", "tokens": 40}
    )
    assert st["pending_context"][0]["summary"] == "db degraded"

    st = _dispatch_control(ctrl, {"cmd": "abort"})
    assert st["abort"] is True
    st = _dispatch_control(ctrl, {"cmd": "clear_abort"})
    assert st["abort"] is False


def test_subscribe_receives_state_changes():
    fresh()
    ctrl = cdb.get_control()
    states = []
    unsub = ctrl.subscribe(lambda st: states.append(st["mode"]))
    ctrl.pause()
    ctrl.resume()
    unsub()
    ctrl.pause()  # not observed after unsub
    assert "paused" in states and "running" in states
    assert states.count("paused") == 1  # only the first pause was observed
