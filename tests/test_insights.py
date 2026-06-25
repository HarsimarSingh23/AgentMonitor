"""Tests for the Insights analytics layer."""

from __future__ import annotations

import contextdb as cdb
from contextdb.control import ControlPlane
from contextdb.store import EventStore


def fresh() -> EventStore:
    store = EventStore()
    cdb.set_store(store)
    cdb.set_control(ControlPlane())
    return store


def test_session_report_basic_counts():
    store = fresh()
    with cdb.session("s") as s:
        s.log_user_message("hello there friend")
        s.log_response("hi")
    rep = cdb.insights(store).session_report("s")
    assert rep.event_count == 2
    assert rep.input_tokens > 0
    assert rep.output_tokens > 0
    assert rep.total_tokens == rep.input_tokens + rep.output_tokens


def test_tool_stats_error_rate_and_avg():
    store = fresh()

    @cdb.log_tool(name="flaky")
    def flaky(ok):
        if not ok:
            raise RuntimeError("x")
        return 1

    with cdb.session("s"):
        flaky(True)
        try:
            flaky(False)
        except RuntimeError:
            pass

    rep = cdb.insights(store).session_report("s")
    stat = {t.name: t for t in rep.tool_stats}["flaky"]
    assert stat.calls == 2
    assert stat.errors == 1
    assert stat.error_rate == 0.5
    assert stat.avg_ms >= 0


def test_carryover_ratio_and_sources():
    store = fresh()
    with cdb.session("s") as s:
        s.log_context([
            cdb.ContextRef("previous_session", "old", "carried", tokens=300),
            cdb.ContextRef("memory", "m1", "pref", tokens=20),
        ])
        s.log_user_message("tiny new ask")
    rep = cdb.insights(store).session_report("s")
    assert rep.carried_context_tokens == 320
    assert 0 < rep.context_carryover_ratio <= 1
    assert rep.context_sources == {"previous_session": 300, "memory": 20}


def test_carryover_ratio_zero_when_no_input():
    store = fresh()
    with cdb.session("s") as s:
        s.log_response("output only")  # no input tokens
    rep = cdb.insights(store).session_report("s")
    assert rep.input_tokens == 0
    assert rep.context_carryover_ratio == 0.0


def test_tool_leaderboard_aggregates_across_sessions():
    store = fresh()

    @cdb.log_tool
    def shared():
        return 1

    with cdb.session("s1"):
        shared()
        shared()
    with cdb.session("s2"):
        shared()

    board = cdb.insights(store).tool_leaderboard()
    top = {t.name: t for t in board}["shared"]
    assert top.calls == 3


def test_context_flow_lists_injections():
    store = fresh()
    with cdb.session("s") as s:
        s.log_context([cdb.ContextRef("rag", "doc-9", "retrieved chunk", tokens=80)])
    flow = cdb.insights(store).context_flow("s")
    assert len(flow) == 1
    assert flow[0]["total_tokens"] == 80
    assert flow[0]["refs"][0]["source_kind"] == "rag"


def test_token_timeline_is_cumulative():
    store = fresh()
    with cdb.session("s") as s:
        s.log_user_message("one two three four")
        s.log_response("five six seven eight")
    timeline = cdb.insights(store).token_timeline("s")
    assert len(timeline) == 2
    assert timeline[1]["cumulative_tokens"] >= timeline[0]["cumulative_tokens"]
    assert timeline[1]["cumulative_tokens"] == sum(t["tokens"] for t in timeline)


def test_summary_keys_and_rollup():
    store = fresh()
    with cdb.session("s1") as s:
        s.log_context([cdb.ContextRef("previous_session", "x", "c", tokens=100)])
        s.log_user_message("ask something")
    summary = cdb.insights(store).summary()
    for key in (
        "sessions", "events", "total_input_tokens", "total_output_tokens",
        "total_carried_context_tokens", "overall_carryover_ratio", "total_errors",
    ):
        assert key in summary
    assert summary["sessions"] == 1
    assert summary["total_carried_context_tokens"] == 100
