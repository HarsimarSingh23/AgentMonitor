"""Tests for the adapters: Anthropic SDK, Claude Code hooks, transcript import."""

from __future__ import annotations

import os

import contextdb as cdb
from contextdb.adapters import anthropic_sdk as asdk
from contextdb.adapters import claude_code_hooks as cch
from contextdb.adapters.transcript_import import import_transcript
from contextdb.control import ControlPlane
from contextdb.dashboard import _event_from_payload
from contextdb.models import EventType
from contextdb.store import EventStore

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_transcript.jsonl")


def fresh() -> EventStore:
    store = EventStore()
    cdb.set_store(store)
    cdb.set_control(ControlPlane())
    return store


# --- Anthropic SDK adapter -------------------------------------------------

def test_anthropic_logs_thinking_tool_and_text():
    store = fresh()
    with cdb.session("a") as s:
        asdk.log_response(s, asdk.mock_response())
    types = [e.type for e in store.events_for("a")]
    assert EventType.THINKING in types
    assert EventType.TOOL_CALL in types
    assert EventType.ASSISTANT_RESPONSE in types


def test_anthropic_attributes_usage_tokens():
    store = fresh()
    with cdb.session("a") as s:
        asdk.log_response(s, asdk.mock_response())
    resp = [e for e in store.events_for("a") if e.type == EventType.ASSISTANT_RESPONSE][0]
    assert resp.tokens.output_tokens == 95  # from mock usage
    assert resp.metadata["usage_input_tokens"] == 320


def test_anthropic_tool_use_recorded_with_id():
    store = fresh()
    with cdb.session("a") as s:
        asdk.log_response(s, asdk.mock_response())
    tool = [e for e in store.events_for("a") if e.type == EventType.TOOL_CALL][0]
    assert tool.name == "web_search"
    assert tool.metadata["tool_use_id"] == "toolu_1"
    assert tool.input_data == {"query": "HTTP retry backoff strategies"}


def test_anthropic_tool_result_backfills_output():
    store = fresh()
    with cdb.session("a") as s:
        asdk.log_response(s, asdk.mock_response())
        asdk.log_tool_result(s, "toolu_1", ["result A", "result B"])
    tool = [e for e in store.events_for("a") if e.type == EventType.TOOL_CALL][0]
    assert tool.output_data == ["result A", "result B"]


def test_anthropic_streaming_logs_blocks():
    store = fresh()
    with cdb.session("a") as s:
        asdk.stream_log(s, asdk.mock_stream())
    types = [e.type for e in store.events_for("a")]
    assert EventType.THINKING in types and EventType.ASSISTANT_RESPONSE in types
    thinking = [e for e in store.events_for("a") if e.type == EventType.THINKING][0]
    assert thinking.content == "Plan: search then answer."


def test_anthropic_accepts_dict_or_object():
    store = fresh()

    class Obj:
        pass

    resp = Obj()
    resp.content = [type("B", (), {"type": "text", "text": "hi"})()]
    resp.usage = type("U", (), {"input_tokens": 5, "output_tokens": 2})()
    resp.stop_reason = "end_turn"
    resp.model = "claude"
    with cdb.session("a") as s:
        asdk.log_response(s, resp)
    assert any(e.content == "hi" for e in store.events_for("a"))


# --- Claude Code hooks adapter ---------------------------------------------

def test_hook_user_prompt_maps_to_user_message():
    ev = cch.hook_to_event({"hook_event_name": "UserPromptSubmit", "session_id": "x", "prompt": "hello"})
    assert ev["type"] == "user_message" and ev["content"] == "hello" and ev["session_id"] == "x"


def test_hook_post_tool_use_maps_to_tool_call():
    ev = cch.hook_to_event({
        "hook_event_name": "PostToolUse", "session_id": "x",
        "tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_response": "a\nb",
    })
    assert ev["type"] == "tool_call" and ev["name"] == "Bash"
    assert ev["input_data"] == {"command": "ls"} and ev["output_data"] == "a\nb"


def test_hook_stop_and_unknown():
    assert cch.hook_to_event({"hook_event_name": "Stop", "session_id": "x"})["type"] == "assistant_response"
    assert cch.hook_to_event({"hook_event_name": "Notification"}) is None


def test_hook_settings_snippet_shape():
    snip = cch.settings_snippet()
    assert "hooks" in snip
    assert {"UserPromptSubmit", "PostToolUse", "Stop"} <= set(snip["hooks"].keys())
    cmd = snip["hooks"]["PostToolUse"][0]["hooks"][0]["command"]
    assert "contextdb.adapters.claude_code_hooks" in cmd


def test_ingest_builds_event_from_hook_payload():
    store = fresh()
    payload = cch.hook_to_event({
        "hook_event_name": "PostToolUse", "session_id": "live",
        "tool_name": "Read", "tool_input": {"file_path": "x.py"}, "tool_response": "contents",
    })
    ev = _event_from_payload(payload, store)
    assert ev.type == EventType.TOOL_CALL
    assert ev.session_id == "live"
    assert ev.tokens.input_tokens > 0  # estimated from input_data


# --- Transcript importer ---------------------------------------------------

def test_transcript_import_event_types():
    store, sid = import_transcript(FIXTURE)
    types = {e.type for e in store.events_for(sid)}
    assert {EventType.USER_MESSAGE, EventType.THINKING, EventType.TOOL_CALL,
            EventType.ASSISTANT_RESPONSE} <= types


def test_transcript_tool_results_backfilled():
    store, sid = import_transcript(FIXTURE)
    grep = [e for e in store.events_for(sid) if e.name == "Grep"][0]
    assert grep.output_data == "docs/retries.md\ndocs/backoff.md"


def test_transcript_insights_run():
    store, sid = import_transcript(FIXTURE)
    rep = cdb.Insights(store).session_report(sid)
    assert rep.input_tokens > 0 and rep.output_tokens > 0
    tool_names = {t.name for t in rep.tool_stats}
    assert {"Grep", "Read"} <= tool_names


def test_transcript_custom_store_and_session_id():
    store = fresh()
    store2, sid = import_transcript(FIXTURE, store=store, session_id="myrun")
    assert store2 is store and sid == "myrun"
    assert len(store.events_for("myrun")) > 0
