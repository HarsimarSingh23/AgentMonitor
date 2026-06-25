"""Adapter for the Anthropic Messages API / Agent SDK.

This is the adapter that can log an agent's **actual reasoning**: when extended
thinking is enabled, the Messages API returns ``thinking`` content blocks
alongside ``text`` and ``tool_use`` blocks. This adapter walks a response (or a
streaming response) and records each block into a contextdb session, so you can
watch the model think → decide to call a tool → answer, live in the dashboard.

No hard dependency on the ``anthropic`` package: inputs are duck-typed, so a
response object, a dict, or the bundled ``mock_response()`` all work.

Typical use::

    import anthropic, contextdb as cdb
    from contextdb.adapters.anthropic_sdk import log_response

    client = anthropic.Anthropic()
    with cdb.session() as s:
        s.log_user_message(user_text)
        resp = client.messages.create(
            model="claude-opus-4-8", max_tokens=1024,
            thinking={"type": "enabled", "budget_tokens": 2048},
            messages=[{"role": "user", "content": user_text}],
        )
        log_response(s, resp)
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from ..models import Event, EventType, TokenUsage
from ..session import Session
from ..tokens import count_tokens


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an object attribute or a dict."""

    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _block_type(block: Any) -> str:
    return _get(block, "type", "") or ""


def _usage_tokens(response: Any) -> tuple[int, int]:
    usage = _get(response, "usage")
    return int(_get(usage, "input_tokens", 0) or 0), int(_get(usage, "output_tokens", 0) or 0)


def log_response(session: Session, response: Any, *, name: str = "assistant") -> list[Event]:
    """Log every content block of a (non-streaming) Messages response.

    Returns the events created, in order. Thinking blocks become THINKING
    events, tool_use blocks become TOOL_CALL events (the model's *intent* to
    call a tool, with its arguments), and text becomes an ASSISTANT_RESPONSE.
    The response's usage is attributed to the text event.
    """

    in_tok, out_tok = _usage_tokens(response)
    blocks = _get(response, "content", []) or []
    events: list[Event] = []
    text_parts: list[str] = []
    text_logged = False

    for block in blocks:
        bt = _block_type(block)
        if bt in ("thinking", "redacted_thinking"):
            text = _get(block, "thinking", "") or _get(block, "data", "[redacted]")
            events.append(session.log_thinking(text, source="anthropic"))
        elif bt == "tool_use":
            tool_input = _get(block, "input", {})
            events.append(
                session.log(
                    Event(
                        type=EventType.TOOL_CALL,
                        session_id=session.id,
                        name=_get(block, "name", "tool"),
                        input_data=tool_input,
                        tokens=TokenUsage(input_tokens=count_tokens(tool_input)),
                        metadata={
                            "source": "anthropic",
                            "tool_use_id": _get(block, "id"),
                            "requested": True,  # model asked; not yet executed
                        },
                    )
                )
            )
        elif bt == "text":
            text_parts.append(_get(block, "text", "") or "")

    if text_parts:
        ev = Event(
            type=EventType.ASSISTANT_RESPONSE,
            session_id=session.id,
            name=name,
            content="".join(text_parts),
            tokens=TokenUsage(output_tokens=out_tok or count_tokens("".join(text_parts))),
            metadata={
                "source": "anthropic",
                "stop_reason": _get(response, "stop_reason"),
                "model": _get(response, "model"),
                "usage_input_tokens": in_tok,
            },
        )
        events.append(session.log(ev))
        text_logged = True

    # If the turn was tool-only (no text), still record the prompt-side token
    # cost so the dashboard's input total stays accurate.
    if in_tok and not text_logged:
        events.append(
            session.log(
                Event(
                    type=EventType.ASSISTANT_RESPONSE,
                    session_id=session.id,
                    name=name,
                    content="",
                    tokens=TokenUsage(input_tokens=in_tok, output_tokens=out_tok),
                    metadata={"source": "anthropic", "stop_reason": _get(response, "stop_reason")},
                )
            )
        )
    return events


def log_tool_result(session: Session, tool_use_id: str, result: Any) -> Optional[Event]:
    """Attach a tool's result to the matching tool_use event and echo it.

    Records a CHANGE event carrying the result and back-fills ``output_data`` on
    the earlier TOOL_CALL with the same ``tool_use_id`` (so its i/o panel is
    complete).
    """

    for ev in reversed(session.store.events_for(session.id)):
        if ev.type == EventType.TOOL_CALL and ev.metadata.get("tool_use_id") == tool_use_id:
            ev.output_data = result
            ev.tokens.output_tokens = count_tokens(result)
            name = ev.name
            break
    else:
        name = "tool"
    return session.log(
        Event(
            type=EventType.CHANGE,
            session_id=session.id,
            name=name,
            content="tool_result",
            output_data=result,
            tokens=TokenUsage(input_tokens=count_tokens(result)),
            metadata={"source": "anthropic", "tool_use_id": tool_use_id},
        )
    )


def stream_log(session: Session, stream: Iterable[Any], *, name: str = "assistant") -> list[Event]:
    """Consume a streaming Messages response, logging blocks as they complete.

    Tolerant of both object- and dict-shaped stream events. Each content block
    is accumulated from its deltas and logged when it stops, so thinking and
    text appear in the dashboard as the model produces them.
    """

    events: list[Event] = []
    blocks: dict[int, dict[str, Any]] = {}
    out_tok = 0

    for event in stream:
        et = _get(event, "type", "")
        if et == "content_block_start":
            idx = _get(event, "index", 0)
            blocks[idx] = {"type": _block_type(_get(event, "content_block")), "text": "",
                           "block": _get(event, "content_block")}
        elif et == "content_block_delta":
            idx = _get(event, "index", 0)
            delta = _get(event, "delta")
            dt = _get(delta, "type", "")
            slot = blocks.setdefault(idx, {"type": "", "text": "", "block": None})
            if dt == "thinking_delta":
                slot["type"] = "thinking"
                slot["text"] += _get(delta, "thinking", "") or ""
            elif dt == "text_delta":
                slot["type"] = "text"
                slot["text"] += _get(delta, "text", "") or ""
            elif dt == "input_json_delta":
                slot["type"] = "tool_use"
                slot["text"] += _get(delta, "partial_json", "") or ""
        elif et == "content_block_stop":
            idx = _get(event, "index", 0)
            slot = blocks.get(idx, {})
            bt = slot.get("type")
            if bt == "thinking":
                events.append(session.log_thinking(slot["text"], source="anthropic"))
            elif bt == "text":
                events.append(session.log_response(slot["text"], source="anthropic"))
            elif bt == "tool_use":
                blk = slot.get("block")
                events.append(
                    session.log(
                        Event(
                            type=EventType.TOOL_CALL, session_id=session.id,
                            name=_get(blk, "name", "tool"),
                            input_data=slot.get("text"),
                            metadata={"source": "anthropic", "tool_use_id": _get(blk, "id"),
                                      "requested": True},
                        )
                    )
                )
        elif et == "message_delta":
            usage = _get(event, "usage")
            out_tok = int(_get(usage, "output_tokens", out_tok) or out_tok)

    return events


# ---------------------------------------------------------------------------
# Mocks so the adapter (and its demo/tests) run with no API key.
# ---------------------------------------------------------------------------

def mock_response() -> dict:
    """A realistic Messages response with thinking + tool_use + text blocks."""

    return {
        "id": "msg_mock",
        "model": "claude-opus-4-8",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 320, "output_tokens": 95},
        "content": [
            {"type": "thinking",
             "thinking": "The user wants retry docs. I'll search internal docs, "
                         "then summarize. Let me call the search tool first."},
            {"type": "tool_use", "id": "toolu_1", "name": "web_search",
             "input": {"query": "HTTP retry backoff strategies"}},
            {"type": "text", "text": "Let me look that up for you."},
        ],
    }


def mock_stream():
    """A minimal streaming sequence (thinking delta then text delta)."""

    return [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Plan: "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "search then answer."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Here are the docs."}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "usage": {"output_tokens": 42}},
    ]
