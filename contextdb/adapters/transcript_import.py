"""Import a Claude Code session transcript (.jsonl) into a contextdb store.

Claude Code records each session as newline-delimited JSON under
``~/.claude/projects/<slug>/<session>.jsonl``. Each line is a user/assistant
turn whose ``message.content`` is a list of blocks (text / thinking / tool_use),
with tool results arriving as ``tool_result`` blocks in subsequent user turns,
plus per-turn token ``usage``.

This importer replays a transcript into a store so you can run the same
``Insights`` (token usage, tool stats, context carryover) over your *real* past
sessions. The parser is deliberately tolerant — transcript schemas vary across
Claude Code versions, so every field access is defensive.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from ..models import Event, EventType, TokenUsage
from ..store import EventStore
from ..tokens import count_tokens


def _content_blocks(message: Any) -> list:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def import_transcript(
    path: str, store: Optional[EventStore] = None, session_id: Optional[str] = None
) -> tuple[EventStore, str]:
    """Load ``path`` into ``store`` (new one if omitted). Returns (store, session_id)."""

    store = store if store is not None else EventStore()
    sid = session_id or os.path.splitext(os.path.basename(path))[0]
    # Map tool_use_id -> its TOOL_CALL event so we can back-fill results.
    pending_tools: dict[str, Event] = {}

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            message = entry.get("message") or {}
            role = message.get("role") or entry.get("type") or ""
            usage = message.get("usage") or {}
            in_tok = int(usage.get("input_tokens", 0) or 0)
            out_tok = int(usage.get("output_tokens", 0) or 0)

            for block in _content_blocks(message):
                bt = block.get("type") if isinstance(block, dict) else None

                if bt == "text" and role == "user":
                    store.add(Event(EventType.USER_MESSAGE, sid, content=block.get("text", ""),
                                    tokens=TokenUsage(input_tokens=count_tokens(block.get("text", ""))),
                                    metadata={"source": "transcript"}))
                elif bt == "text" and role == "assistant":
                    store.add(Event(EventType.ASSISTANT_RESPONSE, sid, content=block.get("text", ""),
                                    tokens=TokenUsage(output_tokens=out_tok or count_tokens(block.get("text", ""))),
                                    metadata={"source": "transcript"}))
                elif bt in ("thinking", "redacted_thinking"):
                    text = block.get("thinking") or block.get("data") or "[redacted]"
                    store.add(Event(EventType.THINKING, sid, content=text,
                                    tokens=TokenUsage(output_tokens=count_tokens(text)),
                                    metadata={"source": "transcript"}))
                elif bt == "tool_use":
                    ev = store.add(Event(EventType.TOOL_CALL, sid, name=block.get("name", "tool"),
                                         input_data=block.get("input"),
                                         tokens=TokenUsage(input_tokens=count_tokens(block.get("input"))),
                                         metadata={"source": "transcript", "tool_use_id": block.get("id")}))
                    if block.get("id"):
                        pending_tools[block["id"]] = ev
                elif bt == "tool_result":
                    tid = block.get("tool_use_id")
                    result = block.get("content")
                    target = pending_tools.get(tid)
                    if target is not None:
                        target.output_data = result
                        target.tokens.output_tokens = count_tokens(result)
                    else:
                        store.add(Event(EventType.CHANGE, sid, name="tool_result",
                                        output_data=result, metadata={"source": "transcript", "tool_use_id": tid}))

            # Some turns carry usage but no text block (e.g. tool-only); make
            # sure prompt-side tokens still land somewhere.
            if in_tok and role == "assistant" and not any(
                isinstance(b, dict) and b.get("type") == "text" for b in _content_blocks(message)
            ):
                store.add(Event(EventType.ASSISTANT_RESPONSE, sid, content="",
                                tokens=TokenUsage(input_tokens=in_tok, output_tokens=out_tok),
                                metadata={"source": "transcript", "tool_only_turn": True}))

    return store, sid


def main() -> None:
    import argparse

    from ..insights import Insights

    ap = argparse.ArgumentParser(description="Import a Claude Code .jsonl transcript")
    ap.add_argument("path", help="path to a session .jsonl transcript")
    args = ap.parse_args()

    store, sid = import_transcript(args.path)
    rep = Insights(store).session_report(sid)
    print(json.dumps(rep.to_dict(), indent=2))


if __name__ == "__main__":
    main()
