"""Adapter that turns Claude Code hook events into contextdb events.

Claude Code can run a command on lifecycle events (UserPromptSubmit, PreToolUse,
PostToolUse, Stop, ...). The harness pipes a JSON payload to that command on
stdin. This module reads that payload and POSTs a normalized event to a running
contextdb dashboard (``/ingest``), so you can **watch a live Claude Code agent**
in the monitor.

Important: hooks give you tool calls, inputs, results, prompts, and token usage
from the transcript — *not* the model's hidden chain-of-thought (that never
leaves Anthropic's servers). For reasoning capture, use the Anthropic SDK
adapter with extended thinking enabled.

Setup
-----
1. Start the collector:        ``python -m contextdb``
2. Register the hook (see ``settings_snippet()`` or the README). Each hook runs:
       ``python -m contextdb.adapters.claude_code_hooks``
3. Work in Claude Code and watch http://127.0.0.1:8765 .

The hook never raises into Claude Code: any error (e.g. collector not running)
is swallowed so it can't disrupt your session.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any, Optional

DEFAULT_URL = os.environ.get("CONTEXTDB_DASHBOARD_URL", "http://127.0.0.1:8765")


def _post(payload: dict, url: str = DEFAULT_URL, timeout: float = 1.5) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/ingest", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    urllib.request.urlopen(req, timeout=timeout).read()


def hook_to_event(hook: dict) -> Optional[dict]:
    """Map a Claude Code hook payload to an /ingest event payload.

    Returns ``None`` for hook events we don't record.
    """

    name = hook.get("hook_event_name") or hook.get("hookEventName") or ""
    session_id = hook.get("session_id") or hook.get("sessionId") or "claude-code"

    if name == "UserPromptSubmit":
        return {
            "type": "user_message",
            "session_id": session_id,
            "content": hook.get("prompt", ""),
            "metadata": {"source": "claude_code", "cwd": hook.get("cwd")},
        }
    if name == "PostToolUse":
        return {
            "type": "tool_call",
            "session_id": session_id,
            "name": hook.get("tool_name", "tool"),
            "input_data": hook.get("tool_input"),
            "output_data": hook.get("tool_response"),
            "metadata": {"source": "claude_code", "hook": name},
        }
    if name == "PreToolUse":
        # Optional: a lightweight "about to call" marker (no result yet).
        return {
            "type": "function_call",
            "session_id": session_id,
            "name": hook.get("tool_name", "tool"),
            "input_data": hook.get("tool_input"),
            "metadata": {"source": "claude_code", "hook": name, "phase": "pre"},
        }
    if name in ("Stop", "SubagentStop"):
        return {
            "type": "assistant_response",
            "session_id": session_id,
            "name": name,
            "content": "[turn complete]",
            "metadata": {"source": "claude_code", "hook": name},
        }
    return None


def settings_snippet(record_pre: bool = False) -> dict:
    """Return a ``.claude/settings.json`` ``hooks`` block wiring up this adapter."""

    cmd = f"{sys.executable} -m contextdb.adapters.claude_code_hooks"
    one = [{"hooks": [{"type": "command", "command": cmd}]}]
    hooks = {"UserPromptSubmit": one, "PostToolUse": one, "Stop": one}
    if record_pre:
        hooks["PreToolUse"] = one
    return {"hooks": hooks}


def main() -> None:
    """Hook entry point: read the payload from stdin, ship it, never crash."""

    try:
        raw = sys.stdin.read()
        hook = json.loads(raw) if raw.strip() else {}
        event = hook_to_event(hook)
        if event is not None:
            _post(event)
    except Exception:  # noqa: BLE001 - a hook must never break the user's session
        pass
    # Exit 0 so Claude Code proceeds normally regardless of collector state.
    sys.exit(0)


if __name__ == "__main__":
    main()
