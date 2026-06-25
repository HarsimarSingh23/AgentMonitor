"""Log an Anthropic agent's reasoning into the live dashboard.

Run with:  python -m examples.anthropic_demo

Uses the bundled mock response so it runs with no API key. To use the real API,
set ANTHROPIC_API_KEY and swap ``mock_response()`` for a real
``client.messages.create(..., thinking={"type": "enabled", ...})`` call, then
pass the response to ``log_response``.
"""

from __future__ import annotations

import os
import time

import contextdb as cdb
from contextdb.adapters.anthropic_sdk import log_response, log_tool_result, mock_response

USE_REAL_API = bool(os.environ.get("ANTHROPIC_API_KEY"))


def get_response(user_text: str):
    if not USE_REAL_API:
        return mock_response()
    import anthropic  # type: ignore

    client = anthropic.Anthropic()
    return client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        thinking={"type": "enabled", "budget_tokens": 2048},
        messages=[{"role": "user", "content": user_text}],
    )


def main() -> None:
    cdb.serve(open_browser=True)
    time.sleep(1.5)
    print(f"Logging an Anthropic turn (real API: {USE_REAL_API}). Watch the dashboard.\n")

    with cdb.session() as s:
        s.log_user_message("Find docs about HTTP retry strategies and summarize.")
        time.sleep(0.8)
        resp = get_response("Find docs about HTTP retry strategies and summarize.")
        events = log_response(s, resp)  # logs thinking + tool_use + text
        time.sleep(0.8)
        # Pretend we executed the requested tool and feed the result back.
        for ev in events:
            if ev.type.value == "tool_call":
                log_tool_result(s, ev.metadata.get("tool_use_id"),
                                ["docs/retries.md", "docs/backoff.md"])

    print(cdb.insights().session_report(s.id).to_dict())
    print("\nDashboard stays live. Ctrl-C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
