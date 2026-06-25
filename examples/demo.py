"""End-to-end demo: simulate a small agent run and print the insights.

Run with:  python -m examples.demo   (from the repo root)
"""

from __future__ import annotations

import json
import time

import contextdb as cdb


# 1. Decorate the agent's tools. Every call is now logged automatically.
@cdb.log_tool
def web_search(query: str) -> list[str]:
    time.sleep(0.01)
    return [f"result for {query!r} #{i}" for i in range(3)]


@cdb.log_tool(name="db.query")
def db_query(sql: str) -> dict:
    time.sleep(0.005)
    if "DROP" in sql.upper():
        raise ValueError("refusing destructive query")
    return {"rows": 42}


def run_conversation(prev_session_id: str | None = None) -> cdb.Session:
    with cdb.session(parent_session_id=prev_session_id) as s:
        s.log_system_prompt("You are a helpful research agent.")

        # Context carried in from a previous conversation + memory.
        if prev_session_id:
            s.log_context(
                [
                    cdb.ContextRef(
                        "previous_session", prev_session_id,
                        "summary of the earlier retries discussion", tokens=180,
                    ),
                    cdb.ContextRef(
                        "memory", "mem-7",
                        "user prefers concise answers", tokens=20,
                    ),
                ]
            )

        s.log_user_message("Find docs about HTTP retry strategies and count them.")

        # A turn groups the nested thinking/tool calls under one parent.
        with s.turn("agent_turn"):
            s.log_thinking("I'll search the web, then query our docs DB.")
            web_search("HTTP retry strategies")
            db_query("SELECT count(*) FROM docs WHERE topic='retries'")
            try:
                db_query("DROP TABLE docs")  # logged as an error event
            except ValueError:
                pass

        s.log_response("Found 3 web results and 42 internal docs on retries.")
        s.log_change("wrote summary to report.md", target="report.md")
        return s


def main() -> None:
    first = run_conversation()
    second = run_conversation(prev_session_id=first.id)

    ins = cdb.insights()

    print("=" * 70)
    print("STORE SUMMARY")
    print(json.dumps(ins.summary(), indent=2))

    print("=" * 70)
    print(f"SESSION REPORT (second run, continues {first.id[:8]})")
    print(json.dumps(ins.session_report(second.id).to_dict(), indent=2))

    print("=" * 70)
    print("CONTEXT FLOW (what was carried into the second run)")
    print(json.dumps(ins.context_flow(second.id), indent=2))

    # Snapshot the in-memory DB for later SQL analysis.
    cdb.get_store().to_sqlite("/tmp/contextdb_demo.sqlite")
    print("=" * 70)
    print("Snapshot written to /tmp/contextdb_demo.sqlite")
    print(f"tiktoken available: {cdb.using_tiktoken()}")


if __name__ == "__main__":
    main()
