"""Live dashboard demo — watch your agent work in real time.

Run with:  python -m examples.live_demo

It launches the dashboard (opens your browser), then simulates an agent
working through several conversations, emitting events one at a time with
small pauses so you can watch them stream into the monitor.
"""

from __future__ import annotations

import random
import time

import contextdb as cdb

PAUSE = 0.9  # seconds between events, so the stream is watchable


def beat(scale: float = 1.0) -> None:
    time.sleep(PAUSE * scale)


@cdb.log_tool
def web_search(query: str) -> list[str]:
    beat(0.4)
    return [f"{query} — result {i}" for i in range(3)]


@cdb.log_tool(name="db.query")
def db_query(sql: str) -> dict:
    beat(0.3)
    if "DROP" in sql.upper():
        raise ValueError("refusing destructive query")
    return {"rows": random.randint(1, 99)}


@cdb.log_tool(name="fs.write")
def fs_write(path: str, content: str) -> str:
    beat(0.2)
    return f"wrote {len(content)} bytes to {path}"


TASKS = [
    "Summarize our HTTP retry strategy docs.",
    "Compare exponential vs. fixed backoff and recommend one.",
    "Draft a runbook section about retry storms.",
]


def run_conversation(task: str, prev_id: str | None) -> cdb.Session:
    with cdb.session(parent_session_id=prev_id) as s:
        s.log_system_prompt("You are a research agent that cites internal docs.")
        beat()

        if prev_id:
            s.log_context(
                [
                    cdb.ContextRef("previous_session", prev_id,
                                   "carried-over summary of the prior task", tokens=random.randint(120, 260)),
                    cdb.ContextRef("memory", "mem-prefs",
                                   "user prefers concise, cited answers", tokens=24),
                ]
            )
            beat()

        s.log_user_message(task)
        beat()

        with s.turn("agent_turn"):
            s.log_thinking("Break the task down: search docs, query counts, then write.")
            beat()
            web_search(task)
            beat(0.5)
            db_query("SELECT count(*) FROM docs WHERE topic='retries'")
            beat(0.5)
            if random.random() < 0.4:
                try:
                    db_query("DROP TABLE docs")  # will be logged as an error
                except ValueError:
                    pass
                beat(0.5)
            fs_write("report.md", "## Retries\n" + "x" * 400)
            beat(0.5)

        s.log_response(f"Done: '{task}'. Wrote findings to report.md.")
        s.log_change("updated report.md", target="report.md")
        beat()
        return s


def main() -> None:
    cdb.serve(open_browser=True)  # non-blocking; dashboard runs in background
    print("Watch the browser. Simulating agent work…  (Ctrl-C to stop)\n")
    time.sleep(1.5)

    prev_id = None
    try:
        for task in TASKS:
            print(f">>> task: {task}")
            s = run_conversation(task, prev_id)
            prev_id = s.id
            beat(2)
        print("\nDone. The dashboard stays live — leave this process running to keep viewing.")
        print("Press Ctrl-C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
