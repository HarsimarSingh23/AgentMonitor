"""Interactive control demo — drive a live agent from the dashboard.

Run with:  python -m examples.control_demo

It launches the dashboard and then loops an agent forever. While it runs, use
the dashboard control bar to:

  * ⏸ pause / ▶ resume / ⤼ step the agent
  * add a breakpoint on a tool (e.g. ``web_search``) to auto-pause before it
  * patch a tool: force ``db.query`` to return ``{"rows": 0}``, or inject an
    error into ``web_search`` (change the scenario live)
  * queue a context injection — it lands at the start of the next turn
  * ⏹ abort to stop the current run (the loop then starts a fresh one)
"""

from __future__ import annotations

import random
import time

import contextdb as cdb

PAUSE = 0.8


def beat(scale: float = 1.0) -> None:
    time.sleep(PAUSE * scale)


@cdb.log_tool
def web_search(query: str) -> list[str]:
    beat(0.4)
    return [f"{query} — result {i}" for i in range(3)]


@cdb.log_tool(name="db.query")
def db_query(sql: str) -> dict:
    beat(0.3)
    return {"rows": random.randint(1, 99)}


@cdb.log_tool(name="fs.write")
def fs_write(path: str, content: str) -> str:
    beat(0.2)
    return f"wrote {len(content)} bytes to {path}"


TASKS = [
    "Summarize our HTTP retry strategy docs.",
    "Compare exponential vs fixed backoff.",
    "Draft a runbook section on retry storms.",
    "Audit which services lack jittered backoff.",
]


def run_conversation(task: str, prev_id):
    with cdb.session(parent_session_id=prev_id) as s:
        s.log_system_prompt("You are a research agent.")
        beat()
        if prev_id:
            s.log_context([cdb.ContextRef("previous_session", prev_id,
                                          "carried summary of prior task", tokens=random.randint(120, 240))])
            beat()
        s.log_user_message(task)
        beat()
        # turn() is a checkpoint: pausing/stepping blocks here, and any context
        # queued from the dashboard is injected at this point.
        with s.turn("agent_turn"):
            s.log_thinking("Plan: search, count, write.")
            beat()
            web_search(task)          # breakpoint/patch target
            beat(0.4)
            db_query("SELECT count(*) FROM docs WHERE topic='retries'")  # patch target
            beat(0.4)
            fs_write("report.md", "## Retries\n" + "x" * 300)
            beat(0.4)
        s.log_response(f"Done: {task}")
        s.log_change("updated report.md", target="report.md")
        beat()
        return s


def main() -> None:
    cdb.serve(open_browser=True)
    print("Dashboard live. Use the control bar to pause/patch/inject. Ctrl-C to quit.\n")
    time.sleep(1.5)
    prev_id = None
    i = 0
    try:
        while True:
            task = TASKS[i % len(TASKS)]
            i += 1
            print(f">>> task: {task}")
            try:
                s = run_conversation(task, prev_id)
                prev_id = s.id
            except cdb.AgentAborted as e:
                print(f"    [aborted] {e} — resetting and starting a fresh run")
                cdb.get_control().clear_abort()
                prev_id = None
            beat(2)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
